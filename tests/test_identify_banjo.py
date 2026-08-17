"""Tests for tune identification and banjo arranging.

The identification tests matter more than they look: identification is used as a
*validation* mechanism, so if the matcher itself is broken we would misread the
transcription engine's quality. Hence the self-identification tests, which pin
down that a correct melody scores 1.0 against its own catalog entry.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from fiddle.abc_io import parse_abc_sections
from fiddle.banjo import (
    BanjoConfig,
    ROLLS,
    arrange,
    choose_capo,
    chord_tones,
    to_tablature,
)
from fiddle.catalog import builtin_catalog, load_catalog, split_tunebook
from fiddle.domain import Measure, Note, Tune, TuneSection
from fiddle.identify import identify, interval_ngrams


# --------------------------------------------------------------------------
# catalog
# --------------------------------------------------------------------------


def test_builtin_catalog_loads_and_every_tune_has_sections():
    catalog = builtin_catalog()
    assert len(catalog) >= 15
    for tune in catalog:
        assert tune.sections, f"{tune.title} has no sections"
        for name, section in tune.sections.items():
            assert section.notes, f"{tune.title} section {name} is empty"
            assert section.bars > 0


def test_catalog_splits_sections_on_repeat_barlines():
    """Real ABC marks sections with repeats, not P: fields."""
    abc = """X:1
T:Two Parter
M:4/4
L:1/8
K:D
|:d2 dd d2 de|f2 e2 d4:|
|:f2 f2 f2 ga|b2 a2 f4:|
"""
    sections = parse_abc_sections(abc)
    assert set(sections) == {"A", "B"}
    assert sections["A"].bars == 2 and sections["B"].bars == 2


def test_information_fields_do_not_leak_into_the_music():
    """An R: or Z: line falling into the body silently destroys a section."""
    abc = """X:1
T:With Fields
R:reel
Z:transcribed by someone
M:4/4
L:1/8
K:D
|:d2 dd d2 de|f2 e2 d4:|
"""
    sections = parse_abc_sections(abc)
    assert sections["A"].bars == 2
    assert len(sections["A"].notes) == 9


def test_tunebook_splitting_finds_every_tune():
    text = "\n".join(
        f"X:{i}\nT:Tune {i}\nM:4/4\nL:1/8\nK:D\n|:d2 dd d2 de|f2 e2 d4:|\n"
        for i in range(1, 4)
    )
    assert len(split_tunebook(text)) == 3


def test_unreadable_catalog_entries_are_skipped_not_fatal(tmp_path):
    """One bad tune in a thousand-file library must not abort the scan."""
    good = "X:1\nT:Good\nM:4/4\nL:1/8\nK:D\n|:d2 dd d2 de|f2 e2 d4:|\n"
    bad = "X:2\nT:Bad Bars\nM:4/4\nL:1/8\nK:D\n|:d2 dd d|\n"
    jig = "X:3\nT:A Jig\nM:6/8\nL:1/8\nK:D\n|:ded cdc|def gfe:|\n"
    (tmp_path / "book.abc").write_text(good + bad + jig)
    catalog = load_catalog(tmp_path)
    assert [t.title for t in catalog] == ["Good"]


# --------------------------------------------------------------------------
# identification
# --------------------------------------------------------------------------


def _catalog_pitches(catalog, title):
    tune = next(t for t in catalog if t.title == title)
    return [n.pitch for s in tune.abc_sections for n in s.notes]


@pytest.mark.parametrize(
    "title", ["Soldier's Joy", "Old Joe Clark", "Blackberry Blossom", "Cluck Old Hen"]
)
def test_a_catalog_tune_identifies_as_itself(title):
    catalog = builtin_catalog()
    ident = identify(_catalog_pitches(catalog, title), catalog)
    assert ident.best.title == title
    assert ident.best.score == pytest.approx(1.0)
    assert ident.is_confident()


def test_identification_is_transposition_invariant():
    """A jam plays a tune in whatever key it likes, and our key may be off."""
    catalog = builtin_catalog()
    pitches = _catalog_pitches(catalog, "Soldier's Joy")
    for shift in (-5, -2, 3, 7):
        ident = identify([p + shift for p in pitches], catalog)
        assert ident.best.title == "Soldier's Joy"
        assert ident.best.score == pytest.approx(1.0)


def test_identification_survives_transcription_errors():
    """Wrong notes should degrade the score, not destroy the match."""
    catalog = builtin_catalog()
    pitches = list(_catalog_pitches(catalog, "Old Joe Clark"))
    for i in range(0, len(pitches), 7):  # corrupt ~14% of notes
        pitches[i] += 1
    ident = identify(pitches, catalog)
    assert ident.best.title == "Old Joe Clark"
    # An n-gram spans five notes, so corrupting one note in seven damages most
    # of them. What must survive is the *separation* from every other tune.
    assert ident.best.score > 0.3
    assert ident.margin > 0.15
    assert ident.is_confident()


def test_noise_does_not_produce_a_confident_match():
    """The failure mode that matters: never claim a match on a bad transcription."""
    import random

    rng = random.Random(0)
    noise = [rng.randint(60, 84) for _ in range(400)]
    ident = identify(noise, builtin_catalog())
    assert not ident.is_confident()


def test_repeated_notes_alone_do_not_identify_anything():
    """A drone-only transcription must not match the most repetitive tune.

    This was a real bug: unweighted n-gram scoring ranked the same tune top for
    every recording, at an identical score, purely on the all-zeros interval
    pattern. IDF weighting is what fixes it.
    """
    ident = identify([69] * 200, builtin_catalog())
    assert not ident.is_confident()
    assert ident.best.score < 0.25


def test_interval_ngrams_are_transposition_invariant():
    a = interval_ngrams([60, 62, 64, 65, 67, 69])
    b = interval_ngrams([67, 69, 71, 72, 74, 76])
    assert a == b


def test_interval_ngrams_need_enough_notes():
    assert interval_ngrams([60, 62]) == {}


# --------------------------------------------------------------------------
# banjo
# --------------------------------------------------------------------------


def _tune_from_catalog(title):
    catalog = builtin_catalog()
    entry = next(t for t in catalog if t.title == title)
    sections = []
    for name, section in sorted(entry.sections.items()):
        bars: dict[int, Measure] = {}
        for n in section.notes:
            idx = int(float(n.start_beats) // 4)
            bars.setdefault(idx, Measure(notes=[], number=idx + 1)).notes.append(
                Note(pitch=n.pitch, start_beats=n.start_beats,
                     duration_beats=n.duration_beats, confidence=1.0)
            )
        sections.append(TuneSection(name=name,
                                    measures=[bars[k] for k in sorted(bars)],
                                    repeats=2))
    return Tune(key=entry.key, meter=entry.meter, tempo=120.0, sections=sections,
                title=entry.title, analysis={"key_sharps": 2})


def test_capo_follows_banjo_practice():
    from fiddle.banjo import G_TUNING

    assert choose_capo("G major", G_TUNING) == 0
    assert choose_capo("A major", G_TUNING) == 2
    assert choose_capo("A mixolydian", G_TUNING) == 2
    assert choose_capo("C major", G_TUNING) == 5
    # D would need capo 7, which nobody does; play out of open position instead.
    assert choose_capo("D major", G_TUNING) == 0


def test_arrangement_is_a_continuous_eighth_note_stream():
    """Scruggs style is defined by the unbroken roll, so there are no gaps."""
    part = arrange(_tune_from_catalog("Old Joe Clark"))
    assert part.notes
    assert all(n.duration_beats == Fraction(1, 2) for n in part.notes)
    starts = [n.start_beats for n in part.notes]
    assert starts == sorted(starts)


def test_every_note_is_physically_playable():
    for title in ("Old Joe Clark", "Soldier's Joy", "Cluck Old Hen"):
        part = arrange(_tune_from_catalog(title))
        for n in part.notes:
            assert 1 <= n.string <= 5
            assert 0 <= n.fret <= 12, f"{title}: fret {n.fret} on string {n.string}"
            assert n.pitch == part.tuning[n.string] + n.fret


def test_fifth_string_is_a_drone_and_never_carries_melody():
    """Using the fifth string melodically is the giveaway of a fake banjo part."""
    part = arrange(_tune_from_catalog("Soldier's Joy"))
    fifth = [n for n in part.notes if n.string == 5]
    assert fifth, "the drone should appear at all"
    assert not any(n.is_melody for n in fifth)
    assert all(n.fret == 0 for n in fifth)


def test_melody_keeps_its_shape_after_the_octave_shift():
    """Per-note octave folding would turn steps into leaps; one shift must not."""
    tune = _tune_from_catalog("Soldier's Joy")
    original = [n.pitch for s in tune.sections for n in s.notes if not n.is_rest]
    part = arrange(tune)
    arranged = [n.pitch for n in part.notes if n.is_melody]
    # Compare interval shape over the notes that survived into the arrangement.
    orig_steps = [b - a for a, b in zip(original, original[1:])]
    banjo_steps = [b - a for a, b in zip(arranged, arranged[1:])]
    big_leaps = sum(1 for s in banjo_steps if abs(s) >= 11)
    assert big_leaps <= sum(1 for s in orig_steps if abs(s) >= 11) + 2


def test_melody_coverage_is_in_a_musical_range():
    part = arrange(_tune_from_catalog("Old Joe Clark"))
    assert 0.2 <= part.melody_coverage <= 0.75


def test_chord_choice_follows_the_melody():
    # A bar full of D-chord tones in D major should pick I, not IV or V.
    assert set(chord_tones("D major", [62, 66, 69, 74])) == {2, 6, 9}
    # A bar of A-chord tones should pick V.
    assert set(chord_tones("D major", [69, 73, 76])) == {9, 1, 4}


def test_arrangement_is_deterministic():
    """Regenerating a part must give the same tab every time."""
    a = to_tablature(arrange(_tune_from_catalog("Soldier's Joy")))
    b = to_tablature(arrange(_tune_from_catalog("Soldier's Joy")))
    assert a == b


def test_all_rolls_are_the_right_length_and_use_real_strings():
    for name, roll in ROLLS.items():
        assert len(roll) == 8, name
        assert all(1 <= s <= 5 for s in roll), name


def test_tablature_renders_five_strings():
    part = arrange(_tune_from_catalog("Cluck Old Hen"), BanjoConfig(vary_rolls=False))
    tab = to_tablature(part, bars_per_line=2)
    assert tab.count("\nD|") >= 2  # first and fourth strings
    assert "\ng|" in tab
    assert "T = thumb" in tab
