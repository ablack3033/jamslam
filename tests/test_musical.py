"""Tests for the musical-inference stages: key, form, consensus, export.

These are the modules encoding domain knowledge, so the tests are written as
statements about old-time music rather than about code: a mixolydian tune must
not be called major, a section must not be confused with a phrase, and three
players agreeing must outvote one who did not.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from fiddle.config import ConsensusConfig, FormConfig, KeyConfig
from fiddle.consensus import build_section
from fiddle.domain import Measure, Note, TimedNote, Tune, TuneSection
from fiddle.form import SectionInstance, analyze_form, slot_similarity
from fiddle.key import infer_key


def _tn(pitch, start, dur=Fraction(1, 2), conf=0.9):
    return TimedNote(
        pitch=pitch, start_beats=Fraction(start).limit_denominator(64),
        duration_beats=Fraction(dur).limit_denominator(64), confidence=conf,
        start_sec=float(start) * 0.5, duration_sec=float(dur) * 0.5,
        pitch_midi=float(pitch),
    )


def _sequence(pitches, start_beat=0.0, step=Fraction(1, 2)):
    return [_tn(p, Fraction(start_beat) + i * step, step)
            for i, p in enumerate(pitches)]


# --------------------------------------------------------------------------
# key
# --------------------------------------------------------------------------


def test_key_detects_plain_d_major():
    # D E F# G A B C# D
    notes = _sequence([62, 64, 66, 67, 69, 71, 73, 74] * 3)
    notes.append(_tn(62, 40, Fraction(2)))  # land on the tonic
    assert infer_key(notes).key == "D major"


def test_key_detects_mixolydian_rather_than_forcing_major():
    """Old Joe Clark is A mixolydian; calling it A major puts a wrong G# on the page.

    This is the single most common key-detection failure on this repertoire, so
    it gets an explicit test.
    """
    # A B C# D E F# G, tonic A, with the flat seventh (G natural) prominent.
    pitches = [69, 71, 73, 74, 76, 78, 79, 69, 79, 76, 69, 79, 74, 69]
    notes = _sequence(pitches * 3)
    notes.append(_tn(69, 60, Fraction(2)))
    analysis = infer_key(notes)
    assert analysis.mode == "mixolydian"
    assert analysis.key == "A mixolydian"


def test_mixolydian_gets_its_parent_major_key_signature():
    """A mixolydian is the fifth mode of D major, so two sharps -- not A major's three.

    Getting this wrong is what puts a G# on every G in an Old Joe Clark
    transcription.
    """
    pitches = [69, 71, 73, 74, 76, 78, 79] * 4
    notes = _sequence(pitches)
    notes.append(_tn(69, 60, Fraction(2)))
    analysis = infer_key(notes)
    assert analysis.mode == "mixolydian"
    assert analysis.sharps == 2


def test_key_override_is_respected_without_reanalysis():
    notes = _sequence([62, 64, 66, 67] * 4)
    analysis = infer_key(notes, KeyConfig(override="G major"))
    assert analysis.key == "G major"
    assert analysis.confidence == 1.0


def test_key_weighting_prefers_long_notes():
    """A held cadential note is stronger evidence than a passing sixteenth."""
    from fiddle.key import pitch_class_weights

    notes = [_tn(62, 0, Fraction(4)), _tn(63, 4, Fraction(1, 4))]
    w = pitch_class_weights(notes)
    assert w[2] > w[3] * 8


# --------------------------------------------------------------------------
# form
# --------------------------------------------------------------------------


def _aabb_notes(bars_per_section=8, beats_per_bar=4, repeats=2):
    """Build a clean AABB performance: two distinct 8-bar sections, each doubled."""
    # 8 notes = 4 beats of eighths, so 8 repetitions fills a 32-beat section.
    a_pitches = [74, 76, 78, 79, 81, 79, 78, 76] * 8
    b_pitches = [69, 71, 73, 74, 76, 74, 73, 71] * 8
    section_beats = bars_per_section * beats_per_bar
    notes = []
    order = ["A", "A", "B", "B"] * repeats
    for i, label in enumerate(order):
        pitches = a_pitches if label == "A" else b_pitches
        notes.extend(_sequence(pitches, start_beat=i * section_beats))
    return notes


def test_form_finds_aabb_with_the_right_section_length():
    notes = _aabb_notes()
    form = analyze_form(notes, beats_per_bar=4)
    assert form.beats_per_section == 32
    assert form.bars_per_section == 8
    assert [s.label for s in form.sections] == list("AABBAABB")


def test_form_does_not_mistake_a_phrase_for_a_section():
    """Each 8-bar section here contains two identical 4-bar phrases.

    A similarity-only criterion picks the 4-bar phrase, because half a section
    repeats just as cleanly as a whole one. Only the repeat-structure prior
    (sections come in runs of two, phrases do not) separates them.
    """
    phrase = [74, 76, 78, 79, 81, 79, 78, 76] * 4  # 4 bars of eighths
    section_a = phrase * 2  # 8 bars: the same phrase twice
    section_b = [69, 71, 73, 74, 76, 74, 73, 71] * 8
    notes = []
    for i, label in enumerate(["A", "A", "B", "B"] * 2):
        pitches = section_a if label == "A" else section_b
        notes.extend(_sequence(pitches, start_beat=i * 32))
    form = analyze_form(notes, beats_per_bar=4)
    assert form.bars_per_section == 8


def test_form_handles_a_crooked_six_bar_tune():
    """AABB is a prior, not a rule; crooked tunes must survive it."""
    a = [74, 76, 78, 79, 81, 79] * 8  # 6 notes = 3 beats, x8 = 24 beats = 6 bars
    b = [67, 69, 71, 72, 74, 72] * 8
    notes = []
    for i, label in enumerate(["A", "A", "B", "B"] * 2):
        notes.extend(_sequence(a if label == "A" else b, start_beat=i * 24))
    form = analyze_form(notes, beats_per_bar=4)
    assert form.bars_per_section == 6
    assert [s.label for s in form.sections] == list("AABBAABB")


def test_form_bars_a_sixteen_beat_section_as_eight_bars_of_two_four():
    """The 8-bar norm decides the meter, not the other way round.

    A 16-beat section is 8 bars in 2/4 and 4 bars in 4/4. Both are literally
    correct -- the meters are metrically nested -- but only one of them is what
    an old-time tune looks like. Beat stress cannot separate 2/4 from 4/4 with
    any confidence; section length can, so form settles it.
    """
    a = [74, 76, 78, 79] * 8  # 4 notes = 2 beats, x8 = 16 beats
    b = [69, 71, 73, 74] * 8
    notes = []
    for i, label in enumerate(["A", "A", "B", "B"] * 2):
        notes.extend(_sequence(a if label == "A" else b, start_beat=i * 16))
    # Deliberately handed the *wrong* prevailing meter; form must not take it.
    form = analyze_form(notes, beats_per_bar=4)
    assert form.beats_per_section == 16
    assert form.bars_per_section == 8
    assert form.meter == "2/4"
    assert form.beats_per_bar == 2


def test_form_finds_a_section_with_a_two_four_bar_dropped_in():
    """Crooked in the way this repertoire is actually crooked.

    Eight bars of 4/4 with one 2/4 bar inserted -- 34 beats rather than 32. The
    section must still be recognised, and must be written as nine bars with one
    of them short rather than padded out to a uniform grid.
    """
    a = [74, 76, 78, 79, 81, 79, 78, 76] * 8 + [83, 81, 79, 78]  # 32 + 2 beats
    b = [69, 71, 73, 74, 76, 74, 73, 71] * 8 + [67, 69, 71, 72]
    notes = []
    for i, label in enumerate(["A", "A", "B", "B"] * 2):
        notes.extend(_sequence(a if label == "A" else b, start_beat=i * 34))
    form = analyze_form(notes, beats_per_bar=4)
    assert form.beats_per_section == 34
    assert form.beats_per_bar == 4
    assert form.is_crooked
    assert sorted(form.bar_lengths) == [2.0] + [4.0] * 8
    assert [s.label for s in form.sections] == list("AABBAABB")


def test_form_tolerates_a_part_played_three_times():
    """Repeat counts vary in a jam; the part order does not."""
    a = [74, 76, 78, 79, 81, 79, 78, 76] * 8
    b = [69, 71, 73, 74, 76, 74, 73, 71] * 8
    notes = []
    for i, label in enumerate(["A", "A", "A", "B", "B", "A", "A", "B", "B"]):
        notes.extend(_sequence(a if label == "A" else b, start_beat=i * 32))
    form = analyze_form(notes, beats_per_bar=4)
    assert form.bars_per_section == 8
    assert [s.label for s in form.sections] == list("AAABBAABB")


def test_form_refuses_a_segmentation_that_barely_separates():
    """No structure must produce no form, not the best-scoring noise.

    Left to itself the search always returns *something*, because some
    hypothesis always scores highest even when every one of them is noise. On a
    real recording that produced a 36-block label sequence with a cluster
    quality of 0.012 -- and consensus would then have averaged unrelated music
    together, which is the one failure this module exists to prevent.
    """
    rng = np.random.default_rng(0)
    notes = _sequence(list(rng.integers(69, 84, size=512)))
    form = analyze_form(notes, beats_per_bar=4)
    assert form.sections == []
    assert form.confidence == 0.0


def test_form_returns_nothing_rather_than_guessing_on_short_input():
    form = analyze_form(_sequence([74, 76, 78, 79] * 2), beats_per_bar=4)
    assert form.sections == []
    assert form.confidence == 0.0


def test_slot_similarity_tolerates_a_one_slot_timing_shift():
    """Human passes land a grid slot apart routinely; exact matching broke form."""
    a = np.array([74, 74, 76, 76, 78, 78, -1, -1])
    b = np.array([74, 76, 76, 78, 78, -1, -1, -1])
    assert slot_similarity(a, b, tolerance=1) > 0.9
    assert slot_similarity(a, b, tolerance=0) < 0.6


def test_slot_similarity_still_separates_different_music():
    # Deliberately not an octave transposition, which similarity treats as a
    # partial match by design.
    a = np.array([74, 74, 76, 76, 78, 78, 79, 79])
    b = np.array([69, 69, 71, 71, 64, 64, 65, 65])
    assert slot_similarity(a, b) < 0.2


# --------------------------------------------------------------------------
# consensus
# --------------------------------------------------------------------------


def _instance(label, pitches, index=0):
    return SectionInstance(
        label=label, index=index, start_beats=0.0, end_beats=4.0,
        notes=_sequence(pitches), similarity_to_reference=1.0,
    )


def test_consensus_outvotes_a_single_deviating_pass():
    """Three passes play F#, one plays F natural: the majority must win.

    This is the project's central mechanism -- repeated passes are independent
    noisy measurements, so voting should recover the underlying melody.
    """
    good = [74, 76, 78, 79]
    bad = [74, 76, 77, 79]  # third note a semitone flat
    instances = [_instance("A", good, 0), _instance("A", good, 1),
                 _instance("A", good, 2), _instance("A", bad, 3)]
    section, report = build_section("A", instances, beats_per_section=2.0,
                                    beats_per_bar=4)
    pitches = [n.pitch for n in section.notes if not n.is_rest]
    assert 78 in pitches
    assert 77 not in pitches
    assert report.n_used == 4


def test_consensus_writes_the_odd_bar_where_the_phrase_breaks():
    """A crooked section is barred by phrasing, not mechanically at the end.

    Form hands consensus the barring it chose -- here three bars of 4/4 and one
    of 2/4. Which bar is the short one is left to consensus, because only it can
    see where notes actually fall. Putting the short bar last would cut a
    four-beat note in half; putting it second does not.
    """
    # Beats 0-3 and 6-13 are eighth-note runs; beats 4-6 are one long note that
    # a barline at beat 12 would slice through.
    notes = [
        *_sequence([74, 76, 78, 79, 81, 79, 78, 76]),          # beats 0-4
        _tn(83, 4, Fraction(2)),                                # beats 4-6
        *_sequence([81, 79, 78, 76, 74, 76, 78, 79], start_beat=6),  # beats 6-10
        *_sequence([81, 83, 84, 83], start_beat=10),            # beats 10-12
        _tn(81, 12, Fraction(2)),                               # beats 12-14: spans 12
    ]
    instances = [SectionInstance("A", i, 0.0, 14.0, notes) for i in range(2)]
    section, _ = build_section(
        "A", instances, beats_per_section=14.0, beats_per_bar=4,
        bar_lengths=(4.0, 4.0, 4.0, 2.0),
    )
    assert len(section.measures) == 4
    lengths = [sum(float(n.duration_beats) for n in m.notes)
               for m in section.measures]
    # Whatever the placement, the section is still 14 beats in four bars...
    assert sum(lengths) == pytest.approx(14.0)
    # ...and the short bar is not the last one, because that placement is the
    # only one that straddles the held note at beat 12.
    assert lengths[-1] > 2.0


def test_consensus_preserves_uncertainty_when_passes_split_evenly():
    """A 50/50 disagreement must produce a low-confidence note, not a confident guess."""
    a = [74, 76, 78, 79]
    b = [74, 76, 77, 79]
    instances = [_instance("A", a, 0), _instance("A", a, 1),
                 _instance("A", b, 2), _instance("A", b, 3)]
    section, _ = build_section("A", instances, beats_per_section=2.0, beats_per_bar=4)
    contested = [n for n in section.notes
                 if not n.is_rest and n.pitch in (77, 78)]
    assert contested
    assert min(n.agreement for n in contested) <= 0.5
    assert any(n.alternatives for n in contested)


def test_consensus_excludes_a_pass_that_does_not_align():
    """Averaging in a mis-segmented pass is worse than ignoring it."""
    good = [74, 76, 78, 79]
    garbage = [55, 57, 59, 60]
    instances = [_instance("A", good, 0), _instance("A", good, 1),
                 _instance("A", good, 2), _instance("A", garbage, 3)]
    section, report = build_section(
        "A", instances, beats_per_section=2.0, beats_per_bar=4,
        config=ConsensusConfig(min_alignment_score=0.55),
    )
    assert 3 in report.excluded
    assert [n.pitch for n in section.notes if not n.is_rest] == good


def test_consensus_disabled_falls_back_to_a_single_pass():
    good = [74, 76, 78, 79]
    bad = [74, 76, 77, 79]
    instances = [_instance("A", good, 0), _instance("A", bad, 1)]
    _, report = build_section(
        "A", instances, beats_per_section=2.0, beats_per_bar=4,
        config=ConsensusConfig(enabled=False),
    )
    assert report.n_used == 1


def test_agreement_raises_confidence_relative_to_disagreement():
    agree = [_instance("A", [74, 76, 78, 79], i) for i in range(4)]
    split = [_instance("A", [74, 76, 78, 79], 0), _instance("A", [74, 76, 78, 79], 1),
             _instance("A", [74, 76, 77, 79], 2), _instance("A", [74, 76, 77, 79], 3)]
    s1, _ = build_section("A", agree, 2.0, 4)
    s2, _ = build_section("A", split, 2.0, 4)
    assert s1.confidence > s2.confidence


# --------------------------------------------------------------------------
# score export
# --------------------------------------------------------------------------


def _simple_tune():
    # Quarter notes, so each 4-note measure really is a full bar of 4/4.
    notes = [Note(pitch=p, start_beats=Fraction(i),
                  duration_beats=Fraction(1), confidence=0.9)
             for i, p in enumerate([74, 76, 78, 79, 81, 79, 78, 76])]
    measures = [Measure(notes=notes[:4], number=1), Measure(notes=notes[4:], number=2)]
    section = TuneSection(name="A", measures=measures, repeats=2)
    return Tune(key="D major", meter="4/4", tempo=120.0, sections=[section],
                title="Test Tune", analysis={"key_sharps": 2})


def test_musicxml_export_roundtrips(tmp_path):
    from music21 import converter

    from fiddle.score import export_musicxml

    path = export_musicxml(_simple_tune(), tmp_path / "t.musicxml")
    stream = converter.parse(str(path))
    assert len(list(stream.flatten().notes)) == 8
    assert stream.flatten().getElementsByClass("KeySignature")[0].sharps == 2


def test_repeated_sections_use_repeat_barlines_not_duplication(tmp_path):
    from music21 import converter

    from fiddle.score import export_musicxml

    path = export_musicxml(_simple_tune(), tmp_path / "t.musicxml")
    stream = converter.parse(str(path))
    measures = list(stream.recurse().getElementsByClass("Measure"))
    assert len(measures) == 2  # not four
    assert any(m.rightBarline is not None and "Repeat" in m.rightBarline.classes
               for m in measures)


def test_midi_export_is_written(tmp_path):
    from fiddle.score import export_midi

    path = export_midi(_simple_tune(), tmp_path / "t.mid")
    assert path.exists() and path.stat().st_size > 50


def test_abc_export_reparses_as_ground_truth(tmp_path):
    """The correction loop depends on our own ABC being readable back in."""
    from fiddle.abc_io import parse_abc_sections
    from fiddle.score import export_abc

    path = export_abc(_simple_tune(), tmp_path / "t.abc")
    sections = parse_abc_sections(path.read_text())
    assert sections["A"].bars == 2
    assert [n.pitch for n in sections["A"].notes] == [74, 76, 78, 79, 81, 79, 78, 76]


def test_uncertain_notes_are_marked_in_the_score():
    from fiddle.score import UNCERTAIN_COLOR, tune_to_stream

    tune = _simple_tune()
    tune.sections[0].measures[0].notes[1].confidence = 0.2
    stream = tune_to_stream(tune, mark_uncertain_below=0.7)
    coloured = [n for n in stream.flatten().notes
                if n.style.color == UNCERTAIN_COLOR]
    assert len(coloured) == 1


def test_abc_spells_accidentals_relative_to_the_key():
    from fiddle.score import _abc_pitch

    # In D major (2 sharps) F# needs no marker, but F natural needs one.
    assert _abc_pitch(78, 2) == "f"
    assert _abc_pitch(77, 2) == "=f"
    # In C major F# needs an explicit sharp.
    assert _abc_pitch(78, 0) == "^f"


# --------------------------------------------------------------------------
# form universe
# --------------------------------------------------------------------------


def test_form_plausibility_ranks_the_known_universe():
    """Old-time form comes from a small set; scoring should reflect that."""
    from fiddle.form import form_plausibility

    assert form_plausibility(list("AABB")) == pytest.approx(1.0)
    # A jam plays the tune several times through.
    assert form_plausibility(list("AABBAABB")) == pytest.approx(1.0)
    # A recording joined mid-tune is the same form, rotated.
    assert form_plausibility(list("BBAABB")) == pytest.approx(1.0)
    # Things that are periodic but are not tunes must score well below.
    for bogus in ("AABBCCD", "AABBCDD", "ABCABC"):
        assert form_plausibility(list(bogus)) < 0.7, bogus


def test_form_plausibility_is_flexible_about_repeat_counts():
    """The part *order* is constrained; how often each part repeats is not.

    A jam plays each part twice as a rule, but three times happens, once
    happens, and the recording is cut off wherever it is cut off. What must
    stay penalised is the run of four, because that is the signature of a
    section-length hypothesis that is wrong by a factor of two.
    """
    from fiddle.form import form_plausibility

    assert form_plausibility(list("AABBB")) > 0.6
    assert form_plausibility(list("AAABBAABB")) > 0.6
    # Cut off mid-repeat: the truncated tail must not be read as evidence.
    assert form_plausibility(list("AABBA")) > 0.6
    # Half-length hypothesis: every part appears to repeat four times.
    assert form_plausibility(list("AAAABBBB")) < 0.25


def test_choose_bar_layout_prefers_eight_bar_sections():
    from fiddle.form import choose_bar_layout

    # 8 bars is reachable under one barring or the other for both of these.
    assert choose_bar_layout(16).bars == 8
    assert choose_bar_layout(16).beats_per_bar == 2
    assert choose_bar_layout(32).bars == 8
    assert choose_bar_layout(32).beats_per_bar == 4
    # Two beats short of eight bars of 4/4: still eight bars, one of them 2/4.
    short = choose_bar_layout(30)
    assert short.bars == 8 and short.is_crooked
    # A length no barring makes plausible still returns something, scored low.
    assert choose_bar_layout(40).plausibility < choose_bar_layout(32).plausibility


def test_form_plausibility_keeps_unusual_forms_reachable():
    """A floor, not a veto: strong acoustic evidence should still win."""
    from fiddle.form import form_plausibility

    assert form_plausibility(list("ABCDEF")) > 0.0


def test_clustering_finds_the_window_between_merge_and_shatter():
    """Regression for a bug that made form detection fail outright on real audio.

    The threshold was chosen from a short hand-picked ladder. On a real
    recording its low rungs merged every block into one cluster and its high
    rungs shattered them past max_sections, so every rung was rejected and the
    search returned nothing -- the viable window sat *between* two rungs.
    Thresholds are now scanned from the observed similarity distribution.
    """
    import numpy as np

    from fiddle.form import _cluster

    # Two groups whose separation is real but narrow, and nowhere near 0.62.
    n = 8
    sim = np.full((n, n), 0.30)
    for i in range(n):
        for j in range(n):
            if (i < 4) == (j < 4):
                sim[i, j] = 0.44
        sim[i, i] = 1.0

    labels, quality = _cluster(sim, 0.62, 3)
    assert quality is not None, "no threshold in the scan produced a clustering"
    assert len(set(labels)) == 2
    assert labels[:4] == [labels[0]] * 4
    assert labels[4:] == [labels[4]] * 4


def test_section_search_stays_within_old_time_lengths():
    from fiddle.config import FormConfig

    cfg = FormConfig()
    # 15- and 16-bar "sections" were real output before this bound; no old-time
    # section is that long, they were a section plus its repeat.
    assert cfg.max_section_bars <= 12
    # One, two or occasionally three parts. Four let the search invent AABBCCD.
    assert cfg.max_sections == 3
    assert 8 in cfg.expected_section_bars
