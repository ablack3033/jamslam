"""Unit tests for individual pipeline stages.

Each test constructs a signal with a known defect and asserts that the stage
responsible removes exactly that defect -- these are the tests that let us
change an algorithm and know immediately whether it still does its job.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from fiddle.abc_io import parse_abc_sections
from fiddle.config import CleanConfig, RhythmConfig, SegmentConfig
from fiddle.domain import PitchContour, hz_to_midi, midi_to_hz
from fiddle.pitch_clean import clean_contour, voiced_runs
from fiddle.segment import segment_notes


# --------------------------------------------------------------------------
# domain
# --------------------------------------------------------------------------


def test_hz_midi_roundtrip():
    midi = np.array([55.0, 62.0, 69.0, 76.0, 88.5])
    assert np.allclose(hz_to_midi(midi_to_hz(midi)), midi)


def test_hz_to_midi_marks_unvoiced_as_nan():
    out = hz_to_midi(np.array([440.0, 0.0, -1.0]))
    assert out[0] == pytest.approx(69.0)
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_voiced_runs_finds_contiguous_regions():
    mask = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    assert voiced_runs(mask) == [(1, 3), (5, 8)]


def test_voiced_runs_handles_edges():
    assert voiced_runs(np.array([1, 1, 0, 1], dtype=bool)) == [(0, 2), (3, 4)]


# --------------------------------------------------------------------------
# pitch cleaning
# --------------------------------------------------------------------------


def _contour(midi, hop=0.01, conf=None):
    midi = np.asarray(midi, dtype=float)
    return PitchContour(
        times=np.arange(len(midi)) * hop,
        midi=midi,
        confidence=np.ones(len(midi)) if conf is None else np.asarray(conf, float),
        hop_seconds=hop,
    )


def test_octave_correction_folds_a_dropped_octave():
    # 2s of A4 with a 0.3s stretch that fell an octave, as the banjo doubling
    # the melody an octave down makes Melodia do.
    midi = np.full(200, 69.0)
    midi[80:110] -= 12.0
    cleaned = clean_contour(
        _contour(midi),
        CleanConfig(enable_median_filter=False, enable_vibrato_smoothing=False),
    )
    assert np.allclose(cleaned.midi, 69.0, atol=0.2)


def test_octave_correction_leaves_a_real_leap_alone():
    # A genuine sustained leap of a fifth must survive untouched.
    midi = np.concatenate([np.full(100, 69.0), np.full(100, 76.0)])
    cleaned = clean_contour(
        _contour(midi),
        CleanConfig(enable_median_filter=False, enable_vibrato_smoothing=False),
    )
    assert cleaned.midi[10] == pytest.approx(69.0, abs=0.2)
    assert cleaned.midi[-10] == pytest.approx(76.0, abs=0.2)


def test_octave_correction_reduces_confidence_where_it_guessed():
    midi = np.full(200, 69.0)
    midi[80:110] -= 12.0
    cleaned = clean_contour(_contour(midi), CleanConfig())
    assert cleaned.confidence[95] < cleaned.confidence[5]


def test_range_gate_removes_impossible_pitches():
    midi = np.full(100, 69.0)
    midi[20:40] = 30.0  # far below any fiddle, and too long to be a dropout
    cleaned = clean_contour(_contour(midi), CleanConfig())
    assert np.all(np.isnan(cleaned.midi[20:40]))


def test_median_filter_removes_isolated_spikes():
    midi = np.full(100, 69.0)
    midi[50] = 76.0
    cleaned = clean_contour(
        _contour(midi),
        CleanConfig(enable_octave_correction=False, enable_vibrato_smoothing=False),
    )
    assert cleaned.midi[50] == pytest.approx(69.0, abs=0.1)


def test_vibrato_smoothing_flattens_wobble_but_keeps_the_note():
    t = np.arange(400) * 0.005
    midi = 69.0 + 0.3 * np.sin(2 * np.pi * 5.5 * t)  # ~30 cents, 5.5 Hz
    cleaned = clean_contour(
        _contour(midi, hop=0.005),
        CleanConfig(enable_octave_correction=False, enable_median_filter=False),
    )
    assert np.std(cleaned.midi[50:-50]) < np.std(midi[50:-50]) / 2
    assert np.mean(cleaned.midi) == pytest.approx(69.0, abs=0.1)


def test_short_gap_fill_bridges_dropout_but_not_a_rest():
    midi = np.full(200, 69.0)
    midi[50:52] = np.nan  # 20ms dropout -> bridge it
    midi[100:150] = np.nan  # 500ms rest -> leave it
    cleaned = clean_contour(_contour(midi), CleanConfig())
    assert not np.any(np.isnan(cleaned.midi[50:52]))
    assert np.all(np.isnan(cleaned.midi[100:150]))


def test_disabled_stages_are_actually_skipped():
    midi = np.full(100, 69.0)
    midi[50] = 76.0
    cfg = CleanConfig(
        enable_median_filter=False, enable_octave_correction=False,
        enable_vibrato_smoothing=False, enable_jump_gate=False,
    )
    cleaned = clean_contour(_contour(midi), cfg)
    assert cleaned.midi[50] == pytest.approx(76.0)


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------


def test_segmentation_splits_on_sustained_pitch_change():
    midi = np.concatenate([np.full(100, 69.0), np.full(100, 71.0),
                           np.full(100, 73.0)])
    notes = segment_notes(_contour(midi), SegmentConfig(use_onsets_to_split=False))
    assert [n.pitch for n in notes] == [69, 71, 73]


def test_segmentation_ignores_brief_ornaments():
    # A 20ms grace-note blip should not become its own note.
    midi = np.full(300, 69.0)
    midi[150:152] = 71.0
    notes = segment_notes(_contour(midi), SegmentConfig(use_onsets_to_split=False))
    assert [n.pitch for n in notes] == [69]


def test_onsets_split_rearticulated_repeated_notes():
    """The characteristic old-time case: 'd2 dd' is three notes, not one.

    Pitch alone cannot see the rearticulation, so without onsets this must
    collapse to a single note and with them it must not.
    """
    midi = np.full(300, 62.0)
    contour = _contour(midi)  # 3 seconds of constant D4
    onsets = np.array([1.0, 2.0])

    without = segment_notes(contour, SegmentConfig(use_onsets_to_split=False))
    with_onsets = segment_notes(
        contour, SegmentConfig(use_onsets_to_split=True), onset_times=onsets
    )
    assert len(without) == 1
    assert len(with_onsets) == 3
    assert all(n.pitch == 62 for n in with_onsets)


def test_short_notes_are_discarded():
    midi = np.concatenate([np.full(100, 69.0), np.full(2, 80.0),
                           np.full(100, 69.0)])
    notes = segment_notes(
        _contour(midi), SegmentConfig(use_onsets_to_split=False, min_note_sec=0.05)
    )
    assert 80 not in [n.pitch for n in notes]


def test_note_pitch_uses_a_robust_centre():
    # An unstable attack must not drag the reported pitch off the note.
    midi = np.concatenate([np.full(10, 66.0), np.full(190, 69.0)])
    notes = segment_notes(_contour(midi), SegmentConfig(use_onsets_to_split=False))
    assert notes[-1].pitch == 69


# --------------------------------------------------------------------------
# rhythm
# --------------------------------------------------------------------------


def test_tempo_folding_corrects_a_halved_estimate():
    from fiddle.rhythm import _fold_tempo_into_range

    beats = np.arange(0, 20, 1.0)  # 60 BPM, i.e. half of 120
    new_beats, tempo, factor = _fold_tempo_into_range(beats, 60.0, RhythmConfig())
    assert tempo == pytest.approx(120.0)
    assert factor == 2.0
    assert len(new_beats) == 2 * len(beats) - 1  # midpoints interpolated


def test_tempo_folding_corrects_a_doubled_estimate():
    from fiddle.rhythm import _fold_tempo_into_range

    beats = np.arange(0, 10, 0.25)  # 240 BPM
    _, tempo, factor = _fold_tempo_into_range(beats, 240.0, RhythmConfig())
    assert tempo == pytest.approx(120.0)
    assert factor == 0.5


def test_tempo_in_range_is_left_alone():
    from fiddle.rhythm import _fold_tempo_into_range

    beats = np.arange(0, 20, 0.5)
    _, tempo, factor = _fold_tempo_into_range(beats, 120.0, RhythmConfig())
    assert factor == 1.0 and tempo == pytest.approx(120.0)


def test_seconds_to_beats_is_linear_between_tracked_beats():
    from fiddle.rhythm import seconds_to_beats

    beats = np.arange(0, 10, 0.5)  # 120 BPM
    out = seconds_to_beats(np.array([0.0, 0.25, 0.5, 1.0]), beats)
    assert out == pytest.approx([0.0, 0.5, 1.0, 2.0])


def test_seconds_to_beats_extrapolates_before_the_first_beat():
    from fiddle.rhythm import seconds_to_beats

    beats = np.arange(1.0, 10.0, 0.5)
    out = seconds_to_beats(np.array([0.5]), beats)
    assert out[0] == pytest.approx(-1.0)


def test_durations_snap_to_simple_values():
    from fiddle.rhythm import _nearest_simple_duration

    cfg = RhythmConfig()
    assert _nearest_simple_duration(0.52, cfg) == Fraction(1, 2)
    assert _nearest_simple_duration(0.98, cfg) == Fraction(1)
    assert _nearest_simple_duration(0.26, cfg) == Fraction(1, 4)
    assert _nearest_simple_duration(2.05, cfg) == Fraction(2)


def test_dotted_values_need_to_fit_clearly_better():
    """Simplicity is preferred on near-ties, so noise cannot invent dotted rhythms."""
    from fiddle.rhythm import _nearest_simple_duration

    cfg = RhythmConfig()
    assert _nearest_simple_duration(0.88, cfg) == Fraction(1)
    assert _nearest_simple_duration(0.76, cfg) == Fraction(3, 4)


# --------------------------------------------------------------------------
# ABC ground truth
# --------------------------------------------------------------------------


def test_abc_sections_parse_with_validated_bar_counts():
    abc = """X:1
M:4/4
L:1/8
K:D
P:A
a2 fa gfef|dcde fdcd|
P:B
f2 df e2 ce|dcde fdcd|
"""
    sections = parse_abc_sections(abc)
    assert set(sections) == {"A", "B"}
    assert sections["A"].bars == 2
    assert sections["A"].notes[0].pitch == 81  # a = A5


def test_abc_rejects_a_bar_that_does_not_add_up():
    """Ground truth that is silently wrong is worse than none, so this must raise."""
    abc = "X:1\nM:4/4\nL:1/8\nK:D\nP:A\na2 fa gfe|\n"
    with pytest.raises(ValueError):
        parse_abc_sections(abc)


def test_abc_handles_a_crooked_section():
    abc = """X:1
M:4/4
L:1/8
K:G
P:A
d2 dB d2 ga|b2 ag e2 d2|G2 A2 G4|
"""
    assert parse_abc_sections(abc)["A"].bars == 3


# --------------------------------------------------------------------------
# melody presence
#
# These guard the check that stops the system emitting a tidy score for a
# recording whose melody it never found -- the failure a musician cannot
# distinguish from success without checking every note.
# --------------------------------------------------------------------------


def _fake_contour(midi, hop=0.01):
    midi = np.asarray(midi, dtype=float)
    return PitchContour(times=np.arange(len(midi)) * hop, midi=midi,
                        confidence=np.ones(len(midi)), hop_seconds=hop)


def test_bass_coupling_detects_a_rigid_octave_above_the_bass():
    """The real failure: the 'melody' is the guitar root, doubled an octave up."""
    from fiddle.presence import _bass_coupling

    rng = np.random.default_rng(0)
    bass = np.repeat(rng.choice([50.0, 55.0, 57.0], size=300), 20)
    melody = bass + 12.0
    assert _bass_coupling(melody, bass) > 0.95


def test_bass_coupling_returns_none_without_enough_data():
    """It must fail CLOSED.

    An earlier version returned 0.0 when there was no trackable bass, which
    reads as "no problem found". That silently invalidated a control: the
    synthetic corpus has no trackable bass at all, so a check that never ran
    was reported as a clean pass.
    """
    from fiddle.presence import _bass_coupling

    assert _bass_coupling(np.full(2000, 69.0), np.full(2000, np.nan)) is None
    assert _bass_coupling(np.zeros(0), np.zeros(0)) is None


def test_bass_coupling_stays_low_for_a_real_melody():
    """A melody wanders relative to the bass even when both are chord-based."""
    from fiddle.presence import _bass_coupling

    rng = np.random.default_rng(1)
    bass = np.repeat(rng.choice([50.0, 55.0, 57.0], size=300), 20)
    melody = np.repeat(rng.choice([74.0, 76.0, 78.0, 79.0, 81.0], size=1200), 5)
    n = min(len(bass), len(melody))
    assert _bass_coupling(melody[:n], bass[:n]) < 0.5


def test_bass_coupling_is_not_fooled_by_a_low_register_melody():
    """A fiddle playing low over a root-position bass often sits an octave above it.

    That is harmony, not harmonics. What marks a harmonic is a *locked*
    interval, so the measure keys on interval stability rather than on the
    interval's value -- an earlier version keyed on the value and would have
    condemned any melody played in the low octave.
    """
    from fiddle.presence import _bass_coupling

    rng = np.random.default_rng(4)
    bass = np.repeat(rng.choice([50.0, 55.0, 57.0], size=300), 20)
    # Melody in the octave above the bass, but moving independently of it.
    melody = np.repeat(rng.choice([62.0, 64.0, 66.0, 67.0, 69.0], size=1200), 5)
    n = min(len(bass), len(melody))
    assert _bass_coupling(melody[:n], bass[:n]) < 0.5


def test_presence_flags_a_drone():
    from fiddle.presence import diagnose_melody

    midi = np.full(4000, 69.0)
    midi[::50] = 71.0  # a little movement, still overwhelmingly one pitch
    p = diagnose_melody(None, contour=_fake_contour(midi),
                        bass_contour=_fake_contour(np.full(4000, np.nan)))
    assert not p.melody_found
    assert any("single pitch" in w for w in p.warnings)
    assert p.bass_coupling is None  # no bass to compare against


def test_presence_flags_a_static_range():
    from fiddle.presence import diagnose_melody

    rng = np.random.default_rng(2)
    midi = 69.0 + rng.choice([0.0, 1.0], size=4000)
    p = diagnose_melody(None, contour=_fake_contour(midi),
                        bass_contour=_fake_contour(np.full(4000, np.nan)))
    assert not p.melody_found


def test_presence_accepts_a_real_melodic_contour():
    """The control: a moving, wide-range contour must NOT be flagged."""
    from fiddle.presence import diagnose_melody

    rng = np.random.default_rng(3)
    tune = [74, 76, 78, 79, 81, 79, 78, 76, 74, 69, 71, 73, 74, 76, 74, 71]
    midi = np.repeat(np.array(tune * 25, dtype=float), 10)
    midi += rng.normal(0, 0.05, len(midi))
    p = diagnose_melody(None, contour=_fake_contour(midi),
                        bass_contour=_fake_contour(np.full(len(midi), np.nan)))
    assert p.melody_found, p.warnings
    assert p.pitch_iqr > 3.0


def test_presence_reports_nothing_voiced():
    from fiddle.presence import diagnose_melody

    p = diagnose_melody(None, contour=_fake_contour(np.full(500, np.nan)),
                        bass_contour=_fake_contour(np.full(500, np.nan)))
    assert not p.melody_found
    assert p.voiced_fraction == 0.0
