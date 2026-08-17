"""End-to-end tests, including regression floors on transcription accuracy.

The accuracy assertions are deliberately loose. Their job is to catch a change
that *breaks* the pipeline, not to encode the current numbers as a target --
tightening them as the system improves is the intended workflow, and the real
measurement lives in `fiddle-transcribe eval`.
"""

from __future__ import annotations

import json

import pytest

from fiddle.eval.metrics import (
    _form_equivalent,
    _reduce_form,
    align_sequences,
    score_transcription,
)


def test_pipeline_produces_a_usable_tune(clean_result):
    result, truth = clean_result
    tune = result.tune
    assert tune.sections, "no sections were produced"
    assert len(tune.notes) > 20
    assert all(55 <= n.pitch <= 100 for n in tune.notes if not n.is_rest)


def test_pipeline_recovers_key_and_tempo(clean_result):
    result, truth = clean_result
    assert result.tune.key == truth.key
    assert result.tune.tempo == pytest.approx(truth.tempo_bpm, rel=0.08)


def test_pipeline_keeps_every_intermediate_stage(clean_result):
    """Diagnosability is a requirement, so it is tested like any other."""
    result, _ = clean_result
    assert len(result.raw_contour.times) > 1000
    assert len(result.clean_contour.times) == len(result.raw_contour.times)
    assert result.raw_notes
    assert result.timed_notes
    assert len(result.rhythm.beat_times) > 10
    assert result.log


def test_cleaning_actually_changed_something(clean_result):
    import numpy as np

    result, _ = clean_result
    raw, clean = result.raw_contour, result.clean_contour
    assert len(clean.history) > len(raw.history)
    both = ~np.isnan(raw.midi) & ~np.isnan(clean.midi)
    assert np.any(np.abs(raw.midi[both] - clean.midi[both]) > 1e-6)


def test_transcription_accuracy_regression_floor(clean_result):
    result, truth = clean_result
    score = score_transcription(
        result.tune, {s.name: s for s in truth.sections},
        truth_key=truth.key, truth_meter=truth.meter,
        truth_tempo=truth.tempo_bpm, truth_form=truth.form,
        slug=truth.slug, difficulty=truth.difficulty,
    )
    # Measured well above these at the time of writing; they are a floor.
    assert score.pitch_accuracy > 0.45
    assert score.sequence_similarity > 0.25
    assert score.octave_error_rate < 0.15


def test_confidence_is_populated_and_bounded(clean_result):
    result, _ = clean_result
    confs = [n.confidence for n in result.tune.notes if not n.is_rest]
    assert confs
    assert all(0.0 <= c <= 1.0 for c in confs)
    assert len(set(round(c, 2) for c in confs)) > 1, "confidence is constant"


# --------------------------------------------------------------------------
# evaluation machinery
# --------------------------------------------------------------------------


def test_alignment_survives_an_inserted_note():
    """A single spurious note must not misalign everything after it."""
    truth = [(74, 0.0), (76, 0.5), (78, 1.0), (79, 1.5)]
    pred = [(74, 0.0), (99, 0.25), (76, 0.5), (78, 1.0), (79, 1.5)]
    pairs = align_sequences(pred, truth)
    matched = [(p, t) for p, t in pairs if p is not None and t is not None]
    assert len(matched) == 4
    assert sum(1 for p, t in pairs if t is None) == 1  # the insertion


def test_alignment_survives_a_dropped_note():
    truth = [(74, 0.0), (76, 0.5), (78, 1.0), (79, 1.5)]
    pred = [(74, 0.0), (78, 1.0), (79, 1.5)]
    pairs = align_sequences(pred, truth)
    assert sum(1 for p, t in pairs if p is None) == 1  # the deletion


def test_form_comparison_ignores_how_many_times_the_jam_repeated():
    assert _form_equivalent("AABB", "AABBAABB")
    assert _form_equivalent("AABBAABB", "AABB")
    assert not _form_equivalent("AABB", "AABBCC")


def test_reduce_form_finds_the_repeating_unit():
    assert _reduce_form("AABBAABB") == "AABB"
    assert _reduce_form("ABC") == "ABC"
    assert _reduce_form("") == ""


def test_confidence_auc_rewards_ranking_errors_low():
    """The product claim depends on this metric, so it gets its own test."""
    import numpy as np

    from fiddle.eval.metrics import _auc

    conf = np.array([0.9, 0.8, 0.2, 0.1])
    wrong = np.array([False, False, True, True])
    assert _auc(conf, wrong) == pytest.approx(1.0)
    assert _auc(conf, ~wrong) == pytest.approx(0.0)
    assert _auc(np.array([0.5, 0.5]), np.array([True, False])) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_writes_every_promised_artifact(tmp_path, clean_render):
    from fiddle.audio import write_wav
    from fiddle.cli import main

    audio, _ = clean_render
    wav = tmp_path / "tune.wav"
    write_wav(wav, audio)

    assert main([str(wav), "-o", str(tmp_path)]) == 0
    for suffix in (".pitch.csv", ".notes.json", ".musicxml", ".mid", ".abc"):
        out = tmp_path / f"tune{suffix}"
        assert out.exists() and out.stat().st_size > 0, f"missing {suffix}"

    payload = json.loads((tmp_path / "tune.notes.json").read_text())
    assert payload["summary"]["n_notes"] > 0
    assert payload["sections"]
    assert "beat_times" in payload["rhythm"]


def test_cli_backends_reports_availability(capsys):
    from fiddle.cli import main

    assert main(["backends"]) == 0
    assert "essentia" in capsys.readouterr().out


def test_config_overrides_are_isolated():
    """Ablations depend on overrides not leaking into the base config."""
    from fiddle.config import Config

    base = Config()
    variant = base.with_overrides(consensus={"enabled": False})
    assert base.consensus.enabled is True
    assert variant.consensus.enabled is False
    assert variant.melody.backend == base.melody.backend


def test_unsectioned_fallback_exports_valid_musicxml(tmp_path):
    """The no-form fallback must still produce a readable score.

    This path is what runs whenever form detection fails, which on real jam
    audio is the common case -- so it is the *most* exercised path in practice,
    not an edge case. It previously crashed music21 deep inside rest generation
    because measures were numbered by absolute bar index, leaving gaps wherever
    the performance had a silent bar.
    """
    from fractions import Fraction

    from music21 import converter

    from fiddle.pipeline import _fallback_sections
    from fiddle.domain import TimedNote, Tune
    from fiddle.score import export_musicxml

    # Notes starting well after beat 0, with a silent bar in the middle.
    starts = [40, 40.5, 41, 41.5, 48, 48.5, 49, 50]
    notes = [
        TimedNote(
            pitch=p, start_beats=Fraction(s).limit_denominator(64),
            duration_beats=Fraction(1, 2), confidence=0.8,
            start_sec=float(s) * 0.5, duration_sec=0.25, pitch_midi=float(p),
        )
        for p, s in zip([74, 76, 78, 79, 81, 79, 78, 76], starts)
    ]
    sections = _fallback_sections(notes, beats_per_bar=4)
    assert sections and sections[0].measures

    # Measures must be contiguous and start at bar 1.
    numbers = [m.number for m in sections[0].measures]
    assert numbers == list(range(1, len(numbers) + 1))
    # Every note must sit inside its own bar.
    for i, m in enumerate(sections[0].measures):
        for n in m.notes:
            assert i * 4 <= float(n.start_beats) < (i + 1) * 4

    tune = Tune(key="D major", meter="4/4", tempo=120.0, sections=sections,
                title="Fallback", analysis={"key_sharps": 2})
    path = export_musicxml(tune, tmp_path / "fallback.musicxml")
    assert len(list(converter.parse(str(path)).flatten().notes)) == 8
