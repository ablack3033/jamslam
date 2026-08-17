"""The transcription pipeline: audio in, canonical Tune out.

This module only wires stages together and carries their intermediate results
into a :class:`TranscriptionResult`. It contains no DSP and no musical logic of
its own -- if you find yourself wanting to add a heuristic here, it belongs in
the stage it concerns.

Every intermediate representation is retained on the result object. That is
what makes the system debuggable: when a wrong note appears, you can see the
raw contour, the cleaned contour, the segmentation, the beat grid, the section
split and the votes that produced it, without re-running anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .config import Config
from .consensus import ConsensusReport, build_tune_sections
from .domain import Audio, Measure, Note, PitchContour, RawNote, TimedNote, Tune, TuneSection
from .form import FormAnalysis, analyze_form
from .key import KeyAnalysis, infer_key
from .melody import get_extractor
from .meter import MeterAnalysis, infer_meter
from .pitch_clean import clean_contour
from .rhythm import RhythmAnalysis, analyze_rhythm, quantize_notes
from .segment import segment_notes


@dataclass
class TranscriptionResult:
    """Everything the pipeline produced, including all intermediate stages."""

    tune: Tune
    audio: Audio
    raw_contour: PitchContour
    clean_contour: PitchContour
    raw_notes: list[RawNote]
    timed_notes: list[TimedNote]
    rhythm: RhythmAnalysis
    meter: MeterAnalysis
    key: KeyAnalysis
    form: FormAnalysis
    consensus_reports: list[ConsensusReport] = field(default_factory=list)
    config: Config = field(default_factory=Config)
    log: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "key": self.tune.key,
            "key_confidence": round(self.key.confidence, 3),
            "meter": self.tune.meter,
            "meter_confidence": round(self.meter.confidence, 3),
            "tempo_bpm": round(self.tune.tempo, 1),
            "form": self.tune.form,
            "form_confidence": round(self.form.confidence, 3),
            "bars_per_section": self.form.bars_per_section,
            "n_notes": len(self.tune.notes),
            "n_uncertain": len(
                self.tune.uncertain_notes(self.config.confidence.uncertain_threshold)
            ),
            "voiced_fraction": round(self.clean_contour.voiced_fraction, 3),
            "melody_backend": self.raw_contour.backend,
        }


def transcribe(audio: Audio, config: Config | None = None) -> TranscriptionResult:
    """Run the full pipeline on preprocessed audio."""
    cfg = config or Config()
    log: list[str] = []

    # 1. Predominant melody -> continuous contour.
    extractor = get_extractor(cfg.melody.backend, cfg.melody)
    raw = extractor.extract(audio)
    log.append(f"melody: {raw.backend}, {len(raw.times)} frames, "
               f"{raw.voiced_fraction:.2f} voiced")

    # 2. Clean the contour, keeping the raw one for diagnostics.
    cleaned = clean_contour(raw, cfg.clean)
    log.extend(f"clean: {h}" for h in cleaned.history[len(raw.history):])

    # 3. Rhythm is inferred from the mix, independently of pitch.
    rhythm = analyze_rhythm(audio, cfg.rhythm)
    log.extend(f"rhythm: {n}" for n in rhythm.notes)

    # 4. Segment the contour into notes, using onsets to split rearticulations.
    raw_notes = segment_notes(cleaned, cfg.segment, onset_times=rhythm.onset_times)
    log.append(f"segment: {len(raw_notes)} raw notes")

    # 5. Quantize onto the raw beat grid, with no phase applied yet.
    #    Phase is deliberately deferred: onset-energy downbeat detection is only
    #    about as good as a coin flip weighted 70/30, and applying a wrong phase
    #    here would translate every note by a beat and corrupt form detection.
    #    Since quantization is a pure translation, applying phase later is
    #    mathematically identical and strictly better informed.
    timed = quantize_notes(raw_notes, rhythm, cfg.rhythm)
    meter_analysis = infer_meter(rhythm, timed, cfg.meter)

    # 6. Form: which stretches are repeats of the same section. This also
    #    returns the offset of the first section boundary, which is a downbeat
    #    by definition and therefore the strongest phase evidence we have.
    form = analyze_form(timed, meter_analysis.beats_per_bar, cfg.form)
    log.extend(f"form: {n}" for n in form.notes)

    if form.sections:
        meter_analysis.downbeat_phase = form.offset_beats % meter_analysis.beats_per_bar
        meter_analysis.notes.append(
            f"downbeat phase taken from section boundary at beat {form.offset_beats:g}"
        )
    log.append(
        f"meter: {meter_analysis.meter} (phase {meter_analysis.downbeat_phase}, "
        f"confidence {meter_analysis.confidence:.2f})"
    )

    # 7. Consensus across repetitions -> canonical sections.
    if form.sections:
        sections, reports = build_tune_sections(
            form, meter_analysis.beats_per_bar, cfg.consensus, cfg.confidence
        )
    else:
        sections, reports = _fallback_sections(timed, meter_analysis.beats_per_bar), []
        log.append("form: no sections found; emitting a single unsectioned part")

    # 8. Key is inferred from the canonical notes, not the raw performance, so
    #    that extraction noise that consensus already removed cannot vote.
    canonical_notes = [n for s in sections for n in s.notes]
    key_analysis = infer_key(canonical_notes or timed, cfg.key)
    log.append(f"key: {key_analysis.key} (confidence {key_analysis.confidence:.2f})")

    tune = Tune(
        key=key_analysis.key,
        meter=meter_analysis.meter,
        tempo=rhythm.tempo_bpm,
        sections=sections,
        title=None,
        analysis={
            "key_sharps": key_analysis.sharps,
            "key_alternatives": [c.name for c in key_analysis.alternatives],
            "downbeat_phase": meter_analysis.downbeat_phase,
            "bars_per_section": form.bars_per_section,
            "form_observed": form.form,
            "tempo_raw_bpm": rhythm.raw_tempo_bpm,
        },
    )

    return TranscriptionResult(
        tune=tune,
        audio=audio,
        raw_contour=raw,
        clean_contour=cleaned,
        raw_notes=raw_notes,
        timed_notes=timed,
        rhythm=rhythm,
        meter=meter_analysis,
        key=key_analysis,
        form=form,
        consensus_reports=reports,
        config=cfg,
        log=log,
    )


def transcribe_file(path, config: Config | None = None) -> TranscriptionResult:
    from .audio import load_audio, preprocess

    cfg = config or Config()
    audio = preprocess(load_audio(path, cfg.melody.sample_rate))
    return transcribe(audio, cfg)


def _fallback_sections(notes: list[TimedNote], beats_per_bar: int) -> list[TuneSection]:
    """Emit one unsectioned part when form detection finds no structure.

    Degrading to a plain transcription is the right failure: a wrong AABB split
    would be worse than none, because consensus would then average unrelated
    music together.
    """
    if not notes:
        return []
    # Measures must be contiguous and section-relative, because the score
    # exporter walks them in order and accumulates bar positions. Numbering by
    # absolute bar index instead would leave gaps wherever the performance had a
    # silent bar, and every later barline would land in the wrong place.
    indices = [int(float(n.start_beats) // beats_per_bar) for n in notes]
    first = min(indices)
    origin = Fraction(first * beats_per_bar)

    measures: dict[int, Measure] = {
        i: Measure(notes=[], number=i + 1)
        for i in range(max(indices) - first + 1)
    }
    for n, idx in zip(notes, indices):
        measures[idx - first].notes.append(
            Note(
                pitch=n.pitch,
                start_beats=n.start_beats - origin,
                duration_beats=n.duration_beats,
                confidence=n.confidence,
                agreement=1.0,
            )
        )
    return [TuneSection(name="A", measures=[measures[k] for k in sorted(measures)],
                        repeats=1)]
