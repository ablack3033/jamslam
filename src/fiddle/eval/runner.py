"""Run the pipeline over a corpus and report per-tune, per-metric results.

Two corpora, one code path:

* the synthetic corpus, rendered on demand with exact ground truth;
* a directory of real recordings, each with a ground-truth ``expected.abc``.

Keeping them on the same scoring code is deliberate. The synthetic corpus tells
us whether a change is sound in principle and catches regressions cheaply; only
the real corpus can answer the project's actual question. Any metric that only
exists for one of them would let the two drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..abc_io import load_ground_truth, meter_from_abc
from ..config import Config
from ..corpus.library import ALL_TUNES, TuneDefinition
from ..corpus.synth import render_tune
from ..pipeline import TranscriptionResult, transcribe
from .metrics import TranscriptionScore, score_transcription

HEADLINE_METRICS = (
    "pitch_accuracy",
    "pitch_class_accuracy",
    "onset_accuracy",
    "duration_accuracy",
    "sequence_similarity",
    "octave_error_rate",
)


@dataclass
class CorpusResult:
    name: str
    scores: list[TranscriptionScore] = field(default_factory=list)
    results: list[TranscriptionResult] = field(default_factory=list)

    def aggregate(self) -> dict:
        if not self.scores:
            return {}
        out = {m: float(np.mean([getattr(s, m) for s in self.scores]))
               for m in HEADLINE_METRICS}
        out["key_accuracy"] = float(np.mean([s.key_correct for s in self.scores]))
        out["meter_accuracy"] = float(np.mean([s.meter_correct for s in self.scores]))
        out["form_accuracy"] = float(np.mean([s.form_correct for s in self.scores]))
        out["tempo_error_pct"] = float(np.mean([s.tempo_error_pct for s in self.scores]))
        cal = [s.calibration for s in self.scores if s.calibration]
        if cal:
            out["confidence_auc"] = float(np.mean([c.auc for c in cal]))
            out["flag_precision"] = float(np.mean([c.flag_precision for c in cal]))
            out["flag_recall"] = float(np.mean([c.flag_recall for c in cal]))
            out["mean_wrong_notes"] = float(np.mean([c.n_wrong for c in cal]))
            out["mean_flagged_notes"] = float(np.mean([c.n_flagged for c in cal]))
        return out


#: Rendered audio is identical across configurations, so an ablation that runs
#: the corpus five times would otherwise spend ~40% of its wall clock
#: re-synthesizing byte-identical audio. Keyed by everything that affects the
#: render, and stores preprocessed audio since preprocessing is deterministic.
_RENDER_CACHE: dict[tuple[str, str, int], tuple] = {}


def _render_cached(tune: TuneDefinition, difficulty: str, seed: int):
    from ..audio import preprocess

    key = (tune.slug, difficulty, seed)
    if key not in _RENDER_CACHE:
        audio, truth = render_tune(tune, difficulty=difficulty, seed=seed)
        _RENDER_CACHE[key] = (preprocess(audio), truth)
    return _RENDER_CACHE[key]


def run_synthetic(
    config: Config | None = None,
    tunes: list[TuneDefinition] | None = None,
    difficulties: tuple[str, ...] = ("clean", "jam", "hard"),
    seed: int = 1,
    keep_results: bool = False,
    name: str = "synthetic",
) -> CorpusResult:
    cfg = config or Config()
    tunes = tunes if tunes is not None else list(ALL_TUNES)
    out = CorpusResult(name=name)

    for tune in tunes:
        for difficulty in difficulties:
            audio, truth = _render_cached(tune, difficulty, seed)
            result = transcribe(audio, cfg)
            score = score_transcription(
                result.tune,
                {s.name: s for s in truth.sections},
                truth_key=truth.key,
                truth_meter=truth.meter,
                truth_tempo=truth.tempo_bpm,
                truth_form=truth.form,
                slug=tune.slug,
                difficulty=difficulty,
                uncertain_threshold=cfg.confidence.uncertain_threshold,
            )
            out.scores.append(score)
            if keep_results:
                out.results.append(result)
    return out


def run_dataset(
    root: str | Path,
    config: Config | None = None,
    keep_results: bool = False,
) -> CorpusResult:
    """Score a directory of real recordings.

    Expected layout::

        datasets/tunes/<slug>/
            recording.m4a        (or .wav/.mp3/.flac)
            expected.abc         (or expected.musicxml)
            meta.json            optional: {"key": "...", "tempo_bpm": ..., "form": "AABB"}

    Anything missing degrades gracefully: with no meta.json we score the notes
    and skip key/tempo checks, rather than refusing to run.
    """
    from ..audio import load_audio, preprocess

    root = Path(root)
    cfg = config or Config()
    out = CorpusResult(name=str(root))

    for tune_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        truth_path = _find(tune_dir, ("expected.abc", "expected.musicxml", "expected.xml"))
        audio_path = _find(tune_dir, ("recording.m4a", "recording.wav", "recording.mp3",
                                      "recording.flac", "recording.ogg"))
        if truth_path is None or audio_path is None:
            continue

        truth_sections = load_ground_truth(truth_path)
        meta = {}
        meta_path = tune_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        abc_text = truth_path.read_text() if truth_path.suffix == ".abc" else ""
        audio = preprocess(load_audio(audio_path, cfg.melody.sample_rate))
        result = transcribe(audio, cfg)

        score = score_transcription(
            result.tune,
            truth_sections,
            truth_key=meta.get("key", result.tune.key),
            truth_meter=meta.get("meter", meter_from_abc(abc_text) if abc_text else result.tune.meter),
            truth_tempo=float(meta.get("tempo_bpm", result.tune.tempo)),
            truth_form=meta.get("form", "".join(sorted(truth_sections)) * 2),
            slug=tune_dir.name,
            difficulty="real",
            uncertain_threshold=cfg.confidence.uncertain_threshold,
        )
        if "key" not in meta:
            score.notes.append("no meta.json: key/tempo/meter checks are vacuous")
        out.scores.append(score)
        if keep_results:
            out.results.append(result)
    return out


def _find(directory: Path, names: tuple[str, ...]) -> Path | None:
    for n in names:
        p = directory / n
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_table(result: CorpusResult) -> str:
    rows = [
        f"{'tune':<22}{'diff':<7}{'pitch':>7}{'pclass':>7}{'onset':>7}"
        f"{'dur':>7}{'seq':>7}{'key':>5}{'form':>6}{'tempo%':>8}"
    ]
    rows.append("-" * 89)
    for s in result.scores:
        rows.append(
            f"{s.slug:<22}{s.difficulty:<7}"
            f"{s.pitch_accuracy:>7.2f}{s.pitch_class_accuracy:>7.2f}"
            f"{s.onset_accuracy:>7.2f}{s.duration_accuracy:>7.2f}"
            f"{s.sequence_similarity:>7.2f}"
            f"{'Y' if s.key_correct else 'n':>5}"
            f"{'Y' if s.form_correct else 'n':>6}"
            f"{s.tempo_error_pct:>8.1f}"
        )
    agg = result.aggregate()
    if agg:
        rows.append("-" * 89)
        rows.append(
            f"{'MEAN':<29}"
            f"{agg['pitch_accuracy']:>7.2f}{agg['pitch_class_accuracy']:>7.2f}"
            f"{agg['onset_accuracy']:>7.2f}{agg['duration_accuracy']:>7.2f}"
            f"{agg['sequence_similarity']:>7.2f}"
            f"{agg['key_accuracy']:>5.2f}{agg['form_accuracy']:>6.2f}"
            f"{agg['tempo_error_pct']:>8.1f}"
        )
        if "confidence_auc" in agg:
            rows.append(
                f"\nconfidence: AUC {agg['confidence_auc']:.3f}  "
                f"flag precision {agg['flag_precision']:.2f}  "
                f"flag recall {agg['flag_recall']:.2f}  "
                f"({agg['mean_flagged_notes']:.1f} flagged / "
                f"{agg['mean_wrong_notes']:.1f} wrong per tune)"
            )
        rows.append(f"octave-error rate: {agg['octave_error_rate']:.3f}")
    return "\n".join(rows)


def compare(results: dict[str, CorpusResult]) -> str:
    """Side-by-side aggregate comparison, for ablations."""
    keys = list(HEADLINE_METRICS) + ["key_accuracy", "form_accuracy", "confidence_auc"]
    header = f"{'metric':<24}" + "".join(f"{n:>16}" for n in results)
    rows = [header, "-" * len(header)]
    for k in keys:
        cells = []
        for r in results.values():
            agg = r.aggregate()
            cells.append(f"{agg[k]:>16.3f}" if k in agg else f"{'-':>16}")
        rows.append(f"{k:<24}" + "".join(cells))
    return "\n".join(rows)


def save_report(result: CorpusResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"name": result.name,
         "aggregate": result.aggregate(),
         "scores": [s.to_dict() for s in result.scores]},
        indent=2,
    ))
    return path
