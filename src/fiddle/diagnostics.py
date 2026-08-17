"""Visual and tabular diagnostics for every intermediate stage.

DSP systems become impossible to improve when their intermediate
representations are invisible. The specific question this module exists to
answer is "why did *that* note appear?", and answering it requires seeing the
raw contour, the cleaned contour and the segmentation on the same time axis,
plus the beat grid that turned seconds into beats.

matplotlib is imported lazily so the transcription engine never depends on it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .pipeline import TranscriptionResult


def write_pitch_csv(result: TranscriptionResult, path: str | Path) -> Path:
    """Frame-level contour: raw and cleaned side by side.

    Both are written to the same file deliberately -- the interesting question
    is almost always what the cleaner changed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw, clean = result.raw_contour, result.clean_contour
    n = min(len(raw.times), len(clean.times))
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "time_sec", "raw_midi", "raw_hz", "confidence",
            "clean_midi", "clean_confidence", "voiced",
        ])
        for i in range(n):
            rm = raw.midi[i]
            cm = clean.midi[i]
            w.writerow([
                f"{raw.times[i]:.4f}",
                "" if np.isnan(rm) else f"{rm:.4f}",
                "" if np.isnan(rm) else f"{440.0 * 2 ** ((rm - 69) / 12):.3f}",
                f"{raw.confidence[i]:.4f}",
                "" if np.isnan(cm) else f"{cm:.4f}",
                f"{clean.confidence[i]:.4f}",
                int(not np.isnan(cm)),
            ])
    return path


def write_notes_json(result: TranscriptionResult, path: str | Path) -> Path:
    """The full note-level record, including provenance and alternatives."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tune = result.tune
    payload = {
        "summary": result.summary(),
        "analysis": tune.analysis,
        "log": result.log,
        "rhythm": {
            "tempo_bpm": result.rhythm.tempo_bpm,
            "raw_tempo_bpm": result.rhythm.raw_tempo_bpm,
            "n_beats": len(result.rhythm.beat_times),
            "beat_times": [round(float(t), 4) for t in result.rhythm.beat_times],
        },
        "meter": {
            "meter": result.meter.meter,
            "beats_per_bar": result.meter.beats_per_bar,
            "downbeat_phase": result.meter.downbeat_phase,
            "confidence": result.meter.confidence,
            "notes": result.meter.notes,
        },
        "key": {
            "key": result.key.key,
            "confidence": result.key.confidence,
            "sharps": result.key.sharps,
            "alternatives": [c.name for c in result.key.alternatives],
            "pitch_class_weights": [round(float(x), 3)
                                    for x in result.key.pitch_class_weights],
        },
        "form": {
            "bars_per_section": result.form.bars_per_section,
            "beats_per_section": result.form.beats_per_section,
            "offset_beats": result.form.offset_beats,
            "confidence": result.form.confidence,
            "observed": result.form.form,
            "passes": [
                {"label": s.label, "index": s.index,
                 "start_beats": round(float(s.start_beats), 3),
                 "similarity_to_reference": round(float(s.similarity_to_reference), 3),
                 "n_notes": len(s.notes)}
                for s in result.form.sections
            ],
            "notes": result.form.notes,
        },
        "consensus": [
            {"label": r.label, "n_observations": r.n_observations,
             "n_used": r.n_used, "excluded": r.excluded,
             "mean_agreement": round(r.mean_agreement, 3), "notes": r.notes}
            for r in result.consensus_reports
        ],
        "raw_notes": [
            {"start_sec": round(n.start_sec, 4), "end_sec": round(n.end_sec, 4),
             "pitch_midi": round(n.pitch_midi, 3), "pitch": n.pitch,
             "confidence": round(n.confidence, 3),
             "spread_cents": round(n.pitch_spread_cents, 1)}
            for n in result.raw_notes
        ],
        "sections": [
            {
                "name": s.name,
                "repeats": s.repeats,
                "confidence": round(s.confidence, 3),
                "measures": [
                    {
                        "number": m.number,
                        "notes": [
                            {
                                "pitch": n.pitch,
                                "is_rest": n.is_rest,
                                "start_beats": str(n.start_beats),
                                "duration_beats": str(n.duration_beats),
                                "confidence": round(n.confidence, 3),
                                "agreement": round(n.agreement, 3),
                                "alternatives": [
                                    {"pitch": a.pitch, "weight": round(a.weight, 3)}
                                    for a in n.alternatives
                                ],
                            }
                            for n in m.notes
                        ],
                    }
                    for m in s.measures
                ],
            }
            for s in tune.sections
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def plot_diagnostics(result: TranscriptionResult, path: str | Path,
                     max_seconds: float = 30.0) -> Path | None:
    """Multi-panel view of the whole pipeline. Returns None if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(16, 16))
    t_max = min(max_seconds, result.audio.duration)

    # 1. Waveform with beats and onsets.
    ax = axes[0]
    sr = result.audio.sample_rate
    n = int(t_max * sr)
    ax.plot(np.arange(n) / sr, result.audio.samples[:n], lw=0.3, color="#888")
    for b in result.rhythm.beat_times[result.rhythm.beat_times < t_max]:
        ax.axvline(b, color="#1f77b4", alpha=0.45, lw=0.8)
    for o in result.rhythm.onset_times[result.rhythm.onset_times < t_max]:
        ax.axvline(o, color="#d62728", alpha=0.3, lw=0.6, ls=":")
    ax.set_title(f"waveform + beats (blue, {result.rhythm.tempo_bpm:.1f} BPM) "
                 f"+ onsets (red dotted)")
    ax.set_xlim(0, t_max)

    # 2. Onset envelope.
    ax = axes[1]
    m = result.rhythm.onset_env_times < t_max
    ax.plot(result.rhythm.onset_env_times[m], result.rhythm.onset_envelope[m],
            color="#d62728", lw=0.8)
    ax.set_title("onset strength envelope")
    ax.set_xlim(0, t_max)

    # 3. Raw vs cleaned contour -- the panel that explains most wrong notes.
    ax = axes[2]
    raw, clean = result.raw_contour, result.clean_contour
    m = raw.times < t_max
    ax.plot(raw.times[m], raw.midi[m], ".", ms=1.2, color="#bbbbbb", label="raw F0")
    mc = clean.times < t_max
    ax.plot(clean.times[mc], clean.midi[mc], ".", ms=1.2, color="#2ca02c",
            label="cleaned F0")
    for note in result.raw_notes:
        if note.start_sec > t_max:
            break
        ax.plot([note.start_sec, note.end_sec], [note.pitch, note.pitch],
                color="#1f77b4", lw=2.4, solid_capstyle="butt")
    ax.set_title("raw F0 (grey) vs cleaned (green) vs segmented notes (blue)")
    ax.set_ylabel("MIDI pitch")
    ax.legend(loc="upper right", markerscale=8)
    ax.set_xlim(0, t_max)

    # 4. Self-similarity between performed passes.
    ax = axes[3]
    sim = result.form.similarity_matrix
    if sim.size:
        im = ax.imshow(sim, cmap="magma", vmin=0, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.025)
        labels = [s.label for s in result.form.sections]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
    ax.set_title(
        f"pass self-similarity -- form {result.form.form or 'none'}, "
        f"{result.form.bars_per_section} bars/section, "
        f"confidence {result.form.confidence:.2f}"
    )

    # 5. Canonical notes coloured by confidence.
    ax = axes[4]
    x = 0
    for section in result.tune.sections:
        for note in section.notes:
            if note.is_rest:
                x += 1
                continue
            colour = plt.cm.RdYlGn(note.confidence)
            ax.bar(x, note.pitch, color=colour, width=0.9)
            x += 1
        ax.axvline(x - 0.5, color="k", lw=1.5)
    ax.set_ylim(50, 95)
    ax.set_title("consensus notes, coloured by confidence (red = uncertain); "
                 "black lines separate sections")
    ax.set_ylabel("MIDI pitch")

    fig.suptitle(
        f"{Path(result.audio.source_path or 'audio').name} | "
        f"{result.tune.key} | {result.tune.meter} | "
        f"{result.tune.tempo:.0f} BPM | form {result.tune.form}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def format_summary(result: TranscriptionResult) -> str:
    """Human-readable console summary, including where we are unsure."""
    s = result.summary()
    lines = [
        f"key           {s['key']}  (confidence {s['key_confidence']:.2f})",
        f"meter         {s['meter']}  (confidence {s['meter_confidence']:.2f})",
        f"tempo         {s['tempo_bpm']:.1f} BPM",
        f"form          {s['form']}  ({s['bars_per_section']} bars/section, "
        f"confidence {s['form_confidence']:.2f})",
        f"notes         {s['n_notes']} canonical, {s['n_uncertain']} flagged uncertain",
        f"melody        {s['melody_backend']} backend, "
        f"{s['voiced_fraction']:.0%} of frames voiced",
    ]
    uncertain = result.tune.uncertain_notes(
        result.config.confidence.uncertain_threshold
    )
    if uncertain:
        lines.append("")
        lines.append("least confident notes:")
        for note in sorted(uncertain, key=lambda n: n.confidence)[:8]:
            alts = ", ".join(
                f"{_name(a.pitch)} {a.weight:.0%}" for a in note.alternatives
            )
            lines.append(
                f"  beat {float(note.start_beats):7.2f}  {_name(note.pitch):<4} "
                f"{note.confidence:.0%}" + (f"   alternatives: {alts}" if alts else "")
            )
    return "\n".join(lines)


_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _name(midi: int) -> str:
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"
