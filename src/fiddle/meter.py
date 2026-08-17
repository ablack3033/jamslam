"""Meter inference and downbeat phase.

An honest statement of what is and is not knowable here:

2/4 and 4/4 are *metrically nested*. The same performance notated in 2/4 with
16 bars and in 4/4 with 8 bars is the same audio, and old-time practice writes
the same tune both ways. So "detecting the meter" from audio is not a
well-posed problem, and any confident answer would be false precision. We infer
a preference, report low confidence when the evidence is weak, and make the
value trivially overridable.

What *is* well-posed and matters far more is the **downbeat phase**: which
tracked beat is beat one. Get that wrong and every barline in the score is
wrong, which is far more visible to a musician than 2/4 vs 4/4. Most of this
module's effort goes there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MeterConfig
from .domain import TimedNote
from .rhythm import RhythmAnalysis, infer_beats_per_bar_evidence


@dataclass
class MeterAnalysis:
    meter: str
    beats_per_bar: int
    # Offset, in beats, from the first tracked beat to the first downbeat.
    downbeat_phase: float
    confidence: float
    notes: list[str] = field(default_factory=list)


def infer_meter(
    rhythm: RhythmAnalysis,
    notes: list[TimedNote] | None = None,
    config: MeterConfig | None = None,
) -> MeterAnalysis:
    cfg = config or MeterConfig()
    if cfg.override:
        num, den = cfg.override.split("/")
        bpb = int(num) * 4 // int(den)
        phase = _best_phase(rhythm, bpb)
        return MeterAnalysis(cfg.override, bpb, phase, 1.0,
                             [f"meter overridden to {cfg.override}"])

    strength = infer_beats_per_bar_evidence(rhythm)
    if len(strength) < 8:
        return MeterAnalysis(cfg.default, 4, 0.0, 0.2,
                             ["too few beats to infer meter; using default"])

    # Phase is estimated on a 4-beat hypothesis because it subsumes the 2-beat
    # one: if the true meter is 2/4, phases 0 and 2 score alike and either is a
    # correct downbeat.
    phase4 = _best_phase(rhythm, 4)
    per_beat = _mean_strength_by_position(strength, 4, phase4)

    # In 4/4 the third beat is a secondary stress, clearly weaker than the
    # first. In 2/4 (barred half as often) it *is* a downbeat and matches it.
    one, three = per_beat[0], per_beat[2]
    ratio = float(three / one) if one > 1e-9 else 1.0
    # The threshold is deliberately strict, so 4/4 wins unless beats one and
    # three are near-indistinguishable. Both readings are musically defensible,
    # and 4/4 is the more common notation for this repertoire, so we bias
    # toward it rather than flipping on weak evidence.
    threshold = 0.985
    if ratio > threshold:
        meter, bpb = "2/4", 2
    else:
        meter, bpb = "4/4", 4

    # Confidence reflects how far the evidence is from the ambiguous midpoint,
    # deliberately capped: this distinction is weak by nature.
    confidence = float(min(0.75, abs(ratio - threshold) * 3.0 + 0.25))
    phase = phase4 % bpb
    return MeterAnalysis(
        meter=meter,
        beats_per_bar=bpb,
        downbeat_phase=phase,
        confidence=confidence,
        notes=[
            f"beat-position strengths (phase {phase4}): "
            + ", ".join(f"{v:.2f}" for v in per_beat),
            f"beat3/beat1 = {ratio:.3f} -> {meter} (2/4 vs 4/4 is weakly identifiable)",
        ],
    )


def _mean_strength_by_position(
    strength: np.ndarray, bpb: int, phase: int
) -> np.ndarray:
    idx = (np.arange(len(strength)) - phase) % bpb
    return np.array([float(np.mean(strength[idx == k])) if np.any(idx == k) else 0.0
                     for k in range(bpb)])


def _best_phase(rhythm: RhythmAnalysis, bpb: int) -> int:
    """Choose the beat offset whose downbeats carry the most onset energy."""
    strength = infer_beats_per_bar_evidence(rhythm)
    if len(strength) < bpb:
        return 0
    scores = []
    for phase in range(bpb):
        idx = (np.arange(len(strength)) - phase) % bpb == 0
        scores.append(float(np.mean(strength[idx])) if np.any(idx) else 0.0)
    return int(np.argmax(scores))
