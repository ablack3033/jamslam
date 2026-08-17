"""The single place where per-note confidence is defined.

A product principle for this project is that the system does not have to be
perfect if it reliably knows where it is unsure. That only holds if confidence
means something, so two rules apply here:

1. The formula lives in exactly one function. Confidence computed ad hoc in
   five modules would be uninterpretable and impossible to calibrate.
2. It is a *ranking* signal, not a probability. Melodia's salience is an
   unbounded magnitude, and agreement across four passes is not a likelihood.
   Calling the output "94%" would be false precision. What we require instead
   is that it be **calibrated in rank order**: notes we score low must be wrong
   more often than notes we score high. The eval harness measures exactly that
   (see eval/metrics.py, confidence calibration), and the weights below are
   expected to move when that measurement says they should.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ConfidenceConfig


@dataclass(frozen=True)
class ConfidenceInputs:
    """Evidence available about one note, each already scaled to roughly 0..1."""

    salience: float  # how strongly the pitch tracker asserted this pitch
    stability: float  # how steady the pitch was inside the note
    quantization: float  # how well the timing fitted a simple rhythmic value
    agreement: float  # share of aligned repetitions that voted for this pitch


def score_note(inputs: ConfidenceInputs, config: ConfidenceConfig | None = None) -> float:
    """Blend the evidence into a single 0..1 confidence.

    A weighted mean rather than a product: a product would drive confidence to
    zero whenever any single signal is missing, and on a solo passage there are
    genuinely no repetitions to agree.
    """
    cfg = config or ConfidenceConfig()
    terms = (
        (cfg.w_salience, inputs.salience),
        (cfg.w_stability, inputs.stability),
        (cfg.w_quantization, inputs.quantization),
        (cfg.w_agreement, inputs.agreement),
    )
    total_w = sum(w for w, _ in terms)
    if total_w <= 0:
        return 0.0
    value = sum(w * float(np.clip(v, 0.0, 1.0)) for w, v in terms) / total_w
    return float(np.clip(value, 0.0, 1.0))


def stability_from_spread(spread_cents: float) -> float:
    """Map pitch spread within a note to a 0..1 stability score.

    50 cents of spread is a quarter tone -- at that point the "note" is really a
    slide or a segmentation error, so the score is near zero there and near one
    for a steadily held pitch.
    """
    return float(np.clip(1.0 - spread_cents / 50.0, 0.0, 1.0))


def quantization_score(error_beats: float, tolerance: float = 0.25) -> float:
    """Map how far a note had to move onto the grid into a 0..1 score."""
    return float(np.clip(1.0 - abs(error_beats) / max(tolerance, 1e-6), 0.0, 1.0))
