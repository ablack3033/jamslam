"""Key and mode inference from an observed pitch-class distribution.

Why not Krumhansl-Schmuckler (the textbook answer, and what music21 does by
default)? Because those profiles are derived from common-practice tonal music
and only offer major and minor. A large fraction of the old-time repertoire is
mixolydian (Old Joe Clark, June Apple, Red Haired Boy) or dorian (Cold Frosty
Morning, Shove the Pig's Foot). Forcing those into major/minor produces a key
signature with a wrong accidental on nearly every phrase -- the single most
visible failure a fiddler would notice in the output.

The approach here is deliberately simple and inspectable:

    score(tonic, mode) = scale fit + tonic evidence + repertoire prior

Scale fit alone cannot distinguish a mode from its relatives (D major and B
minor are the same seven pitch classes), so tonic evidence -- weight on the
tonic and fifth, and which pitch the tune lands on -- does the real work.

This module consumes *observed pitches* and produces a *musical
interpretation*. It never touches audio, so re-keying a transcription is
instantaneous and requires no re-analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import KeyConfig
from .domain import Note, TimedNote

# Scale degrees as semitone offsets from the tonic.
MODE_INTERVALS: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "minor": (0, 2, 3, 5, 7, 8, 10),
}

# How many sharps(+)/flats(-) the *relative major* of each mode implies, as an
# offset from the tonic's own major key signature.
_MODE_SIGNATURE_OFFSET = {"major": 0, "mixolydian": -1, "dorian": -2, "minor": -3}

_PC_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC_NAMES_FLAT = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]
# Number of sharps in the major key on each pitch class (negative = flats).
_MAJOR_SHARPS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}


@dataclass
class KeyCandidate:
    tonic_pc: int
    mode: str
    score: float
    name: str

    def __repr__(self) -> str:
        return f"{self.name} ({self.score:.3f})"


@dataclass
class KeyAnalysis:
    key: str  # e.g. "A mixolydian"
    tonic_pc: int
    mode: str
    sharps: int  # key signature, positive sharps / negative flats
    confidence: float
    pitch_class_weights: np.ndarray = field(default_factory=lambda: np.zeros(12))
    alternatives: list[KeyCandidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def infer_key(
    notes: list[TimedNote] | list[Note],
    config: KeyConfig | None = None,
) -> KeyAnalysis:
    cfg = config or KeyConfig()
    weights = pitch_class_weights(notes)

    if cfg.override:
        tonic_pc, mode = _parse_key(cfg.override)
        return KeyAnalysis(
            key=cfg.override, tonic_pc=tonic_pc, mode=mode,
            sharps=_signature_for(tonic_pc, mode), confidence=1.0,
            pitch_class_weights=weights, notes=["key overridden by user"],
        )

    if weights.sum() <= 0:
        return KeyAnalysis("D major", 2, "major", 2, 0.0, weights,
                           notes=["no pitches observed; defaulting to D major"])

    w = weights / weights.sum()
    final_pc = _final_pitch_class(notes)
    candidates: list[KeyCandidate] = []

    for tonic in range(12):
        for mode in cfg.modes:
            intervals = MODE_INTERVALS[mode]
            scale = [(tonic + i) % 12 for i in intervals]

            # 1. How much of the tune's weight falls inside the scale.
            fit = float(w[scale].sum())

            # 2. Tonic evidence. The tonic and fifth carry disproportionate
            #    weight in fiddle tunes (drone strings, cadences), and the tune
            #    almost always lands on the tonic.
            tonic_w = float(w[tonic])
            fifth_w = float(w[(tonic + 7) % 12])
            tonic_ev = 1.6 * tonic_w + 0.5 * fifth_w
            if final_pc is not None and final_pc == tonic:
                tonic_ev += 0.12

            # 3. Repertoire priors, applied gently so evidence can override.
            name = _key_name(tonic, mode)
            root = name.split()[0]
            prior = cfg.key_prior.get(root, 0.2) * cfg.mode_prior.get(mode, 0.3)

            score = fit + tonic_ev + cfg.prior_weight * prior
            candidates.append(KeyCandidate(tonic, mode, score, name))

    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0]
    runner = candidates[1]
    # Confidence is the margin over the best alternative, squashed. A tune that
    # fits two readings equally well should say so rather than pick one boldly.
    margin = (best.score - runner.score) / max(best.score, 1e-6)
    confidence = float(np.clip(0.45 + margin * 6.0, 0.0, 0.99))

    return KeyAnalysis(
        key=best.name,
        tonic_pc=best.tonic_pc,
        mode=best.mode,
        sharps=_signature_for(best.tonic_pc, best.mode),
        confidence=confidence,
        pitch_class_weights=weights,
        alternatives=candidates[1:5],
        notes=[
            f"top: {best.name} {best.score:.3f}; next: {runner.name} {runner.score:.3f}",
            f"final note pitch class: {final_pc}",
        ],
    )


def pitch_class_weights(notes) -> np.ndarray:
    """Duration-and-confidence weighted pitch-class histogram.

    Weighting by duration matters: a passing sixteenth is much weaker evidence
    about the key than a held cadential note. Weighting by confidence keeps
    likely extraction errors from voting.
    """
    w = np.zeros(12)
    for n in notes:
        if getattr(n, "is_rest", False):
            continue
        dur = float(n.duration_beats)
        w[n.pitch % 12] += dur * max(0.1, float(n.confidence))
    return w


def _final_pitch_class(notes) -> int | None:
    for n in reversed(list(notes)):
        if not getattr(n, "is_rest", False):
            return int(n.pitch) % 12
    return None


def _key_name(tonic_pc: int, mode: str) -> str:
    sharps = _signature_for(tonic_pc, mode)
    names = _PC_NAMES_FLAT if sharps < 0 else _PC_NAMES_SHARP
    return f"{names[tonic_pc]} {mode}"


def _signature_for(tonic_pc: int, mode: str) -> int:
    """Key signature for a modal tonic, expressed in sharps (negative = flats).

    A mode takes the key signature of its parent major scale. A mixolydian is
    the fifth mode of D major, so it is written with two sharps rather than A
    major's three -- that one-accidental correction is what stops every G in an
    Old Joe Clark transcription from coming out as G#.
    """
    base = _MAJOR_SHARPS[tonic_pc % 12]
    sharps = base + _MODE_SIGNATURE_OFFSET.get(mode, 0)
    # Fold enharmonically absurd signatures back into readable territory.
    while sharps > 7:
        sharps -= 12
    while sharps < -7:
        sharps += 12
    return sharps


def _parse_key(text: str) -> tuple[int, str]:
    parts = text.replace("-", "-").split()
    root = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "major"
    if mode not in MODE_INTERVALS:
        mode = "major"
    for pc, name in enumerate(_PC_NAMES_SHARP):
        if name == root:
            return pc, mode
    for pc, name in enumerate(_PC_NAMES_FLAT):
        if name == root:
            return pc, mode
    raise ValueError(f"cannot parse key {text!r}")
