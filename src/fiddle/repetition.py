"""Measure how strongly a pitch contour repeats, without any ground truth.

This is the objective function for tuning extraction on real recordings, where
we have no transcription to score against. It rests on a fact about the material
rather than on a model: a jam plays a tune round and round, so **if we are
tracking the melody, the contour must be self-similar at the tune's period.**
If we are tracking a drone, the guitar, or noise, it will not be -- or it will be
similar at *every* lag, which is just as diagnostic.

Two properties make this usable as a target to optimize:

* It needs no ground truth, so it works on any recording anyone sends.
* It sits upstream of segmentation, quantization and form analysis, so it
  measures the extractor alone and cannot be confounded by their failures.

The score deliberately measures *excess* similarity at the best lag over the
typical lag. A constant drone scores high everywhere and therefore has no peak,
which is exactly the failure mode we are trying to detect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .domain import PitchContour


@dataclass
class RepetitionScore:
    """How strongly, and at what period, a contour repeats."""

    peak_lag_sec: float
    peak_similarity: float
    baseline_similarity: float
    voiced_fraction: float
    #: The headline number: excess similarity at the best lag over a typical
    #: lag. Near zero means no melodic repetition was tracked.
    excess: float = 0.0
    lags: np.ndarray = field(default_factory=lambda: np.zeros(0))
    curve: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __repr__(self) -> str:
        return (
            f"RepetitionScore(excess={self.excess:.3f}, "
            f"peak={self.peak_similarity:.3f} at {self.peak_lag_sec:.1f}s, "
            f"voiced={self.voiced_fraction:.2f})"
        )


def repetition_score(
    contour: PitchContour,
    min_lag_sec: float = 4.0,
    max_lag_sec: float = 45.0,
    resolution_hz: float = 20.0,
    tolerance_semitones: float = 0.6,
) -> RepetitionScore:
    """Score how strongly ``contour`` repeats, and at what period.

    Similarity at a lag is the fraction of mutually-voiced frames whose pitches
    agree within ``tolerance_semitones``. Frames where either side is unvoiced
    are excluded rather than counted as disagreement, so a contour with gaps is
    not penalised for the gaps themselves -- only for what it does where it
    speaks.
    """
    midi = _resample(contour, resolution_hz)
    voiced = ~np.isnan(midi)
    voiced_fraction = float(np.mean(voiced))
    if voiced_fraction < 0.02:
        return RepetitionScore(0.0, 0.0, 0.0, voiced_fraction, 0.0)

    lags = np.arange(
        int(min_lag_sec * resolution_hz),
        int(max_lag_sec * resolution_hz),
        max(1, int(0.05 * resolution_hz)),
    )
    sims = np.array([_similarity_at_lag(midi, voiced, int(k), tolerance_semitones)
                     for k in lags])
    valid = ~np.isnan(sims)
    if not np.any(valid):
        return RepetitionScore(0.0, 0.0, 0.0, voiced_fraction, 0.0)

    sims = np.where(valid, sims, 0.0)
    peak = int(np.argmax(sims))
    baseline = float(np.median(sims[valid]))
    return RepetitionScore(
        peak_lag_sec=float(lags[peak] / resolution_hz),
        peak_similarity=float(sims[peak]),
        baseline_similarity=baseline,
        voiced_fraction=voiced_fraction,
        excess=float(sims[peak] - baseline),
        lags=lags / resolution_hz,
        curve=sims,
    )


def _resample(contour: PitchContour, hz: float) -> np.ndarray:
    """Downsample the contour to a coarse uniform grid, taking a robust median.

    Coarse on purpose. We are looking for structure on the scale of seconds, and
    a 3 ms hop would make the comparison dominated by vibrato and micro-timing.
    """
    times = np.asarray(contour.times, dtype=float)
    midi = np.asarray(contour.midi, dtype=float)
    if len(times) == 0:
        return np.zeros(0)
    n = int(times[-1] * hz) + 1
    out = np.full(n, np.nan)
    idx = np.minimum((times * hz).astype(int), n - 1)
    order = np.argsort(idx, kind="stable")
    idx_sorted, midi_sorted = idx[order], midi[order]
    starts = np.searchsorted(idx_sorted, np.arange(n), side="left")
    ends = np.searchsorted(idx_sorted, np.arange(n), side="right")
    for i in range(n):
        chunk = midi_sorted[starts[i]:ends[i]]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk):
            out[i] = np.median(chunk)
    return out


def _similarity_at_lag(
    midi: np.ndarray, voiced: np.ndarray, lag: int, tolerance: float
) -> float:
    if lag <= 0 or lag >= len(midi):
        return np.nan
    a, b = midi[:-lag], midi[lag:]
    both = voiced[:-lag] & voiced[lag:]
    if np.sum(both) < 50:
        return np.nan
    diff = np.abs(a[both] - b[both])
    # Octave-equivalent agreement counts: an octave error is still the same
    # melodic material, and we are asking whether the tune repeats.
    ok = (diff < tolerance) | (np.abs(diff - 12.0) < tolerance)
    return float(np.mean(ok))


def contour_on_beat_grid(
    contour: PitchContour, beat_times: np.ndarray, samples_per_beat: int = 8
) -> np.ndarray:
    """Resample a pitch contour onto a uniform beat grid.

    Form analysis needs blocks that are equal in *beats*, not in seconds, so
    that tempo drift inside a performance does not smear the comparison. Taking
    a median within each grid cell keeps the result robust to isolated frames.
    """
    from .rhythm import seconds_to_beats

    times = np.asarray(contour.times, dtype=float)
    midi = np.asarray(contour.midi, dtype=float)
    if len(times) == 0 or len(beat_times) < 2:
        return np.zeros(0)

    beats = seconds_to_beats(times, np.asarray(beat_times, dtype=float))
    cells = np.floor(beats * samples_per_beat).astype(int)
    cells -= cells.min()
    n = int(cells.max()) + 1
    out = np.full(n, np.nan)
    order = np.argsort(cells, kind="stable")
    cells_sorted, midi_sorted = cells[order], midi[order]
    starts = np.searchsorted(cells_sorted, np.arange(n), side="left")
    ends = np.searchsorted(cells_sorted, np.arange(n), side="right")
    for i in range(n):
        chunk = midi_sorted[starts[i]:ends[i]]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk):
            out[i] = np.median(chunk)
    return out
