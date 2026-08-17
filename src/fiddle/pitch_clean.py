"""Clean a raw predominant-pitch contour without destroying musical detail.

The guiding principle is asymmetry of harm. Leaving an artifact in costs us one
wrong note that the confidence model can flag; aggressively "fixing" real
playing costs us a note that was right and is now silently wrong, with high
confidence. So every stage here is conservative, local, and reversible in the
sense that it lowers confidence on what it touches rather than pretending the
correction was free.

Each stage is individually switchable via CleanConfig so that the eval corpus
can answer "does this stage help?" instead of us assuming it does.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .config import CleanConfig
from .domain import OPEN_STRINGS_MIDI, PitchContour


def clean_contour(contour: PitchContour, config: CleanConfig | None = None) -> PitchContour:
    """Run the enabled cleaning stages in order, returning a new contour."""
    cfg = config or CleanConfig()
    midi = np.array(contour.midi, dtype=float)
    conf = np.array(contour.confidence, dtype=float)
    hop = contour.hop_seconds
    history = list(contour.history)

    stages = [
        (cfg.enable_confidence_gate, _confidence_gate),
        (cfg.enable_range_gate, _range_gate),
        (cfg.enable_jump_gate, _jump_gate),
        (cfg.enable_octave_correction, _octave_correction),
        (cfg.enable_median_filter, _median_filter),
        (cfg.enable_vibrato_smoothing, _vibrato_smoothing),
        (cfg.enable_drone_suppression, _drone_suppression),
        (cfg.enable_short_gap_fill, _short_gap_fill),
    ]
    for enabled, stage in stages:
        if not enabled:
            continue
        midi, conf, note = stage(midi, conf, hop, cfg)
        history.append(note)

    return contour.copy_with(midi=midi, confidence=conf, history=history)


# ---------------------------------------------------------------------------
# Stages. Each takes (midi, conf, hop, cfg) and returns (midi, conf, log line).
# ---------------------------------------------------------------------------


def _confidence_gate(midi, conf, hop, cfg):
    """Drop frames whose salience is negligible relative to this recording."""
    kill = conf < cfg.min_confidence
    n = int(np.sum(kill & ~np.isnan(midi)))
    midi = midi.copy()
    midi[kill] = np.nan
    return midi, conf, f"confidence_gate: unvoiced {n} frames"


def _range_gate(midi, conf, hop, cfg):
    """Anything outside the fiddle's physical range is an extraction error."""
    midi = midi.copy()
    out = (midi < cfg.min_midi) | (midi > cfg.max_midi)
    n = int(np.sum(out & ~np.isnan(midi)))
    midi[out] = np.nan
    return midi, conf, f"range_gate: removed {n} frames"


def _jump_gate(midi, conf, hop, cfg):
    """Remove single frames reached by a physically impossible slew rate.

    A fiddler can leap an octave; they cannot do it in one 3 ms hop. Only
    isolated frames are removed -- a sustained leap is real music and survives,
    because both its entry and exit would have to be impossible.
    """
    midi = midi.copy()
    v = ~np.isnan(midi)
    idx = np.flatnonzero(v)
    if len(idx) < 3:
        return midi, conf, "jump_gate: skipped (too few voiced frames)"

    max_step = cfg.max_jump_semitones_per_sec * hop
    removed = 0
    for a, b, c in zip(idx, idx[1:], idx[2:]):
        if (b - a) > 3 or (c - b) > 3:
            continue  # spans a gap; not a slew-rate question
        up = abs(midi[b] - midi[a])
        down = abs(midi[c] - midi[b])
        outer = abs(midi[c] - midi[a])
        # Spike shape: jumps away and immediately back.
        if up > max_step and down > max_step and outer < up * 0.5:
            midi[b] = np.nan
            removed += 1
    return midi, conf, f"jump_gate: removed {removed} spike frames"


def _octave_correction(midi, conf, hop, cfg):
    """Fold frames sitting exactly an octave off the local melodic centre.

    Octave errors are the dominant Melodia failure on this material: the banjo
    doubles the melody an octave down and the salience function occasionally
    prefers it, or locks onto the second harmonic. The correction is only
    applied when the frame is within a tolerance of *exactly* 12 semitones from
    a robust local reference, which real melodic leaps essentially never are for
    a sustained stretch.
    """
    midi = midi.copy()
    conf = conf.copy()
    win = max(3, int(cfg.octave_window_sec / hop) | 1)

    corrected = 0
    for _ in range(2):  # two passes: fixing some frames improves the reference
        ref = _rolling_nanmedian(midi, win)
        valid = ~np.isnan(midi) & ~np.isnan(ref)
        delta = midi - ref
        for sign in (12.0, -12.0):
            hit = valid & (np.abs(delta - sign) < cfg.octave_tolerance_semitones)
            if not np.any(hit):
                continue
            midi[hit] -= sign
            # The frame is now plausible but we guessed; say so.
            conf[hit] *= 0.85
            corrected += int(np.sum(hit))
    return midi, conf, f"octave_correction: shifted {corrected} frames"


def _median_filter(midi, conf, hop, cfg):
    """Median-filter inside voiced runs to kill isolated one-off estimates.

    Applied per run so the filter never smears a real note across a rest, and
    kept short enough (~45 ms) that a sixteenth note at dance tempo survives.
    """
    midi = midi.copy()
    k = max(3, int(cfg.median_filter_sec / hop) | 1)
    changed = 0
    for s, e in voiced_runs(~np.isnan(midi)):
        if e - s < k:
            continue
        seg = midi[s:e]
        filt = median_filter(seg, size=k, mode="nearest")
        changed += int(np.sum(np.abs(filt - seg) > 0.05))
        midi[s:e] = filt
    return midi, conf, f"median_filter: adjusted {changed} frames (k={k})"


def _vibrato_smoothing(midi, conf, hop, cfg):
    """Flatten small periodic pitch wobble while preserving real motion.

    A moving average slightly longer than one vibrato cycle removes the wobble,
    but applying it everywhere would round off slides and grace notes. So the
    smoothed value is only accepted where the frame's deviation from it is
    small -- i.e. where the motion looks like vibrato rather than melody.
    """
    midi = midi.copy()
    w = max(3, int(cfg.vibrato_window_sec / hop) | 1)
    depth = cfg.vibrato_max_depth_cents / 100.0
    smoothed_frames = 0
    for s, e in voiced_runs(~np.isnan(midi)):
        if e - s < w:
            continue
        seg = midi[s:e]
        sm = uniform_filter1d(seg, size=w, mode="nearest")
        take = np.abs(seg - sm) < depth
        seg[take] = sm[take]
        midi[s:e] = seg
        smoothed_frames += int(np.sum(take))
    return midi, conf, f"vibrato_smoothing: smoothed {smoothed_frames} frames"


def _drone_suppression(midi, conf, hop, cfg):
    """Remove long stretches parked on an open string under a moving melody.

    When a fiddler holds a double-stop drone, the extractor sometimes follows
    the drone instead of the melody. Such a stretch is recognisable: it sits
    within a few cents of an open-string pitch, it is unusually static, and the
    surrounding material is elsewhere. This is off by default -- it is a real
    risk to legitimate long open-string notes, so it must earn its place on the
    eval corpus before being switched on.
    """
    midi = midi.copy()
    conf = conf.copy()
    tol = cfg.drone_tolerance_cents / 100.0
    min_frames = int(cfg.drone_min_duration_sec / hop)
    removed = 0
    for s, e in voiced_runs(~np.isnan(midi)):
        seg = midi[s:e]
        for open_midi in OPEN_STRINGS_MIDI:
            near = np.abs(seg - open_midi) < tol
            for rs, re_ in voiced_runs(near):
                if re_ - rs < min_frames:
                    continue
                # Only suppress if the melody is demonstrably elsewhere nearby.
                ctx = _context_median(midi, s + rs, s + re_, int(1.0 / hop))
                if ctx is not None and abs(ctx - open_midi) > 2.0:
                    midi[s + rs : s + re_] = np.nan
                    removed += re_ - rs
    return midi, conf, f"drone_suppression: removed {removed} frames"


def _short_gap_fill(midi, conf, hop, cfg):
    """Bridge micro-dropouts inside what is clearly one sustained note.

    Only fills when both sides agree in pitch, so this never invents a note
    across a real rest. Filled frames get reduced confidence.
    """
    midi = midi.copy()
    conf = conf.copy()
    max_frames = int(cfg.max_gap_fill_sec / hop)
    filled = 0
    nan = np.isnan(midi)
    for s, e in voiced_runs(nan):
        if e - s > max_frames or s == 0 or e >= len(midi):
            continue
        left, right = midi[s - 1], midi[e]
        if np.isnan(left) or np.isnan(right) or abs(left - right) > 1.0:
            continue
        midi[s:e] = np.linspace(left, right, e - s)
        conf[s:e] = min(conf[s - 1], conf[e]) * 0.8
        filled += e - s
    return midi, conf, f"short_gap_fill: filled {filled} frames"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def voiced_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) index ranges where ``mask`` is True."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def _rolling_nanmedian(x: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware rolling median via a strided view; window is forced odd."""
    n = len(x)
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    strides = np.lib.stride_tricks.sliding_window_view(padded, window)
    with np.errstate(all="ignore"):
        # All-NaN windows (a long rest) legitimately have no median; they come
        # back as NaN, which downstream code already treats as "no reference".
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.nanmedian(strides, axis=-1)
    return out[:n]


def _context_median(midi: np.ndarray, s: int, e: int, radius: int) -> float | None:
    lo, hi = max(0, s - radius), min(len(midi), e + radius)
    ctx = np.concatenate([midi[lo:s], midi[e:hi]])
    ctx = ctx[~np.isnan(ctx)]
    return float(np.median(ctx)) if len(ctx) >= 5 else None
