"""Beat tracking, tempo estimation and quantization.

Rhythm analysis is deliberately independent of pitch analysis: it looks only at
the onset envelope of the full mix. In a jam that is an advantage rather than a
compromise, because the guitar and bass define the beat far more clearly than
the fiddle does.

The two things that actually go wrong here, in order:

1. Tempo octave errors. librosa happily reports half or double the dance tempo.
   Old-time has a narrow, well-known tempo band, which is exactly the kind of
   domain prior this project exists to exploit, so we fold the estimate into
   that band and rebuild the beat grid to match.
2. Quantization that encodes the performance instead of the tune. We search for
   the simplest plausible reading, keep the original seconds alongside it, and
   record how far we had to move each note so the choice stays auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction

import numpy as np

from .config import RhythmConfig
from .domain import Audio, RawNote, TimedNote

# Durations we are willing to notate, in quarter-note beats. Restricting the
# vocabulary is the point: old-time tunes are built from these, and admitting
# quintuplets would let noise masquerade as notation.
_SIMPLE_DURATIONS: tuple[Fraction, ...] = (
    Fraction(1, 4),   # sixteenth
    Fraction(1, 2),   # eighth
    Fraction(3, 4),   # dotted eighth
    Fraction(1, 1),   # quarter
    Fraction(3, 2),   # dotted quarter
    Fraction(2, 1),   # half
    Fraction(3, 1),   # dotted half
    Fraction(4, 1),   # whole
)


@dataclass
class RhythmAnalysis:
    tempo_bpm: float
    beat_times: np.ndarray
    onset_times: np.ndarray
    onset_envelope: np.ndarray
    onset_env_times: np.ndarray
    # Kept so diagnostics can show what the raw tracker said before correction.
    raw_tempo_bpm: float = 0.0
    tempo_correction: float = 1.0
    notes: list[str] = field(default_factory=list)


def analyze_rhythm(audio: Audio, config: RhythmConfig | None = None) -> RhythmAnalysis:
    import librosa

    cfg = config or RhythmConfig()
    y = np.asarray(audio.samples, dtype=np.float32)
    sr = audio.sample_rate
    hop = 512

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop)

    raw_tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames",
        start_bpm=cfg.preferred_bpm,
    )
    raw_tempo = float(np.atleast_1d(raw_tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    beat_times, tempo, factor = _fold_tempo_into_range(beat_times, raw_tempo, cfg)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=True,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)

    return RhythmAnalysis(
        tempo_bpm=tempo,
        beat_times=np.asarray(beat_times, dtype=float),
        onset_times=np.asarray(onset_times, dtype=float),
        onset_envelope=onset_env,
        onset_env_times=env_times,
        raw_tempo_bpm=raw_tempo,
        tempo_correction=factor,
        notes=[f"raw tempo {raw_tempo:.1f} -> {tempo:.1f} BPM (x{factor:g})"],
    )


def _fold_tempo_into_range(
    beat_times: np.ndarray, tempo: float, cfg: RhythmConfig
) -> tuple[np.ndarray, float, float]:
    """Correct tempo-octave errors and rebuild the beat grid to match.

    We pick the power-of-two multiple of the reported tempo that lands inside
    the old-time dance band and sits closest to the typical tempo. Halving keeps
    every other beat; doubling interpolates midpoints, which is safe because the
    subdivision is even by construction.
    """
    if tempo <= 0 or len(beat_times) < 2:
        return beat_times, max(tempo, cfg.preferred_bpm), 1.0

    candidates = [0.25, 0.5, 1.0, 2.0, 4.0]
    scored = []
    for f in candidates:
        t = tempo * f
        if not (cfg.min_bpm <= t <= cfg.max_bpm):
            continue
        scored.append((abs(np.log2(t / cfg.preferred_bpm)), f, t))
    if not scored:
        return beat_times, tempo, 1.0

    _, factor, new_tempo = min(scored)
    bt = np.asarray(beat_times, dtype=float)
    f = factor
    while f > 1.0 + 1e-9:  # double the grid
        mid = (bt[:-1] + bt[1:]) / 2.0
        bt = np.sort(np.concatenate([bt, mid]))
        f /= 2.0
    while f < 1.0 - 1e-9:  # halve the grid
        bt = bt[::2]
        f *= 2.0
    return bt, float(new_tempo), factor


def seconds_to_beats(times: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    """Map wall-clock seconds onto a continuous beat axis.

    Linear interpolation between tracked beats means tempo drift inside the
    performance is absorbed here rather than corrupting the quantization.
    """
    bt = np.asarray(beat_times, dtype=float)
    if len(bt) < 2:
        return np.asarray(times, dtype=float)
    idx = np.arange(len(bt), dtype=float)
    period = float(np.median(np.diff(bt)))
    t = np.asarray(times, dtype=float)
    out = np.interp(t, bt, idx)
    # np.interp clamps outside the grid; extrapolate at the median tempo so
    # notes before the first or after the last tracked beat are still placed.
    before = t < bt[0]
    after = t > bt[-1]
    out[before] = (t[before] - bt[0]) / period
    out[after] = (len(bt) - 1) + (t[after] - bt[-1]) / period
    return out


def quantize_notes(
    notes: list[RawNote],
    rhythm: RhythmAnalysis,
    config: RhythmConfig | None = None,
    beat_offset: float = 0.0,
) -> list[TimedNote]:
    """Place notes on a simple beat grid, preserving their original timing.

    Durations are derived from the *quantized gap to the next note* rather than
    from the measured note length. A fiddler's note-off is governed by bowing,
    not by notation: a written quarter note routinely sounds for 0.7 of a beat.
    Tiling from onset to onset recovers the notated value; genuine rests are
    detected separately as gaps too large to be articulation.
    """
    cfg = config or RhythmConfig()
    if not notes:
        return []

    starts = seconds_to_beats(np.array([n.start_sec for n in notes]), rhythm.beat_times)
    ends = seconds_to_beats(np.array([n.end_sec for n in notes]), rhythm.beat_times)
    starts = starts - beat_offset
    ends = ends - beat_offset

    grid = cfg.quantize_grid
    q_starts = np.round(starts * grid) / grid

    # Enforce strict monotonicity: two notes cannot start on the same tick.
    for i in range(1, len(q_starts)):
        if q_starts[i] <= q_starts[i - 1]:
            q_starts[i] = q_starts[i - 1] + 1.0 / grid

    out: list[TimedNote] = []
    for i, n in enumerate(notes):
        start_q = Fraction(int(round(q_starts[i] * grid)), grid)
        if i + 1 < len(notes):
            span = q_starts[i + 1] - q_starts[i]
        else:
            span = max(ends[i] - q_starts[i], 1.0 / grid)
        dur = _nearest_simple_duration(span, cfg)
        err = abs(starts[i] - q_starts[i])
        out.append(
            TimedNote(
                pitch=n.pitch,
                start_beats=start_q,
                duration_beats=dur,
                confidence=n.confidence,
                start_sec=n.start_sec,
                duration_sec=n.duration_sec,
                pitch_midi=n.pitch_midi,
                quantization_error_beats=float(err),
            )
        )
    return _absorb_interlopers(out, cfg)


def _absorb_interlopers(notes: list[TimedNote], cfg: RhythmConfig) -> list[TimedNote]:
    """Remove notes far shorter than the line around them.

    Melody selection picks the strongest note at each instant, so whenever an
    accompaniment note is briefly louder than the fiddle it appears as a short
    note wedged between two melody notes. Those interlopers are what fragment
    the line: measured, 45% of quantized notes came out a sixteenth long while
    the tune is written almost entirely in eighths, and each one splits the note
    it interrupts.

    The test is *relative* to the local note length rather than absolute,
    because a genuine run of sixteenths is short everywhere -- there the median
    is short too and nothing is absorbed. Only a note much shorter than its own
    neighbourhood is treated as an artifact, and its time is given back to the
    note it interrupted.
    """
    if cfg.interloper_ratio <= 0.0 or len(notes) < 3:
        return notes

    durations = np.array([float(n.duration_beats) for n in notes])
    out: list[TimedNote] = []
    half = max(1, cfg.interloper_window // 2)
    for i, note in enumerate(notes):
        lo, hi = max(0, i - half), min(len(notes), i + half + 1)
        local = np.median(np.delete(durations[lo:hi], min(i - lo, hi - lo - 1)))
        interloper = (
            out
            and float(note.duration_beats) < cfg.interloper_ratio * local
            and note.pitch != out[-1].pitch
        )
        if interloper:
            # Hand the time back to the note this one interrupted.
            out[-1] = replace(
                out[-1],
                duration_beats=out[-1].duration_beats + note.duration_beats,
            )
            continue
        out.append(note)
    return out


def _nearest_simple_duration(span: float, cfg: RhythmConfig) -> Fraction:
    """Snap a measured span to the simplest duration that plausibly explains it.

    Ties and near-ties resolve toward the *simpler* value (fewer dots, larger
    denominator only when clearly needed), because the goal is the tune as it
    would be written, not a transcription of this particular bow stroke.
    """
    allowed = [d for d in _SIMPLE_DURATIONS if cfg.allow_dotted or d.denominator in (1, 2, 4) and d.numerator != 3]
    if not allowed:
        allowed = list(_SIMPLE_DURATIONS)
    span = max(span, float(allowed[0]))
    # Simplicity penalty: dotted values must fit meaningfully better to win.
    best, best_cost = allowed[0], float("inf")
    for d in allowed:
        penalty = 0.04 if d.numerator == 3 else 0.0
        cost = abs(span - float(d)) + penalty
        if cost < best_cost:
            best, best_cost = d, cost
    return best


def infer_beats_per_bar_evidence(rhythm: RhythmAnalysis) -> np.ndarray:
    """Onset strength sampled at each tracked beat.

    Exposed for the meter module, which needs to know which beats are strong
    without re-running onset detection.
    """
    if len(rhythm.beat_times) == 0:
        return np.zeros(0)
    return np.interp(rhythm.beat_times, rhythm.onset_env_times, rhythm.onset_envelope)
