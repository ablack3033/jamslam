"""Continuous melody tracking: a Viterbi decode over pitch-salience candidates.

The previous selector scored contours and then let them compete frame by frame.
That is a local decision, and it shows: the chosen line hops between voices
whenever the accompaniment is momentarily louder than the fiddle, which is often.

A melody is a *path*, so decode it as one. Each frame offers ~20 salience peaks;
the decode chooses one per frame, trading the strength of each candidate against
the cost of jumping to it from the previous choice. A brief burst of loud guitar
no longer captures the line, because switching to it and back costs more than
staying on a slightly quieter fiddle.

The trap in doing this is that continuity alone loves a drone: a sustained open
string is both salient and perfectly continuous, so an unmodified decode parks on
it forever. That is why the emission term carries a **persistence penalty**,
computed before decoding: a spectral peak that sits at one pitch for a second or
more is a drone or a held chord tone, and is penalised in proportion to how long
it has held. This is the term that makes continuity safe to use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Candidate pitches are binned this finely when measuring how long a peak holds.
_PERSISTENCE_BIN = 0.5  # semitones
#: A peak holding one pitch longer than this is drone-like rather than melodic.
#:
#: This was 0.9s and that was wrong: a held note at the end of a phrase easily
#: lasts a second, so the rule condemned legitimate melody notes along with
#: drones. Every contour in a test case came back classified as a drone. At
#: dance tempo 2.5s is about five beats -- long past any note a fiddler holds
#: mid-tune, but comfortably above a phrase-ending long note.
_DRONE_SECONDS = 2.5


@dataclass(frozen=True)
class TrackingWeights:
    """Everything the decode trades off, in one place so it can be swept."""

    salience: float = 1.0
    #: Reward for sitting above the other voices. The melody is usually the top
    #: line. Measured on bebop_1, contour-level decoding needs this fairly high
    #: (0.8) to leave the accompaniment at all; below ~0.4 it collapses onto a
    #: sustained drone and covers almost none of the recording.
    height: float = 0.8
    #: Penalty for a candidate whose pitch has been held for a long time.
    drone: float = 2.2
    #: Cost per semitone of movement between consecutive frames.
    continuity: float = 1.4
    #: Extra cost for jumping an octave, the classic extraction error.
    octave_jump: float = 1.5
    #: Movement up to this size is free -- vibrato and intonation drift.
    free_motion: float = 0.4
    #: Frames whose chosen candidate is weaker than this fraction of the typical
    #: chosen salience are marked unvoiced. Without it the decode is obliged to
    #: emit a pitch in every frame, including silence and applause, and those
    #: invented frames wreck everything downstream.
    voicing_threshold: float = 0.35


def persistence_seconds(
    pitches: list[np.ndarray], hop_seconds: float, n_bins_pad: float = 0.0
) -> list[np.ndarray]:
    """For each candidate, how long a peak has continuously held that pitch.

    Computed on a coarse pitch grid so that vibrato does not break a held note
    into fragments. Runs are measured through short gaps for the same reason.
    """
    if not pitches:
        return []
    finite = [p[np.isfinite(p)] for p in pitches if len(p)]
    if not any(len(p) for p in finite):
        return [np.zeros(len(p)) for p in pitches]

    lo = min(float(p.min()) for p in finite if len(p))
    hi = max(float(p.max()) for p in finite if len(p))
    n_bins = max(1, int(np.ceil((hi - lo) / _PERSISTENCE_BIN)) + 1)
    n_frames = len(pitches)

    present = np.zeros((n_frames, n_bins), dtype=bool)
    for t, p in enumerate(pitches):
        if len(p) == 0:
            continue
        idx = np.clip(((p - lo) / _PERSISTENCE_BIN).astype(int), 0, n_bins - 1)
        present[t, idx] = True
        # Mark neighbours too, so a note wobbling across a bin edge still counts
        # as one held pitch rather than two alternating ones.
        present[t, np.clip(idx - 1, 0, n_bins - 1)] = True
        present[t, np.clip(idx + 1, 0, n_bins - 1)] = True

    run = np.zeros((n_frames, n_bins), dtype=np.float32)
    for b in range(n_bins):
        col = present[:, b]
        if not col.any():
            continue
        run[:, b] = _run_lengths(col) * hop_seconds

    out = []
    for t, p in enumerate(pitches):
        if len(p) == 0:
            out.append(np.zeros(0))
            continue
        idx = np.clip(((p - lo) / _PERSISTENCE_BIN).astype(int), 0, n_bins - 1)
        out.append(run[t, idx])
    return out


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Length of the True-run containing each position (0 where False)."""
    out = np.zeros(len(mask), dtype=np.float32)
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out[start:i] = i - start
            start = None
    if start is not None:
        out[start:] = len(mask) - start
    return out


def decode_melody(
    pitches: list[np.ndarray],
    saliences: list[np.ndarray],
    hop_seconds: float,
    weights: TrackingWeights | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Viterbi-decode a single continuous melodic line through the candidates.

    Returns (midi, confidence) arrays with one entry per frame; frames with no
    candidate at all come back as NaN.
    """
    w = weights or TrackingWeights()
    n_frames = len(pitches)
    if n_frames == 0:
        return np.zeros(0), np.zeros(0)

    max_k = max((len(p) for p in pitches), default=0)
    if max_k == 0:
        return np.full(n_frames, np.nan), np.zeros(n_frames)

    persistence = persistence_seconds(pitches, hop_seconds)
    emission = _emission_scores(pitches, saliences, persistence, max_k, w)

    pitch_grid = np.full((n_frames, max_k), np.nan)
    for t, p in enumerate(pitches):
        pitch_grid[t, : len(p)] = p

    # Forward pass. Kept as an explicit loop over frames because the transition
    # cost depends on the candidate pitches, which differ every frame.
    score = np.full((n_frames, max_k), -np.inf)
    back = np.zeros((n_frames, max_k), dtype=np.int16)
    score[0] = emission[0]

    for t in range(1, n_frames):
        prev_pitch = pitch_grid[t - 1]
        cur_pitch = pitch_grid[t]
        valid_prev = np.isfinite(prev_pitch) & np.isfinite(score[t - 1])
        if not valid_prev.any():
            score[t] = emission[t]
            continue
        # transition[j, k]: cost of moving from previous candidate j to current k
        delta = np.abs(cur_pitch[None, :] - prev_pitch[:, None])
        cost = _movement_cost(delta, w)
        total = score[t - 1][:, None] - cost
        total[~valid_prev, :] = -np.inf
        best = np.argmax(total, axis=0)
        score[t] = total[best, np.arange(max_k)] + emission[t]
        back[t] = best

    # Backward pass.
    midi = np.full(n_frames, np.nan)
    conf = np.zeros(n_frames)
    last_valid = np.isfinite(score[-1])
    if not last_valid.any():
        return midi, conf
    k = int(np.argmax(np.where(last_valid, score[-1], -np.inf)))
    for t in range(n_frames - 1, -1, -1):
        if k < len(pitches[t]):
            midi[t] = pitches[t][k]
            conf[t] = saliences[t][k] if k < len(saliences[t]) else 0.0
        k = int(back[t][k])
    # Voicing. The decode always chooses *something*, so silence has to be
    # removed afterwards, by comparing each chosen candidate against how strong
    # a chosen candidate typically is in this recording.
    chosen = conf[np.isfinite(midi)]
    if len(chosen):
        reference = float(np.median(chosen[chosen > 0])) if np.any(chosen > 0) else 0.0
        if reference > 0:
            midi = np.where(conf < w.voicing_threshold * reference, np.nan, midi)
    peak = float(np.max(conf)) if np.any(conf > 0) else 1.0
    return midi, np.clip(conf / peak, 0.0, 1.0)


def _emission_scores(pitches, saliences, persistence, max_k, w) -> np.ndarray:
    """Per-candidate desirability, before any continuity is considered."""
    n_frames = len(pitches)
    out = np.full((n_frames, max_k), -np.inf)

    all_p = np.concatenate([p for p in pitches if len(p)]) if any(
        len(p) for p in pitches) else np.zeros(1)
    lo, hi = np.percentile(all_p, 10), np.percentile(all_p, 90)
    span = max(float(hi - lo), 1e-6)

    all_s = np.concatenate([s for s in saliences if len(s)]) if any(
        len(s) for s in saliences) else np.zeros(1)
    s_ref = float(np.percentile(all_s, 95)) or 1.0

    for t in range(n_frames):
        p, s = pitches[t], saliences[t]
        if len(p) == 0:
            continue
        k = len(p)
        strength = np.log1p(np.clip(s[:k], 0, None) / s_ref * 9.0) / np.log(10.0)
        height = np.clip((p - lo) / span, 0.0, 1.0)
        held = persistence[t][:k] if len(persistence) > t else np.zeros(k)
        drone = np.clip(held / _DRONE_SECONDS, 0.0, 2.0)
        out[t, :k] = (
            w.salience * strength + w.height * height - w.drone * drone
        )
    return out


def _movement_cost(delta: np.ndarray, w: TrackingWeights) -> np.ndarray:
    """Cost of moving ``delta`` semitones between consecutive frames.

    Free below a threshold so vibrato is not fought, then linear, with an extra
    bump around the octave because octave switching is the characteristic
    failure of salience-based tracking rather than a thing melodies do often.
    """
    d = np.nan_to_num(delta, nan=99.0)
    cost = w.continuity * np.clip(d - w.free_motion, 0.0, None)
    cost += w.octave_jump * (np.abs(d - 12.0) < 1.0)
    return cost


# ---------------------------------------------------------------------------
# Contour-level decoding
#
# Frame-level decoding turned out to be the wrong unit. Measured on a real
# recording it cured the symptoms -- bass locking fell from 63% to 8%, drone
# parking from 42% to 11% -- while making the result *worse* on the measure that
# matters, repetition at the tune's period, which collapsed from 0.287 to 0.04.
# Low locking and low drone-parking are necessary but not sufficient: aimless
# wandering also scores well on both. The decode had cured the drone by
# following nothing at all.
#
# Essentia's PitchContours already solves the hard part -- grouping frames into
# coherent pitch fragments under continuity constraints. The real question was
# never "which peak this frame" but "which fragment next", so the decode belongs
# at that level.
# ---------------------------------------------------------------------------


def decode_contours(
    contour_pitches: list[np.ndarray],
    contour_saliences: list[np.ndarray],
    start_frames: list[int],
    n_frames: int,
    hop_seconds: float,
    weights: TrackingWeights | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a sequence of pitch contours forming one continuous melodic line.

    A shortest-path over fragments: each contour is worth its salience and
    length, discounted if it is a drone, and moving between contours costs
    according to the pitch jump and the silence between them.
    """
    w = weights or TrackingWeights()
    n = len(contour_pitches)
    midi = np.full(n_frames, np.nan)
    conf = np.zeros(n_frames)
    if n == 0:
        return midi, conf

    order = np.argsort(np.asarray(start_frames))
    ends, values = [], []
    all_medians = np.array([float(np.median(p)) if len(p) else np.nan
                            for p in contour_pitches])
    finite = all_medians[np.isfinite(all_medians)]
    lo = float(np.percentile(finite, 10)) if len(finite) else 0.0
    hi = float(np.percentile(finite, 90)) if len(finite) else 1.0
    span = max(hi - lo, 1e-6)

    for i in range(n):
        p = contour_pitches[i]
        s = contour_saliences[i]
        ends.append(start_frames[i] + len(p))
        duration = len(p) * hop_seconds
        spread = float(np.percentile(p, 90) - np.percentile(p, 10)) if len(p) > 2 else 0.0
        mean_sal = float(np.mean(s)) if len(s) else 0.0
        height = float(np.clip((np.median(p) - lo) / span, 0.0, 1.0)) if len(p) else 0.0
        # A long fragment that does not move is a drone, however loud.
        # The penalty must be MULTIPLICATIVE. An additive one cannot work here,
        # because a contour's value already scales with its duration, so a long
        # drone simply outgrows any fixed subtraction -- which is exactly what
        # happened: the decode selected a single sustained A4 covering 6% of the
        # recording and called it the melody.
        drone_factor = 0.05 if (duration > _DRONE_SECONDS and spread < 0.5) else 1.0
        values.append(
            duration * (w.salience * mean_sal + w.height * height) * drone_factor
        )

    best = np.full(n, -np.inf)
    prev = np.full(n, -1, dtype=int)
    for oi, i in enumerate(order):
        best[i] = values[i]
        for j in order[:oi]:
            # Contours may overlap: Essentia emits several candidate fragments
            # covering the same instant. Requiring strict succession starved the
            # path down to a handful of fragments. What matters is that the line
            # advances in time; a later fragment simply takes over where they
            # overlap.
            if start_frames[i] <= start_frames[j]:
                continue
            gap_frames = start_frames[i] - ends[j]
            if gap_frames * hop_seconds > 2.0:
                continue  # too far apart to be one line
            jump = abs(float(np.median(contour_pitches[i]))
                       - float(np.median(contour_pitches[j])))
            cost = w.continuity * max(0.0, jump - w.free_motion) * 0.25
            cost += w.octave_jump * (abs(jump - 12.0) < 1.0)
            cost += 0.5 * max(0.0, gap_frames) * hop_seconds
            candidate = best[j] - cost + values[i]
            if candidate > best[i]:
                best[i] = candidate
                prev[i] = j

    node = int(np.argmax(best))
    while node >= 0:
        p, s = contour_pitches[node], contour_saliences[node]
        a = max(0, start_frames[node])
        b = min(n_frames, a + len(p))
        if b > a:
            # Fill only frames no later contour in the path has claimed. The
            # path is walked backwards and contours overlap, so without this an
            # early long contour silently overwrites every choice after it --
            # which made a single sustained drone swallow the whole decode.
            free = np.isnan(midi[a:b])
            idx = np.arange(a, b)[free]
            midi[idx] = p[: b - a][free]
            if len(s) >= b - a:
                conf[idx] = s[: b - a][free]
            elif len(s):
                conf[idx] = float(np.mean(s))
        node = int(prev[node])

    peak = float(np.max(conf)) if np.any(conf > 0) else 1.0
    return midi, np.clip(conf / peak, 0.0, 1.0)
