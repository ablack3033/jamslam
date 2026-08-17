"""Turn a cleaned continuous pitch contour into discrete note events.

Two failure modes bound this problem from either side:

* Over-segmentation: treating every wobble as a note change. Guarded by
  requiring a sustained departure from the note's running median.
* Under-segmentation: merging rearticulated repeated notes into one long note.
  This is *the* characteristic old-time failure -- bowing patterns rearticulate
  the same pitch constantly ("d2 dd d2 de") and pitch alone cannot see it. The
  only evidence is the onset envelope, so we use it.
"""

from __future__ import annotations

import numpy as np

from .config import SegmentConfig
from .domain import PitchContour, RawNote
from .pitch_clean import voiced_runs


def segment_notes(
    contour: PitchContour,
    config: SegmentConfig | None = None,
    onset_times: np.ndarray | None = None,
    beat_seconds: float | None = None,
) -> list[RawNote]:
    """Segment ``contour`` into notes, optionally splitting at ``onset_times``.

    ``onset_times`` comes from the rhythm module rather than being computed
    here, so that pitch and rhythm analysis stay independent and each can be
    replaced on its own. ``beat_seconds`` comes from the same place and makes
    the onset-splitting guard musical rather than tempo-blind -- see
    :func:`_split_by_onsets`.
    """
    cfg = config or SegmentConfig()
    midi = np.asarray(contour.midi, dtype=float)
    conf = np.asarray(contour.confidence, dtype=float)
    times = np.asarray(contour.times, dtype=float)
    hop = contour.hop_seconds

    boundaries: list[tuple[int, int]] = []
    for s, e in voiced_runs(~np.isnan(midi)):
        boundaries.extend(_split_run_by_pitch(midi, s, e, cfg))

    if cfg.use_onsets_to_split and onset_times is not None and len(onset_times):
        boundaries = _split_by_onsets(boundaries, times, midi, onset_times, cfg,
                                      hop, beat_seconds)

    notes = [_make_note(midi, conf, times, hop, s, e) for s, e in boundaries]
    notes = [n for n in notes if n.duration_sec >= cfg.min_note_sec]
    notes = [n for n in notes if n.voiced_fraction >= cfg.min_voiced_fraction]
    notes = _merge_adjacent_same_pitch(notes, cfg)
    return notes


def _split_run_by_pitch(
    midi: np.ndarray, start: int, end: int, cfg: SegmentConfig
) -> list[tuple[int, int]]:
    """Cut a voiced run wherever pitch settles at a new level.

    The reference is a running median of the current note rather than the
    previous frame, so a slide into a note does not itself start a new note --
    only arriving somewhere and *staying* does.
    """
    out: list[tuple[int, int]] = []
    seg_start = start
    accum: list[float] = []
    pending = 0
    # A change must persist this long to count; shorter is ornament or slide.
    min_persist = max(2, int(0.030 / max(1e-6, cfg.min_note_sec) * 2))

    for i in range(start, end):
        p = midi[i]
        if not accum:
            accum.append(p)
            continue
        ref = float(np.median(accum[-64:]))
        if abs(p - ref) > cfg.pitch_change_semitones:
            pending += 1
            if pending >= min_persist:
                cut = i - pending + 1
                if cut > seg_start:
                    out.append((seg_start, cut))
                seg_start = cut
                accum = [p]
                pending = 0
        else:
            pending = 0
            accum.append(p)

    if end > seg_start:
        out.append((seg_start, end))
    return out


def _split_by_onsets(
    boundaries: list[tuple[int, int]],
    times: np.ndarray,
    midi: np.ndarray,
    onset_times: np.ndarray,
    cfg: SegmentConfig,
    hop: float,
    beat_seconds: float | None = None,
) -> list[tuple[int, int]]:
    """Subdivide long constant-pitch segments at detected onsets.

    Guarded three ways so we do not shred legitimate long notes: the segment
    must be long enough to hold two notes, both halves must clear the minimum
    duration, and the onset must fall comfortably inside the segment.

    That minimum is measured in *beats* when the tempo is known. Stated in
    seconds it is tempo-blind, and at old-time dance tempo 0.10 s is under half
    an eighth note -- so any onset landing mid-note split it in two, and a jam
    texture supplies such onsets constantly. Measured on the corpus, that
    produced 220 sixteenth notes where the tune has 94 eighths.
    """
    out: list[tuple[int, int]] = []
    min_seconds = cfg.onset_split_min_sec
    if beat_seconds and beat_seconds > 0:
        min_seconds = max(min_seconds, cfg.onset_split_min_beats * beat_seconds)
    min_frames = max(1, int(min_seconds / hop))
    for s, e in boundaries:
        if e - s < 2 * min_frames:
            out.append((s, e))
            continue
        t0, t1 = times[s], times[min(e, len(times) - 1)]
        inside = onset_times[(onset_times > t0 + min_seconds)
                             & (onset_times < t1 - min_seconds)]
        cuts = [s]
        for t in inside:
            idx = int(np.searchsorted(times, t))
            if idx - cuts[-1] >= min_frames and e - idx >= min_frames:
                cuts.append(idx)
        cuts.append(e)
        out.extend(list(zip(cuts, cuts[1:])))
    return out


def _make_note(
    midi: np.ndarray,
    conf: np.ndarray,
    times: np.ndarray,
    hop: float,
    s: int,
    e: int,
) -> RawNote:
    seg = midi[s:e]
    voiced = ~np.isnan(seg)
    vals = seg[voiced]
    if len(vals) == 0:
        return RawNote(float(times[s]), float(times[s]) + hop, 0.0, 0.0,
                       frame_range=(s, e))

    # Trim the attack: the first ~15% of a bowed note is where the pitch is
    # still settling, and including it biases the median.
    trim = max(0, int(len(vals) * 0.12))
    core = vals[trim:] if len(vals) - trim >= 3 else vals
    pitch = float(np.median(core))
    spread = float(np.percentile(np.abs(core - pitch), 90) * 100.0)

    cseg = conf[s:e][voiced]
    salience = float(np.mean(cseg)) if len(cseg) else 0.0
    start_t = float(times[s])
    end_t = float(times[e - 1]) + hop
    return RawNote(
        start_sec=start_t,
        end_sec=end_t,
        pitch_midi=pitch,
        confidence=salience,
        pitch_spread_cents=spread,
        voiced_fraction=float(np.mean(voiced)),
        frame_range=(s, e),
    )


def _merge_adjacent_same_pitch(
    notes: list[RawNote], cfg: SegmentConfig
) -> list[RawNote]:
    """Rejoin fragments of one note that a brief dropout split apart.

    Only merges across very short gaps and only when the rounded pitch matches,
    so genuinely rearticulated notes (which the onset splitter deliberately
    separated) are left alone -- their gap is zero, but they were split *by an
    onset*, which is stronger evidence than pitch continuity.
    """
    if not notes:
        return notes
    out = [notes[0]]
    for n in notes[1:]:
        prev = out[-1]
        gap = n.start_sec - prev.end_sec
        same = prev.pitch == n.pitch
        if same and 0 < gap <= cfg.merge_same_pitch_gap_sec:
            w0, w1 = prev.duration_sec, n.duration_sec
            out[-1] = RawNote(
                start_sec=prev.start_sec,
                end_sec=n.end_sec,
                pitch_midi=(prev.pitch_midi * w0 + n.pitch_midi * w1) / (w0 + w1),
                confidence=(prev.confidence * w0 + n.confidence * w1) / (w0 + w1),
                pitch_spread_cents=max(prev.pitch_spread_cents, n.pitch_spread_cents),
                voiced_fraction=(prev.voiced_fraction * w0 + n.voiced_fraction * w1)
                / (w0 + w1),
                frame_range=(prev.frame_range[0], n.frame_range[1]),
            )
        else:
            out.append(n)
    return out
