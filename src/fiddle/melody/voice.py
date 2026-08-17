"""Follow a single voice through a polyphonic note stream.

The problem this solves, stated precisely. Basic Pitch returns overlapping note
events for every instrument at once -- around seven simultaneous notes at the
thresholds this project uses. Choosing the melody *per instant* ("keep the
loudest note sounding now") makes every decision independently, so whenever the
guitar happens to be louder than the fiddle for a moment the line jumps to it
and jumps back. Measured, that produced output spread across more pitch classes
than a real tune contains: 41% of notes in the top two pitch classes against 53%
for a real tune, which is not a melody but a line switching between voices.

A melody is a *path*, not a sequence of independent choices. So score whole
paths and take the best one: reward covering time with loud notes, charge for
moving in pitch, for silence, and for overlap. Dynamic programming finds the
global optimum exactly, and the result is by construction a single connected
line rather than a per-frame argmax.

Why this is not the frame-level Viterbi that failed earlier: that decoded over
*frames* of a salience function, where the state space has no notion of a note
and the decoder is free to wander within one. Here the unit is a detected note
event with a real onset, duration and amplitude, so "stay on this note" costs
nothing and "switch" pays a price proportional to how far it moved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


#: A typical melodic note at old-time dance tempo, in seconds. Used only to
#: normalise the reward so that taking one ordinary note scores about 1.0 and
#: the transition costs below can be read as fractions of a note.
_TYPICAL_NOTE_SECONDS = 0.20


@dataclass(frozen=True)
class VoiceWeights:
    """Costs and rewards for path selection, in units of "one ordinary note".

    The scaling is the whole ballgame and getting it wrong is not a matter of
    degree. The first version rewarded ``amplitude * duration`` directly, which
    for a typical note is about 0.09, while charging 0.10 per semitone of
    movement -- so moving a whole tone cost more than twice what taking a note
    was worth, and the optimal path was mathematically forced never to move.
    It duly locked onto a drone: concentration got *worse* than the per-instant
    selector it replaced, 88% to 93% of notes in two pitch classes.

    So the reward is normalised to ~1.0 per note and every cost below is
    expressed as a fraction of that.
    """

    #: Reward for taking a note, scaled by amplitude. About 1.0 for an ordinary
    #: note of typical length and middling amplitude.
    amplitude: float = 2.0

    #: No extra credit for holding a note longer than this. The accompaniment's
    #: signature in this material is sustain, so paying by the second hands the
    #: contest to the guitar before melodic plausibility is considered at all.
    max_note_seconds: float = 0.60

    #: Cost per semitone of movement between consecutive notes. Deliberately
    #: small: the fiddle is the voice that *moves*, and the accompaniment is
    #: what stays put, so a large movement cost selects against the melody.
    step: float = 0.02

    #: Movement beyond this is a leap rather than a step.
    leap_threshold: float = 5.0
    #: Additional cost per semitone beyond the leap threshold. Real melodies do
    #: leap, so this is a discouragement rather than a bar.
    leap: float = 0.05

    #: An octave or more is usually a different instrument rather than the
    #: fiddle jumping. Charged on top of the leap cost.
    octave: float = 0.40

    #: Cost per second of silence between one note ending and the next
    #: beginning. Keeps the path contiguous instead of hopping between
    #: convenient loud notes far apart.
    gap: float = 1.00

    #: Cost per second of overlap. A monophonic line does not sound two notes
    #: at once; some tolerance is needed because detected onsets are imprecise.
    overlap: float = 1.50

    #: Cost for starting a fresh phrase rather than continuing. Without it the
    #: decoder restarts constantly and the path fragments back into per-instant
    #: choices, which is the behaviour this module exists to remove.
    restart: float = 0.80

    #: Cost per semitone of distance from the recording's own melodic centre,
    #: charged on every note rather than on movement. A fiddler stays within
    #: roughly an octave and a half; the point of a band term is that it
    #: constrains *where* the path lives without penalising motion inside it,
    #: which is what a movement cost gets wrong. Zero disables it.
    register: float = 0.0
    #: Semitones either side of the centre that cost nothing.
    register_tolerance: float = 9.0


#: Only notes starting within this many seconds before the current one are
#: considered as predecessors. Bounds the search; a melody note essentially
#: never follows one that began more than a couple of bars earlier.
_LOOKBACK_SECONDS = 2.0


def select_voice(
    events, melody_floor: int, weights: VoiceWeights | None = None
) -> list[tuple[float, float, int, float]]:
    """Choose the single best path through overlapping note events.

    ``events`` is Basic Pitch's output: tuples of
    ``(start_sec, end_sec, pitch_midi, amplitude, ...)``. Returns the chosen
    notes in time order.
    """
    w = weights or VoiceWeights()
    notes = [
        (float(e[0]), float(e[1]), int(e[2]), float(e[3]))
        for e in events
        if int(e[2]) >= melody_floor and float(e[1]) > float(e[0])
    ]
    if not notes:
        return []
    notes.sort(key=lambda n: (n[0], n[1]))

    starts = np.array([n[0] for n in notes])
    ends = np.array([n[1] for n in notes])
    pitches = np.array([n[2] for n in notes], dtype=float)
    amps = np.array([n[3] for n in notes])

    # Reward for taking a note at all. Duration is capped before it counts, so
    # a three-second sustained chord tone is worth no more than a crotchet --
    # otherwise the accompaniment wins on sustain alone.
    durations = np.minimum(ends - starts, w.max_note_seconds)
    reward = w.amplitude * amps * durations / _TYPICAL_NOTE_SECONDS

    if w.register > 0.0:
        # The centre is taken from the note pool itself, weighted by how long
        # each pitch sounds, so it adapts to the recording rather than assuming
        # a key or a register.
        order = np.argsort(pitches)
        cumulative = np.cumsum(durations[order])
        centre = float(pitches[order][np.searchsorted(cumulative,
                                                      cumulative[-1] / 2.0)])
        excess = np.maximum(0.0, np.abs(pitches - centre) - w.register_tolerance)
        reward -= w.register * excess

    best = np.full(len(notes), -np.inf)
    parent = np.full(len(notes), -1, dtype=int)

    # Predecessors are a sliding window: notes sorted by start, so once a
    # candidate begins earlier than the lookback it can be dropped for good.
    window_start = 0
    for i in range(len(notes)):
        while starts[window_start] < starts[i] - _LOOKBACK_SECONDS:
            window_start += 1

        # Starting a new phrase here.
        best[i] = reward[i] - w.restart
        parent[i] = -1

        if window_start < i:
            j = slice(window_start, i)
            # A predecessor must actually precede. Sorting is by (start, end),
            # so notes sounding together are adjacent in the array and would
            # otherwise be chained to one another -- the "single voice" could
            # then sound two notes at once, which is the one property this
            # module exists to guarantee.
            precedes = starts[j] < starts[i]
            delta = np.abs(pitches[j] - pitches[i])
            cost = w.step * delta
            cost += w.leap * np.maximum(0.0, delta - w.leap_threshold)
            cost += w.octave * (delta >= 11.5)
            # Positive where the previous note ended before this one started
            # (a gap), negative where they overlap.
            separation = starts[i] - ends[j]
            cost += w.gap * np.maximum(0.0, separation)
            cost += w.overlap * np.maximum(0.0, -separation)

            candidates = np.where(precedes, best[j] + reward[i] - cost, -np.inf)
            k = int(np.argmax(candidates))
            if candidates[k] > best[i]:
                best[i] = float(candidates[k])
                parent[i] = window_start + k

    # Walk back from the best endpoint.
    path = []
    node = int(np.argmax(best))
    while node >= 0:
        path.append(notes[node])
        node = int(parent[node])
    path.reverse()
    return path
