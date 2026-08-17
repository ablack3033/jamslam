"""Detect repeated sections (AABB and friends) in a performed tune.

The point of this module is *not* to avoid printing four sections. It is to
discover which stretches of the performance are repeated measurements of the
same underlying music, so the consensus builder can average away the noise.
Form detection is therefore the gateway to the project's central hypothesis,
and its failure mode is severe: a wrong segmentation makes consensus actively
harmful, averaging together things that were never the same. Everything here is
built to fail loudly (low confidence, few sections) rather than quietly.

Method:
  1. Rasterize the quantized notes onto a fixed eighth-note slot grid, so any
     two equal-length stretches can be compared elementwise.
  2. Search over section lengths in *bars* and over bar-aligned start offsets,
     partitioning the timeline into blocks of that length.
  3. Score each hypothesis by how cleanly the blocks cluster: high similarity
     within a cluster, low between clusters.
  4. Label clusters A, B, C... in order of first appearance.

An 8-bar section gets a small bonus, because it is genuinely the most common
length -- but it is a bonus, not a constraint. The corpus includes a crooked
6-bar tune specifically to keep that honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import FormConfig
from .domain import TimedNote

REST = -1
FORM_GRID = 2  # slots per beat; eighth notes are enough to see structure


@dataclass
class SectionInstance:
    """One performed pass of a section."""

    label: str
    index: int
    start_beats: float
    end_beats: float
    notes: list[TimedNote]  # start_beats made relative to the section start
    similarity_to_reference: float = 0.0


@dataclass
class FormAnalysis:
    sections: list[SectionInstance]
    bars_per_section: int
    beats_per_section: float
    confidence: float
    # Beat position of the first section boundary. This doubles as the best
    # available downbeat estimate -- see analyze_form's docstring.
    offset_beats: float = 0.0
    similarity_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    notes: list[str] = field(default_factory=list)

    @property
    def form(self) -> str:
        return "".join(s.label for s in self.sections)

    @property
    def labels(self) -> list[str]:
        seen: list[str] = []
        for s in self.sections:
            if s.label not in seen:
                seen.append(s.label)
        return seen

    def observations(self, label: str) -> list[SectionInstance]:
        return [s for s in self.sections if s.label == label]


def analyze_form(
    notes: list[TimedNote],
    beats_per_bar: int,
    config: FormConfig | None = None,
) -> FormAnalysis:
    cfg = config or FormConfig()
    if len(notes) < 8:
        return FormAnalysis([], 0, 0.0, 0.0, notes=["too few notes for form analysis"])

    first_note_beat = _tune_start_beat(notes)
    start = float(notes[0].start_beats)
    end = float(notes[-1].end_beats)
    # Snap the analysis window to a barline; quantization already placed beat 0
    # on a downbeat, so flooring to a bar keeps blocks bar-aligned.
    start = np.floor(start / beats_per_bar) * beats_per_bar
    total_beats = end - start
    if total_beats < cfg.min_section_bars * beats_per_bar * 2:
        return FormAnalysis([], 0, 0.0, 0.0,
                            notes=["performance too short to show repetition"])

    best = None
    log: list[str] = []
    # Search section length in BEATS, not bars. Bars are a notational choice --
    # 2/4 and 4/4 are metrically nested, so a section that is 8 bars of 4/4 is
    # 16 bars of 2/4 and both are correct. Tying the search to bar counts made
    # a benign meter ambiguity split real sections in half. Beats are the
    # meter-independent quantity, and stepping by beats_per_bar keeps blocks
    # bar-aligned either way.
    min_block = max(cfg.min_section_bars * beats_per_bar, beats_per_bar)
    max_block = min(cfg.max_section_bars * 4, int(total_beats // 2))
    candidates: list[tuple] = []

    # Step by ONE BEAT, not one bar. Old-time is full of crooked tunes -- a
    # section with a 2-beat or 6-beat bar in it, so the section total is not a
    # multiple of the bar length. Stepping by whole bars made those sections
    # literally unrepresentable: the correct hypothesis was never generated, and
    # the search silently returned the nearest wrong one. This costs more
    # hypotheses, paid for by a narrower offset search below.
    for block_beats in range(min_block, max_block + 1):
        n_blocks_max = int(total_beats // block_beats)
        if n_blocks_max < 3:  # need at least one repeat plus a contrast
            continue
        bars = block_beats / beats_per_bar
        # Offsets are searched at BEAT resolution, not bar resolution. The
        # downbeat phase from onset energy is only about 70% reliable, and if it
        # is off by one beat then bar-aligned block starts can never coincide
        # with a real section boundary -- every hypothesis is then wrong by a
        # constant, which looks like a melody failure but is not. Searching
        # every beat removes that dependency, and the winning offset becomes our
        # best downbeat estimate, since a section boundary is by definition a
        # downbeat and is far better evidence than onset energy.
        for offset_beats in range(0, min(block_beats, 8)):
            offset = start + offset_beats
            blocks = _blocks(notes, offset, block_beats, total_beats + start)
            # A jam plays at least AABB, so a hypothesis yielding fewer than
            # four blocks is almost always a doubled section length that
            # survived only because it had too little data to contradict it.
            if len(blocks) < 4:
                continue
            sim = _similarity_matrix(blocks)
            labels, quality = _cluster(sim, cfg.similarity_threshold, cfg.max_sections)
            if quality is None:
                continue
            coverage = len(blocks) * block_beats / total_beats
            # Whole-bar sections are much more common, so they get the bonus;
            # crooked ones stay reachable at a mild disadvantage.
            prior = 1.08 if bars in cfg.expected_section_bars else 1.0
            if bars != int(bars):
                prior *= 0.92
            # AABB dominates the repertoire, so two distinct sections is the
            # single most likely answer. A gentle bonus, not a constraint --
            # one-part and three-part tunes must still be reachable.
            n_labels = len(set(labels))
            if n_labels == 2:
                prior *= 1.10
            # Reward hypotheses that explain more of the recording and that
            # actually find repetition; a segmentation where every block is its
            # own cluster explains nothing.
            repetition = 1.0 - (n_labels - 1) / max(1, len(blocks) - 1)
            # THE domain prior: old-time sections are played twice in a row.
            # This is what separates a section from a phrase -- a half-length
            # block clusters just as cleanly but yields runs of four.
            structure = _repeat_structure_score(labels)
            # Coverage is squared: a hypothesis that leaves a quarter of the
            # recording unexplained is not a quarter worse, it is probably the
            # wrong section length being propped up by having less data to
            # disagree with it.
            #
            # The offset term deserves explanation. Similarity-based scoring is
            # completely FLAT in offset: shifting every block by the same amount
            # leaves all pairwise similarities unchanged, so clustering cannot
            # locate the section boundary at all, only its period. External
            # evidence is required, and the reliable piece is that a recording
            # of a tune starts at the top of the tune. So we prefer offsets that
            # place the first boundary at the first note, tolerating a couple of
            # beats of pickup before it. When that assumption fails (someone hit
            # record mid-tune) the result is a constant shift, not a breakdown.
            score = (
                quality * prior
                * (0.3 + 0.7 * coverage ** 2)
                * (0.5 + 0.5 * repetition)
                * (0.35 + 0.95 * structure)
                * _offset_plausibility(offset, first_note_beat)
            )
            candidates.append((score, bars, offset, blocks, sim, labels, quality,
                               coverage, block_beats))

    if not candidates:
        return FormAnalysis([], 0, 0.0, 0.0,
                            notes=["no section hypothesis produced clean clusters"])

    # Take the best-scoring hypothesis outright. An earlier version preferred
    # the longest hypothesis scoring within 5% of the best, as a guard against
    # mistaking a phrase for a section; the repeat-structure term now handles
    # that distinction directly, and the length preference had become actively
    # harmful, letting implausible 10- and 15-bar sections win on a technicality.
    best = max(candidates, key=lambda c: c[0])

    score, bars, offset, blocks, sim, labels, quality, coverage, block_beats = best

    sections: list[SectionInstance] = []
    for i, (label, (b_start, _)) in enumerate(zip(labels, blocks)):
        b_end = b_start + block_beats
        local = [
            _rebase(n, b_start)
            for n in notes
            if b_start <= float(n.start_beats) < b_end
        ]
        sections.append(
            SectionInstance(label=label, index=i, start_beats=b_start,
                            end_beats=b_end, notes=local)
        )

    # Similarity of each pass to the first pass of its own label, used both as
    # a report to the user and as the gate on consensus merging.
    for s in sections:
        peers = [o for o in sections if o.label == s.label]
        ref = peers[0]
        s.similarity_to_reference = float(sim[s.index][ref.index])

    log.append(
        f"section length {bars:g} bars ({block_beats} beats), offset {offset:g}, "
        f"{len(sections)} passes, cluster quality {quality:.3f}, coverage {coverage:.2f}"
    )
    confidence = float(np.clip(quality * (0.6 + 0.4 * coverage), 0.0, 0.99))
    return FormAnalysis(
        sections=sections,
        bars_per_section=int(round(bars)),
        beats_per_section=float(block_beats),
        confidence=confidence,
        offset_beats=float(offset),
        similarity_matrix=sim,
        notes=log,
    )


# ---------------------------------------------------------------------------
# Slot rasterization and comparison
# ---------------------------------------------------------------------------


def notes_to_slots(
    notes: list[TimedNote], start_beats: float, length_beats: float, grid: int
) -> np.ndarray:
    """Rasterize notes onto a fixed grid. Empty slots are REST.

    A slot array is the workhorse representation for both similarity and
    consensus: it makes two performances of the same section directly
    comparable elementwise without any alignment bookkeeping.
    """
    n = max(1, int(round(length_beats * grid)))
    out = np.full(n, REST, dtype=int)
    for note in notes:
        s = (float(note.start_beats) - start_beats) * grid
        e = (float(note.end_beats) - start_beats) * grid
        i0 = max(0, int(round(s)))
        i1 = min(n, max(i0 + 1, int(round(e))))
        if i0 >= n:
            continue
        out[i0:i1] = note.pitch
    return out


def _blocks(
    notes: list[TimedNote], offset: float, block_beats: float, end: float
) -> list[tuple[float, np.ndarray]]:
    out = []
    t = offset
    while t + block_beats <= end + 1e-9:
        out.append((t, notes_to_slots(notes, t, block_beats, FORM_GRID)))
        t += block_beats
    return out


def slot_similarity(a: np.ndarray, b: np.ndarray, tolerance: int = 1) -> float:
    """Agreement between two slot arrays, ignoring mutually-silent slots.

    Two refinements over exact elementwise matching, both of which turned out to
    matter more than any other choice in this module:

    * **Time tolerance.** A slot is considered matched if the same pitch appears
      within ``tolerance`` slots. Two passes of the same section are played by
      humans, so note boundaries land a grid slot apart routinely; requiring
      exact slot alignment made genuinely identical sections look only ~30%
      similar on noisy recordings, which collapsed form detection entirely.
    * **Octave equivalence.** An octave error is a different reading of the same
      melodic material, so passes differing only by one should still be
      recognized as the same section. Scored partially, since it is still wrong.
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    active = (a != REST) | (b != REST)
    n_active = int(np.sum(active))
    if n_active == 0:
        return 1.0

    exact = np.zeros(len(a), dtype=bool)
    octave = np.zeros(len(a), dtype=bool)
    voiced_a = a != REST
    for shift in range(-tolerance, tolerance + 1):
        bs = np.roll(b, shift)
        exact |= voiced_a & (a == bs)
        octave |= voiced_a & (bs != REST) & (np.abs(a - bs) % 12 == 0)
    # Rest-vs-rest slots are excluded above; rest-vs-note counts as a mismatch.
    exact &= active
    octave &= active & ~exact
    return float((np.sum(exact) + 0.6 * np.sum(octave)) / n_active)


def _similarity_matrix(blocks: list[tuple[float, np.ndarray]]) -> np.ndarray:
    n = len(blocks)
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            s = slot_similarity(blocks[i][1], blocks[j][1])
            sim[i, j] = sim[j, i] = s
    return sim


def _cluster(
    sim: np.ndarray, threshold: float, max_sections: int
) -> tuple[list[str], float | None]:
    """Greedy single-pass clustering, labelling in order of first appearance.

    Greedy is the right choice here: sections appear in a known temporal order,
    so the first block of a new letter really is the natural representative, and
    a global method would only add opacity.
    """
    n = len(sim)
    labels = [""] * n
    reps: list[int] = []
    for i in range(n):
        best_j, best_s = -1, threshold
        for k, rep in enumerate(reps):
            if sim[i, rep] >= best_s:
                best_j, best_s = k, sim[i, rep]
        if best_j >= 0:
            labels[i] = chr(ord("A") + best_j)
        else:
            if len(reps) >= max_sections:
                return labels, None  # hypothesis produces implausibly many sections
            labels[i] = chr(ord("A") + len(reps))
            reps.append(i)

    if len(reps) < 2:
        return labels, None  # no contrast: everything is "A", explains nothing

    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if labels[i] == labels[j] else between).append(sim[i, j])
    if not within or not between:
        return labels, None
    quality = float(np.mean(within) - np.mean(between))
    return labels, quality


# How plausible a run of N identical consecutive sections is. Old-time practice
# is to play each section exactly twice before moving on, so a run of 2 is the
# signature of a correct section length. The other values are what you observe
# when the hypothesis is wrong by a factor of two:
#   runs of 4  -> the block is half a section (each half repeats within it)
#   runs of 1  -> the block swallowed a section and its repeat
# Both remain reachable, because tunes really are sometimes played through
# without repeats, or with a section tripled at the end of a set.
_RUN_LENGTH_PLAUSIBILITY = {1: 0.50, 2: 1.00, 3: 0.55, 4: 0.30}


#: Beats of pickup (anacrusis) we consider normal before the first downbeat.
_MAX_PICKUP_BEATS = 2.0


def _tune_start_beat(notes: list[TimedNote]) -> float:
    """Where the playing really begins, ignoring stray leading events.

    The literal first segmented note is a poor anchor: it is often a count-in
    click, a tuning scrape, or a spurious fragment picked out of the room noise
    before anyone starts. Since the section-offset prior leans on this value, we
    want the first note that is followed by *sustained* playing.
    """
    for i, note in enumerate(notes):
        window_end = float(note.start_beats) + 4.0
        following = sum(
            1 for n in notes[i:] if float(n.start_beats) < window_end
        )
        if following >= 4:
            return float(note.start_beats)
    return float(notes[0].start_beats)


def _offset_plausibility(offset: float, first_note_beat: float) -> float:
    """Prefer section boundaries that coincide with the start of the playing.

    ``lead`` is how far the first note falls *after* the candidate boundary.
    Zero is ideal. A slightly negative value means the first note precedes the
    boundary, which is exactly what a pickup looks like and is not penalized up
    to :data:`_MAX_PICKUP_BEATS`.
    """
    lead = first_note_beat - offset
    excess = max(0.0, lead) + max(0.0, -lead - _MAX_PICKUP_BEATS)
    return 1.0 / (1.0 + 0.25 * excess)


def _repeat_structure_score(labels: list[str]) -> float:
    """How well the label sequence matches old-time repeat practice.

    This is the single most informative prior available for choosing a section
    length, and it is sharp in a way that plain similarity is not: a half-length
    hypothesis clusters just as cleanly as the correct one, but produces runs of
    four rather than runs of two, and a double-length hypothesis produces runs
    of one. Only the correct length yields AABB.
    """
    if not labels:
        return 0.0
    runs: list[int] = []
    count = 1
    for prev, cur in zip(labels, labels[1:]):
        if cur == prev:
            count += 1
        else:
            runs.append(count)
            count = 1
    runs.append(count)
    # The final run is often truncated by the recording ending mid-repeat, so it
    # is excluded when we have enough other evidence.
    scored = runs[:-1] if len(runs) > 2 else runs
    return float(np.mean([_RUN_LENGTH_PLAUSIBILITY.get(r, 0.2) for r in scored]))


def _rebase(note: TimedNote, origin: float) -> TimedNote:
    from fractions import Fraction

    return TimedNote(
        pitch=note.pitch,
        start_beats=note.start_beats - Fraction(origin).limit_denominator(64),
        duration_beats=note.duration_beats,
        confidence=note.confidence,
        start_sec=note.start_sec,
        duration_sec=note.duration_sec,
        pitch_midi=note.pitch_midi,
        quantization_error_beats=note.quantization_error_beats,
    )
