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
  2. Search over section lengths in *beats* and over start offsets,
     partitioning the timeline into blocks of that length.
  3. Score each hypothesis by how cleanly the blocks cluster: high similarity
     within a cluster, low between clusters.
  4. Label clusters A, B, C... in order of first appearance.
  5. Bar the winning length -- choose 2/4 or 4/4, and where any odd bar goes.

Note the order of steps 2 and 5. Section length in beats is what the audio
determines; the barring is a notational choice made afterwards, from that
length. This inverts the obvious arrangement, and it matters: 2/4 and 4/4 are
metrically nested, so the same recording is correctly barred either way, and
asking the onset envelope to choose is close to hopeless. Asking instead which
barring makes the section a plausible number of *bars* is easy, because the
8-bar section is close to universal in this repertoire. Meter therefore comes
out of form rather than going into it.

The priors here are bonuses rather than constraints, and they are deliberately
loose in the two directions this music is actually loose: a section may carry a
bar of odd length (a 2/4 bar dropped into a 4/4 tune is idiomatic), and a part
may be repeated any plausible number of times rather than exactly twice. The
corpus includes a crooked 6-bar tune specifically to keep the first honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import FormConfig
from .domain import TimedNote

REST = -1
FORM_GRID = 2  # slots per beat; eighth notes are enough to see structure

#: The universe of *part orders*, with how common each is.
#:
#: This is the strongest constraint available. The search would otherwise range
#: freely over section length, offset and cluster count, and happily return
#: things like AABBCCD with 16 bars per section -- structure that is periodic
#: but is not a tune.
#:
#: Note what is and is not enumerated here. The part order is a small closed
#: set; the *repeat counts* are not. A jam plays each part twice as a rule, but
#: three times happens, once happens, and the recording is cut off wherever it
#: is cut off. So an earlier version that enumerated whole label strings
#: ("AABB", "AABBB", "AAABBB", ...) was fighting a combinatorial explosion it
#: could not win, and scored perfectly ordinary performances as unrecognised.
#: Repeat counts are scored separately, per part, by
#: :data:`_REPEAT_PLAUSIBILITY` -- which is what makes the prior flexible about
#: repeats without loosening the part-order constraint at all.
PART_SEQUENCES: dict[str, float] = {
    "AB": 1.00,   # two-part tunes: overwhelmingly the most common
    "ABC": 0.50,  # three-part tunes
    "A": 0.35,    # one-part tunes
    "ABCD": 0.15,
}

#: How plausible it is that a part was played N times in a row.
#:
#: Two is the norm. The other values are what you observe when the section
#: length hypothesis is wrong by a factor of two: a half-length block makes each
#: part appear to repeat four times, and a double-length block makes it appear
#: once. Both stay reachable, because tunes really are played straight through
#: sometimes, and a part really does get tripled at the end of a set.
_REPEAT_PLAUSIBILITY: dict[int, float] = {1: 0.35, 2: 1.00, 3: 0.50, 4: 0.15}

#: How plausible a section is when notated with this many bars. Derived from
#: ``FormConfig.expected_section_bars``, whose order is meaningful.
_BARS_RANK_WEIGHT = (1.00, 0.50, 0.45, 0.35, 0.35, 0.30, 0.25)
_UNEXPECTED_BARS_WEIGHT = 0.12

#: How plausible a single odd bar of this many beats is, inside a section whose
#: other bars are regular. A 2-beat bar dropped into a 4/4 tune is completely
#: idiomatic in this repertoire; a 1-beat bar almost never is.
_ODD_BAR_PLAUSIBILITY = {1: 0.30, 2: 0.95, 3: 0.70, 5: 0.35, 6: 0.55, 7: 0.25}

#: 4/4 and 2/4 are both standard for old-time; the tie is broken by bar count,
#: not by this, which is deliberately almost flat.
_METER_PLAUSIBILITY = {4: 1.00, 2: 0.95}


@dataclass(frozen=True)
class BarLayout:
    """How a section of a given length in beats gets written down as bars."""

    beats_per_bar: int
    bar_lengths: tuple[float, ...]
    plausibility: float

    @property
    def bars(self) -> int:
        return len(self.bar_lengths)

    @property
    def meter(self) -> str:
        return f"{self.beats_per_bar}/4"

    @property
    def is_crooked(self) -> bool:
        return any(length != self.beats_per_bar for length in self.bar_lengths)


def choose_bar_layout(
    block_beats: float, config: FormConfig | None = None
) -> BarLayout | None:
    """Pick the most plausible barring of a section this many beats long.

    Section length in beats is what the audio actually determines; bars are a
    notational choice on top of it, and 2/4 and 4/4 are metrically nested, so
    both are literally correct for the same recording. Choosing between them by
    beat stress is nearly hopeless (see :mod:`fiddle.meter`). Choosing by *bar
    count* is not, because the 8-bar section is close to universal in this
    repertoire: a 16-beat section is 8 bars of 2/4 and 4 bars of 4/4, and the
    first of those is a tune while the second is half of one.

    Odd beats are handled the way a transcriber handles them -- either a short
    bar is inserted (a 2/4 bar in an otherwise 4/4 tune, which is extremely
    common here) or the spare beats are absorbed into one longer bar. Whichever
    reads better wins. Returns ``None`` when no barring gives a bar count inside
    the configured range.
    """
    cfg = config or FormConfig()
    total = int(round(block_beats))
    best: BarLayout | None = None

    for beats_per_bar in cfg.meter_candidates:
        meter_weight = _METER_PLAUSIBILITY.get(beats_per_bar, 0.8)
        full, remainder = divmod(total, beats_per_bar)
        variants: list[tuple[list[float], float, int]] = []
        if remainder == 0:
            variants.append(([float(beats_per_bar)] * full, 1.0, full))
        else:
            # An inserted short bar.
            variants.append((
                [float(beats_per_bar)] * full + [float(remainder)],
                _ODD_BAR_PLAUSIBILITY.get(remainder, 0.20),
                full,
            ))
            # ...or the spare beats absorbed into one long bar.
            if full >= 1:
                merged = beats_per_bar + remainder
                variants.append((
                    [float(beats_per_bar)] * (full - 1) + [float(merged)],
                    _ODD_BAR_PLAUSIBILITY.get(merged, 0.20),
                    full,
                ))

        for lengths, odd_weight, full_bars in variants:
            bars = len(lengths)
            if not cfg.min_section_bars <= bars <= cfg.max_section_bars:
                continue
            plausibility = _bars_plausibility(bars, cfg) * odd_weight * meter_weight
            if bars != full_bars:
                # "Eight bars with a 2/4 bar dropped in" is how a fiddler
                # describes this, and it is much more plausible than the nine-bar
                # section the raw count suggests. Score it both ways, keep the
                # better reading, and pay a small penalty for the irregularity.
                plausibility = max(
                    plausibility,
                    0.9 * _bars_plausibility(full_bars, cfg) * odd_weight * meter_weight,
                )
            layout = BarLayout(beats_per_bar, tuple(lengths), plausibility)
            if best is None or layout.plausibility > best.plausibility:
                best = layout
    return best


#: Ratios of the measured repetition period that a section may occupy. A
#: section is the period, or half of it (when the tune's two parts repeat as a
#: unit and the peak lands on the pair), or twice it (when a section's halves
#: resemble each other and the peak lands on the half), or a quarter (when the
#: peak found the whole AABB cycle).
_PERIOD_RATIOS = (0.25, 0.5, 1.0, 2.0)


def _candidate_lengths(
    min_block: int, max_block: int, period_beats: float | None
) -> list[int]:
    """Which section lengths are worth testing.

    When the repetition detector found a period, this is a *constraint* rather
    than a preference, and that distinction was worth a measurement. A block
    length incommensurate with the tune's period slides its phase forward by a
    fixed amount each block, so blocks an even number apart come back into phase
    and blocks an odd number apart do not. Clustering reads that as a clean A/B
    contrast and reports ``ABABAB...`` -- with a cluster quality several times
    better than the *correct* length achieves, because at the correct length
    every block is in phase and so everything resembles everything.

    Measured on a real recording: at block lengths of 16, 32 and 48 beats, mean
    block similarity was 0.50 at every lag (flat -- in phase); at 12, 20, 28, 36
    and 44 it alternated between 0.20 and 0.52 (phase rotation). The alternating
    hypotheses won the search outright. Down-weighting them was not enough, so
    incommensurate lengths are no longer generated at all.

    The ratios are applied exactly. Widening them by a beat either way was
    tried, on the theory that a crooked section might miss a whole ratio -- and
    it put the artifact straight back, one beat off the true period instead of
    four, with cluster quality of 0.04 and label sequences that were plainly
    noise. It was also unnecessary: the period is measured from the audio, so a
    crooked tune has a crooked period, and the exact ratio already fits it.
    """
    if not period_beats or period_beats <= 0:
        return list(range(min_block, max_block + 1))
    lengths = {int(round(period_beats * ratio)) for ratio in _PERIOD_RATIOS}
    within = sorted(b for b in lengths if min_block <= b <= max_block)
    # If the period rules out every plausible section length, it is more likely
    # wrong than the whole repertoire is; fall back to the unconstrained scan.
    return within or list(range(min_block, max_block + 1))


def _bars_plausibility(bars: int, cfg: FormConfig) -> float:
    if bars in cfg.expected_section_bars:
        rank = cfg.expected_section_bars.index(bars)
        return _BARS_RANK_WEIGHT[min(rank, len(_BARS_RANK_WEIGHT) - 1)]
    return _UNEXPECTED_BARS_WEIGHT


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
    # How the section is barred. The prevailing bar length and the per-bar
    # lengths within one section; the two differ when the section is crooked.
    # This is form's output rather than its input -- see choose_bar_layout.
    beats_per_bar: int = 4
    bar_lengths: tuple[float, ...] = ()
    similarity_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    notes: list[str] = field(default_factory=list)

    @property
    def meter(self) -> str:
        return f"{self.beats_per_bar}/4"

    @property
    def is_crooked(self) -> bool:
        return any(length != self.beats_per_bar for length in self.bar_lengths)

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
    contour_grid: np.ndarray | None = None,
    period_beats: float | None = None,
) -> FormAnalysis:
    """Find repeated sections.

    ``contour_grid`` is an optional beat-aligned resampling of the *pitch
    contour* (see :func:`fiddle.repetition.contour_on_beat_grid`). When given,
    block similarity is measured from it instead of from the segmented notes.

    That matters more than it sounds. Measured on a real jam recording, the
    contour repeats clearly -- similarity 0.55 at the tune's period against a
    0.25 baseline -- while the *same recording's* segmented notes give
    within-cluster and between-cluster similarity that are statistically
    indistinguishable (cluster quality 0.02). Segmentation and quantization
    destroy the very structure form detection needs. So when the contour is
    available we cluster on it, and use the notes only to fill the sections.
    """
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
    if total_beats < cfg.min_section_beats * 2:
        return FormAnalysis([], 0, 0.0, 0.0,
                            notes=["performance too short to show repetition"])

    best = None
    log: list[str] = []
    # Search section length in BEATS, not bars. Bars are a notational choice --
    # 2/4 and 4/4 are metrically nested, so a section that is 8 bars of 4/4 is
    # 16 bars of 2/4 and both are correct. Tying the search to bar counts made
    # a benign meter ambiguity split real sections in half. Beats are the
    # meter-independent quantity, and the barring is chosen afterwards, from
    # the winning length, by choose_bar_layout.
    min_block = max(cfg.min_section_beats, min(cfg.meter_candidates))
    max_block = min(cfg.max_section_bars * max(cfg.meter_candidates),
                    int(total_beats // 2))
    block_lengths = _candidate_lengths(min_block, max_block, period_beats)
    candidates: list[tuple] = []

    # Step by ONE BEAT, not one bar. Old-time is full of crooked tunes -- a
    # section with a 2-beat or 6-beat bar in it, so the section total is not a
    # multiple of the bar length. Stepping by whole bars made those sections
    # literally unrepresentable: the correct hypothesis was never generated, and
    # the search silently returned the nearest wrong one. This costs more
    # hypotheses, paid for by a narrower offset search below.
    for block_beats in block_lengths:
        n_blocks_max = int(total_beats // block_beats)
        if n_blocks_max < 3:  # need at least one repeat plus a contrast
            continue
        layout = choose_bar_layout(block_beats, cfg)
        if layout is None:
            continue  # no barring of this length gives a plausible bar count
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
            if contour_grid is not None:
                blocks = _contour_blocks(contour_grid, offset, block_beats)
            else:
                blocks = _blocks(notes, offset, block_beats, total_beats + start)
            # A jam plays at least AABB, so a hypothesis yielding fewer than
            # four blocks is almost always a doubled section length that
            # survived only because it had too little data to contradict it.
            if len(blocks) < 4:
                continue
            sim = (_contour_similarity_matrix(blocks) if contour_grid is not None
                   else _similarity_matrix(blocks))
            labels, quality = _cluster(sim, cfg.similarity_threshold, cfg.max_sections)
            if quality is None:
                continue
            coverage = len(blocks) * block_beats / total_beats
            # How well this length can be written down as bars at all. An
            # 8-bar section under either barring scores full marks; a length
            # that can only be barred as 5 or 11 bars is penalised heavily,
            # because it is nearly always the search carving periodic structure
            # at the wrong scale rather than a genuinely odd tune.
            prior = 0.5 + 0.8 * layout.plausibility
            # AABB dominates the repertoire, so two distinct sections is the
            # single most likely answer. A gentle bonus, not a constraint --
            # one-part and three-part tunes must still be reachable.
            n_labels = len(set(labels))
            if n_labels == 2:
                prior *= 1.10
            # The measured repetition period is direct evidence about section
            # length, obtained upstream of segmentation and therefore far more
            # reliable than anything derived from the notes. A section must be a
            # whole number of those periods -- one period per section, or two
            # when the section's halves resemble each other.
            if period_beats:
                ratio = block_beats / period_beats
                whole = abs(ratio - round(ratio)) < 0.05
                # One period per section is the default reading; two is the
                # common case where a section's halves resemble each other, so
                # the repetition peak lands on the half. Beyond that we are
                # almost certainly swallowing several sections into one block.
                prior *= {1: 1.45, 2: 1.30}.get(round(ratio), 0.85) if whole else 0.7
            # Reward hypotheses that explain more of the recording and that
            # actually find repetition; a segmentation where every block is its
            # own cluster explains nothing.
            repetition = 1.0 - (n_labels - 1) / max(1, len(blocks) - 1)
            # THE domain prior: the part order is drawn from a small known
            # universe and each part is repeated a plausible number of times.
            # Sharp enough to reject AABBCCD outright rather than merely
            # scoring it a little lower.
            plausible_form = form_plausibility(labels)
            structure = 0.3 * _repeat_structure_score(labels) + 0.7 * plausible_form
            structure *= 0.35 + 0.65 * plausible_form
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
            candidates.append((score, layout, offset, blocks, sim, labels, quality,
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

    score, layout, offset, blocks, sim, labels, quality, coverage, block_beats = best

    # Refuse a segmentation whose clusters barely separate. This is the module's
    # stated failure mode -- fail loudly rather than quietly -- and without the
    # floor the search always returns *something*, because some hypothesis
    # always scores highest even when every one of them is noise.
    #
    # The floor applies to the hypothesis actually being returned, not to the
    # best quality anywhere in the candidate set. Those differ: scoring weighs
    # quality against the musical priors, so the winner is regularly not the
    # best-separated hypothesis, and checking the wrong one let a segmentation
    # of quality 0.071 through a floor of 0.08.
    if quality < cfg.min_cluster_quality:
        return FormAnalysis(
            [], 0, 0.0, 0.0,
            notes=[f"best hypothesis separates clusters by only {quality:.3f}, "
                   f"under the {cfg.min_cluster_quality:.2f} floor; "
                   f"no section structure found"],
        )

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
        f"section length {layout.bars} bars of {layout.meter} ({block_beats} beats), "
        f"offset {offset:g}, {len(sections)} passes, "
        f"cluster quality {quality:.3f}, coverage {coverage:.2f}"
    )
    if layout.is_crooked:
        odd = [f"{length:g}" for length in layout.bar_lengths
               if length != layout.beats_per_bar]
        log.append(
            f"crooked section: bar lengths {', '.join(odd)} beats against "
            f"{layout.beats_per_bar} elsewhere"
        )
    confidence = float(np.clip(quality * (0.6 + 0.4 * coverage), 0.0, 0.99))
    return FormAnalysis(
        sections=sections,
        bars_per_section=layout.bars,
        beats_per_section=float(block_beats),
        confidence=confidence,
        offset_beats=float(offset),
        beats_per_bar=layout.beats_per_bar,
        bar_lengths=layout.bar_lengths,
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


#: Samples per beat used when clustering on the contour.
CONTOUR_GRID = 8


def _contour_blocks(grid: np.ndarray, offset: float, block_beats: float):
    """Cut the beat-aligned contour into equal blocks."""
    n = int(round(block_beats * CONTOUR_GRID))
    start = int(round(offset * CONTOUR_GRID))
    out = []
    i = max(0, start)
    while i + n <= len(grid):
        out.append((i / CONTOUR_GRID, grid[i:i + n]))
        i += n
    return out


def _contour_similarity(a: np.ndarray, b: np.ndarray,
                        tolerance: float = 0.6) -> float:
    """Agreement between two contour blocks, ignoring mutually-unvoiced samples.

    Same rule as the repetition detector, which is the point: that measure is
    demonstrably able to find a tune's period on real audio, so section
    clustering should use the same evidence rather than a weaker proxy.
    """
    both = ~np.isnan(a) & ~np.isnan(b)
    if int(np.sum(both)) < max(8, len(a) // 8):
        return 0.0
    diff = np.abs(a[both] - b[both])
    ok = (diff < tolerance) | (np.abs(diff - 12.0) < tolerance)
    return float(np.mean(ok))


def _contour_similarity_matrix(blocks) -> np.ndarray:
    n = len(blocks)
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            sim[i, j] = sim[j, i] = _contour_similarity(blocks[i][1], blocks[j][1])
    return sim


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
    """Cluster blocks, scanning for a threshold that yields a plausible count.

    A fixed absolute threshold cannot work: similarity values do not mean the
    same thing on every recording. A clean solo fiddle reaches ~1.0 between
    repeats of a section, while a phone recording of a loud circle tops out near
    0.8 with a mean of 0.4.

    An earlier version tried a short hand-picked ladder of thresholds, and that
    failed in a way that was invisible until instrumented: on real audio its low
    rungs merged everything into a single cluster and its high rungs shattered
    the blocks past ``max_sections``, so *every* rung was rejected and form
    detection returned nothing at all. The viable window sat between two rungs.

    So derive the candidates from the data instead -- quantiles of the observed
    off-diagonal similarities -- and keep the best-scoring clustering that
    yields between two and ``max_sections`` labels.
    """
    off_diagonal = sim[~np.eye(len(sim), dtype=bool)]
    if not len(off_diagonal):
        return [""] * len(sim), None

    quantiles = np.percentile(off_diagonal, np.linspace(40.0, 99.5, 28))
    candidates = sorted({round(float(t), 4) for t in (*quantiles, threshold)})

    best: tuple[list[str], float] | None = None
    for candidate in candidates:
        labels, quality = _cluster_at(sim, candidate, max_sections)
        if quality is None:
            continue
        # Prefer the clustering that separates best, breaking ties toward the
        # form we actually expect to see.
        score = quality * (0.5 + 0.5 * form_plausibility(labels))
        if best is None or score > best[1]:
            best = (labels, score)
    if best is None:
        return [""] * len(sim), None
    # Recompute the honest quality for the winner; the tie-break above must not
    # leak into the number the caller scores hypotheses with.
    _, quality = _cluster_at(sim, _threshold_for(sim, best[0]), max_sections)
    return best[0], quality if quality is not None else best[1]


def _threshold_for(sim: np.ndarray, labels: list[str]) -> float:
    """The lowest within-cluster similarity implied by a labelling."""
    within = [
        sim[i, j]
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
        if labels[i] == labels[j]
    ]
    return float(min(within)) if within else 0.0


def _cluster_at(
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


def form_plausibility(labels: list[str]) -> float:
    """How well a label sequence looks like a known form played several times.

    Scored on the run-length encoding rather than the raw string, which is what
    makes it flexible about repeats. ``AABB``, ``AABBB`` and ``AAABBAABB`` all
    encode to the same part order ``AB`` cycling, differing only in how many
    times each part was held -- and repeat counts genuinely vary in a jam. An
    earlier version enumerated whole label strings, so every extra repeat needed
    its own entry and anything unlisted scored as noise.

    Two allowances beyond that:

    * **Rotation**, because a recording may be joined mid-tune, so the first
      part heard need not be the tune's A part.
    * **A truncated tail**, because recordings stop wherever they stop. The
      final run is therefore not scored for length when there is other evidence
      -- a part cut off after one pass says nothing about how often it repeats.

    Returns a small floor rather than zero for unrecognised sequences, so an
    unusual but real form stays reachable if the acoustic evidence is strong.
    """
    if not labels:
        return 0.0
    runs = _run_length_encode(labels)
    letters = "".join(letter for letter, _ in runs)

    # The recording stops mid-repeat far more often than not, so drop the last
    # run's count from the average once there is enough left to average over.
    counts = [count for _, count in runs]
    scored_counts = counts[:-1] if len(counts) > 2 else counts
    repeat_score = float(np.mean(
        [_REPEAT_PLAUSIBILITY.get(c, 0.10) for c in scored_counts]
    ))

    best = 0.05
    for sequence, weight in PART_SEQUENCES.items():
        for rotation in range(len(sequence)):
            rotated = sequence[rotation:] + sequence[:rotation]
            cycled = (rotated * (len(letters) // len(rotated) + 2))[: len(letters)]
            match = sum(1 for a, b in zip(letters, cycled) if a == b) / len(letters)
            best = max(best, weight * match * repeat_score)
    return best


def _run_length_encode(labels: list[str]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    for label in labels:
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1] + 1)
        else:
            runs.append((label, 1))
    return runs


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
    runs = [count for _, count in _run_length_encode(labels)]
    # The final run is often truncated by the recording ending mid-repeat, so it
    # is excluded when we have enough other evidence.
    scored = runs[:-1] if len(runs) > 2 else runs
    return float(np.mean([_REPEAT_PLAUSIBILITY.get(r, 0.10) for r in scored]))


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
