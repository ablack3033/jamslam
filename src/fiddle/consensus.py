"""Build a canonical section from several noisy performances of it.

This is the project's central bet. A jam plays the A part four or eight times.
Each pass is an independent noisy measurement of one underlying melody, and the
errors -- an octave slip here, a missed segmentation there, a slide read as an
extra note -- are largely uncorrelated between passes. Voting across passes
should therefore beat any single pass, and it should also tell us *where* the
passes disagreed, which is precisely the information the correction UI needs.

Three design decisions worth stating:

* We vote on a slot grid, not on note lists. Note-level alignment has to solve
  insertion/deletion matching before it can vote; slot voting dissolves that
  problem, because an extra ornament simply occupies few slots and loses.
* We gate on alignment quality. If a pass does not resemble the reference, it
  is excluded from the vote and recorded as an outlier. Averaging things that
  were never the same is worse than not averaging at all.
* We never resolve disagreement silently. A slot where the passes split
  produces a note with low confidence and its runners-up attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .config import ConfidenceConfig, ConsensusConfig
from .confidence import ConfidenceInputs, quantization_score, score_note
from .domain import (
    Measure,
    Note,
    NoteAlternative,
    SectionObservation,
    TimedNote,
    TuneSection,
)
from .form import FORM_GRID, REST, FormAnalysis, SectionInstance, notes_to_slots

CONSENSUS_GRID = 4  # sixteenth-note slots; finer than form analysis needs


@dataclass
class ConsensusReport:
    label: str
    n_observations: int
    n_used: int
    excluded: list[int] = field(default_factory=list)
    mean_agreement: float = 0.0
    notes: list[str] = field(default_factory=list)


def build_section(
    label: str,
    instances: list[SectionInstance],
    beats_per_section: float,
    beats_per_bar: int,
    config: ConsensusConfig | None = None,
    confidence_config: ConfidenceConfig | None = None,
) -> tuple[TuneSection, ConsensusReport]:
    """Derive one canonical :class:`TuneSection` from its performed passes."""
    cfg = config or ConsensusConfig()
    ccfg = confidence_config or ConfidenceConfig()
    report = ConsensusReport(label=label, n_observations=len(instances), n_used=0)

    grids = [notes_to_slots(inst.notes, 0.0, beats_per_section, CONSENSUS_GRID)
             for inst in instances]

    if not cfg.enabled or len(instances) < cfg.min_observations:
        # Fall back to the single most confident pass. Being able to switch
        # consensus off entirely is what makes its benefit measurable.
        best = int(np.argmax([_mean_conf(i.notes) for i in instances]))
        report.n_used = 1
        report.notes.append("consensus disabled or too few passes; using best pass")
        voted = grids[best]
        agreement = np.ones(len(voted))
        alternatives = [[] for _ in voted]
        used = [best]
    else:
        ref = _choose_reference(grids, instances)
        used, excluded = [], []
        for i, g in enumerate(grids):
            score = _alignment_score(g, grids[ref])
            instances[i].similarity_to_reference = score
            if i == ref or score >= cfg.min_alignment_score:
                used.append(i)
            else:
                excluded.append(i)
        if len(used) < cfg.min_observations:
            used, excluded = [ref], [i for i in range(len(grids)) if i != ref]
            report.notes.append("passes did not align; falling back to single pass")

        report.n_used = len(used)
        report.excluded = excluded
        weights = [
            max(0.05, _mean_conf(instances[i].notes)) if cfg.weight_by_confidence else 1.0
            for i in used
        ]
        voted, agreement, alternatives = _vote(
            [grids[i] for i in used], weights
        )
        report.mean_agreement = float(np.mean(agreement)) if len(agreement) else 0.0
        report.notes.append(
            f"voted over passes {used}"
            + (f", excluded {excluded} (poor alignment)" if excluded else "")
        )

    salience = _mean_scalar_slots(
        [instances[i] for i in used], beats_per_section, "confidence"
    )
    qerror = _mean_scalar_slots(
        [instances[i] for i in used], beats_per_section, "quantization_error_beats"
    )
    notes = _slots_to_notes(
        voted, agreement, alternatives, instances, ccfg, salience, qerror
    )
    measures = _to_measures(notes, beats_per_bar, beats_per_section)

    observations = [
        SectionObservation(
            label=f"{label}{i + 1}",
            start_sec=_first_sec(inst.notes),
            end_sec=_last_sec(inst.notes),
            notes=inst.notes,
            alignment_score=inst.similarity_to_reference,
        )
        for i, inst in enumerate(instances)
    ]

    section = TuneSection(
        name=label,
        measures=measures,
        repeats=2 if len(instances) >= 2 else 1,
        observations=observations,
        confidence=float(np.mean([n.confidence for n in notes])) if notes else 0.0,
    )
    return section, report


def build_tune_sections(
    form: FormAnalysis,
    beats_per_bar: int,
    config: ConsensusConfig | None = None,
    confidence_config: ConfidenceConfig | None = None,
) -> tuple[list[TuneSection], list[ConsensusReport]]:
    sections, reports = [], []
    for label in form.labels:
        instances = form.observations(label)
        section, report = build_section(
            label, instances, form.beats_per_section, beats_per_bar,
            config, confidence_config,
        )
        sections.append(section)
        reports.append(report)
    return sections, reports


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------


def _vote(
    grids: list[np.ndarray], weights: list[float]
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, float]]]]:
    """Weighted per-slot plurality vote.

    Octave-collapsed voting matters here: if three passes say A5 and one says
    A4, the pitch class is unanimous and only the octave is contested. Voting on
    pitch class first and then resolving the octave among the winners keeps a
    single octave slip from dragging the whole slot into low confidence.
    """
    n = len(grids[0])
    voted = np.full(n, REST, dtype=int)
    agreement = np.zeros(n)
    alternatives: list[list[tuple[int, float]]] = [[] for _ in range(n)]

    for s in range(n):
        tally: dict[int, float] = {}
        for g, w in zip(grids, weights):
            tally[int(g[s])] = tally.get(int(g[s]), 0.0) + w
        total = sum(tally.values())
        if total <= 0:
            continue

        # Resolve pitch class first, octave second.
        pc_tally: dict[int, float] = {}
        for p, w in tally.items():
            key = REST if p == REST else p % 12
            pc_tally[key] = pc_tally.get(key, 0.0) + w
        best_pc = max(pc_tally, key=lambda k: pc_tally[k])
        if best_pc == REST:
            voted[s] = REST
            agreement[s] = pc_tally[REST] / total
        else:
            octs = {p: w for p, w in tally.items() if p != REST and p % 12 == best_pc}
            best_p = max(octs, key=lambda k: octs[k])
            voted[s] = best_p
            agreement[s] = pc_tally[best_pc] / total

        ranked = sorted(tally.items(), key=lambda kv: kv[1], reverse=True)
        alternatives[s] = [(p, w / total) for p, w in ranked[1:3] if p != voted[s]]

    return voted, agreement, alternatives


def _choose_reference(grids: list[np.ndarray], instances) -> int:
    """Pick the medoid pass: the one most similar to all the others.

    Using the medoid rather than the first pass matters because the first time
    through a tune is often the *least* representative -- players are still
    finding it.
    """
    n = len(grids)
    if n == 1:
        return 0
    scores = []
    for i in range(n):
        sims = [_alignment_score(grids[i], grids[j]) for j in range(n) if j != i]
        conf = _mean_conf(instances[i].notes)
        scores.append(float(np.mean(sims)) * 0.85 + conf * 0.15)
    return int(np.argmax(scores))


def _alignment_score(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity allowing a small global shift, to absorb downbeat wobble.

    Full DTW is unnecessary once both passes are on a quantized beat grid: what
    remains is a possible whole-slot offset, so a short shift search is both
    sufficient and far easier to reason about.
    """
    from .form import slot_similarity

    best = slot_similarity(a, b)
    for shift in (-2, -1, 1, 2):
        shifted = np.roll(b, shift)
        best = max(best, slot_similarity(a, shifted))
    return best


def _mean_scalar_slots(
    instances: list[SectionInstance], beats_per_section: float, attr: str
) -> np.ndarray:
    """Rasterize a per-note scalar (salience, quantization error) onto the grid.

    Averaged across the passes that were used, so the resulting confidence
    reflects how strong the *evidence* was, not just how well the passes
    happened to agree.
    """
    n = max(1, int(round(beats_per_section * CONSENSUS_GRID)))
    total = np.zeros(n)
    count = np.zeros(n)
    for inst in instances:
        for note in inst.notes:
            i0 = max(0, int(round(float(note.start_beats) * CONSENSUS_GRID)))
            i1 = min(n, max(i0 + 1, int(round(float(note.end_beats) * CONSENSUS_GRID))))
            if i0 >= n:
                continue
            total[i0:i1] += float(getattr(note, attr))
            count[i0:i1] += 1
    return total / np.maximum(count, 1.0)


def _slots_to_notes(
    voted: np.ndarray,
    agreement: np.ndarray,
    alternatives: list[list[tuple[int, float]]],
    instances: list[SectionInstance],
    ccfg: ConfidenceConfig,
    salience: np.ndarray,
    qerror: np.ndarray,
) -> list[Note]:
    """Collapse the voted slot array back into notes.

    Runs of the same pitch become one note. This does lose rearticulation of
    identical pitches across a slot boundary, which the segmenter recovered via
    onsets -- so we keep note boundaries from the reference pass where they
    exist rather than blindly merging.
    """
    notes: list[Note] = []
    n = len(voted)
    # Slots at which some pass started a note; used to preserve rearticulation.
    starts = np.zeros(n, dtype=bool)
    for inst in instances:
        for note in inst.notes:
            idx = int(round(float(note.start_beats) * CONSENSUS_GRID))
            if 0 <= idx < n:
                starts[idx] = True

    i = 0
    while i < n:
        pitch = int(voted[i])
        j = i + 1
        while j < n and int(voted[j]) == pitch and not starts[j]:
            j += 1
        span_agree = float(np.mean(agreement[i:j])) if j > i else 0.0
        dur = Fraction(j - i, CONSENSUS_GRID)
        start = Fraction(i, CONSENSUS_GRID)
        if pitch == REST:
            # Only notate a rest if it is long enough to be musically real;
            # shorter gaps are bow lifts and belong to the previous note.
            if dur >= Fraction(1, 2):
                notes.append(Note(pitch=0, start_beats=start, duration_beats=dur,
                                  confidence=span_agree, is_rest=True))
            elif notes:
                notes[-1].duration_beats += dur
        else:
            alt = alternatives[i] if i < len(alternatives) else []
            conf = score_note(
                ConfidenceInputs(
                    salience=float(np.mean(salience[i:j])),
                    # Duration in slots is a proxy for stability: a note held
                    # for several slots was tracked consistently, while a
                    # one-slot note is often a segmentation artifact.
                    stability=float(np.clip((j - i) / 2.0, 0.0, 1.0)),
                    quantization=quantization_score(float(np.mean(qerror[i:j]))),
                    agreement=span_agree,
                ),
                ccfg,
            )
            notes.append(
                Note(
                    pitch=pitch,
                    start_beats=start,
                    duration_beats=dur,
                    confidence=conf,
                    agreement=span_agree,
                    alternatives=[NoteAlternative(p, w) for p, w in alt if p != REST],
                    provenance=list(range(len(instances))),
                )
            )
        i = j
    return notes


def _to_measures(
    notes: list[Note], beats_per_bar: int, beats_per_section: float
) -> list[Measure]:
    n_bars = max(1, int(round(beats_per_section / beats_per_bar)))
    measures = [Measure(notes=[], number=i + 1) for i in range(n_bars)]
    for note in notes:
        idx = min(n_bars - 1, int(float(note.start_beats) // beats_per_bar))
        measures[idx].notes.append(note)
    return measures


def _mean_conf(notes: list[TimedNote]) -> float:
    return float(np.mean([n.confidence for n in notes])) if notes else 0.0


def _first_sec(notes: list[TimedNote]) -> float:
    return float(notes[0].start_sec) if notes else 0.0


def _last_sec(notes: list[TimedNote]) -> float:
    return float(notes[-1].start_sec + notes[-1].duration_sec) if notes else 0.0
