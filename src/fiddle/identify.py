"""Identify which known tune a transcription is, by matching against a catalog.

This exists as a **validation mechanism**, not as a product feature. Hand-writing
ground truth for a jam recording is slow, which is exactly why the regression
corpus is hard to grow. But if a transcription matches a known tune strongly,
that match is itself evidence the transcription is roughly right -- and the
catalog setting can then serve as approximate ground truth, closing the loop:

    transcribe -> identify -> use the matched setting as expected.abc -> measure

Two design decisions carry most of the weight:

**Interval n-grams, not note sequences.** Comparing melodic *intervals* rather
than pitches makes matching automatically transposition-invariant, which matters
because a jam plays a tune in whatever key it likes and because our own key
inference may be off by a step. Using n-grams rather than whole-sequence
alignment makes it robust to insertions and deletions: one spurious note
destroys only the handful of n-grams that span it, instead of misaligning
everything downstream.

**Match against the raw note stream, not the detected sections.** Form detection
is the least reliable stage in the pipeline, and on real jam audio it frequently
fails outright. Identification that depended on it would fail exactly when it is
most needed. So we match the flat sequence of transcribed notes, and a tune is
recognised whether or not we ever worked out its form.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .catalog import CatalogTune, builtin_catalog

#: Number of consecutive intervals per n-gram. Four intervals spans five notes,
#: which is long enough to be distinctive in a diatonic repertoire but short
#: enough that a wrong note only destroys a few of them.
NGRAM = 4

#: Intervals beyond this are extraction errors, not melody; clamping them stops
#: one octave glitch from generating a family of unique, unmatchable n-grams.
MAX_INTERVAL = 12


@dataclass
class TuneMatch:
    title: str
    key: str
    score: float  # share of the catalog tune's n-grams present in the recording
    coverage: float  # share of the recording's n-grams belonging to this tune
    # How many standard deviations above chance this tune's score is, given how
    # easily it matches meaningless input. This, not `score`, is what ranks.
    z: float = 0.0
    chance_score: float = 0.0
    section_scores: dict[str, float] = field(default_factory=dict)
    source: str = ""
    tune: CatalogTune | None = None

    def __repr__(self) -> str:
        return f"TuneMatch({self.title!r}, z={self.z:.1f}, score={self.score:.3f})"


@dataclass
class Identification:
    matches: list[TuneMatch]
    n_notes: int
    catalog_size: int

    @property
    def best(self) -> TuneMatch | None:
        return self.matches[0] if self.matches else None

    @property
    def margin(self) -> float:
        """Gap in z between the best and second-best match.

        A high score with no margin means the catalog holds several similar
        tunes, not that we identified this one -- so margin is reported
        separately rather than folded into the score.
        """
        if len(self.matches) < 2:
            return self.matches[0].z if self.matches else 0.0
        return self.matches[0].z - self.matches[1].z

    def is_confident(self, min_z: float = 6.0, min_margin: float = 2.0) -> bool:
        """Confident only when the match is far above what chance would give.

        Thresholds are in units of the null model's standard deviation, not raw
        score, because raw score is not comparable across tunes: a repetitive,
        stepwise tune scores far higher on meaningless input than an arpeggiated
        one does. See :func:`null_baseline`.
        """
        return bool(
            self.matches
            and self.matches[0].z >= min_z
            and self.margin >= min_margin
        )


def interval_ngrams(pitches: list[int], n: int = NGRAM) -> Counter:
    """Multiset of n-gram interval patterns, which is transposition-invariant."""
    if len(pitches) < n + 1:
        return Counter()
    intervals = [
        max(-MAX_INTERVAL, min(MAX_INTERVAL, b - a))
        for a, b in zip(pitches, pitches[1:])
    ]
    return Counter(
        tuple(intervals[i : i + n]) for i in range(len(intervals) - n + 1)
    )


def _ngram_weights(catalog: list[CatalogTune]) -> dict[tuple, float]:
    """Inverse-document-frequency weight per n-gram.

    Without this, identification is dominated by melodically empty patterns.
    The all-zeros n-gram (four repeated notes) occurs in most tunes *and* in any
    noisy transcription, so an unweighted score ranked the most repetitive
    catalog entry top for every single recording, at an identical score --
    a match on nothing at all. Weighting by rarity means a tune is identified by
    the phrases that make it distinctive.
    """
    import math

    df: Counter = Counter()
    for entry in catalog:
        grams: set = set()
        for section in entry.sections.values():
            grams |= set(interval_ngrams([n.pitch for n in section.notes]))
        df.update(grams)
    n = max(1, len(catalog))
    return {g: math.log(1.0 + n / c) for g, c in df.items()}


def _weighted_share(grams: set, observed: set, weights: dict[tuple, float]) -> float:
    total = sum(weights.get(g, 1.0) for g in grams)
    if total <= 0:
        return 0.0
    hit = sum(weights.get(g, 1.0) for g in grams & observed)
    return hit / total


#: Cache of null baselines, keyed by catalog identity. Computing one costs a few
#: dozen n-gram set operations, which is cheap, but not free per call.
_NULL_CACHE: dict[tuple, dict[str, tuple[float, float]]] = {}


def _random_diatonic_walk(rng, length: int = 500) -> list[int]:
    """A meaningless but *plausible* melody: a stepwise walk in a major scale.

    The null model has to look like a bad transcription, not like white noise.
    Uniformly random pitches score near zero against everything and would prove
    nothing; a noisy transcription of a real jam is full of small stepwise moves
    and repeated notes, which is exactly what accidentally matches simple tunes.
    """
    scale = (0, 2, 4, 5, 7, 9, 11)
    root = rng.choice((62, 67, 69))
    pitch = root
    out = []
    for _ in range(length):
        pitch += rng.choice((-2, -1, 0, 1, 2)) * rng.choice((1, 2))
        pitch = max(60, min(84, pitch))
        out.append(
            min(range(60, 85),
                key=lambda c: (abs(c - pitch), (c - root) % 12 not in scale))
        )
    return out


def null_baseline(
    catalog: list[CatalogTune], trials: int = 40, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """Mean and spread of each tune's score against meaningless input.

    This exists because raw match scores are **not comparable between tunes**.
    Measured over random diatonic walks, one particular catalog tune won 60 out
    of 60 trials at a mean score of 0.394 -- higher than any real recording had
    scored. Ranking on raw score therefore reported that tune as the best match
    for essentially every input, which is a match on nothing.

    Normalising by this baseline turns the score into "how much better than
    chance", which is the quantity we actually meant all along.
    """
    key = tuple(sorted(t.title for t in catalog))
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]

    import random
    import statistics

    weights = _ngram_weights(catalog)
    tune_grams = {
        t.title: set().union(*(
            set(interval_ngrams([n.pitch for n in s.notes]))
            for s in t.sections.values()
        )) for t in catalog
    }
    samples: dict[str, list[float]] = {t.title: [] for t in catalog}
    for trial in range(trials):
        rng = random.Random(seed + trial)
        observed = set(interval_ngrams(_random_diatonic_walk(rng)))
        for title, grams in tune_grams.items():
            samples[title].append(_weighted_share(grams, observed, weights))

    baseline = {
        title: (
            statistics.fmean(values),
            max(statistics.pstdev(values), 1e-3),
        )
        for title, values in samples.items()
    }
    _NULL_CACHE[key] = baseline
    return baseline


def identify(
    pitches: list[int],
    catalog: list[CatalogTune] | None = None,
    top_k: int = 5,
) -> Identification:
    """Rank catalog tunes by how far above chance their material appears in ``pitches``."""
    catalog = catalog if catalog is not None else builtin_catalog()
    baseline = null_baseline(catalog)
    observed = interval_ngrams(pitches)
    observed_set = set(observed)
    weights = _ngram_weights(catalog)

    matches: list[TuneMatch] = []
    for entry in catalog:
        section_scores: dict[str, float] = {}
        all_grams: set = set()
        for name, section in entry.sections.items():
            grams = set(interval_ngrams([n.pitch for n in section.notes]))
            all_grams |= grams
            section_scores[name] = _weighted_share(grams, observed_set, weights)
        if not all_grams:
            continue
        hit = all_grams & observed_set
        score = _weighted_share(all_grams, observed_set, weights)
        coverage = (
            sum(observed[g] for g in hit) / sum(observed.values())
            if observed else 0.0
        )
        chance, spread = baseline.get(entry.title, (0.0, 1e-3))
        matches.append(
            TuneMatch(
                title=entry.title, key=entry.key, score=score, coverage=coverage,
                z=(score - chance) / spread, chance_score=chance,
                section_scores=section_scores, source=entry.source, tune=entry,
            )
        )

    matches.sort(key=lambda m: (m.z, m.coverage), reverse=True)
    return Identification(
        matches=matches[:top_k], n_notes=len(pitches), catalog_size=len(catalog)
    )


def identify_result(result, catalog=None, top_k: int = 5) -> Identification:
    """Identify from a :class:`TranscriptionResult`.

    Uses ``timed_notes`` -- the flat quantized stream -- rather than the
    consensus sections, so identification still works when form detection has
    failed. On a real jam recording that is the common case, not an edge case.
    """
    pitches = [n.pitch for n in result.timed_notes]
    if not pitches:
        pitches = [n.pitch for n in result.tune.notes if not n.is_rest]
    return identify(pitches, catalog, top_k)


def format_identification(ident: Identification) -> str:
    lines = [
        f"searched {ident.catalog_size} catalog tunes against "
        f"{ident.n_notes} transcribed notes",
        "",
        f"{'tune':<30}{'key':<16}{'z':>7}{'match':>7}{'chance':>8}   sections",
    ]
    lines.append("-" * 84)
    for m in ident.matches:
        secs = " ".join(f"{k}={v:.2f}" for k, v in sorted(m.section_scores.items()))
        lines.append(
            f"{m.title:<30}{m.key:<16}{m.z:>7.1f}{m.score:>7.3f}"
            f"{m.chance_score:>8.3f}   {secs}"
        )
    lines.append("")
    if ident.is_confident():
        lines.append(
            f"=> confident match: {ident.best.title} "
            f"({ident.best.z:.1f} sigma above chance, "
            f"{ident.margin:.1f} clear of the next candidate)"
        )
    elif ident.best:
        lines.append(
            f"=> no confident match (best {ident.best.title} at "
            f"{ident.best.z:.1f} sigma, margin {ident.margin:.1f}). "
            f"Either the tune is not in the catalog, or the transcription is too "
            f"noisy to match."
        )
    lines.append(
        "   'chance' is what this tune scores against meaningless stepwise input; "
        "raw scores are not comparable between tunes."
    )
    return "\n".join(lines)
