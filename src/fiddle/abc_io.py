"""ABC notation as the ground-truth format.

The spec proposed ``expected.musicxml`` per recording. ABC is a better choice
for this corpus, for a practical reason: the corpus only becomes valuable at
20-50 recordings, and that only happens if writing ground truth is cheap. An
old-time A part is one line of ABC that a fiddler can type from memory and
eyeball for correctness; the equivalent MusicXML is a few hundred lines nobody
will proofread. Most of the repertoire is also already published in ABC.

MusicXML ground truth is still supported (see :func:`load_ground_truth`) --
this is about lowering the cost of the common case, not closing off the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

_HEADER_KEYS = ("X:", "T:", "M:", "L:", "K:")


@dataclass(frozen=True)
class TruthNote:
    """A note in the canonical (notated) tune, in beats from the section start."""

    pitch: int
    start_beats: Fraction
    duration_beats: Fraction
    measure: int


@dataclass(frozen=True)
class TruthSection:
    name: str
    notes: tuple[TruthNote, ...]
    bars: int

    @property
    def beats(self) -> Fraction:
        return sum((n.duration_beats for n in self.notes), Fraction(0))


def parse_abc_sections(abc: str) -> dict[str, TruthSection]:
    """Split a multi-part ABC tune into named sections.

    music21 parses ABC well but handles ``P:`` part markers inconsistently, so
    we split on them ourselves and re-parse each section with a reconstructed
    header. Explicit and boring beats clever here -- ground truth that is subtly
    wrong is worse than no ground truth at all.
    """
    from music21 import converter

    lines = [ln.strip() for ln in abc.strip().splitlines() if ln.strip()]
    header = [ln for ln in lines if ln[:2] in _HEADER_KEYS]

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for ln in lines:
        if ln.startswith("P:"):
            current = ln[2:].strip()
            sections[current] = []
        elif ln[:2] in _HEADER_KEYS:
            continue
        elif current is not None:
            sections[current].append(ln)

    if not sections:  # single-section tune with no P: markers
        sections["A"] = [ln for ln in lines if ln[:2] not in _HEADER_KEYS]

    beats_per_bar = beats_per_bar_from_header(header)
    out: dict[str, TruthSection] = {}
    for name, body in sections.items():
        stream = converter.parse("\n".join(header + body), format="abc")
        # Measure numbers are derived from note offsets rather than read from
        # music21 Measure objects: for short excerpts music21 leaves the notes
        # directly on the Part with no Measure containers at all, and offsets
        # are unambiguous for the un-syncopated material we parse here.
        notes = tuple(
            TruthNote(
                pitch=int(n.pitch.midi),
                start_beats=Fraction(n.offset).limit_denominator(64),
                duration_beats=Fraction(n.duration.quarterLength).limit_denominator(64),
                measure=int(Fraction(n.offset) / beats_per_bar) + 1,
            )
            for n in stream.flatten().notes
            if n.isNote
        )
        total = sum((n.duration_beats for n in notes), Fraction(0))
        out[name] = TruthSection(name=name, notes=notes,
                                 bars=_validate_bars(name, total, beats_per_bar))
    return out


def beats_per_bar_from_header(header) -> Fraction:
    for ln in header:
        if ln.startswith("M:"):
            num, den = ln[2:].strip().split("/")
            # Beats are counted in quarter notes throughout the pipeline.
            return Fraction(int(num) * 4, int(den))
    return Fraction(4)


def meter_from_abc(abc: str) -> str:
    for ln in abc.splitlines():
        if ln.strip().startswith("M:"):
            return ln.strip()[2:].strip()
    return "4/4"


def _validate_bars(name: str, total: Fraction, beats_per_bar: Fraction) -> int:
    """Fail loudly on mis-typed notation rather than corrupting ground truth."""
    bars = total / beats_per_bar
    if bars != int(bars):
        raise ValueError(
            f"ABC section {name!r} has {total} beats, which is not a whole "
            f"number of {beats_per_bar}-beat bars"
        )
    return int(bars)


def load_abc_file(path: str | Path) -> dict[str, TruthSection]:
    return parse_abc_sections(Path(path).read_text())


def load_musicxml_sections(path: str | Path) -> dict[str, TruthSection]:
    """Load ground truth from MusicXML, treating repeat brackets as sections.

    Supported so that transcriptions exported by this tool, or scores from other
    software, can be used as ground truth without conversion.
    """
    from music21 import converter

    stream = converter.parse(str(path))
    measures = list(stream.recurse().getElementsByClass("Measure"))
    if not measures:
        raise ValueError(f"no measures found in {path}")

    # Split at repeat-end barlines; failing that, treat the whole score as "A".
    groups: list[list] = [[]]
    for m in measures:
        groups[-1].append(m)
        right = getattr(m, "rightBarline", None)
        if right is not None and right.classes[0] == "Repeat":
            groups.append([])
    groups = [g for g in groups if g]

    out: dict[str, TruthSection] = {}
    for i, group in enumerate(groups):
        name = chr(ord("A") + i)
        notes: list[TruthNote] = []
        origin = float(group[0].offset)
        for m in group:
            for n in m.notes:
                if not n.isNote:
                    continue
                notes.append(
                    TruthNote(
                        pitch=int(n.pitch.midi),
                        start_beats=Fraction(
                            float(m.offset) + float(n.offset) - origin
                        ).limit_denominator(64),
                        duration_beats=Fraction(
                            n.duration.quarterLength
                        ).limit_denominator(64),
                        measure=int(m.number),
                    )
                )
        out[name] = TruthSection(name=name, notes=tuple(notes), bars=len(group))
    return out


def load_ground_truth(path: str | Path) -> dict[str, TruthSection]:
    """Load ground truth from either .abc or .musicxml/.xml."""
    path = Path(path)
    if path.suffix.lower() in (".abc",):
        return load_abc_file(path)
    return load_musicxml_sections(path)
