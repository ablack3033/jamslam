"""Traditional tune definitions used to build the synthetic corpus.

These are public-domain old-time tunes written in ABC. They are *settings* --
approximations of common versions -- and that is fine, because the evaluation
compares a transcription against this exact notation, not against the folk
tradition. What matters is that they are musically realistic: correct fiddle
range, idiomatic contours, real modal content.

The corpus deliberately spans the axes we expect to break the pipeline:

  soldiers_joy       D major,      fast, continuous eighth notes, high register
  angeline_the_baker D major,      long held notes + wide leaps, low register
  cripple_creek      A major,      mixed rhythm, repeated notes (bowing test)
  old_joe_clark      A mixolydian, flat-7 -- breaks major/minor key detection
  sandy_river_belles G major,      6-bar crooked A part -- breaks the 8-bar prior
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TuneDefinition:
    slug: str
    title: str
    key: str  # music21-style, e.g. "D major", "A mixolydian"
    meter: str
    abc: str
    # One chord symbol per bar, per section. Drives the synthetic accompaniment.
    chords: dict[str, list[str]] = field(default_factory=dict)
    tempo_bpm: float = 120.0
    # How the jam actually plays it: sections in performance order.
    performance: tuple[str, ...] = ("A", "A", "B", "B", "A", "A", "B", "B")
    notes: str = ""


SOLDIERS_JOY = TuneDefinition(
    slug="soldiers_joy",
    title="Soldier's Joy",
    key="D major",
    meter="4/4",
    tempo_bpm=124.0,
    abc="""X:1
T:Soldier's Joy
M:4/4
L:1/8
K:D
P:A
a2 fa gfef|dcde fdcd|a2 fa gfeg|fdcd e2 de|
a2 fa gfef|dcde fdcd|a2 fa gfeg|fdcd e2 d2|
P:B
f2 df e2 ce|dcde fdcd|f2 df e2 ce|fdcd e2 de|
f2 df e2 ce|dcde fdcd|a2 fa gfeg|fdcd e2 d2|
""",
    chords={
        "A": ["D", "D", "G", "D", "D", "D", "A", "D"],
        "B": ["D", "A", "D", "D", "D", "A", "A", "D"],
    },
    notes="Dense continuous eighths in the upper octave; stresses segmentation.",
)

ANGELINE_THE_BAKER = TuneDefinition(
    slug="angeline_the_baker",
    title="Angeline the Baker",
    key="D major",
    meter="4/4",
    tempo_bpm=112.0,
    abc="""X:1
T:Angeline the Baker
M:4/4
L:1/8
K:D
P:A
d2 dd d2 de|f2 f2 f2 ed|d2 dd d2 de|f2 e2 d4|
d2 dd d2 de|f2 f2 f2 ed|A2 AA B2 AG|F2 D2 D4|
P:B
f2 f2 f2 ga|b2 b2 b2 af|f2 f2 f2 ed|d2 e2 d4|
f2 f2 f2 ga|b2 b2 b2 af|A2 AA B2 AG|F2 D2 D4|
""",
    chords={
        "A": ["D", "D", "D", "D", "D", "D", "A", "D"],
        "B": ["D", "D", "D", "A", "D", "D", "A", "D"],
    },
    notes="Repeated pitches and long held notes; stresses onset-based splitting.",
)

CRIPPLE_CREEK = TuneDefinition(
    slug="cripple_creek",
    title="Cripple Creek",
    key="A major",
    meter="4/4",
    tempo_bpm=132.0,
    abc="""X:1
T:Cripple Creek
M:4/4
L:1/8
K:A
P:A
a2 ab agef|gefg e2 d2|a2 ab agef|g2 e2 a4|
a2 ab agef|gefg e2 d2|e2 dB A2 B2|d2 B2 A4|
P:B
A2 AB A2 e2|f2 ef e2 d2|A2 AB A2 e2|d2 B2 A4|
A2 AB A2 e2|f2 ef e2 d2|e2 dB A2 B2|d2 B2 A4|
""",
    chords={
        "A": ["A", "A", "A", "A", "A", "A", "E", "A"],
        "B": ["A", "D", "A", "E", "A", "D", "E", "A"],
    },
    notes="Octave leap between A and B parts; stresses octave-error handling.",
)

OLD_JOE_CLARK = TuneDefinition(
    slug="old_joe_clark",
    title="Old Joe Clark",
    key="A mixolydian",
    meter="4/4",
    tempo_bpm=128.0,
    abc="""X:1
T:Old Joe Clark
M:4/4
L:1/8
K:AMix
P:A
a2 ab a2 ge|a2 ab a4|a2 ab agef|g2 e2 d2 B2|
a2 ab a2 ge|a2 ab a4|a2 ab agef|g2 e2 a4|
P:B
e2 e2 e2 dB|d2 d2 d2 B2|e2 e2 e2 dB|A2 B2 A4|
e2 e2 e2 dB|d2 d2 d2 B2|a2 ab agef|g2 e2 a4|
""",
    chords={
        "A": ["A", "A", "A", "G", "A", "A", "G", "A"],
        "B": ["E", "D", "E", "A", "E", "D", "G", "A"],
    },
    notes="Mixolydian flat-7 (G natural in A); breaks naive major/minor key finding.",
)

SANDY_RIVER_BELLE = TuneDefinition(
    slug="sandy_river_belle",
    title="Sandy River Belle (crooked setting)",
    key="G major",
    meter="4/4",
    tempo_bpm=118.0,
    # A deliberately crooked A part: 6 bars, not 8. Old-time is full of these,
    # and a pipeline that hard-codes 8-bar sections will mangle them.
    abc="""X:1
T:Sandy River Belle
M:4/4
L:1/8
K:G
P:A
d2 dB d2 ga|b2 ag e2 d2|d2 dB d2 ga|
b2 ag e2 d2|G2 AB d2 BA|G2 A2 G4|
P:B
g2 ga b2 ag|e2 d2 B2 A2|G2 AB d2 ga|
b2 ag e2 d2|G2 AB d2 BA|G2 A2 G4|
""",
    chords={
        "A": ["G", "D", "G", "D", "C", "G"],
        "B": ["G", "D", "G", "D", "C", "G"],
    },
    performance=("A", "A", "B", "B", "A", "A", "B", "B"),
    notes="Crooked 6-bar sections; regression guard against hard-coded 8-bar form.",
)


ALL_TUNES: tuple[TuneDefinition, ...] = (
    SOLDIERS_JOY,
    ANGELINE_THE_BAKER,
    CRIPPLE_CREEK,
    OLD_JOE_CLARK,
    SANDY_RIVER_BELLE,
)


def get_tune(slug: str) -> TuneDefinition:
    for t in ALL_TUNES:
        if t.slug == slug:
            return t
    raise KeyError(f"unknown tune {slug!r}; have {[t.slug for t in ALL_TUNES]}")
