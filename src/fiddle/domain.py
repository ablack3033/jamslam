"""Core musical domain model.

Design rules that this module exists to enforce:

* MIDI is an *export* format, never the internal representation. Everything
  the pipeline reasons about lives here as typed Python.
* Every stage carries confidence forward. A transcription that is 90% right
  is only useful if it knows which 10% to flag, so confidence is not an
  optional decoration -- it is part of every note-like type.
* Continuous measurements (fractional MIDI pitch, seconds) are preserved
  alongside their discrete musical interpretation (integer pitch, beats).
  Re-interpreting a tune in a different key or meter must never require
  re-running audio analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

# Fiddle range, generously bounded. Standard violin tuning is G3-E5 open
# strings; E7 is about as high as an old-time player ever goes and G3 is the
# bottom open string. Anything outside this is an extraction error, not music.
FIDDLE_MIN_MIDI = 55  # G3
FIDDLE_MAX_MIDI = 100  # E7
OPEN_STRINGS_MIDI = (55, 62, 69, 76)  # G3 D4 A4 E5


def hz_to_midi(freq: np.ndarray | float) -> np.ndarray | float:
    """Fractional MIDI pitch. Unvoiced (<=0 Hz) frames map to NaN."""
    f = np.asarray(freq, dtype=float)
    out = np.full(f.shape, np.nan, dtype=float)
    voiced = f > 0
    out[voiced] = 69.0 + 12.0 * np.log2(f[voiced] / 440.0)
    return out if out.shape else float(out)


def midi_to_hz(midi: np.ndarray | float) -> np.ndarray | float:
    return 440.0 * np.power(2.0, (np.asarray(midi, dtype=float) - 69.0) / 12.0)


@dataclass(frozen=True)
class Audio:
    """Mono float32 samples plus provenance. Immutable by convention."""

    samples: np.ndarray
    sample_rate: int
    source_path: str | None = None

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    def __repr__(self) -> str:  # keep numpy arrays out of reprs/logs
        return (
            f"Audio(duration={self.duration:.2f}s, sr={self.sample_rate}, "
            f"source={self.source_path!r})"
        )


@dataclass
class PitchContour:
    """Continuous predominant-melody estimate.

    Deliberately *not* rounded to semitones. Vibrato, slides, octave errors and
    intonation drift are all still visible here, which is what lets the cleaner
    make informed decisions instead of guessing from quantized data.

    ``midi`` is NaN wherever the frame is unvoiced.
    """

    times: np.ndarray  # seconds, shape (n,)
    midi: np.ndarray  # fractional MIDI or NaN, shape (n,)
    confidence: np.ndarray  # 0..1 salience/voicing strength, shape (n,)
    hop_seconds: float
    backend: str = "unknown"
    # Free-form record of what each cleaning stage did, for diagnostics.
    history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.times)
        if not (len(self.midi) == len(self.confidence) == n):
            raise ValueError("PitchContour arrays must be the same length")

    @property
    def voiced(self) -> np.ndarray:
        return ~np.isnan(self.midi)

    @property
    def voiced_fraction(self) -> float:
        return float(np.mean(self.voiced)) if len(self.midi) else 0.0

    def copy_with(self, **kw) -> "PitchContour":
        return replace(self, **kw)


@dataclass
class RawNote:
    """A segmented note event, still in seconds and still fractional in pitch.

    ``pitch_midi`` is the robust (median) fractional pitch over the note's
    frames; keeping it unrounded lets downstream code see that a note sat 40
    cents flat, which is evidence about intonation rather than about identity.
    """

    start_sec: float
    end_sec: float
    pitch_midi: float
    confidence: float
    # Diagnostics: how stable the pitch was inside the note, in cents.
    pitch_spread_cents: float = 0.0
    voiced_fraction: float = 1.0
    frame_range: tuple[int, int] = (0, 0)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def pitch(self) -> int:
        return int(round(self.pitch_midi))


@dataclass
class TimedNote:
    """A note placed on the beat grid, retaining its original wall-clock timing.

    Both representations are kept because quantization is a *hypothesis*. If the
    meter or downbeat phase is later revised, we re-quantize from seconds rather
    than compounding an earlier rounding error.
    """

    pitch: int
    start_beats: Fraction
    duration_beats: Fraction
    confidence: float
    start_sec: float
    duration_sec: float
    pitch_midi: float = 0.0
    # How far the performance was from the quantized grid, in beats. Large
    # values mean the quantization is suspect, not that the player was sloppy.
    quantization_error_beats: float = 0.0

    @property
    def end_beats(self) -> Fraction:
        return self.start_beats + self.duration_beats


@dataclass
class NoteAlternative:
    """A runner-up interpretation retained so uncertainty is inspectable."""

    pitch: int
    weight: float


@dataclass
class Note:
    """A note in the final canonical tune.

    ``confidence`` is a blend of pitch salience, segmentation stability and
    cross-repetition agreement -- see fiddle.confidence for the exact formula.
    It is a *ranking* signal calibrated against the eval corpus, not a
    probability handed down by any single algorithm.
    """

    pitch: int  # MIDI number; rests are represented by pitch=None subclassing
    start_beats: Fraction
    duration_beats: Fraction
    confidence: float
    is_rest: bool = False
    alternatives: list[NoteAlternative] = field(default_factory=list)
    # Which observed repetitions voted for this note (indices into
    # TuneSection.observations). Empty when the note came from a single pass.
    provenance: list[int] = field(default_factory=list)
    # Fraction of aligned repetitions that agreed on this pitch.
    agreement: float = 1.0

    @property
    def end_beats(self) -> Fraction:
        return self.start_beats + self.duration_beats


@dataclass
class Measure:
    notes: list[Note]
    number: int = 0

    @property
    def total_beats(self) -> Fraction:
        return sum((n.duration_beats for n in self.notes), Fraction(0))


@dataclass
class SectionObservation:
    """One performed pass of a section, kept as evidence behind the consensus."""

    label: str  # e.g. "A1"
    start_sec: float
    end_sec: float
    notes: list[TimedNote]
    alignment_score: float = 0.0  # similarity to the section's reference pass


@dataclass
class TuneSection:
    name: str  # "A", "B", ...
    measures: list[Measure]
    repeats: int = 2
    # The individual performances this section was distilled from. Preserved so
    # the UI can eventually show "3 of 4 passes played F#, one played F".
    observations: list[SectionObservation] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def notes(self) -> list[Note]:
        return [n for m in self.measures for n in m.notes]


@dataclass
class Tune:
    key: str  # e.g. "D major", "A mixolydian"
    meter: str  # e.g. "4/4"
    tempo: float  # BPM of the notated beat
    sections: list[TuneSection]
    title: str | None = None
    # Everything the pipeline wants to explain about how it got here.
    analysis: dict = field(default_factory=dict)

    @property
    def notes(self) -> list[Note]:
        return [n for s in self.sections for n in s.notes]

    @property
    def form(self) -> str:
        """Compact form string, e.g. 'AABB'."""
        return "".join(s.name * max(1, s.repeats) for s in self.sections)

    def uncertain_notes(self, threshold: float = 0.7) -> list[Note]:
        return [n for n in self.notes if n.confidence < threshold and not n.is_rest]


def flatten_note_sequence(notes: Iterable[Note]) -> list[int]:
    """Pitch-only sequence, used by similarity and evaluation code."""
    return [n.pitch for n in notes if not n.is_rest]


def sequence_to_intervals(pitches: Sequence[int]) -> list[int]:
    """Transposition-invariant view of a melody."""
    return [b - a for a, b in zip(pitches, pitches[1:])]
