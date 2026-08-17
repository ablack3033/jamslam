"""Decide whether a recording actually contains a recoverable fiddle melody.

The most damaging thing this system can do is emit a confident, tidy score for a
recording whose melody it never found. A musician has no way to tell that apart
from a good transcription without checking every note, which is exactly the work
the product exists to avoid.

So before transcribing, ask a blunter question: **is there a melody in here at
all?** Three symptoms distinguish a tracked melody from the usual impostors, and
each is cheap to measure:

* **Drone.** A sustained open string or a held chord tone parks the contour on
  one pitch. A melody does not spend half its time on a single note.
* **Accompaniment doubling.** When the guitar or bass is closer to the mic than
  the fiddle, the tracker locks onto its harmonics, and the "melody" comes out
  as a rigid octave above the bass line.
* **Static range.** A fiddle tune covers roughly an octave. A contour with an
  interquartile range of two or three semitones is not a tune.

This was built after a real recording produced a plausible-looking transcription
that was, on inspection, the guitar's root note doubled an octave up: 63% of its
frames sat exactly twelve semitones above the bass contour, and above 400 Hz
three quarters of what little remained was one sustained A4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MelodyConfig
from .domain import Audio, PitchContour

#: Intervals at which a contour that is really tracking the bass will sit.
_HARMONIC_INTERVALS = (0.0, 7.0, 12.0, 19.0, 24.0)


@dataclass
class MelodyPresence:
    """Evidence about whether a melody was found, and what was found instead."""

    voiced_fraction: float
    modal_pitch: int
    modal_pitch_share: float  # share of voiced time on the single commonest pitch
    pitch_iqr: float  # interquartile range, in semitones
    bass_coupling: float  # share of frames sitting at a fixed interval above the bass
    verdict: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def melody_found(self) -> bool:
        return not self.warnings

    def __repr__(self) -> str:
        return f"MelodyPresence({self.verdict!r})"


def diagnose_melody(
    audio: Audio,
    config: MelodyConfig | None = None,
    contour: PitchContour | None = None,
    bass_contour: PitchContour | None = None,
) -> MelodyPresence:
    """Assess whether ``audio`` yields a usable melodic contour.

    Extracts a second contour restricted to the bass register, so that
    accompaniment doubling can be detected by comparison rather than guessed at.
    """
    from .melody import get_extractor

    cfg = config or MelodyConfig()
    if contour is None:
        contour = get_extractor("essentia", cfg).extract(audio)

    midi = np.asarray(contour.midi, dtype=float)
    voiced = midi[~np.isnan(midi)]
    voiced_fraction = float(np.mean(~np.isnan(midi))) if len(midi) else 0.0
    if len(voiced) < 200:
        return MelodyPresence(
            voiced_fraction, 0, 0.0, 0.0, 0.0,
            verdict="almost nothing was voiced; no melody to speak of",
            warnings=["no pitched content found"],
        )

    rounded = np.round(voiced)
    values, counts = np.unique(rounded, return_counts=True)
    modal_pitch = int(values[int(np.argmax(counts))])
    modal_share = float(counts.max() / counts.sum())
    iqr = float(np.percentile(voiced, 75) - np.percentile(voiced, 25))

    if bass_contour is None:
        bass_cfg = MelodyConfig(
            min_frequency=70.0, max_frequency=max(140.0, cfg.min_frequency * 1.2),
            hop_size=cfg.hop_size, frame_size=cfg.frame_size,
        )
        bass_contour = get_extractor("essentia", bass_cfg).extract(audio)
    coupling = _bass_coupling(midi, np.asarray(bass_contour.midi, dtype=float))

    warnings: list[str] = []
    if coupling > 0.5:
        warnings.append(
            f"{coupling:.0%} of the contour sits at a fixed interval above the "
            f"bass register: this is most likely the accompaniment, not the fiddle"
        )
    if modal_share > 0.45:
        warnings.append(
            f"{modal_share:.0%} of voiced time is on a single pitch "
            f"(MIDI {modal_pitch}): most likely a drone or held chord tone"
        )
    if iqr < 3.0:
        warnings.append(
            f"pitch range is only {iqr:.1f} semitones across the middle half of "
            f"the recording: too static to be a tune"
        )
    if voiced_fraction < 0.15:
        warnings.append(
            f"only {voiced_fraction:.0%} of frames are voiced: the melody "
            f"instrument may be too quiet or too distant"
        )

    verdict = (
        "melodic content present"
        if not warnings
        else "no recoverable fiddle melody found"
    )
    return MelodyPresence(
        voiced_fraction=voiced_fraction, modal_pitch=modal_pitch,
        modal_pitch_share=modal_share, pitch_iqr=iqr, bass_coupling=coupling,
        verdict=verdict, warnings=warnings,
    )


def _bass_coupling(melody: np.ndarray, bass: np.ndarray) -> float:
    """Share of frames where the melody sits a fixed harmonic interval above the bass.

    A real melody wanders relative to the bass line; a harmonic of it does not.
    """
    n = min(len(melody), len(bass))
    if n == 0:
        return 0.0
    m, b = melody[:n], bass[:n]
    both = ~np.isnan(m) & ~np.isnan(b)
    if int(np.sum(both)) < 100:
        return 0.0
    diff = np.abs(m[both] - b[both])
    locked = np.zeros(int(np.sum(both)), dtype=bool)
    for interval in _HARMONIC_INTERVALS:
        locked |= np.abs(diff - interval) < 0.6
    return float(np.mean(locked))


def format_presence(p: MelodyPresence) -> str:
    lines = [
        f"melody check   {p.verdict}",
        f"  voiced           {p.voiced_fraction:.0%} of frames",
        f"  commonest pitch  MIDI {p.modal_pitch} for {p.modal_pitch_share:.0%} of voiced time",
        f"  pitch spread     {p.pitch_iqr:.1f} semitones (interquartile)",
        f"  bass coupling    {p.bass_coupling:.0%} locked to the bass register",
    ]
    for w in p.warnings:
        lines.append(f"  WARNING: {w}")
    return "\n".join(lines)
