"""Render synthetic old-time jam recordings with exact ground truth.

Why synthesize at all, when the product targets real recordings?

Because "did the transcription work?" is unanswerable without ground truth, and
hand-transcribing real jam audio is slow and itself error-prone. Synthetic jams
give us a corpus where the correct answer is known exactly, so every algorithm
change produces a number. They are a *necessary but not sufficient* validation:
passing here means the engine is not broken; it does not prove the engine works
on real audio. The same evaluation harness runs on real recordings the moment
you supply them with a ground-truth .abc file.

The synthesis is engineered around the failure modes we actually fear, not
around sounding pleasant:

  * fiddle open-string drones / double stops competing with the melody
  * banjo doubling the melody an octave down (invites octave errors)
  * a second fiddle slightly detuned and slightly behind the beat
  * guitar and bass filling the low-mid salience the melody must beat
  * timing jitter plus slow tempo drift, so beat tracking is not trivial
  * imperfect intonation, vibrato, and room noise
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable

import numpy as np

from ..abc_io import (
    TruthNote,
    TruthSection,
    beats_per_bar_from_header,
    parse_abc_sections,
)
from ..domain import Audio, midi_to_hz
from .library import TuneDefinition

# ---------------------------------------------------------------------------
# Ground truth
#
# The canonical notation comes from abc_io (shared with the evaluation harness,
# so synthetic and real recordings are scored by identical code). What this
# module adds is where each section actually occurred in the rendered audio,
# which is the only truth that is specific to a performance.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerformedSection:
    name: str
    index: int  # 0-based position in the performance order
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class GroundTruth:
    slug: str
    title: str
    key: str
    meter: str
    tempo_bpm: float
    sections: tuple[TruthSection, ...]
    performance: tuple[PerformedSection, ...]
    difficulty: str

    @property
    def form(self) -> str:
        return "".join(p.name for p in self.performance)

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "title": self.title,
            "key": self.key,
            "meter": self.meter,
            "tempo_bpm": self.tempo_bpm,
            "difficulty": self.difficulty,
            "form": self.form,
            "sections": [
                {
                    "name": s.name,
                    "bars": s.bars,
                    "notes": [
                        {
                            "pitch": n.pitch,
                            "start_beats": str(n.start_beats),
                            "duration_beats": str(n.duration_beats),
                            "measure": n.measure,
                        }
                        for n in s.notes
                    ],
                }
                for s in self.sections
            ],
            "performance": [
                {
                    "name": p.name,
                    "index": p.index,
                    "start_sec": round(p.start_sec, 4),
                    "end_sec": round(p.end_sec, 4),
                }
                for p in self.performance
            ],
        }


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Difficulty:
    name: str
    fiddle_gain: float = 1.0
    guitar_gain: float = 0.0
    banjo_gain: float = 0.0
    bass_gain: float = 0.0
    mandolin_gain: float = 0.0
    second_fiddle_gain: float = 0.0
    noise_db: float = -90.0
    reverb_wet: float = 0.0
    timing_jitter_sec: float = 0.004
    intonation_cents: float = 3.0
    tempo_drift_pct: float = 0.0
    drone_probability: float = 0.0
    vibrato_cents: float = 12.0


DIFFICULTIES: dict[str, Difficulty] = {
    # Solo fiddle, near-perfect. If the pipeline fails here the problem is our
    # DSP, not the jam. This is the control condition.
    "clean": Difficulty(
        name="clean",
        timing_jitter_sec=0.004,
        intonation_cents=3.0,
        vibrato_cents=8.0,
    ),
    # A realistic, well-recorded jam: full band, one clear lead fiddle.
    "jam": Difficulty(
        name="jam",
        fiddle_gain=1.0,
        guitar_gain=0.34,
        banjo_gain=0.30,
        bass_gain=0.30,
        mandolin_gain=0.16,
        noise_db=-38.0,
        reverb_wet=0.13,
        timing_jitter_sec=0.017,
        intonation_cents=12.0,
        tempo_drift_pct=1.5,
        drone_probability=0.22,
        vibrato_cents=18.0,
    ),
    # A phone in the middle of a loud circle: two fiddles, banjo forward,
    # heavy droning, sloppier time.
    "hard": Difficulty(
        name="hard",
        fiddle_gain=0.9,
        guitar_gain=0.46,
        banjo_gain=0.46,
        bass_gain=0.38,
        mandolin_gain=0.34,
        second_fiddle_gain=0.42,
        noise_db=-30.0,
        reverb_wet=0.22,
        timing_jitter_sec=0.030,
        intonation_cents=22.0,
        tempo_drift_pct=2.6,
        drone_probability=0.42,
        vibrato_cents=26.0,
    ),
}

_CHORD_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_OPEN_STRINGS = (55, 62, 69, 76)


def render_tune(
    tune: TuneDefinition,
    difficulty: str | Difficulty = "jam",
    sample_rate: int = 44100,
    seed: int = 0,
) -> tuple[Audio, GroundTruth]:
    """Render one performance of ``tune`` and return audio plus ground truth.

    ``difficulty`` accepts a :class:`Difficulty` directly as well as a preset
    name, so tests can construct conditions the presets do not cover -- notably
    a loud sustained bass, which the presets deliberately keep quiet and which
    is needed to validate the accompaniment-tracking check.
    """
    diff = DIFFICULTIES[difficulty] if isinstance(difficulty, str) else difficulty
    rng = np.random.default_rng(seed)
    sections = parse_abc_sections(tune.abc)
    beats_per_bar = beats_per_bar_from_header(tune.abc.splitlines())

    # A slow tempo random walk, sampled per section, so beat tracking must cope
    # with drift the way it must on real recordings.
    order = list(tune.performance)
    drift = 1.0 + rng.normal(0, diff.tempo_drift_pct / 100.0, size=len(order) + 1)
    drift = np.clip(drift, 0.94, 1.06)

    total_beats = sum(
        float(sum((n.duration_beats for n in sections[name].notes), Fraction(0)))
        for name in order
    )
    est_sec = total_beats * 60.0 / tune.tempo_bpm * 1.15 + 4.0
    n_samples = int(est_sec * sample_rate)
    mix = {k: np.zeros(n_samples, dtype=np.float64) for k in
           ("fiddle", "banjo", "guitar", "bass", "mandolin")}

    t_sec = 1.0  # a beat of lead-in silence
    performed: list[PerformedSection] = []

    for idx, name in enumerate(order):
        section = sections[name]
        bpm = tune.tempo_bpm * float(drift[idx])
        spb = 60.0 / bpm  # seconds per beat
        sec_start = t_sec
        chords = tune.chords.get(name, [])

        # --- melody -------------------------------------------------------
        for note in section.notes:
            nominal = t_sec + float(note.start_beats) * spb
            dur = float(note.duration_beats) * spb
            onset = nominal + rng.normal(0, diff.timing_jitter_sec)
            cents = rng.normal(0, diff.intonation_cents)
            _add(mix["fiddle"], _fiddle(
                note.pitch + cents / 100.0, dur * 0.96, sample_rate, rng,
                vibrato_cents=diff.vibrato_cents), onset, sample_rate,
                gain=diff.fiddle_gain)

            # Double-stop drone on an open string below the melody note. This
            # is the single most dangerous confuser for predominant-melody
            # extraction, so it must be present in the test data.
            if diff.drone_probability > 0 and rng.random() < diff.drone_probability:
                below = [s for s in _OPEN_STRINGS if note.pitch - 12 <= s < note.pitch]
                if below:
                    _add(mix["fiddle"], _fiddle(
                        float(below[-1]), dur * 0.9, sample_rate, rng,
                        vibrato_cents=0.0), onset, sample_rate,
                        gain=diff.fiddle_gain * 0.55)

            if diff.second_fiddle_gain > 0:
                _add(mix["fiddle"], _fiddle(
                    note.pitch + rng.normal(8.0, 6.0) / 100.0, dur * 0.94,
                    sample_rate, rng, vibrato_cents=diff.vibrato_cents * 1.3),
                    onset + abs(rng.normal(0.022, 0.010)), sample_rate,
                    gain=diff.second_fiddle_gain)

            if diff.mandolin_gain > 0:
                _add(mix["mandolin"], _mandolin(
                    float(note.pitch), dur, sample_rate, rng),
                    onset + rng.normal(0, 0.008), sample_rate,
                    gain=diff.mandolin_gain)

            # Banjo shadows the melody an octave down on longer notes -- the
            # classic source of octave errors in the extracted contour.
            if diff.banjo_gain > 0 and note.duration_beats >= Fraction(1, 2):
                if rng.random() < 0.55:
                    _add(mix["banjo"], _pluck(
                        float(note.pitch - 12), min(dur * 2.2, 0.9), sample_rate,
                        brightness=1.5), onset + rng.normal(0, 0.012),
                        sample_rate, gain=diff.banjo_gain * 0.8)

        # --- accompaniment -------------------------------------------------
        bars = section.bars
        for bar in range(bars):
            chord = chords[bar % len(chords)] if chords else "D"
            bar_start = t_sec + bar * float(beats_per_bar) * spb
            for beat in range(int(beats_per_bar)):
                b_t = bar_start + beat * spb
                if diff.bass_gain > 0 and beat % 2 == 0:
                    root = _chord_root_midi(chord, low=36)
                    _add(mix["bass"], _bass(root, spb * 0.9, sample_rate),
                         b_t + rng.normal(0, diff.timing_jitter_sec * 0.6),
                         sample_rate, gain=diff.bass_gain)
                if diff.guitar_gain > 0:
                    # Boom-chuck: bass note on the beat, chord on the offbeat.
                    off = b_t + spb * 0.5
                    _add(mix["guitar"], _strum(chord, spb * 0.6, sample_rate, rng),
                         off + rng.normal(0, diff.timing_jitter_sec * 0.6),
                         sample_rate, gain=diff.guitar_gain)
                if diff.banjo_gain > 0:
                    # Clawhammer 5th-string drone on the offbeat (G4).
                    _add(mix["banjo"], _pluck(67.0, 0.32, sample_rate, brightness=1.8),
                         b_t + spb * 0.75 + rng.normal(0, 0.010), sample_rate,
                         gain=diff.banjo_gain * 0.55)

        sec_beats = float(sum((n.duration_beats for n in section.notes), Fraction(0)))
        t_sec += sec_beats * spb
        performed.append(PerformedSection(name=name, index=idx,
                                          start_sec=sec_start, end_sec=t_sec))

    total = sum(mix.values())
    total = total[: int((t_sec + 1.0) * sample_rate)]

    if diff.reverb_wet > 0:
        total = _reverb(total, sample_rate, wet=diff.reverb_wet, rng=rng)
    if diff.noise_db > -80:
        amp = 10 ** (diff.noise_db / 20.0)
        total = total + rng.normal(0, amp, size=total.shape)

    peak = float(np.max(np.abs(total))) or 1.0
    total = (total / peak * 0.85).astype(np.float32)

    truth = GroundTruth(
        slug=tune.slug,
        title=tune.title,
        key=tune.key,
        meter=tune.meter,
        tempo_bpm=tune.tempo_bpm,
        sections=tuple(sections[k] for k in sorted(sections)),
        performance=tuple(performed),
        difficulty=diff.name,
    )
    return Audio(samples=total, sample_rate=sample_rate,
                 source_path=f"synth:{tune.slug}:{difficulty}"), truth


# ---------------------------------------------------------------------------
# Voices. Simple additive/subtractive models -- realism only matters where it
# affects pitch salience, so effort goes into harmonic structure and envelopes.
# ---------------------------------------------------------------------------


def _env(n: int, sr: int, attack: float, release: float) -> np.ndarray:
    e = np.ones(n)
    a = min(int(attack * sr), n // 2)
    r = min(int(release * sr), n // 2)
    if a > 0:
        e[:a] = np.linspace(0.0, 1.0, a) ** 1.5
    if r > 0:
        e[-r:] *= np.linspace(1.0, 0.0, r) ** 1.2
    return e


def _fiddle(midi: float, dur: float, sr: int, rng: np.random.Generator,
            vibrato_cents: float = 15.0) -> np.ndarray:
    n = max(int(dur * sr), 16)
    t = np.arange(n) / sr
    f0 = float(midi_to_hz(midi))

    # Vibrato ramps in after the attack, as a real bowed note does.
    if vibrato_cents > 0 and dur > 0.22:
        rate = rng.uniform(4.8, 6.4)
        ramp = np.clip((t - 0.09) / 0.18, 0.0, 1.0)
        cents = vibrato_cents * ramp * np.sin(2 * np.pi * rate * t)
    else:
        cents = np.zeros(n)
    freq = f0 * (2 ** (cents / 1200.0))
    phase = 2 * np.pi * np.cumsum(freq) / sr

    out = np.zeros(n)
    # Violin spectrum: strong low harmonics, gentle rolloff, bright enough that
    # the 2nd harmonic can plausibly beat the fundamental in salience.
    for k in range(1, 15):
        amp = 1.0 / (k ** 0.92)
        if k in (2, 3):
            amp *= 1.25
        out += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    out *= _env(n, sr, attack=0.028, release=0.045)
    # Bow noise
    out += rng.normal(0, 0.012, n) * _env(n, sr, 0.01, 0.05)
    return out / 4.0


def _pluck(midi: float, dur: float, sr: int, brightness: float = 1.4) -> np.ndarray:
    n = max(int(dur * sr), 16)
    t = np.arange(n) / sr
    f0 = float(midi_to_hz(midi))
    out = np.zeros(n)
    for k in range(1, 10):
        # Higher partials decay faster, as on a real plucked string.
        decay = np.exp(-t * (5.0 + k * brightness))
        out += (1.0 / (k ** 1.15)) * decay * np.sin(2 * np.pi * f0 * k * t)
    out *= _env(n, sr, attack=0.003, release=0.02)
    return out / 3.0


def _bass(midi: float, dur: float, sr: int) -> np.ndarray:
    n = max(int(dur * sr), 16)
    t = np.arange(n) / sr
    f0 = float(midi_to_hz(midi))
    decay = np.exp(-t * 3.2)
    out = decay * (np.sin(2 * np.pi * f0 * t) + 0.32 * np.sin(4 * np.pi * f0 * t))
    return out * _env(n, sr, 0.006, 0.05) / 1.4


def _mandolin(midi: float, dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Tremolo: repeated short plucks, the mandolin's sustain mechanism."""
    n = max(int(dur * sr), 16)
    out = np.zeros(n)
    period = int(sr / 11.0)
    pos = 0
    while pos < n:
        grain = _pluck(midi, 0.13, sr, brightness=2.2)
        end = min(pos + len(grain), n)
        out[pos:end] += grain[: end - pos]
        pos += period + int(rng.normal(0, sr * 0.004))
    return out / 2.0


def _chord_root_midi(chord: str, low: int = 40) -> float:
    pc = _CHORD_PC[chord[0].upper()]
    root = pc
    while root < low:
        root += 12
    return float(root)


def _strum(chord: str, dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    root = _chord_root_midi(chord, low=48)
    voicing = [root, root + 7, root + 12, root + 16, root + 19]
    n = max(int(dur * sr), 16)
    out = np.zeros(n)
    for i, m in enumerate(voicing):
        grain = _pluck(m, dur, sr, brightness=1.1)
        off = int(i * 0.006 * sr)
        end = min(off + len(grain), n)
        out[off:end] += grain[: end - off] * (0.9 ** i)
    return out / 2.5


def _reverb(x: np.ndarray, sr: int, wet: float, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import fftconvolve

    n = int(0.55 * sr)
    t = np.arange(n) / sr
    ir = rng.normal(0, 1, n) * np.exp(-t / 0.19)
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir ** 2))
    wet_sig = fftconvolve(x, ir, mode="full")[: len(x)]
    return (1 - wet) * x + wet * wet_sig


def _add(buf: np.ndarray, sig: np.ndarray, at_sec: float, sr: int,
         gain: float = 1.0) -> None:
    start = max(0, int(at_sec * sr))
    end = min(len(buf), start + len(sig))
    if end > start:
        buf[start:end] += sig[: end - start] * gain
