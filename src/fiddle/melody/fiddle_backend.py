"""Melody extraction with fiddle-specific voice selection.

Melodia's weakness on real jam recordings is not its pitch tracking, which is
good, but its **melody selection**. It builds a salience function over the whole
mix, tracks pitch contours through it, and then picks the contour it believes is
the lead using generic criteria. In a jam recorded on a phone, the guitar and
bass are often physically closer to the microphone than the fiddle, so the
generic criteria pick them, and the "melody" comes out as the accompaniment root
doubled an octave up.

We know things about this material that a generic selector does not:

* **A melody moves; a drone does not.** A contour that holds one pitch for a
  second or more is an open string or a held chord tone, not a tune -- however
  salient it is.
* **The melody is usually the top voice.** The fiddle sits above the guitar and
  bass even when it is quieter.
* **A contour a fixed octave above a lower one is that lower one's harmonic.**

So this backend reuses Essentia's salience and contour tracking -- the parts that
work -- and replaces only the final selection step. It is the custom extractor
the :class:`~fiddle.melody.MelodyExtractor` interface was designed to allow.
"""

from __future__ import annotations

import numpy as np

from ..config import MelodyConfig
from ..domain import Audio, PitchContour

#: Salience-function reference; bins are measured in cents above this.
_REFERENCE_HZ = 55.0
_BIN_RESOLUTION = 10.0  # cents per bin

#: A contour this long that barely moves is a drone, not a melodic line.
_DRONE_MIN_DURATION = 0.7
_DRONE_MAX_SPREAD_SEMITONES = 0.35


class FiddleMelodyExtractor:
    """Salience-based extraction with melody selection tuned to fiddle jams."""

    name = "fiddle"

    def __init__(self, config: MelodyConfig | None = None) -> None:
        self.config = config or MelodyConfig()

    def extract(self, audio: Audio) -> PitchContour:
        import essentia
        import essentia.standard as es

        essentia.log.infoActive = False
        essentia.log.warningActive = False

        cfg = self.config
        sr = float(audio.sample_rate)
        hop = max(128, cfg.hop_size)
        frame_size = max(2048, cfg.frame_size)
        x = np.ascontiguousarray(audio.samples, dtype=np.float32)

        window = es.Windowing(type="hann")
        spectrum = es.Spectrum()
        peaks = es.SpectralPeaks(
            sampleRate=sr, maxPeaks=100, magnitudeThreshold=0.0,
            minFrequency=40.0, maxFrequency=5000.0, orderBy="magnitude",
        )
        salience = es.PitchSalienceFunction(
            binResolution=_BIN_RESOLUTION, referenceFrequency=_REFERENCE_HZ,
            numberHarmonics=20, harmonicWeight=0.8, magnitudeThreshold=40,
        )
        salience_peaks = es.PitchSalienceFunctionPeaks(
            binResolution=_BIN_RESOLUTION, referenceFrequency=_REFERENCE_HZ,
            minFrequency=cfg.min_frequency, maxFrequency=cfg.max_frequency,
        )

        all_bins, all_sals = [], []
        for frame in es.FrameGenerator(x, frameSize=frame_size, hopSize=hop,
                                       startFromZero=True):
            f, m = peaks(spectrum(window(frame)))
            b, s = salience_peaks(salience(f, m))
            all_bins.append(np.asarray(b, dtype=np.float32))
            all_sals.append(np.asarray(s, dtype=np.float32))

        hop_seconds = hop / sr
        contours = es.PitchContours(
            binResolution=_BIN_RESOLUTION, hopSize=hop, sampleRate=sr,
            peakDistributionThreshold=0.9, peakFrameThreshold=0.9,
            pitchContinuity=27.5625, timeContinuity=100.0, minDuration=100.0,
        )
        bins, saliences, start_times, _ = contours(all_bins, all_sals)

        n_frames = len(all_bins)
        midi, conf = _select_melody(
            bins, saliences, start_times, n_frames, hop_seconds
        )
        return PitchContour(
            times=np.arange(n_frames) * hop_seconds,
            midi=midi,
            confidence=conf,
            hop_seconds=hop_seconds,
            backend=self.name,
            history=["essentia:PitchSalienceFunction", "fiddle:voice_selection"],
        )


def _bins_to_midi(bins: np.ndarray) -> np.ndarray:
    hz = _REFERENCE_HZ * 2.0 ** (np.asarray(bins, dtype=float) * _BIN_RESOLUTION / 1200.0)
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def _select_melody(bins, saliences, start_times, n_frames, hop_seconds):
    """Choose, for each frame, which tracked contour carries the melody.

    Scores every contour once, then lets contours compete frame by frame. The
    scoring is where the domain knowledge lives; the competition is deliberately
    simple so the outcome stays explainable.
    """
    tracks = []
    for contour_bins, contour_sals, t0 in zip(bins, saliences, start_times):
        pitches = _bins_to_midi(np.asarray(contour_bins))
        if len(pitches) == 0:
            continue
        start = int(round(float(t0) / hop_seconds))
        duration = len(pitches) * hop_seconds
        spread = float(np.percentile(pitches, 90) - np.percentile(pitches, 10))
        mean_sal = float(np.mean(contour_sals)) if len(contour_sals) else 0.0

        # A long contour that does not move is a drone or a held chord tone.
        # This is the single most important term: it is what stops the selector
        # locking onto the loudest sustained note in the room.
        if duration > _DRONE_MIN_DURATION and spread < _DRONE_MAX_SPREAD_SEMITONES:
            drone_penalty = 0.15
        else:
            drone_penalty = 1.0

        tracks.append({
            "start": start,
            "pitches": pitches,
            "saliences": np.asarray(contour_sals, dtype=float),
            "median": float(np.median(pitches)),
            "score": mean_sal * drone_penalty,
        })

    midi = np.full(n_frames, np.nan)
    conf = np.zeros(n_frames)
    if not tracks:
        return midi, conf

    # Melody is usually the top voice, so a contour is rewarded for sitting above
    # the others. Measured against the overall distribution rather than pairwise,
    # which keeps the decision stable when contours overlap only partially.
    medians = np.array([t["median"] for t in tracks])
    lo, hi = np.percentile(medians, 10), np.percentile(medians, 90)
    span = max(hi - lo, 1e-6)
    for t in tracks:
        height = float(np.clip((t["median"] - lo) / span, 0.0, 1.0))
        t["score"] *= 0.5 + 0.9 * height

    best = np.full(n_frames, -np.inf)
    for t in tracks:
        s, e = t["start"], min(t["start"] + len(t["pitches"]), n_frames)
        if e <= s:
            continue
        span_len = e - s
        wins = t["score"] > best[s:e]
        idx = np.arange(s, e)[wins]
        midi[idx] = t["pitches"][:span_len][wins]
        conf[idx] = t["saliences"][:span_len][wins] if len(t["saliences"]) >= span_len \
            else t["score"]
        best[s:e] = np.where(wins, t["score"], best[s:e])

    peak = float(np.max(conf)) if np.any(conf > 0) else 1.0
    return midi, np.clip(conf / peak, 0.0, 1.0)
