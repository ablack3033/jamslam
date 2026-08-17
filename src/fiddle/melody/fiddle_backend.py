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

So this backend reuses Essentia's salience function -- the part that works -- and
replaces the selection step with a Viterbi decode that follows one line
continuously (see :mod:`fiddle.melody.tracking`). It is the custom extractor the
:class:`~fiddle.melody.MelodyExtractor` interface was designed to allow.
"""

from __future__ import annotations

import numpy as np

from ..config import MelodyConfig
from ..domain import Audio, PitchContour
from .tracking import TrackingWeights, decode_contours

#: Salience-function reference; bins are measured in cents above this.
_REFERENCE_HZ = 55.0
_BIN_RESOLUTION = 10.0  # cents per bin

class FiddleMelodyExtractor:
    """Salience-based extraction with melody selection tuned to fiddle jams."""

    name = "fiddle"

    def __init__(self, config: MelodyConfig | None = None,
                 weights: TrackingWeights | None = None) -> None:
        self.config = config or MelodyConfig()
        self.weights = weights or TrackingWeights()

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
        n_frames = len(all_bins)

        # Let Essentia group frames into coherent fragments; we decide which
        # fragments form the melody. Frame-level decoding was tried first and
        # measured worse -- see fiddle.melody.tracking.
        contours = es.PitchContours(
            binResolution=_BIN_RESOLUTION, hopSize=hop, sampleRate=sr,
            peakDistributionThreshold=0.9, peakFrameThreshold=0.9,
            pitchContinuity=27.5625, timeContinuity=100.0, minDuration=100.0,
        )
        c_bins, c_sals, c_starts, _ = contours(all_bins, all_sals)
        pitches = [_bins_to_midi(np.asarray(b)) for b in c_bins if len(b)]
        saliences = [np.asarray(s, dtype=float) for s, b in zip(c_sals, c_bins) if len(b)]
        starts = [int(round(float(t) / hop_seconds))
                  for t, b in zip(c_starts, c_bins) if len(b)]
        midi, conf = decode_contours(pitches, saliences, starts, n_frames,
                                     hop_seconds, self.weights)
        return PitchContour(
            times=np.arange(n_frames) * hop_seconds,
            midi=midi,
            confidence=conf,
            hop_seconds=hop_seconds,
            backend=self.name,
            history=["essentia:PitchSalienceFunction", "fiddle:viterbi_tracking"],
        )


def _bins_to_midi(bins: np.ndarray) -> np.ndarray:
    hz = _REFERENCE_HZ * 2.0 ** (np.asarray(bins, dtype=float) * _BIN_RESOLUTION / 1200.0)
    return 69.0 + 12.0 * np.log2(hz / 440.0)
