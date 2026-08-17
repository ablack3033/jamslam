"""Essentia PredominantPitchMelodia backend.

Melodia is the right first choice for this problem specifically because it does
not try to transcribe everything: it builds a harmonic salience function over
the whole mix, tracks pitch contours through it, and then *selects* the one
contour it believes is the lead. That is exactly the "one dominant melodic
line" assumption old-time fiddle tunes satisfy.

Known weaknesses we inherit and must handle downstream, not here:
  * octave errors when the salience function locks onto a harmonic
  * attraction to a fiddle's open-string drone during double stops
  * confidence is a salience magnitude, not a calibrated probability
The cleaner and the confidence model deal with these; keeping this module a
faithful, unopinionated wrapper is what makes backend comparison meaningful.
"""

from __future__ import annotations

import numpy as np

from ..config import MelodyConfig
from ..domain import Audio, PitchContour, hz_to_midi


class EssentiaMelodyExtractor:
    name = "essentia"

    def __init__(self, config: MelodyConfig | None = None) -> None:
        self.config = config or MelodyConfig()

    def extract(self, audio: Audio) -> PitchContour:
        try:
            import essentia
            import essentia.standard as es
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "essentia is not installed; `pip install essentia` or use "
                "backend='pyin'"
            ) from exc

        essentia.log.infoActive = False
        essentia.log.warningActive = False

        cfg = self.config
        x = np.ascontiguousarray(audio.samples, dtype=np.float32)

        melodia = es.PredominantPitchMelodia(
            frameSize=cfg.frame_size,
            hopSize=cfg.hop_size,
            sampleRate=float(audio.sample_rate),
            minFrequency=cfg.min_frequency,
            maxFrequency=cfg.max_frequency,
            voicingTolerance=cfg.voicing_tolerance,
            filterIterations=cfg.filter_iterations,
            # Let unvoiced frames stay unvoiced. Guessing fills silence with
            # invented pitch, which manufactures notes between phrases.
            guessUnvoiced=False,
            # Melodia's own vibrato handling keeps contours from fragmenting;
            # residual vibrato is smoothed by the PitchCleaner.
            voiceVibrato=True,
        )
        freqs, conf = melodia(x)

        freqs = np.asarray(freqs, dtype=float)
        conf = np.asarray(conf, dtype=float)
        n = min(len(freqs), len(conf))
        freqs, conf = freqs[:n], conf[:n]

        hop_seconds = cfg.hop_size / float(audio.sample_rate)
        times = np.arange(n) * hop_seconds
        midi = hz_to_midi(freqs)

        return PitchContour(
            times=times,
            midi=midi,
            confidence=_normalize_confidence(conf),
            hop_seconds=hop_seconds,
            backend=self.name,
            history=["essentia:PredominantPitchMelodia"],
        )


def _normalize_confidence(conf: np.ndarray) -> np.ndarray:
    """Scale salience to 0..1 relative to this recording's own maximum.

    Melodia's confidence is an unbounded salience magnitude that depends on
    recording level and instrumentation, so an absolute threshold is meaningless
    across recordings. Relative scaling makes the value comparable *within* a
    recording, which is all any downstream stage actually needs.
    """
    conf = np.nan_to_num(np.asarray(conf, dtype=float), nan=0.0)
    peak = float(np.percentile(conf[conf > 0], 95)) if np.any(conf > 0) else 0.0
    if peak <= 1e-9:
        return np.zeros_like(conf)
    return np.clip(conf / peak, 0.0, 1.0)
