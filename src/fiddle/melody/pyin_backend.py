"""librosa pYIN backend.

pYIN is a *monophonic* pitch tracker, so on a full jam it is the wrong tool in
principle. It is here for two reasons, both of them methodological:

1. It is the honest baseline. If Melodia does not beat pYIN on real jam
   recordings, the "predominant melody extraction" premise of this project is
   weaker than assumed and we should know that number.
2. It removes Essentia from the critical path, so the pipeline and its tests
   run anywhere.
"""

from __future__ import annotations

import numpy as np

from ..config import MelodyConfig
from ..domain import Audio, PitchContour, hz_to_midi


class PyinMelodyExtractor:
    name = "pyin"

    def __init__(self, config: MelodyConfig | None = None) -> None:
        self.config = config or MelodyConfig()

    def extract(self, audio: Audio) -> PitchContour:
        import librosa

        cfg = self.config
        # pYIN is expensive at a 128-sample hop and gains nothing from it, since
        # its own analysis window is far longer. 256 keeps runtime sane while
        # staying well under the shortest note we care about.
        hop = max(cfg.hop_size, 256)
        f0, voiced_flag, voiced_prob = librosa.pyin(
            np.asarray(audio.samples, dtype=np.float32),
            fmin=cfg.min_frequency,
            fmax=cfg.max_frequency,
            sr=audio.sample_rate,
            frame_length=max(cfg.frame_size, 2048),
            hop_length=hop,
        )
        f0 = np.asarray(f0, dtype=float)
        f0 = np.where(np.isfinite(f0), f0, 0.0)
        f0 = np.where(np.asarray(voiced_flag, dtype=bool), f0, 0.0)

        hop_seconds = hop / float(audio.sample_rate)
        times = np.arange(len(f0)) * hop_seconds
        return PitchContour(
            times=times,
            midi=hz_to_midi(f0),
            confidence=np.nan_to_num(np.asarray(voiced_prob, dtype=float), nan=0.0),
            hop_seconds=hop_seconds,
            backend=self.name,
            history=["librosa:pyin"],
        )
