"""Spotify Basic Pitch backend (optional, isolated install).

Basic Pitch is a polyphonic note-level transcriber, so adapting it to this
pipeline requires a *melody selection* step that Melodia does internally: we
take its note events and keep the highest-confidence line, resolving overlaps
in favour of the upper voice. That is a real assumption -- in a fiddle jam the
melody is usually but not always the top line -- and it is stated here rather
than hidden, because it is part of what a Melodia-vs-BasicPitch comparison is
actually measuring.

Install note: basic-pitch depends on TensorFlow 2.15 and pins numpy<2, which
conflicts with the main environment. Install it into a separate venv:

    python -m venv .venv-basicpitch
    .venv-basicpitch/bin/pip install basic-pitch jamslam

and run the CLI with `--melody-backend basicpitch` from that venv. The pipeline
never imports this module unless the backend is explicitly requested.
"""

from __future__ import annotations

import numpy as np

from ..config import MelodyConfig
from ..domain import Audio, PitchContour


class BasicPitchMelodyExtractor:
    name = "basicpitch"

    def __init__(self, config: MelodyConfig | None = None) -> None:
        self.config = config or MelodyConfig()

    def extract(self, audio: Audio) -> PitchContour:
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import predict
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "basic-pitch is not installed. It pins numpy<2 and TensorFlow, "
                "so install it in a separate venv (see module docstring)."
            ) from exc

        import tempfile
        from pathlib import Path

        from ..audio import write_wav

        # Basic Pitch's public API takes a path, not an array.
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "input.wav"
            write_wav(wav, audio)
            _, _, note_events = predict(str(wav), ICASSP_2022_MODEL_PATH)

        return self._notes_to_contour(note_events, audio)

    def _notes_to_contour(self, note_events, audio: Audio) -> PitchContour:
        """Rasterize note events onto the same frame grid the other backends use.

        Producing a contour rather than notes keeps the interface honest: every
        backend hands downstream code the same type, and Basic Pitch's note
        segmentation is then subject to the same cleaning and segmentation we
        apply to everyone else, so the comparison isolates pitch quality.
        """
        hop_seconds = self.config.hop_size / float(audio.sample_rate)
        n_frames = int(np.ceil(audio.duration / hop_seconds)) + 1
        times = np.arange(n_frames) * hop_seconds
        midi = np.full(n_frames, np.nan)
        conf = np.zeros(n_frames)

        # note_events: (start_sec, end_sec, pitch_midi, amplitude, bends)
        for ev in note_events:
            start, end, pitch, amplitude = ev[0], ev[1], int(ev[2]), float(ev[3])
            i0 = max(0, int(round(start / hop_seconds)))
            i1 = min(n_frames, int(round(end / hop_seconds)))
            if i1 <= i0:
                continue
            span = slice(i0, i1)
            # Melody-selection rule: prefer the higher pitch, break ties by
            # amplitude. Stated as one line so it is easy to change and measure.
            existing = midi[span]
            better = np.isnan(existing) | (pitch > existing)
            idx = np.arange(i0, i1)[better]
            midi[idx] = pitch
            conf[idx] = np.clip(amplitude, 0.0, 1.0)

        return PitchContour(
            times=times,
            midi=midi,
            confidence=conf,
            hop_seconds=hop_seconds,
            backend=self.name,
            history=["basic_pitch:predict", "melody_selection:highest_pitch"],
        )
