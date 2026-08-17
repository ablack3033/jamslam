"""Melody extraction: the one place that knows about audio DSP libraries.

Downstream modules receive a :class:`PitchContour` and must never be able to
tell which backend produced it. That constraint is what lets us swap Essentia
for Basic Pitch for a future trained model and measure the difference on the
same eval corpus.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import MelodyConfig
from ..domain import Audio, PitchContour


@runtime_checkable
class MelodyExtractor(Protocol):
    """Estimate the predominant melodic line from polyphonic audio."""

    name: str

    def extract(self, audio: Audio) -> PitchContour: ...


def get_extractor(name: str, config: MelodyConfig | None = None) -> MelodyExtractor:
    """Resolve a backend by name, importing it lazily.

    Lazy import matters: Basic Pitch drags in TensorFlow and pins numpy<2, so it
    must never be imported unless explicitly requested.
    """
    config = config or MelodyConfig()
    key = name.lower()
    if key == "essentia":
        from .essentia_backend import EssentiaMelodyExtractor

        return EssentiaMelodyExtractor(config)
    if key == "pyin":
        from .pyin_backend import PyinMelodyExtractor

        return PyinMelodyExtractor(config)
    if key in ("basicpitch", "basic_pitch"):
        from .basicpitch_backend import BasicPitchMelodyExtractor

        return BasicPitchMelodyExtractor(config)
    raise ValueError(f"unknown melody backend {name!r}")


def available_backends() -> dict[str, bool]:
    """Which backends can actually run in this environment."""
    out = {}
    for name, module in (
        ("essentia", "essentia.standard"),
        ("pyin", "librosa"),
        ("basicpitch", "basic_pitch.inference"),
    ):
        try:
            __import__(module)
            out[name] = True
        except Exception:
            out[name] = False
    return out


__all__ = ["MelodyExtractor", "get_extractor", "available_backends"]
