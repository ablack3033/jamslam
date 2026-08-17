"""Melody extraction: the one place that knows about audio DSP libraries.

Downstream modules receive a :class:`PitchContour` and must never be able to
tell which backend produced it. That constraint is what lets us swap Essentia
for Basic Pitch for a future trained model and measure the difference on the
same eval corpus.
"""

from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable

from ..config import MelodyConfig
from ..domain import Audio, PitchContour


@runtime_checkable
class MelodyExtractor(Protocol):
    """Estimate the predominant melodic line from polyphonic audio."""

    name: str

    def extract(self, audio: Audio) -> PitchContour: ...


#: Tried in order when the requested backend cannot be imported.
FALLBACK_ORDER = ("basicpitch", "essentia", "pyin")


def get_extractor(name: str, config: MelodyConfig | None = None) -> MelodyExtractor:
    """Resolve a backend by name, importing it lazily.

    Lazy import matters: Basic Pitch pulls in TensorFlow and pins numpy<2, so it
    must never be imported unless it is actually wanted.

    Basic Pitch is the default because it measured best on real recordings, but
    it cannot be installed alongside the rest of this package. Rather than make
    the default configuration fail on a fresh checkout, an unavailable backend
    falls back to the next best one **and says so on stderr**. A silent fallback
    would be worse than a crash: the whole point of choosing a backend is that
    the choice changes the result, so a user must never be left believing they
    are running Basic Pitch when they are not.
    """
    config = config or MelodyConfig()
    key = name.lower()
    try:
        return _build(key, config)
    except (ImportError, RuntimeError) as exc:
        for alternative in FALLBACK_ORDER:
            if alternative == key:
                continue
            try:
                extractor = _build(alternative, config)
            except (ImportError, RuntimeError):
                continue
            print(
                f"melody backend {name!r} unavailable ({exc}); "
                f"falling back to {alternative!r}",
                file=sys.stderr,
            )
            return extractor
        raise


def _build(key: str, config: MelodyConfig) -> MelodyExtractor:
    if key == "essentia":
        from .essentia_backend import EssentiaMelodyExtractor

        return EssentiaMelodyExtractor(config)
    if key == "fiddle":
        from .fiddle_backend import FiddleMelodyExtractor

        return FiddleMelodyExtractor(config)
    if key == "pyin":
        from .pyin_backend import PyinMelodyExtractor

        return PyinMelodyExtractor(config)
    if key in ("basicpitch", "basic_pitch"):
        # Probe for the dependency HERE, at construction. The extractor imports
        # basic_pitch lazily inside extract(), which meant an unavailable
        # backend constructed fine and only failed minutes later, after the
        # audio had been decoded -- and the fallback below never fired.
        import importlib.util

        if importlib.util.find_spec("basic_pitch") is None:
            raise ImportError("basic-pitch is not installed in this environment")
        from .basicpitch_backend import BasicPitchMelodyExtractor

        return BasicPitchMelodyExtractor(config)
    raise ValueError(f"unknown melody backend {key!r}")


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
