"""Tune catalogs: libraries of known settings to identify a transcription against.

A catalog is just ABC. That choice matters, because the point is to be able to
aim this at a *real* tune library rather than at whatever we happened to type
in: any directory of ``.abc`` files, or a single multi-tune tunebook, works.

    fiddle-transcribe identify recording.m4a --catalog /path/to/library

The built-in starter catalog exists so the feature works out of the box, not
because it is authoritative. Old-time tunes have no canonical setting -- every
player's version differs -- so the matcher is built to tolerate that rather than
to demand an exact library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..abc_io import TruthSection, parse_abc_sections

_BUILTIN = Path(__file__).parent / "oldtime_starter.abc"


@dataclass
class CatalogTune:
    title: str
    key: str
    meter: str
    sections: dict[str, TruthSection]
    source: str = ""
    rhythm: str = ""  # R: field, e.g. "reel", "hornpipe"

    @property
    def slug(self) -> str:
        return "".join(
            c.lower() if c.isalnum() else "_" for c in self.title
        ).strip("_")

    @property
    def abc_sections(self) -> list[TruthSection]:
        return [self.sections[k] for k in sorted(self.sections)]

    def __repr__(self) -> str:
        return f"CatalogTune({self.title!r}, {self.key}, {len(self.sections)} sections)"


def split_tunebook(text: str) -> list[str]:
    """Split a multi-tune ABC file into individual tunes on ``X:`` headers."""
    tunes: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("%") and not line.startswith("%%"):
            continue  # comment
        if line.startswith("X:"):
            tunes.append([])
        if tunes:
            tunes[-1].append(line)
    return ["\n".join(t) for t in tunes if any(ln.startswith("K:") for ln in t)]


def parse_catalog_tune(abc: str, source: str = "") -> CatalogTune | None:
    """Parse one ABC tune into a CatalogTune, or None if it cannot be read.

    Returns None rather than raising: a real library will contain tunes in
    meters we do not handle, and one bad entry must not abort a scan of a
    thousand files.
    """
    title = _field(abc, "T:") or "Untitled"
    key = _field(abc, "K:") or "D"
    meter = _field(abc, "M:") or "4/4"
    if meter in ("6/8", "9/8", "12/8", "3/4"):
        return None  # jigs and waltzes are out of scope for this pipeline
    try:
        sections = parse_abc_sections(abc)
    except Exception:
        return None
    if not sections or not any(s.notes for s in sections.values()):
        return None
    return CatalogTune(
        title=title, key=_normalize_key(key), meter=meter,
        sections=sections, source=source, rhythm=_field(abc, "R:") or "",
    )


def load_catalog(path: str | Path | None = None) -> list[CatalogTune]:
    """Load a catalog from a file, a directory of ABC files, or the builtin."""
    if path is None:
        return load_catalog(_BUILTIN)

    path = Path(path)
    files: list[Path]
    if path.is_dir():
        files = sorted(path.rglob("*.abc")) + sorted(path.rglob("*.ABC"))
    else:
        files = [path]

    out: list[CatalogTune] = []
    for f in files:
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for chunk in split_tunebook(text):
            tune = parse_catalog_tune(chunk, source=str(f))
            if tune is not None:
                out.append(tune)
    return out


def builtin_catalog() -> list[CatalogTune]:
    return load_catalog(_BUILTIN)


def _field(abc: str, prefix: str) -> str | None:
    for line in abc.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


_MODE_SUFFIX = {
    "mix": "mixolydian", "dor": "dorian", "min": "minor", "m": "minor",
    "maj": "major", "aeo": "minor", "ion": "major",
}


def _normalize_key(k: str) -> str:
    """Turn an ABC key field ("AMix", "Ador", "Bm") into "A mixolydian" etc."""
    k = k.strip().split()[0] if k.strip() else "D"
    root = k[0].upper()
    rest = k[1:]
    if rest[:1] in ("#", "b"):
        root += "#" if rest[0] == "#" else "-"
        rest = rest[1:]
    mode = _MODE_SUFFIX.get(rest[:3].lower(), None)
    if mode is None:
        mode = _MODE_SUFFIX.get(rest[:1].lower(), "major") if rest else "major"
    return f"{root} {mode}"
