"""Render exported MIDI to audio, so a transcription can be heard rather than read.

Hearing the output is the fastest honest check available while there is no
ground truth: a transcription that has locked onto a drone is obvious in two
seconds of playback and takes far longer to see in notation.

Requires fluidsynth with a General MIDI soundfont, and ffmpeg.

Usage:

    python tools/render_audio.py work/final --out work/audio
    python tools/render_audio.py work/final --out work/audio --compare memory_of_home
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOUNDFONTS = (
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/default.sf2",
)

#: Source audio we may publish alongside a rendering. Only the gold-standard
#: reference recording is committed to this repository.
SOURCE_AUDIO = {"memory_of_home": "datasets/tunes/memory_of_home/recording.m4a"}

#: Seconds of each side to use when building an A/B comparison.
COMPARE_SECONDS = 25


def find_soundfont() -> str:
    for path in SOUNDFONTS:
        if Path(path).exists():
            return path
    raise SystemExit(
        "No General MIDI soundfont found. Install one, e.g.\n"
        "  apt-get install fluidsynth fluid-soundfont-gm"
    )


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed:\n{result.stderr[-2000:]}")


def render(midi: Path, out_dir: Path, soundfont: str) -> Path:
    """MIDI -> normalised mp3."""
    out = out_dir / f"{midi.stem}.mp3"
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "raw.wav"
        run(["fluidsynth", "-ni", "-F", str(wav), "-r", "44100",
             soundfont, str(midi)])
        # The exporter writes modest velocities, so a straight render peaks
        # around -29 dBFS and sounds broken rather than quiet. Normalise to a
        # conventional listening level.
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
             "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             # A single synthesised melody line: mono at a modest bitrate is
             # transparent here and keeps a five-minute take near 3 MB.
             "-ac", "1", "-codec:a", "libmp3lame", "-b:a", "96k", str(out)])
    return out


def compare(name: str, rendered: Path, out_dir: Path) -> Path | None:
    """Original recording followed by its transcription, for an A/B listen."""
    source = ROOT / SOURCE_AUDIO[name]
    if not source.exists():
        return None
    out = out_dir / f"{name}.compare.mp3"
    with tempfile.TemporaryDirectory() as tmp:
        original = Path(tmp) / "original.wav"
        transcription = Path(tmp) / "transcription.wav"
        run(["ffmpeg", "-y", "-loglevel", "error", "-t", str(COMPARE_SECONDS),
             "-i", str(source), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-ar", "44100", "-ac", "2", str(original)])
        run(["ffmpeg", "-y", "-loglevel", "error", "-t", str(COMPARE_SECONDS),
             "-i", str(rendered), "-ar", "44100", "-ac", "2",
             str(transcription)])
        listing = Path(tmp) / "list.txt"
        listing.write_text(f"file '{original}'\nfile '{transcription}'\n")
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-codec:a", "libmp3lame", "-b:a", "192k",
             str(out)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="directory holding .mid output")
    ap.add_argument("names", nargs="*", help="restrict to these stems")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare", action="append", default=[],
                    help="also build an original-then-transcription A/B file")
    args = ap.parse_args()

    for tool in ("fluidsynth", "ffmpeg"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} is not installed")

    out_dir = args.out or args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    soundfont = find_soundfont()

    paths = sorted(args.source.glob("*.mid"))
    if args.names:
        paths = [p for p in paths if p.stem in set(args.names)]

    for midi in paths:
        rendered = render(midi, out_dir, soundfont)
        print("wrote", rendered)
        if midi.stem in args.compare and midi.stem in SOURCE_AUDIO:
            ab = compare(midi.stem, rendered, out_dir)
            if ab:
                print("wrote", ab)


if __name__ == "__main__":
    main()
