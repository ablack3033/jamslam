"""Audio loading and preprocessing.

Kept deliberately thin. The only real jobs are: get any container the user
throws at us (m4a from a phone, wav, mp3, flac) into mono float32 at a known
sample rate, and apply preprocessing that is *uncontroversial*.

Notably absent: source separation, denoising, EQ. Those change what the melody
extractor sees in ways we cannot yet measure, so they do not belong here until
the eval corpus can score them.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .domain import Audio

_FFMPEG = shutil.which("ffmpeg")


def load_audio(path: str | Path, sample_rate: int = 44100) -> Audio:
    """Decode ``path`` to mono float32 at ``sample_rate``.

    Tries ffmpeg first when available because it handles the phone-recording
    formats (m4a/aac) that soundfile cannot, then falls back to soundfile and
    finally librosa.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if _FFMPEG is not None:
        try:
            return _load_with_ffmpeg(path, sample_rate)
        except (subprocess.CalledProcessError, OSError):
            pass  # fall through to the pure-python decoders

    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1).astype(np.float32)
        if sr != sample_rate:
            mono = _resample(mono, sr, sample_rate)
        return Audio(samples=mono, sample_rate=sample_rate, source_path=str(path))
    except Exception:
        import librosa

        mono, _ = librosa.load(str(path), sr=sample_rate, mono=True)
        return Audio(
            samples=mono.astype(np.float32),
            sample_rate=sample_rate,
            source_path=str(path),
        )


def _load_with_ffmpeg(path: Path, sample_rate: int) -> Audio:
    cmd = [
        _FFMPEG, "-nostdin", "-v", "error",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sample_rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if samples.size == 0:
        raise OSError(f"ffmpeg produced no audio for {path}")
    return Audio(samples=samples, sample_rate=sample_rate, source_path=str(path))


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    import librosa

    return librosa.resample(x, orig_sr=sr_in, target_sr=sr_out)


def preprocess(audio: Audio, *, target_peak: float = 0.9) -> Audio:
    """Normalize level and remove DC offset.

    Peak normalization matters because Melodia's salience threshold is not
    fully level-invariant, and phone recordings of jams vary wildly in level.
    We do not compress or gate: dynamics carry onset information.
    """
    x = np.asarray(audio.samples, dtype=np.float32)
    if x.size == 0:
        return audio
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x)))
    if peak > 1e-6:
        x = x * (target_peak / peak)
    return Audio(
        samples=x.astype(np.float32),
        sample_rate=audio.sample_rate,
        source_path=audio.source_path,
    )


def write_wav(path: str | Path, audio: Audio) -> None:
    """Write a wav using the standard library, avoiding a soundfile dependency."""
    import wave

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio.samples, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio.sample_rate)
        w.writeframes(pcm16.tobytes())


def emphasize_sustained(
    audio: Audio,
    power: float = 2.0,
    lookback_sec: float = 0.15,
    n_fft: int = 2048,
    hop_length: int = 256,
) -> Audio:
    """Attenuate plucked instruments relative to the bowed one.

    In an old-time jam the fiddle is the **only bowed instrument** -- banjo,
    guitar, mandolin and bass are all plucked. That is a physical asymmetry a
    generic melody extractor cannot know about, and it is directly measurable:
    a plucked note decays from a sharp attack, so its magnitude sits well below
    its own recent peak, while a bowed note sustains near it.

    So each time-frequency bin is scaled by how close it is to its recent
    maximum. Bins that are still ringing near their peak survive; bins in the
    tail of a decay are attenuated. ``power`` sets how aggressive that is.

    Measured on a real recording, raising ``power`` monotonically reduces both
    drone-parking and bass-tracking in the extracted melody. It is a modest
    effect on its own, but it costs nothing and needs no model.
    """
    import librosa
    from scipy.ndimage import maximum_filter1d

    if power <= 0:
        return audio

    y = np.asarray(audio.samples, dtype=np.float32)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)

    lookback = max(1, int(lookback_sec * audio.sample_rate / hop_length))
    recent_peak = maximum_filter1d(
        magnitude, size=2 * lookback + 1, axis=1, origin=-lookback // 2
    )
    sustain = np.clip(magnitude / (recent_peak + 1e-9), 0.0, 1.0) ** power

    masked = librosa.istft(
        magnitude * sustain * np.exp(1j * phase), hop_length=hop_length, length=len(y)
    )
    peak = float(np.max(np.abs(masked)))
    if peak > 1e-9:
        masked = masked / peak * 0.9
    return Audio(
        samples=masked.astype(np.float32),
        sample_rate=audio.sample_rate,
        source_path=audio.source_path,
    )
