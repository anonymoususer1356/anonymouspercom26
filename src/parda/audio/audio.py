import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# Convert any ffmpeg-readable input into a mono floating-point WAV.
def load_audio(input_path: Path, working_wav: Path, sample_rate: int) -> np.ndarray:
    working_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-y",
        str(working_wav),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")

    audio, actual_rate = sf.read(working_wav, dtype="float32", always_2d=False)
    if actual_rate != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz audio, received {actual_rate} Hz")
    if audio.ndim != 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    return np.asarray(audio, dtype=np.float32)


# Resample a mono waveform using a polyphase filter.
def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    divisor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate // divisor)
    down = int(source_rate // divisor)
    return resample_poly(audio, up, down).astype(np.float32)


# Apply the envelope-following AGC used by the earlier diarisation pipeline.
def apply_agc(
    audio: np.ndarray,
    sample_rate: int,
    attack_ms: float = 50,
    release_ms: float = 500,
    target_level: float = 0.3,
    percentile: float = 25,
) -> np.ndarray:
    envelope = np.abs(audio)
    block_size = max(sample_rate, 1)
    block_thresholds = []

    for start in range(0, len(envelope), block_size):
        block = envelope[start : start + block_size]
        if len(block):
            block_thresholds.append(float(np.percentile(block, percentile)))

    if not block_thresholds:
        return audio.copy()

    positions = np.arange(len(block_thresholds)) * block_size
    threshold = np.interp(np.arange(len(audio)), positions, block_thresholds)
    attack = 1 - np.exp(-1 / (attack_ms * sample_rate / 1000))
    release = 1 - np.exp(-1 / (release_ms * sample_rate / 1000))
    gain = np.ones_like(audio)
    smoothed_envelope = 0.0

    for index, level in enumerate(envelope):
        coefficient = attack if level > smoothed_envelope else release
        smoothed_envelope = (
            smoothed_envelope * (1 - coefficient) + level * coefficient
        )
        if smoothed_envelope > threshold[index]:
            gain[index] = target_level / (smoothed_envelope + 1e-8)

    return (np.tanh(audio * gain) * 0.95).astype(np.float32)


# Apply a short fade to avoid clicks at separated-region boundaries.
def apply_edge_fade(audio: np.ndarray, sample_rate: int, fade_ms: float = 10) -> np.ndarray:
    fade_samples = min(int(sample_rate * fade_ms / 1000), len(audio) // 2)
    if fade_samples <= 1:
        return audio

    faded = audio.copy()
    ramp = np.sin(np.linspace(0, np.pi / 2, fade_samples, dtype=np.float32)) ** 2
    faded[:fade_samples] *= ramp
    faded[-fade_samples:] *= ramp[::-1]
    return faded
