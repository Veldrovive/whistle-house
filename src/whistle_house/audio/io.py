"""Audio file helpers."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def to_mono_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def load_audio(path: Path | str, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Read any libsndfile format (wav/flac/ogg/...) -> mono float32, optionally resampled."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    data = to_mono_float(data)
    if sample_rate is not None and sample_rate != sr:
        return resample(data, sr, sample_rate), sample_rate
    return data, sr


def save_wav(path: Path | str, samples: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), np.asarray(samples, np.float32), sample_rate, subtype="PCM_16")
