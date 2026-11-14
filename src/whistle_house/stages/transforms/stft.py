"""Streaming STFT core + spectrogram transforms.

Ported from v1 ``audio_lib.AudioStreamProcessor._process_chunk`` (the part of v1 that worked
well) and restructured so that:

* the STFT core (``StreamingSTFT``) is independent of what is done with the power spectrum, so
  new front-ends (mel, linear, CQT-ish, learned filterbanks, ...) reuse it;
* chunk size no longer has to be a multiple of the hop length;
* offline processing of a whole signal is literally ``push(signal)`` on a fresh instance, so the
  training data pipeline sees *exactly* the features that the live pipeline sees.

v1 maths preserved: symmetric Hann window (``np.hanning``), ``|rfft|^2`` power, Slaney mel
filterbank (librosa defaults), ``power_to_db(ref=1, amin=1e-10, top_db=80)`` applied per frame,
frame time = centre of the analysis window. One deliberate fix: v1 labelled mel bins with
``mel_frequencies(n_mels)``, which is slightly off from the real filter centres; we use the true
centres (``mel_frequencies(n_mels + 2)[1:-1]``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import Field

from whistle_house.stages.base import StageConfig, StageContext, Transform, register
from whistle_house.types import FrameBlock, FrequencyAxis

# --------------------------------------------------------------------------- mel helpers
# Numpy re-implementation of librosa's Slaney-style mel scale (htk=False, norm="slaney") so the
# runtime doesn't need librosa/numba. tests/test_transform.py checks parity with librosa.

_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    f = np.asanyarray(f, dtype=np.float64)
    mel = f / _F_SP
    return np.where(
        f >= _MIN_LOG_HZ,
        _MIN_LOG_MEL + np.log(np.maximum(f, 1e-12) / _MIN_LOG_HZ) / _LOGSTEP,
        mel,
    )


def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    m = np.asanyarray(m, dtype=np.float64)
    return np.where(
        m >= _MIN_LOG_MEL, _MIN_LOG_HZ * np.exp(_LOGSTEP * (m - _MIN_LOG_MEL)), _F_SP * m
    )


def mel_frequencies(n: int, fmin: float, fmax: float) -> np.ndarray:
    return mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n))


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """(n_mels, n_fft//2+1) Slaney-normalised triangular filters (== librosa.filters.mel)."""
    fftfreqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_f = mel_frequencies(n_mels + 2, fmin, fmax)
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    return (weights * enorm[:, None]).astype(np.float32)


def power_to_db(
    power: np.ndarray, amin: float = 1e-10, ref: float = 1.0, top_db: float | None = 80.0
) -> np.ndarray:
    """librosa.power_to_db applied independently to each frame (row), as v1 did."""
    db = 10.0 * np.log10(np.maximum(amin, power)) - 10.0 * np.log10(max(amin, ref))
    if top_db is not None and db.size:
        db = np.maximum(db, db.max(axis=-1, keepdims=True) - top_db)
    return db.astype(np.float32)


# --------------------------------------------------------------------------- STFT core


class StreamingSTFT:
    """Sliding-window power spectrum over a stream of samples.

    A frame is emitted every ``hop`` samples. The analysis buffer starts zero-filled (as in v1),
    so frame k covers samples ``[(k+1)*hop - n_fft, (k+1)*hop)`` and its time is the window
    centre ``((k+1)*hop - n_fft/2) / sr``.
    """

    def __init__(self, sample_rate: int, n_fft: int, hop: int):
        if hop <= 0 or hop > n_fft:
            raise ValueError("hop must be in (0, n_fft]")
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop = hop
        self.window = np.hanning(n_fft).astype(np.float32)
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(self.n_fft, np.float32)  # last n_fft samples consumed
        self._pending = np.zeros(0, np.float32)  # < hop samples not yet forming a frame
        self._frames_emitted = 0

    def push(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (power (n_frames, n_fft//2+1) float32, times (n_frames,) float64)."""
        samples = np.asarray(samples, np.float32).reshape(-1)
        data = np.concatenate([self._pending, samples]) if self._pending.size else samples
        n_hops = data.shape[0] // self.hop
        used = n_hops * self.hop
        self._pending = data[used:].copy()
        if n_hops == 0:
            return np.zeros((0, self.n_fft // 2 + 1), np.float32), np.zeros(0)

        ext = np.concatenate([self._buf, data[:used]])
        frames = sliding_window_view(ext, self.n_fft)[self.hop :: self.hop]
        self._buf = ext[-self.n_fft :].copy()

        spec = np.fft.rfft(frames * self.window, n=self.n_fft, axis=-1)
        power = (spec.real**2 + spec.imag**2).astype(np.float32)

        k = self._frames_emitted + np.arange(n_hops)
        times = ((k + 1) * self.hop - self.n_fft / 2) / self.sr
        self._frames_emitted += n_hops
        return power, times


# --------------------------------------------------------------------------- transforms


class _StftConfig(StageConfig):
    n_fft: int = Field(2048, description="FFT window length in samples")
    hop: int = Field(512, description="Samples between frames")
    fmin: float = Field(300.0, description="Lowest frequency kept (Hz)")
    fmax: float = Field(4000.0, description="Highest frequency kept (Hz)")
    top_db: float | None = Field(
        80.0, description="Per-frame dynamic range floor (dB below frame max); null disables"
    )
    output: Literal["db", "power"] = Field("db", description="Output units")


class _StftTransform(Transform):
    Config = _StftConfig

    def __init__(self, config: _StftConfig, ctx: StageContext):
        super().__init__(config, ctx)
        self.stft = StreamingSTFT(ctx.sample_rate, config.n_fft, config.hop)

    def reset(self) -> None:
        self.stft.reset()

    def _project(self, power: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def push(self, samples: np.ndarray) -> FrameBlock:
        power, times = self.stft.push(samples)
        proj = self._project(power)
        values = (
            power_to_db(proj, top_db=self.config.top_db) if self.config.output == "db" else proj
        )
        return FrameBlock(values.astype(np.float32), times, self.axis)


@register
class MelSpectrogram(_StftTransform):
    """STFT -> mel filterbank -> dB (the v1 transform)."""

    name = "stft_mel"

    class Config(_StftConfig):
        n_mels: int = Field(128, description="Number of mel bands between fmin and fmax")

    def __init__(self, config: Config, ctx: StageContext):
        super().__init__(config, ctx)
        self.basis = mel_filterbank(
            ctx.sample_rate, config.n_fft, config.n_mels, config.fmin, config.fmax
        )
        centres = mel_frequencies(config.n_mels + 2, config.fmin, config.fmax)[1:-1]
        self._axis = FrequencyAxis(centres, "mel")

    @property
    def axis(self) -> FrequencyAxis:
        return self._axis

    def _project(self, power: np.ndarray) -> np.ndarray:
        return power @ self.basis.T


@register
class LinearSpectrogram(_StftTransform):
    """STFT -> raw FFT bins cropped to [fmin, fmax] -> dB. Better pitch resolution up high."""

    name = "stft_linear"

    def __init__(self, config: _StftConfig, ctx: StageContext):
        super().__init__(config, ctx)
        freqs = np.fft.rfftfreq(config.n_fft, 1.0 / ctx.sample_rate)
        self._sel = np.flatnonzero((freqs >= config.fmin) & (freqs <= config.fmax))
        self._axis = FrequencyAxis(freqs[self._sel], "linear")

    @property
    def axis(self) -> FrequencyAxis:
        return self._axis

    def _project(self, power: np.ndarray) -> np.ndarray:
        return power[:, self._sel]
