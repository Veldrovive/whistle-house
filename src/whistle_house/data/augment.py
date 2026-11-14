"""Waveform-domain augmentation + synthetic audio.

Clean whistles are recorded in a quiet room, so robustness to background sound has to be
manufactured: every training example is (clean whistle or nothing) + (random interferer at a
random SNR) + (random gain/EQ/reverb). Interferers come from a ``NoiseBank``:

* negatives recorded on the deployment mic (``speech``/``noise``/``music``/``silence`` labels),
* external corpora dropped into ``<data_dir>/noise/`` (e.g. MUSAN speech/music/noise, ESC-50,
  DEMAND, LibriSpeech - see AGENTS.md),
* synthetic voiced "pseudo-speech" (harmonic stacks + moving formants), which reproduces the
  property of speech that fooled v1: strong, gliding harmonic peaks.

All functions take a ``np.random.Generator`` so data generation is reproducible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, lfilter

from whistle_house.audio.io import load_audio

log = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


# --------------------------------------------------------------------------- primitives


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """signal + noise scaled so that 20*log10(rms(signal)/rms(noise)) == snr_db."""
    noise = fit_length(noise, len(signal))
    scale = rms(signal) / (rms(noise) * 10 ** (snr_db / 20))
    return (signal + scale * noise).astype(np.float32)


def fit_length(x: np.ndarray, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random crop or tile x to exactly n samples."""
    if len(x) == 0:
        return np.zeros(n, np.float32)
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))))
    start = int(rng.integers(0, len(x) - n + 1)) if rng is not None else 0
    return x[start : start + n].astype(np.float32)


def random_gain(x: np.ndarray, rng: np.random.Generator, lo_db=-20.0, hi_db=6.0) -> np.ndarray:
    return (x * 10 ** (rng.uniform(lo_db, hi_db) / 20)).astype(np.float32)


def _peaking_biquad(f0: float, gain_db: float, q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a])
    return b / den[0], den / den[0]


def random_eq(x: np.ndarray, sr: int, rng: np.random.Generator, n_bands: int = 3) -> np.ndarray:
    """A few random peaking filters: simulates different mic/room colourations."""
    for _ in range(n_bands):
        b, a = _peaking_biquad(
            rng.uniform(150, min(7000, sr / 2 - 100)), rng.uniform(-8, 8), rng.uniform(0.5, 2.0), sr
        )
        x = lfilter(b, a, x)
    return x.astype(np.float32)


def random_reverb(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Convolve with a synthetic exponentially-decaying noise impulse response."""
    rt60 = rng.uniform(0.1, 0.8)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0 / rng.uniform(0.3, 1.0)  # direct path vs. reverb ratio
    ir /= np.sqrt(np.sum(ir**2))
    return fftconvolve(x, ir)[: len(x)].astype(np.float32)


# --------------------------------------------------------------------------- synthesis


@dataclass
class SynthWhistle:
    audio: np.ndarray
    f0: np.ndarray  # per-sample fundamental in Hz, 0 where silent


def synth_whistle(
    sr: int, duration: float, rng: np.random.Generator, n_notes: int | None = None
) -> SynthWhistle:
    """Random whistled phrase: notes with glides/vibrato, breath noise and weak harmonics."""
    n = int(duration * sr)
    audio = np.zeros(n, np.float32)
    f0 = np.zeros(n, np.float32)
    n_notes = n_notes or int(rng.integers(1, 6))
    t = 0.2 * rng.random()
    for _ in range(n_notes):
        dur = rng.uniform(0.08, 0.9)
        s, e = int(t * sr), min(n, int((t + dur) * sr))
        if e - s < int(0.05 * sr):
            break
        m = e - s
        tt = np.arange(m) / sr
        start_hz = rng.uniform(600, 3200)
        glide = rng.choice([0.0, 0.0, rng.uniform(-12, 12)])  # semitones over the note
        curve = start_hz * 2 ** (glide * (tt / max(tt[-1], 1e-3)) ** rng.uniform(0.5, 2) / 12)
        vib = rng.uniform(0, 0.3) * np.sin(2 * np.pi * rng.uniform(4, 7) * tt)  # semitones
        hz = curve * 2 ** (vib / 12)
        phase = 2 * np.pi * np.cumsum(hz) / sr
        env = np.minimum(1, np.minimum(tt / 0.02, (tt[-1] - tt) / 0.03 + 1e-3))
        env *= 1 + 0.15 * rng.standard_normal() * np.sin(2 * np.pi * 3 * tt)
        tone = np.sin(phase) + rng.uniform(0, 0.08) * np.sin(2 * phase)
        breath = rng.uniform(0.0, 0.15) * rng.standard_normal(m)
        audio[s:e] += (env * (tone + breath) * rng.uniform(0.2, 0.8)).astype(np.float32)
        f0[s:e] = hz
        t += dur + rng.uniform(0.05, 0.6)
        if t >= duration:
            break
    return SynthWhistle(audio, f0)


def _resonator(f: float, bw: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.exp(-np.pi * bw / sr)
    return np.array([1 - r]), np.array([1, -2 * r * np.cos(2 * np.pi * f / sr), r * r])


def synth_pseudo_speech(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    """Crude voiced-speech stand-in: jittery glottal pulse train through moving formants,
    with syllable-rate amplitude modulation. Not intelligible - just speech-like spectra."""
    n = int(duration * sr)
    out = np.zeros(n, np.float32)
    seg = int(0.12 * sr)
    f0_base = rng.uniform(85, 260)
    phase = 0.0
    for s in range(0, n, seg):
        m = min(seg, n - s)
        f0 = f0_base * 2 ** (rng.normal(0, 2) / 12)
        hz = f0 * (1 + 0.01 * rng.standard_normal(m))
        ph = phase + 2 * np.pi * np.cumsum(hz) / sr
        phase = ph[-1] % (2 * np.pi)
        src = (2 * ((ph / (2 * np.pi)) % 1) - 1).astype(np.float32)  # sawtooth: all harmonics
        src += 0.05 * rng.standard_normal(m).astype(np.float32)
        y = np.zeros(m, np.float32)
        for fc, bw in (
            (rng.uniform(300, 900), 80),
            (rng.uniform(900, 2400), 120),
            (rng.uniform(2300, 3300), 160),
        ):
            b, a = _resonator(fc, bw, sr)
            y += lfilter(b, a, src).astype(np.float32)
        out[s : s + m] = y
    syll = np.clip(
        np.sin(2 * np.pi * rng.uniform(3, 6) * np.arange(n) / sr + rng.uniform(0, 6)), 0, 1
    )
    out *= syll.astype(np.float32)
    return out / (np.abs(out).max() + 1e-9) * 0.5


# --------------------------------------------------------------------------- noise bank


class NoiseBank:
    """Random interferers from files on disk (+ synthetic pseudo-speech)."""

    def __init__(
        self,
        sample_rate: int,
        files: list[Path],
        synthetic_speech_prob: float = 0.25,
        max_cache: int = 200,
    ):
        self.sr = sample_rate
        self.files = files
        self.synthetic_speech_prob = synthetic_speech_prob if files else 1.0
        self._cache: dict[Path, np.ndarray] = {}
        self._max_cache = max_cache

    @staticmethod
    def scan(directories: list[Path]) -> list[Path]:
        out = []
        for d in directories:
            if d.exists():
                out += [p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
        return sorted(out)

    def _load(self, p: Path) -> np.ndarray:
        if p not in self._cache:
            if len(self._cache) >= self._max_cache:
                self._cache.pop(next(iter(self._cache)))
            try:
                self._cache[p] = load_audio(p, self.sr)[0]
            except Exception as e:  # noqa: BLE001 - corrupt/unsupported file must not kill training
                log.warning("skipping noise file %s: %s", p, e)
                self._cache[p] = np.zeros(0, np.float32)
        return self._cache[p]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < self.synthetic_speech_prob:
            return synth_pseudo_speech(self.sr, n / self.sr, rng)
        for _ in range(5):
            x = self._load(self.files[int(rng.integers(len(self.files)))])
            if len(x) and rms(x) > 1e-5:
                return fit_length(x, n, rng)
        return synth_pseudo_speech(self.sr, n / self.sr, rng)
