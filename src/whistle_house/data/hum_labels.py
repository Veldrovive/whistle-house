"""Pitch auto-labels for hum recordings (pYIN), the hum counterpart of the whistle labeller.

Whistles are labelled by running ``peak_tracker`` on the front-end output, which assumes one
clean spectral peak per frame. A hum is a harmonic stack instead: on the recordings so far the
fundamental (~80-200 Hz) is the strongest harmonic in most frames, h2 is a few dB below it, and
h3+ fall off fast. pYIN works on the waveform (autocorrelation over whole periods), so it uses
every harmonic at once, finds the fundamental even when one harmonic is weak, and labels clean
hums almost completely (99-100% of frames voiced, ~0.1 semitone frame-to-frame jitter).

pYIN is too slow for live use on the Pi; this is for offline labelling (training machine,
manager inspection). It needs librosa (the ``ml`` extra), imported lazily.

Labels do not depend on the pipeline front-end (they are Hz from the waveform), so the cache
key is only the labeller config. Tracks are float16 f0 at ``LABEL_RATE`` (0 = unvoiced), the
same format as whistle labels.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from whistle_house.audio.io import resample, to_mono_float
from whistle_house.data.store import RecordingStore
from whistle_house.data.training import LABEL_RATE, LabelCache


@dataclass(frozen=True)
class HumLabelConfig:
    fmin: float = 60.0  # lowest fundamental searched (Hz); low male hums reach ~80 Hz
    fmax: float = 600.0  # highest fundamental; covers high voices with margin
    sample_rate: int = 16000  # analysis rate (hums have nothing useful above ~3 kHz)
    frame_length: int = 1024  # pYIN window at sample_rate (64 ms, > 2 periods of fmin)
    # pYIN's HMM decides voicing; frames it calls voiced but below this probability are also
    # dropped (0 = trust the HMM, which labels clean hums ~99% voiced)
    min_voiced_prob: float = 0.0
    min_duration_s: float = 0.15  # drop voiced runs shorter than this (clicks, breaths)

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def _drop_short_runs(voiced: np.ndarray, min_len: int) -> np.ndarray:
    out = voiced.copy()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(np.int8), [0]])))
    for s, e in zip(edges[::2], edges[1::2], strict=True):
        if e - s < min_len:
            out[s:e] = False
    return out


def hum_f0(audio: np.ndarray, sr: int, cfg: HumLabelConfig | None = None) -> np.ndarray:
    """Mono audio -> float16 f0 track at LABEL_RATE (frame i is centred at i / LABEL_RATE)."""
    import librosa

    cfg = cfg or HumLabelConfig()
    x = resample(np.asarray(audio, np.float32), sr, cfg.sample_rate)
    n = int(len(x) / cfg.sample_rate * LABEL_RATE) + 1
    if len(x) < cfg.frame_length:
        return np.zeros(n, np.float16)
    f0, vflag, prob = librosa.pyin(
        x,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        sr=cfg.sample_rate,
        frame_length=cfg.frame_length,
        hop_length=cfg.sample_rate // LABEL_RATE,
        center=True,
    )
    voiced = vflag & np.isfinite(f0) & (prob >= cfg.min_voiced_prob)
    voiced = _drop_short_runs(voiced, round(cfg.min_duration_s * LABEL_RATE))
    out = np.zeros(n, np.float32)
    m = min(n, len(f0))
    out[:m] = np.where(voiced[:m], f0[:m], 0.0)
    return out.astype(np.float16)


def harmonic_levels(
    audio: np.ndarray, sr: int, f0: np.ndarray, n_harmonics: int = 8, n_fft: int = 8192
) -> np.ndarray:
    """Median level (dB) of harmonics h1..hN relative to each frame's strongest one.

    ``f0`` is a LABEL_RATE track (e.g. from ``hum_f0``). NaN for a harmonic that never fits
    under Nyquist. Used to answer "which harmonics carry the pitch on this mic?".
    """
    x = np.asarray(audio, np.float32)
    hop = sr // LABEL_RATE
    if len(x) < n_fft:
        return np.full(n_harmonics, np.nan)
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, n_fft // 2), n_fft)[::hop]
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    m = min(len(frames), len(f0))
    idx = np.flatnonzero(np.asarray(f0[:m], np.float32) > 0)
    if not idx.size:
        return np.full(n_harmonics, np.nan)
    mag = np.abs(np.fft.rfft(frames[idx] * np.hanning(n_fft), axis=-1))
    levels = np.full((len(idx), n_harmonics), np.nan)
    for h in range(1, n_harmonics + 1):
        fh = h * f0[idx].astype(np.float64)
        ok = fh < sr / 2 - 50
        k = np.searchsorted(freqs, fh[ok])
        win = np.clip(k[:, None] + np.arange(-2, 3)[None, :], 0, len(freqs) - 1)
        levels[ok, h - 1] = 20 * np.log10(mag[np.flatnonzero(ok)[:, None], win].max(1) + 1e-12)
    rel = levels - np.nanmax(levels, axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        return np.nanmedian(rel, axis=0) if np.isfinite(rel).any() else rel[0]


def summarize(f0: np.ndarray) -> dict:
    """Voiced fraction and pitch range of a LABEL_RATE f0 track."""
    v = np.asarray(f0, np.float32)
    v = v[v > 0]
    out: dict = {"voiced_fraction": float(len(v) / max(1, len(f0)))}
    if len(v):
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        out.update(f0_median=float(p50), f0_p5=float(p5), f0_p95=float(p95))
    return out


def _hum_label_job(args) -> tuple[str, np.ndarray]:
    key, path, segments, cfg = args
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    f0 = hum_f0(to_mono_float(audio), sr, cfg)
    if segments:  # restrict to annotated hum segments
        mask = np.zeros(len(f0), bool)
        for s, e in segments:
            mask[int(s * LABEL_RATE) : int(e * LABEL_RATE) + 1] = True
        f0[~mask] = 0
    return key, f0


def hum_label_cache(cache_dir: Path, cfg: HumLabelConfig | None = None) -> LabelCache:
    return LabelCache(cache_dir, "hum", (cfg or HumLabelConfig()).key())


def ensure_hum_labels(
    store: RecordingStore,
    cache_dir: Path,
    rec_ids: list[str] | None = None,
    cfg: HumLabelConfig | None = None,
    workers: int = 1,
) -> dict[str, np.ndarray]:
    """f0 tracks for the given recordings (default: every ``hum`` recording), cached."""
    cfg = cfg or HumLabelConfig()
    recs = [r for r in store.list() if (r.id in rec_ids if rec_ids else r.label == "hum")]
    items, ids = [], []
    for r in recs:
        segs = [(s.start, s.end) for s in r.segments if s.label == "hum"]
        key = r.id + (":" + hashlib.sha1(repr(segs).encode()).hexdigest()[:8] if segs else "")
        items.append((key, str(store.audio_path(r.id)), segs))
        ids.append(r.id)
    labels = hum_label_cache(cache_dir, cfg).ensure(
        items, workers=workers, job=_hum_label_job, job_args=(cfg,)
    )
    return {rid: labels[k] for rid, (k, _, _) in zip(ids, items, strict=True)}
