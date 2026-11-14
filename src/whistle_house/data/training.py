"""Training example generation for frame-level whistle detectors (numpy only, no torch).

An example is ``(features (T, F), f0 (T,))`` where features are the *pipeline front-end output*
(transform + filters, exactly what the detector sees live) and f0 is the target pitch in Hz per
frame (0 = no whistle).

Pitch labels for real whistle recordings are produced automatically by running the heuristic
``peak_tracker`` on the clean (quiet-room) recording - in clean conditions it is accurate. If a
recording has ``segments``, labels are restricted to its ``whistle`` segments. The labels are
then kept while noise/speech is mixed in, which is what teaches the model to ignore it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from whistle_house.data.augment import (
    NoiseBank,
    mix_at_snr,
    random_eq,
    random_gain,
    random_reverb,
    rms,
    synth_whistle,
)
from whistle_house.data.store import NEGATIVE_LABELS, Recording, RecordingStore
from whistle_house.pipeline import Frontend
from whistle_house.stages.base import StageContext, build_stage
from whistle_house.types import WhistleEvent

log = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    window_s: float = 3.0
    warmup_s: float = 1.0  # discarded lead-in so stateful filters (noise floor) are settled
    p_real_whistle: float = 0.45
    p_synth_whistle: float = 0.25  # remainder: negative-only examples
    p_interferer: float = 0.85
    snr_db: tuple[float, float] = (-3.0, 25.0)
    p_reverb: float = 0.3
    p_eq: float = 0.5
    # peak_tracker params used to auto-label clean whistle recordings
    labeler_params: dict = field(default_factory=dict)


def events_to_f0(events: list[WhistleEvent], n_samples: int, sr: int) -> np.ndarray:
    """Per-sample f0 (Hz, 0 = none) from event contours (linear interpolation inside events)."""
    f0 = np.zeros(n_samples, np.float32)
    for ev in events:
        ts = np.array([p.t for p in ev.contour])
        hz = np.array([p.hz for p in ev.contour])
        s, e = max(0, int(ev.start * sr)), min(n_samples, int(ev.end * sr) + 1)
        if e > s:
            f0[s:e] = np.interp(np.arange(s, e) / sr, ts, hz)
    return f0


class ExampleGenerator:
    def __init__(
        self,
        frontend_spec: dict,
        store: RecordingStore | None,
        noise_dirs: list[Path],
        splits: tuple[str, ...] = ("train",),
        config: GeneratorConfig | None = None,
        data_dir: Path | None = None,
    ):
        self.cfg = config or GeneratorConfig()
        self.frontend_spec = frontend_spec
        self.frontend = Frontend.from_dict(frontend_spec, data_dir)
        self.sr = self.frontend.sample_rate
        self.positives: list[tuple[np.ndarray, np.ndarray]] = []
        neg_files: list[Path] = []
        if store is not None:
            for rec in store.list():
                if rec.split not in splits:
                    continue
                if rec.label == "whistle":
                    audio, _ = store.load(rec.id, self.sr)
                    self.positives.append((audio, self._auto_label(rec, audio)))
                elif rec.label in NEGATIVE_LABELS:
                    neg_files.append(store.audio_path(rec.id))
        self.noise = NoiseBank(self.sr, neg_files + NoiseBank.scan(noise_dirs))
        log.info(
            "generator: %d whistle recordings, %d interferer files",
            len(self.positives),
            len(self.noise.files),
        )

    def _auto_label(self, rec: Recording, audio: np.ndarray) -> np.ndarray:
        self.frontend.reset()
        _, filt = self.frontend.push(audio)
        det = build_stage(
            "detector",
            "peak_tracker",
            self.cfg.labeler_params,
            StageContext(self.sr, axis=self.frontend.axis),
        )
        events = det.process(filt).events + det.flush()
        f0 = events_to_f0(events, len(audio), self.sr)
        segs = [s for s in rec.segments if s.label == "whistle"]
        if segs:
            mask = np.zeros(len(audio), bool)
            for s in segs:
                mask[int(s.start * self.sr) : int(s.end * self.sr)] = True
            f0[~mask] = 0
        return f0

    @property
    def axis(self):
        return self.frontend.axis

    def example(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        c, sr = self.cfg, self.sr
        n = int((c.window_s + c.warmup_s) * sr)
        u = rng.random()
        p_real = c.p_real_whistle if self.positives else 0.0
        p_synth = c.p_synth_whistle + (c.p_real_whistle - p_real)
        if u < p_real:
            audio, f0_full = self.positives[int(rng.integers(len(self.positives)))]
            if len(audio) <= n:
                off = -int(rng.integers(0, n - len(audio) + 1))
            else:
                off = int(rng.integers(0, len(audio) - n + 1))
            clean, f0 = np.zeros(n, np.float32), np.zeros(n, np.float32)
            src = slice(max(off, 0), max(off, 0) + n)
            dst = slice(max(-off, 0), max(-off, 0) + len(audio[src]))
            clean[dst], f0[dst] = audio[src], f0_full[src]
        elif u < p_real + p_synth:
            sw = synth_whistle(sr, c.window_s + c.warmup_s, rng)
            clean, f0 = sw.audio, sw.f0
        else:
            clean, f0 = np.zeros(n, np.float32), np.zeros(n, np.float32)

        if rng.random() < c.p_reverb and clean.any():
            clean = random_reverb(clean, sr, rng)
        has_signal = rms(clean) > 1e-4
        if rng.random() < c.p_interferer or not has_signal:
            noise = random_gain(self.noise.sample(n, rng), rng, -30, 0)
            mix = mix_at_snr(clean, noise, rng.uniform(*c.snr_db)) if has_signal else noise
        else:
            mix = clean
        mix = mix + 10 ** rng.uniform(-5, -3) * rng.standard_normal(n).astype(np.float32)
        if rng.random() < c.p_eq:
            mix = random_eq(mix, sr, rng)
        mix = random_gain(mix, rng, -12, 6)
        mix = np.clip(mix, -1, 1).astype(np.float32)

        _, block = self.frontend.process_signal(mix)
        keep = block.times >= c.warmup_s
        idx = np.clip(np.round(block.times[keep] * sr).astype(int), 0, n - 1)
        return block.values[keep], f0[idx]

    def batch(self, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = zip(*(self.example(rng) for _ in range(size)), strict=True)
        t = min(x.shape[0] for x in xs)
        return np.stack([x[:t] for x in xs]), np.stack([y[:t] for y in ys])


def f0_to_bins(f0: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Hz -> fractional bin index on the axis (NaN where f0 == 0 or out of range)."""
    b = np.interp(f0, freqs, np.arange(len(freqs)), left=np.nan, right=np.nan)
    b[f0 <= 0] = np.nan
    return b


__all__ = ["ExampleGenerator", "GeneratorConfig", "events_to_f0", "f0_to_bins"]
