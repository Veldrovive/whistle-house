"""Spectral gating against a measured noise profile (stationary noise suppression).

The same idea as ``noisereduce``'s stationary mode / classic spectral gating: a silence recording
gives each bin's noise mean ``mu`` and std ``sigma`` (see ``whistle_house.noise_profiles``). A bin
passes when its level is clearly above the noise, ``x - mu > n_std * sigma``. Raw per-frame noise
in dB fluctuates by several dB, so the level used for the decision is first smoothed over time
with a short causal (one-pole) average, which suppresses "musical noise" speckles without
needing lookahead. The decision is soft (a sigmoid ``softness_std`` wide), optionally smoothed
across neighbouring bins, and has a slow release so a note's tail is not chopped.

Output: ``mask * max(level, 0)``, the smoothed height above the noise mean in dB, 0 where gated.
That is the same "0 = floor" convention as ``noise_floor``, so ``peak_tracker`` defaults work.
Unlike ``noise_floor`` the floor is fixed, so a long held note is not slowly absorbed into it.

Design notes (from tests on the real mic + synthetic mixes):

* ``smooth_output=False`` outputs the raw frame instead of the smoothed level. That lets
  single-frame spikes in broadband sound (breath, speech) through as isolated, very prominent
  peaks, because their gated neighbours drop to 0; it gave ~20x more false events.
* A "keep absolute dB, attenuate gated bins" output was tried and rejected for the same reason.

Everything is per-frame causal state, so streaming == offline for any chunk size.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from whistle_house.noise_profiles import NoiseProfileStore, upstream_fingerprint
from whistle_house.stages.base import Filter, StageConfig, StageContext, register
from whistle_house.types import FrameBlock


@register
class SpectralGate(Filter):
    """Suppress the mic's stationary noise using a profile measured from silence recordings."""

    name = "spectral_gate"

    class Config(StageConfig):
        profile: str = Field(
            "",
            description="Noise profile name (data/noise_profiles/<name>.json). Build one from "
            "silence recordings in the manager's filter editor or `whistle-house noise-profile "
            "create`. Empty = pass frames through unchanged.",
        )
        n_std: float = Field(
            2.0, description="Pass bins more than this many noise std-devs above the noise mean"
        )
        softness_std: float = Field(
            0.5, description="Width (in std-devs) of the soft transition around the threshold"
        )
        smooth_frames: float = Field(
            2.0,
            description="Time constant (frames) of the causal level smoothing used for the "
            "gate decision. 0 = decide on raw frames.",
        )
        release_frames: float = Field(
            3.0, description="Time constant (frames) for the gate closing after a sound ends"
        )
        freq_smooth_bins: int = Field(
            1, description="Smooth the gate mask over this many neighbouring bins (1 = off)"
        )
        smooth_output: bool = Field(
            True,
            description="Output the time-smoothed level (smooth_frames) instead of the raw "
            "frame, so single-frame noise spikes do not survive the gate",
        )

    def __init__(self, config: Config, ctx: StageContext):
        super().__init__(config, ctx)
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        if config.profile:
            self._load(config.profile, ctx)
        k = max(1, int(config.freq_smooth_bins))
        self._kernel = np.bartlett(k + 2)[1:-1] if k > 1 else None
        if self._kernel is not None:
            self._kernel /= self._kernel.sum()
        self._a_smooth = _pole(config.smooth_frames)
        self._a_release = _pole(config.release_frames)
        self.reset()

    def _load(self, name: str, ctx: StageContext) -> None:
        if ctx.data_dir is None:
            raise ValueError("spectral_gate needs a data directory to load its noise profile")
        prof = NoiseProfileStore(ctx.data_dir / "noise_profiles").get(name)
        freqs, mu, sigma = prof.arrays()
        if ctx.axis is not None and (
            freqs.shape != ctx.axis.freqs.shape or not np.allclose(freqs, ctx.axis.freqs)
        ):
            raise ValueError(
                f"noise profile {name!r} was measured on {freqs.size} bins, this filter sees "
                f"{ctx.axis.n_bins}: rebuild the profile for this pipeline"
            )
        up = ctx.extra.get("upstream")
        if up is not None:
            here = upstream_fingerprint(up["sample_rate"], up["transform"], up["filters"])
            if here != prof.upstream:
                raise ValueError(
                    f"noise profile {name!r} was measured with a different transform/filter "
                    "chain before this filter; rebuild it from silence for this pipeline"
                )
        self._mu, self._sigma = mu, sigma

    def reset(self) -> None:
        self._level: np.ndarray | None = None  # smoothed (x - mu)
        self._mask: np.ndarray | None = None

    def process(self, block: FrameBlock) -> FrameBlock:
        if self._mu is None or block.n_frames == 0:
            return block
        c = self.config
        mu, sigma = self._mu, self._sigma
        thr = c.n_std * sigma
        width = np.maximum(c.softness_std * sigma, 1e-3)
        out = np.empty_like(block.values)
        for i, frame in enumerate(block.values):
            d = frame - mu
            if self._level is None:
                self._level = d.copy()
                self._mask = np.zeros_like(d)
            else:
                self._level += (1.0 - self._a_smooth) * (d - self._level)
            # decide on the smoothed level, but never miss a strong onset the smoothing lags
            lvl = np.maximum(self._level, d - thr)
            m = 1.0 / (1.0 + np.exp(-(lvl - thr) / width))
            if self._kernel is not None:
                m = np.convolve(m, self._kernel, mode="same")
            # instant attack, slow release
            self._mask = np.maximum(m, self._a_release * self._mask)
            out[i] = self._mask * np.maximum(self._level if c.smooth_output else d, 0.0)
        return FrameBlock(out.astype(np.float32), block.times, block.axis)


def _pole(tau_frames: float) -> float:
    """One-pole coefficient for a time constant in frames (0 -> no smoothing)."""
    return float(np.exp(-1.0 / tau_frames)) if tau_frames > 0 else 0.0
