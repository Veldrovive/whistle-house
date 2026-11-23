"""Spectral gating against whichever of several noise profiles the room is in right now.

``spectral_gate`` assumes one stationary background. A room has several (quiet, fridge running,
bathroom fan with the door open or closed), so this filter holds several profiles (see
``whistle_house.noise_profiles``), estimates per frame which one is playing, and gates with the
predicted noise. The gate itself is exactly ``spectral_gate``'s; with one profile the output is
the same (``tests/test_adaptive_gate.py`` checks it).

Estimator: a discrete Bayes filter (the forward pass of an HMM) over the state (profile k,
foreground a). ``a = 1`` means some other sound (speech, a whistle, a door) is on top of the
background. Per frame ``x`` (dB per bin):

* Predict: ``p = A^T p B``. Profiles change rarely (``A``: stay with probability
  ``exp(-1/(dwell_s * fps))``, the rest spread evenly over the others); foreground comes and
  goes (``B``: leave background with time constant ``fg_onset_s``, leave foreground with
  ``fg_release_s``).
* Update, background (``a = 0``): the mean over bins of
  ``log[(1 - rise_prob) N(x; mu_k, sd_k) + rise_prob rise_k(x)]``. ``rise_k`` is the profile's
  Gaussian convolved with a uniform rise of 0..``rise_db`` dB: a bin raised by some quiet other
  sound, so one loud bin cannot refute a profile.
* Update, foreground (``a = 1``): ``-log(fg_width_db)`` for every profile, i.e. a foreground
  frame says nothing about which profile is on; the belief over profiles just holds while
  someone talks or whistles, instead of jumping to the loudest profile.
* Bins are not independent, so the per-bin mean log-likelihood is multiplied by
  ``evidence_fraction * n_bins`` (an effective number of independent bins) before it is turned
  into a probability.

Gate: the predicted noise is the posterior mixture of the profiles, moment-matched in dB
(``mu = sum w_k mu_k``, ``var = sum w_k (sd_k^2 + (mu_k - mu)^2)``). The gate then does what
``spectral_gate`` does, except that it smooths ``x`` over time and subtracts the current ``mu``
(spectral_gate smooths ``x - mu``; the same thing when ``mu`` is constant).

All state is a per-frame recursion, so streaming == offline for any chunk size.

Prototype (5 real silence takes on the deployment mic, profiles from the first half of each,
test on the second half, with synthetic speech/whistle/hum/white bursts on top): 96-100% of
frames picked the right profile. ``telemetry()`` reports the per-frame posterior for the
manager's Live page.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field
from scipy.special import ndtr

from whistle_house.noise_profiles import NoiseProfileStore, canonical_stage, upstream_fingerprint
from whistle_house.stages.base import Filter, StageConfig, StageContext, StageKind, register
from whistle_house.stages.filters.spectral_gate import _pole
from whistle_house.types import FrameBlock

_LIK_CHUNK = 256  # frames per vectorised likelihood evaluation (bounds memory on long clips)
_FLOOR = 1e-12  # likelihood floor (a bin far below every profile's mean)


@register
class AdaptiveGate(Filter):
    """Spectral gate that tracks which of several noise profiles (fridge, fan, quiet) is on."""

    name = "adaptive_gate"
    relative_output = True

    class Config(StageConfig):
        profiles: list[str] = Field(
            default_factory=list,
            description="Noise profile names (data/noise_profiles/<name>.json), one per "
            "background the room can be in (quiet, fridge, fan, ...), all measured at this "
            "filter's position. Empty = pass frames through unchanged.",
        )
        dwell_s: float = Field(
            20.0, description="Typical time (s) the room stays in one background before switching"
        )
        fg_onset_s: float = Field(
            0.5, description="Time constant (s) of a foreground sound (speech, whistle) starting"
        )
        fg_release_s: float = Field(
            0.7, description="Time constant (s) of a foreground sound ending"
        )
        rise_prob: float = Field(
            0.05,
            description="Share of bins assumed raised by some other quiet sound in a background "
            "frame (keeps single loud bins from refuting a profile)",
        )
        rise_db: float = Field(
            40.0, description="How far (dB) such a raised bin may be above the noise"
        )
        fg_width_db: float = Field(
            20.0,
            description="Spread (dB) of a foreground frame's level: smaller = frames that fit "
            "no profile are called foreground sooner",
        )
        evidence_fraction: float = Field(
            0.25,
            description="Effective share of independent bins (bins are correlated): scales "
            "how strongly one frame moves the belief",
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
        if len(set(config.profiles)) != len(config.profiles):
            raise ValueError("adaptive_gate: each noise profile may be listed only once")
        mus, sds = [], []
        for name in config.profiles:
            mu, sd = _load_profile(name, ctx)
            if mus and mu.shape != mus[0].shape:
                raise ValueError(f"noise profile {name!r} has a different number of bins")
            mus.append(mu)
            sds.append(sd)
        self._mu = np.stack(mus).astype(np.float64) if mus else None  # (K, F)
        self._sd = np.stack(sds).astype(np.float64) if sds else None
        k = max(1, int(config.freq_smooth_bins))
        self._kernel = np.bartlett(k + 2)[1:-1] if k > 1 else None
        if self._kernel is not None:
            self._kernel /= self._kernel.sum()
        self._a_smooth = _pole(config.smooth_frames)
        self._a_release = _pole(config.release_frames)
        fps = _frame_rate(ctx)
        n = len(config.profiles)
        stay = float(np.exp(-1.0 / (config.dwell_s * fps))) if config.dwell_s > 0 else 0.0
        if n <= 1:
            self._A = np.eye(max(n, 1))
        else:
            self._A = np.full((n, n), (1.0 - stay) / (n - 1))
            np.fill_diagonal(self._A, stay)
        s_on = _pole(config.fg_onset_s * fps)  # P(stay background)
        s_off = _pole(config.fg_release_s * fps)  # P(stay foreground)
        self._B = np.array([[s_on, 1.0 - s_on], [1.0 - s_off, s_off]])
        self._telemetry: dict | None = None
        self.reset()

    @property
    def profile_names(self) -> list[str]:
        return list(self.config.profiles)

    def reset(self) -> None:
        n = max(1, len(self.config.profiles))
        self._p = np.full((n, 2), 0.5 / n)  # belief over (profile, foreground)
        self._xs: np.ndarray | None = None  # smoothed input frame
        self._mask: np.ndarray | None = None
        self._last_mu: np.ndarray | None = None
        self._last_x: np.ndarray | None = None
        self._telemetry = None

    # ------------------------------------------------------------------ estimator

    def _background_loglik(self, x: np.ndarray) -> np.ndarray:
        """(T, F) frames -> (T, K) mean over bins of the background log-likelihood."""
        c = self.config
        mu, sd = self._mu[None], self._sd[None]  # (1, K, F)
        z = (x[:, None, :] - mu) / sd
        g = np.exp(-0.5 * z * z) / (sd * np.sqrt(2 * np.pi))
        rise = (ndtr(z) - ndtr(z - c.rise_db / sd)) / c.rise_db
        return np.log((1 - c.rise_prob) * g + c.rise_prob * rise + _FLOOR).mean(axis=2)

    def _posteriors(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the forward recursion over the frames -> (profile probs (T, K), fg prob (T,))."""
        c = self.config
        T, K = x.shape[0], self._mu.shape[0]
        scale = c.evidence_fraction * x.shape[1]
        l1 = -np.log(c.fg_width_db)
        probs, fg = np.empty((T, K)), np.empty(T)
        for s in range(0, T, _LIK_CHUNK):
            l0 = self._background_loglik(x[s : s + _LIK_CHUNK])
            for j, row in enumerate(l0):
                p = self._A.T @ self._p @ self._B
                ll = scale * np.stack([row, np.full(K, l1)], axis=1)
                p = p * np.exp(ll - ll.max())
                p /= p.sum()
                self._p = p
                probs[s + j] = p.sum(axis=1)
                fg[s + j] = p[:, 1].sum()
        return probs, fg

    # ------------------------------------------------------------------ gate

    def process(self, block: FrameBlock) -> FrameBlock:
        if self._mu is None:
            return block
        if block.n_frames == 0:
            self._telemetry = self._pack(np.zeros((0, self._mu.shape[0])), np.zeros(0))
            return block
        c = self.config
        x = block.values.astype(np.float64)
        probs, fg = self._posteriors(x)
        out = np.empty_like(block.values)
        for i, frame in enumerate(x):
            w = probs[i][:, None]
            mu = (w * self._mu).sum(axis=0)
            sigma = np.sqrt((w * (self._sd**2 + (self._mu - mu) ** 2)).sum(axis=0))
            thr = c.n_std * sigma
            width = np.maximum(c.softness_std * sigma, 1e-3)
            d = frame - mu
            if self._xs is None:
                self._xs = frame.copy()
                self._mask = np.zeros_like(d)
            else:
                self._xs += (1.0 - self._a_smooth) * (frame - self._xs)
            level = self._xs - mu
            lvl = np.maximum(level, d - thr)
            m = 1.0 / (1.0 + np.exp(-(lvl - thr) / width))
            if self._kernel is not None:
                m = np.convolve(m, self._kernel, mode="same")
            self._mask = np.maximum(m, self._a_release * self._mask)
            out[i] = self._mask * np.maximum(level if c.smooth_output else d, 0.0)
            self._last_mu = mu
        self._last_x = x[-1]
        self._telemetry = self._pack(probs, fg)
        return FrameBlock(out.astype(np.float32), block.times, block.axis)

    # ------------------------------------------------------------------ telemetry

    def _pack(self, probs: np.ndarray, fg: np.ndarray) -> dict:
        def r(v, nd):
            return None if v is None else np.round(v, nd).tolist()

        return {
            "profiles": self.profile_names,
            "series": {"probs": r(probs, 3), "foreground": r(fg, 3)},
            "mu": r(self._last_mu, 1),  # effective noise mean of the last frame
            "input": r(self._last_x, 1),  # the last input frame (this filter's input)
        }

    def telemetry(self) -> dict | None:
        return self._telemetry

    def telemetry_info(self) -> dict | None:
        if self._mu is None:
            return None
        return {
            "profiles": self.profile_names,
            "means": np.round(self._mu, 1).tolist(),
            "freqs": self.ctx.axis.freqs.tolist() if self.ctx.axis is not None else None,
        }


def _load_profile(name: str, ctx: StageContext) -> tuple[np.ndarray, np.ndarray]:
    """(mean, std) of a profile, checked against this filter's axis and upstream chain."""
    if ctx.data_dir is None:
        raise ValueError("adaptive_gate needs a data directory to load its noise profiles")
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
    return mu, sigma


def _frame_rate(ctx: StageContext) -> float:
    """Frames per second, from the upstream transform's hop (default 512 samples)."""
    up = ctx.extra.get("upstream")
    hop = 512
    if up is not None:
        t = canonical_stage(StageKind.transform, up["transform"]["name"], up["transform"]["params"])
        hop = int(t["params"].get("hop", hop))
    return ctx.sample_rate / hop
