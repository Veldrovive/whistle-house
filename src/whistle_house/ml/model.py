"""Lightweight learned detector for whistles and hums (requires the ``ml`` extra: torch).

Input:  (B, T, F) filtered spectrogram frames (normalised with stats stored in the checkpoint).
Output, per class in ``cfg.classes`` (K of them: ``whistle``, and ``hum`` for models that have it):
        presence logits (B, T, K)    - "is someone whistling / humming in this frame?"
        salience logits (B, T, K, F) - "which frequency bin is its pitch in?" (for a hum: the
                                       fundamental, never a harmonic)

Convolutions are *causal in time* (only past frames are used), so the model can run on a stream:
``forward_stream`` takes new frames plus a small state (each block's last input frames) and
gives exactly the offline result, computing every frame once.

``lookahead`` > 0 gives a delayed model: the causal network's output at frame ``t`` is the
prediction for frame ``t - lookahead``, so it has seen that much of the future. ``forward``
hides the delay (it pads the end with ``lookahead`` zero frames and drops the first outputs, so
outputs line up with inputs); a streaming caller uses ``forward_stream`` and handles the delay
itself (see ``stages/detectors/learned.py``).

Harmonic input (``cfg.harmonics``, the ``harmonic`` arch)
---------------------------------------------------------
A whistle is one spectral line. A hum is a periodic signal: lines at f0, 2 f0, 3 f0, ... They are
equally spaced in Hz, but the spacing *is* the pitch, so on a linear axis the pattern stretches
with every note; on a log axis the pattern is the same at every pitch (a pitch change is a
shift, which is what a convolution shares weights over) but the lines are unevenly spaced
(+12, +19, +24, +27.9 ... semitones), so small kernels never line up with them.

``HarmonicStack`` gets both properties at once. For every pitch candidate (a bin of the
log-frequency front end) it reads the spectrum at fixed multiples of that bin's own frequency:
channel ``j`` at bin ``b`` holds the level at ``harmonics[j] * f_b``. That is a linear frequency
axis measured in units of the candidate pitch. Along it, the harmonics of a note at ``f_b`` sit
at 1, 2, 3, ... (equally spaced, in the same channels whatever the pitch) and the gaps between
them at 1.5, 2.5, ... . On a log axis a multiple is a constant bin shift, so the layer is a few
fixed gathers with no parameters.

The network then sees, at each pitch bin, the whole comb in its input channels:

* whistle: energy in channel 1 only;
* hum: channels 1, 2, 3, ... filled with a falling level, the half-integer channels empty;
* the wrong candidates give themselves away: at 2 f0 or 3 f0 the sub-harmonic channels (1/2,
  1/3) are filled, at f0/2 only the even channels are.

With harmonics up to 8 every pitch bin has a frequency receptive field from a third of its
frequency to 8 times it. ``freq_context`` adds the rest: each block pools its features over the
whole frequency axis and feeds that back to every bin, so a bin also knows what the frame holds
anywhere else (broadband speech energy, a stronger comb elsewhere).

``ARCHS`` names the architectures training can build. ``small`` is the whistle-only Pi model.
``large`` is a whistle-only reference model for "how well can this be done". ``harmonic`` is the
whistle + hum model described above; it needs a log-frequency front end (``stft_log``).
``harmonic_deep`` is the same with twice the blocks (twice the time receptive field).
``harmonic_long`` reaches further in time in both directions: dilations up to 32 (2.9 s
receptive field) and 32 frames of lookahead.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Multiples of a pitch candidate's frequency that the harmonic model looks at: harmonics 1-8 and
# the midpoints between them (an equally spaced linear axis in units of the pitch), plus the
# sub-harmonics 1/2 and 1/3, which are only filled when the candidate is itself a harmonic.
HARMONIC_GRID: tuple[float, ...] = (1 / 3, *(0.5 * i for i in range(1, 17)))


@dataclass
class WhistleNetConfig:
    n_bins: int = 128
    channels: int = 16
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    kernel_t: int = 3
    kernel_f: int = 3
    dilations_f: tuple[int, ...] = ()  # per-block frequency dilation; () = 1 everywhere
    lookahead: int = 0  # frames of future context (output is delayed by this much when streaming)
    classes: tuple[str, ...] = ("whistle",)  # one presence + salience output per class
    # Harmonic input: frequency multiples stacked as input channels (see module docstring).
    # () = the plain spectrogram as a single channel. Needs ``bins_per_octave`` of the (log)
    # frequency axis the model is trained on.
    harmonics: tuple[float, ...] = ()
    bins_per_octave: float = 0.0
    freq_context: bool = False  # per-block pooling over the whole frequency axis

    @property
    def receptive_field(self) -> int:
        """Input frames one output depends on (``lookahead`` of them are after its frame)."""
        return 1 + sum((self.kernel_t - 1) * d for d in self.dilations)


ARCHS: dict[str, dict] = {
    "small": {},  # ~7k params, causal: the whistle-only Pi model
    "large": {  # ~190k params, 16 frames of lookahead: the reference / upper-bound model
        "channels": 48,
        "dilations": (1, 2, 4, 8, 16, 1, 2, 4, 8, 16),
        "dilations_f": (1, 2, 4, 8, 1, 1, 2, 4, 8, 1),
        "lookahead": 16,
    },
    # Whistles + hums on a log-frequency front end (stft_log). ~31k params. Time receptive field
    # 63 frames (0.73 s at hop 512 / 44.1 kHz) plus 12 frames of lookahead: telling a hum from
    # speech is about how the comb moves over time, not what one frame looks like.
    "harmonic": {
        "channels": 24,
        "dilations": (1, 2, 4, 8, 16),
        "lookahead": 12,
        "classes": ("whistle", "hum"),
        "harmonics": HARMONIC_GRID,
        "freq_context": True,
    },
    # ``harmonic`` twice as deep: two dilation cycles, 10 blocks, ~60k params. Time receptive
    # field 125 frames (1.45 s) + the same 12 frames of lookahead, about twice the cost per
    # frame. For the open problem of the harmonic model: voiced speech has a real comb at its
    # pitch, so only how it moves over time (over several syllables) tells it from a hum.
    "harmonic_deep": {
        "channels": 24,
        "dilations": (1, 2, 4, 8, 16, 1, 2, 4, 8, 16),
        "lookahead": 12,
        "classes": ("whistle", "hum"),
        "harmonics": HARMONIC_GRID,
        "freq_context": True,
    },
    # Longer context in both directions, for the same problem. Two dilation cycles up to 32:
    # 12 blocks, ~75k params, time receptive field 253 frames (2.94 s), about 1.3x the cost per
    # frame of ``harmonic_deep`` (dilation is free; the streaming state grows to ~250 frames).
    # 32 frames (0.37 s) of that are lookahead, so 2.6 s are past: a decision about a frame
    # sees whether the sound went on as a held note or broke up into syllables. Events arrive
    # 0.37 s late; a tune only fires after ``quiet_after_s`` of silence anyway, but the
    # recognizer's ``settle_s`` must cover the delay (0.5 instead of the default 0.25).
    # Training uses longer windows for it (``ml/train.py``: 1.5x the receptive field).
    "harmonic_long": {
        "channels": 24,
        "dilations": (1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32),
        "lookahead": 32,
        "classes": ("whistle", "hum"),
        "harmonics": HARMONIC_GRID,
        "freq_context": True,
    },
}


def log_bins_per_octave(freqs: np.ndarray, tol: float = 1e-3) -> float:
    """Bins per octave of a log-spaced frequency axis; raises if the axis is not log-spaced."""
    steps = np.diff(np.log2(np.asarray(freqs, np.float64)))
    if len(steps) < 2 or np.any(steps <= 0) or np.ptp(steps) > tol * steps.mean():
        raise ValueError(
            "the harmonic model needs a log-frequency front end (every bin the same number of "
            "cents apart), e.g. the stft_log transform"
        )
    return round(float(1.0 / steps.mean()), 6)


class HarmonicStack(nn.Module):
    """(B, T, F) log-frequency frames -> (B, H, T, F): the spectrum at multiples of each bin.

    Channel ``j`` at bin ``b`` is the input at frequency ``multiples[j] * f_b``, i.e. shifted by
    ``bins_per_octave * log2(multiples[j])`` bins (linear interpolation between the two
    neighbouring bins). Positions off the axis are filled with the frame's lowest level
    ("nothing there"). No parameters; per frame, so it streams trivially.
    """

    def __init__(self, n_bins: int, bins_per_octave: float, multiples: tuple[float, ...]):
        super().__init__()
        if bins_per_octave <= 0:
            raise ValueError("harmonic input needs bins_per_octave of the frequency axis")
        shift = torch.tensor([bins_per_octave * math.log2(m) for m in multiples])
        pos = torch.arange(n_bins)[None, :] + shift[:, None]  # (H, F) fractional source bin
        lo = pos.floor().clamp(0, max(n_bins - 2, 0))
        self.register_buffer("idx", lo.long(), persistent=False)
        self.register_buffer("frac", (pos - lo).float(), persistent=False)
        self.register_buffer("valid", (pos >= 0) & (pos <= n_bins - 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x[..., self.idx], x[..., self.idx + 1]  # (B, T, H, F)
        y = a + self.frac * (b - a)
        floor = x.amin(-1, keepdim=True).unsqueeze(-1)
        return torch.where(self.valid, y, floor).permute(0, 2, 1, 3)


class _CausalBlock(nn.Module):
    def __init__(
        self, cin: int, cout: int, kt: int, kf: int, dil: int, dil_f: int = 1, ctx: bool = False
    ):
        super().__init__()
        self.pad_t = (kt - 1) * dil
        self.pad_f = (kf // 2) * dil_f
        self.conv = nn.Conv2d(cin, cout, (kt, kf), dilation=(dil, dil_f))
        self.norm = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        # Frequency context: mean + max of the block's features over all bins, fed back to every
        # bin. Starts at zero so the block begins as the plain convolution.
        self.ctx = nn.Conv1d(2 * cout, cout, 1) if ctx else None
        if self.ctx is not None:
            nn.init.zeros_(self.ctx.weight)
            nn.init.zeros_(self.ctx.bias)

    def _body(self, padded: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """padded: x with ``pad_t`` earlier frames in front; returns one output per frame of x."""
        y = F.gelu(self.norm(self.conv(F.pad(padded, (self.pad_f, self.pad_f, 0, 0)))))
        if self.ctx is not None:
            pooled = torch.cat([y.mean(-1), y.amax(-1)], dim=1)  # (B, 2C, T)
            y = y + self.ctx(pooled).unsqueeze(-1)
        return y + self.skip(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T, F)
        return self._body(F.pad(x, (0, 0, self.pad_t, 0)), x)

    def step(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Streaming: ``state`` is the last ``pad_t`` input frames (zeros before the stream)."""
        padded = torch.cat([state, x], dim=2)
        return self._body(padded, x), padded[:, :, padded.shape[2] - self.pad_t :]


class WhistleNet(nn.Module):
    def __init__(self, cfg: WhistleNetConfig):
        super().__init__()
        self.cfg = cfg
        c, k = cfg.channels, len(cfg.classes)
        if k < 1 or len(set(cfg.classes)) != k:
            raise ValueError("classes must be a non-empty list of distinct names")
        self.stack = (
            HarmonicStack(cfg.n_bins, cfg.bins_per_octave, cfg.harmonics) if cfg.harmonics else None
        )
        blocks, cin = [], len(cfg.harmonics) or 1
        dil_f = cfg.dilations_f or (1,) * len(cfg.dilations)
        if len(dil_f) != len(cfg.dilations):
            raise ValueError("dilations_f must be empty or match dilations in length")
        for d, df in zip(cfg.dilations, dil_f, strict=True):
            blocks.append(_CausalBlock(cin, c, cfg.kernel_t, cfg.kernel_f, d, df, cfg.freq_context))
            cin = c
        self.blocks = nn.Sequential(*blocks)
        self.salience = nn.Conv2d(c, k, 1)
        self.presence = nn.Linear(2 * c, k)

    @property
    def receptive_field(self) -> int:
        return self.cfg.receptive_field

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, F) -> presence (B, T, K), salience (B, T, K, F), aligned with the input."""
        la = self.cfg.lookahead
        if not la:
            return self.forward_causal(x)
        pres, sal = self.forward_causal(F.pad(x, (0, 0, 0, la)))
        return pres[:, la:], sal[:, la:]

    def _embed(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F) -> (B, C_in, T, F)
        return self.stack(x) if self.stack is not None else x.unsqueeze(1)

    def _heads(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sal = self.salience(h).permute(0, 2, 1, 3)  # (B, T, K, F)
        pooled = torch.cat([h.mean(-1), h.amax(-1)], dim=1).transpose(1, 2)  # (B, T, 2C)
        return self.presence(pooled), sal  # presence (B, T, K)

    def forward_causal(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Strictly causal pass: output ``t`` is the prediction for frame ``t - lookahead``."""
        return self._heads(self.blocks(self._embed(x)))

    def init_state(self, batch: int = 1) -> list[torch.Tensor]:
        """Streaming state for a fresh stream (the zero padding the offline pass starts with)."""
        ref = self.salience.weight
        return [
            ref.new_zeros(batch, blk.conv.in_channels, blk.pad_t, self.cfg.n_bins)
            for blk in self.blocks
        ]

    def forward_stream(
        self, x: torch.Tensor, state: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """``forward_causal`` for the next frames of a stream: (presence, salience, new state).

        Feeding a signal through in pieces gives the same outputs as one ``forward_causal``
        call, and each frame is computed once."""
        h, new_state = self._embed(x), []
        for blk, st in zip(self.blocks, state, strict=True):
            h, st = blk.step(h, st)
            new_state.append(st)
        return *self._heads(h), new_state


def save_checkpoint(
    path: Path, model: WhistleNet, norm: tuple[float, float], frontend: dict, meta: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")  # write then rename: a crash never leaves half a file
    torch.save(
        {
            "arch": asdict(model.cfg),
            "state_dict": model.state_dict(),
            "norm": {"mean": norm[0], "std": norm[1]},
            "frontend": frontend,  # {"sample_rate", "transform": {...}, "filters": [...]}
            "meta": meta,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> tuple[WhistleNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = dict(ck["arch"])
    for key in ("dilations", "dilations_f", "classes", "harmonics"):
        if key in arch:
            arch[key] = tuple(arch[key])
    model = WhistleNet(WhistleNetConfig(**arch))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
