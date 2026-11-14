"""Lightweight learned whistle detector (requires the ``ml`` extra: torch).

Input:  (B, T, F) filtered spectrogram frames (normalised with stats stored in the checkpoint).
Output: presence logits (B, T)    - "is someone whistling in this frame?"
        salience logits (B, T, F) - "which frequency bin is the whistle in?" (pitch)

Convolutions are *causal in time* (only past frames are used), so running the model on a
stream with ``receptive_field - 1`` frames of history gives exactly the offline result. This is
what lets the live detector and the training/eval code share one model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class WhistleNetConfig:
    n_bins: int = 128
    channels: int = 16
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    kernel_t: int = 3
    kernel_f: int = 3


class _CausalBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kt: int, kf: int, dil: int):
        super().__init__()
        self.pad_t = (kt - 1) * dil
        self.pad_f = kf // 2
        self.conv = nn.Conv2d(cin, cout, (kt, kf), dilation=(dil, 1))
        self.norm = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T, F)
        y = F.pad(x, (self.pad_f, self.pad_f, self.pad_t, 0))
        return F.gelu(self.norm(self.conv(y))) + self.skip(x)


class WhistleNet(nn.Module):
    def __init__(self, cfg: WhistleNetConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        blocks, cin = [], 1
        for d in cfg.dilations:
            blocks.append(_CausalBlock(cin, c, cfg.kernel_t, cfg.kernel_f, d))
            cin = c
        self.blocks = nn.Sequential(*blocks)
        self.salience = nn.Conv2d(c, 1, 1)
        self.presence = nn.Linear(2 * c, 1)

    @property
    def receptive_field(self) -> int:
        return 1 + sum((self.cfg.kernel_t - 1) * d for d in self.cfg.dilations)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(x.unsqueeze(1))  # (B, C, T, F)
        sal = self.salience(h).squeeze(1)  # (B, T, F)
        pooled = torch.cat([h.mean(-1), h.amax(-1)], dim=1).transpose(1, 2)  # (B, T, 2C)
        pres = self.presence(pooled).squeeze(-1)  # (B, T)
        return pres, sal


def save_checkpoint(
    path: Path, model: WhistleNet, norm: tuple[float, float], frontend: dict, meta: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": asdict(model.cfg),
            "state_dict": model.state_dict(),
            "norm": {"mean": norm[0], "std": norm[1]},
            "frontend": frontend,  # {"sample_rate", "transform": {...}, "filters": [...]}
            "meta": meta,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[WhistleNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = dict(ck["arch"])
    arch["dilations"] = tuple(arch["dilations"])
    model = WhistleNet(WhistleNetConfig(**arch))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
