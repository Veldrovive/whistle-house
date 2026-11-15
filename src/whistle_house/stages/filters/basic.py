"""Simple frequency-space filters. The pipeline applies a *list* of filters in order."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from whistle_house.stages.base import Filter, StageConfig, StageContext, register
from whistle_house.types import FrameBlock


@register
class Identity(Filter):
    """Pass frames through unchanged."""

    name = "identity"

    def process(self, block: FrameBlock) -> FrameBlock:
        return block


@register
class NoiseFloor(Filter):
    """Subtract an adaptive per-bin noise floor (fast to fall, slow to rise).

    Stationary hum/fans/fridges get removed; short tonal events (whistles) stick out.
    """

    name = "noise_floor"
    relative_output = True

    class Config(StageConfig):
        rise_rate: float = Field(
            0.01, description="Per-frame fraction the floor rises toward the signal"
        )
        fall_rate: float = Field(
            0.5, description="Per-frame fraction the floor falls toward the signal"
        )
        offset_db: float = Field(3.0, description="Extra margin subtracted above the floor")
        clip_zero: bool = Field(True, description="Clip negative results to 0")

    def __init__(self, config: Config, ctx: StageContext):
        super().__init__(config, ctx)
        self.reset()

    def reset(self) -> None:
        self._floor: np.ndarray | None = None

    def process(self, block: FrameBlock) -> FrameBlock:
        c = self.config
        out = np.empty_like(block.values)
        for i, frame in enumerate(block.values):
            if self._floor is None:
                self._floor = frame.copy()
            diff = frame - self._floor
            self._floor += np.where(diff > 0, c.rise_rate, c.fall_rate) * diff
            out[i] = frame - self._floor - c.offset_db
        if c.clip_zero:
            np.maximum(out, 0, out=out)
        return FrameBlock(out, block.times, block.axis)


@register
class FrameMedian(Filter):
    """Subtract each frame's median across bins (removes broadband loudness changes)."""

    name = "frame_median"
    relative_output = True

    class Config(StageConfig):
        clip_zero: bool = Field(True, description="Clip negative results to 0")

    def process(self, block: FrameBlock) -> FrameBlock:
        if block.n_frames == 0:
            return block
        out = block.values - np.median(block.values, axis=1, keepdims=True)
        if self.config.clip_zero:
            np.maximum(out, 0, out=out)
        return FrameBlock(out.astype(np.float32), block.times, block.axis)


@register
class BandReject(Filter):
    """Replace bins inside the given frequency bands with the frame minimum."""

    name = "band_reject"

    class Config(StageConfig):
        bands: list[tuple[float, float]] = Field(
            default_factory=list, description="List of [low_hz, high_hz] bands to suppress"
        )

    def __init__(self, config: Config, ctx: StageContext):
        super().__init__(config, ctx)
        f = ctx.axis.freqs if ctx.axis is not None else np.zeros(0)
        mask = np.zeros(f.shape, bool)
        for lo, hi in config.bands:
            mask |= (f >= lo) & (f <= hi)
        self._mask = mask

    def process(self, block: FrameBlock) -> FrameBlock:
        if not self._mask.any() or block.n_frames == 0:
            return block
        out = block.values.copy()
        out[:, self._mask] = out.min(axis=1, keepdims=True)
        return FrameBlock(out, block.times, block.axis)
