"""Learned whistle detector: WhistleNet presence/pitch + contour tracking.

Train a checkpoint with ``whistle-house train-detector`` (or the manager's Models page); it is
saved under ``<data_dir>/models/<name>.pt``. The checkpoint records the front-end (transform +
filters) it was trained on; pipelines using it must use the same front-end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import Field

from whistle_house.stages.base import StageConfig, StageContext, WhistleDetector, register
from whistle_house.stages.detectors.common import ContourTracker, refine_peak
from whistle_house.types import DetectorOutput, FrameBlock, WhistleEvent


@register
class LearnedDetector(WhistleDetector):
    """WhistleNet (small causal CNN) presence + pitch, linked into contours."""

    name = "learned"

    class Config(StageConfig):
        checkpoint: str = Field(
            "whistlenet.pt", description="Checkpoint file in <data_dir>/models (or absolute path)"
        )
        on_threshold: float = Field(0.6, description="Presence probability to start a note")
        off_threshold: float = Field(0.4, description="Presence probability to continue a note")
        max_jump_semitones: float = Field(1.5, description="Max pitch change between frames")
        max_gap_frames: int = Field(2, description="Frames a note may vanish before it ends")
        min_duration_s: float = Field(0.08, description="Shortest note reported")

    def __init__(self, config: Config, ctx: StageContext):
        super().__init__(config, ctx)
        import torch  # deferred: torch is an optional extra

        from whistle_house.ml.model import load_checkpoint

        path = Path(config.checkpoint)
        if not path.is_absolute():
            path = (ctx.data_dir or Path(".")) / "models" / path
        if not path.exists():
            raise FileNotFoundError(f"learned detector checkpoint not found: {path}")
        self._torch = torch
        self.model, ck = load_checkpoint(path)
        self.mean, self.std = ck["norm"]["mean"], ck["norm"]["std"]
        n_bins = ck["arch"]["n_bins"]
        if ctx.axis is not None and ctx.axis.n_bins != n_bins:
            raise ValueError(
                f"checkpoint expects {n_bins} bins but the front-end produces {ctx.axis.n_bins}; "
                f"use the front-end it was trained with: {ck['frontend']}"
            )
        self.history_len = self.model.receptive_field - 1
        self.tracker = ContourTracker(
            config.max_jump_semitones, config.max_gap_frames, config.min_duration_s
        )
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self._history = np.zeros((0, self.ctx.axis.n_bins if self.ctx.axis else 0), np.float32)

    def infer(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(T, F) -> presence prob (T,), salience prob (T, F). Stateless, for offline use."""
        x = self._torch.from_numpy(((values - self.mean) / self.std).astype(np.float32))[None]
        with self._torch.no_grad():
            pres, sal = self.model(x)
        return self._torch.sigmoid(pres)[0].numpy(), self._torch.sigmoid(sal)[0].numpy()

    def process(self, block: FrameBlock) -> DetectorOutput:
        if block.n_frames == 0:
            return DetectorOutput(frame_scores=np.zeros(0, np.float32))
        ctx_vals = (
            np.concatenate([self._history, block.values]) if self._history.size else block.values
        )
        pres, sal = self.infer(ctx_vals)
        pres, sal = pres[-block.n_frames :], sal[-block.n_frames :]
        self._history = ctx_vals[-self.history_len :] if self.history_len else ctx_vals[:0]

        c = self.config
        events: list[WhistleEvent] = []
        for t, p, s in zip(block.times, pres, sal, strict=True):
            thr = c.off_threshold if self.tracker.tracks else c.on_threshold
            cands = []
            if p >= thr:
                b = int(np.argmax(s))
                cands = [(block.axis.bin_to_hz(refine_peak(s, b)), float(p))]
            events.extend(self.tracker.step(float(t), cands))
        return DetectorOutput(
            events=events, frame_scores=pres.astype(np.float32), active=self.tracker.active()
        )

    def flush(self) -> list[WhistleEvent]:
        return self.tracker.flush()
