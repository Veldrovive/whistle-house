"""Learned detector: WhistleNet presence/pitch per kind (whistle, hum) + contour tracking.

Train a checkpoint with ``whistle-house train-detector`` (or the manager's Models page); it is
saved under ``<data_dir>/models/<name>.pt``. The checkpoint records the front-end (transform +
filters) it was trained on; pipelines using it must use the same front-end.

A checkpoint has one output per kind of sound it was trained on (``whistle``; the ``harmonic``
arch adds ``hum``). Each kind gets its own contour tracker, so a hum never continues a whistle,
and its events carry ``kind``.

Frames go through the model once (``WhistleNet.forward_stream`` keeps the few past frames each
layer needs), so the cost per chunk does not grow with the model's receptive field.
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
        kinds: list[str] = Field(
            default_factory=list,
            description="Kinds of sound to report, of those the checkpoint detects (whistle, "
            "hum). Empty = all of them",
        )

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
        self.classes: tuple[str, ...] = tuple(self.model.cfg.classes)
        unknown = [k for k in config.kinds if k not in self.classes]
        if unknown:
            raise ValueError(
                f"checkpoint {path.name} detects {list(self.classes)}, not {unknown}; "
                "fix the detector's `kinds`"
            )
        # (class index, kind) reported by this detector
        self.kinds = [(i, k) for i, k in enumerate(self.classes) if k in (config.kinds or [k])]
        self.lookahead = self.model.cfg.lookahead
        self.trackers = {
            k: ContourTracker(
                config.max_jump_semitones, config.max_gap_frames, config.min_duration_s, kind=k
            )
            for _, k in self.kinds
        }
        self.reset()

    def reset(self) -> None:
        for tr in self.trackers.values():
            tr.reset()
        self._state = self.model.init_state()
        self._skip = self.lookahead  # causal outputs still to drop (they predate frame 0)
        self._times: list[float] = []  # times of frames whose prediction has not come out yet
        self._axis = self.ctx.axis

    def _norm(self, values: np.ndarray):
        return self._torch.from_numpy(((values - self.mean) / self.std).astype(np.float32))[None]

    def infer(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(T, F) -> presence prob (T, K), salience prob (T, K, F) for the model's K classes
        (``self.classes``), aligned with the input frames. Stateless (offline)."""
        with self._torch.no_grad():
            pres, sal = self.model(self._norm(values))
        return self._torch.sigmoid(pres)[0].numpy(), self._torch.sigmoid(sal)[0].numpy()

    def _advance(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Push new frames through the causal network; outputs lag by the model's lookahead."""
        with self._torch.no_grad():
            pres, sal, self._state = self.model.forward_stream(self._norm(values), self._state)
        pres = self._torch.sigmoid(pres)[0].numpy()
        sal = self._torch.sigmoid(sal)[0].numpy()
        drop = min(self._skip, len(pres))
        self._skip -= drop
        return pres[drop:], sal[drop:]

    def _track(self, pres: np.ndarray, sal: np.ndarray) -> list[WhistleEvent]:
        c = self.config
        events: list[WhistleEvent] = []
        times, self._times = self._times[: len(pres)], self._times[len(pres) :]
        for t, p, s in zip(times, pres, sal, strict=True):
            for i, kind in self.kinds:
                tracker = self.trackers[kind]
                thr = c.off_threshold if tracker.tracks else c.on_threshold
                cands = []
                if p[i] >= thr:
                    b = int(np.argmax(s[i]))
                    cands = [(self._axis.bin_to_hz(refine_peak(s[i], b)), float(p[i]))]
                events.extend(tracker.step(float(t), cands))
        return events

    def process(self, block: FrameBlock) -> DetectorOutput:
        if block.n_frames == 0:
            return DetectorOutput(frame_scores=np.zeros(0, np.float32))
        self._axis = block.axis
        self._times.extend(float(t) for t in block.times)
        pres, sal = self._advance(block.values)
        events = self._track(pres, sal)
        # With lookahead the scores for this block's last frames are not known yet: scores are
        # reported `lookahead` frames late (zeros until the first ones arrive).
        kind_scores = {k: np.zeros(block.n_frames, np.float32) for _, k in self.kinds}
        if len(pres):
            for i, k in self.kinds:
                kind_scores[k][-len(pres) :] = pres[:, i]
        return DetectorOutput(
            events=events,
            frame_scores=np.max(list(kind_scores.values()), axis=0),
            active=[e for tr in self.trackers.values() for e in tr.active()],
            kind_scores=kind_scores,
        )

    def flush(self) -> list[WhistleEvent]:
        events: list[WhistleEvent] = []
        if self._times:  # lookahead: the last frames' predictions need "future" frames; pad
            n_bins = self.model.cfg.n_bins
            pad = np.full((len(self._times), n_bins), self.mean, np.float32)
            events = self._track(*self._advance(pad))
        return events + [e for tr in self.trackers.values() for e in tr.flush()]
