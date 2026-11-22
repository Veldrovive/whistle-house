"""Learned detector: WhistleNet presence/pitch per kind (whistle, hum) + contour tracking.

Train a checkpoint with ``whistle-house train-detector`` (or the manager's Models page); it is
saved under ``<data_dir>/models/<name>.pt``. The checkpoint records the front-end (transform +
filters) it was trained on; pipelines using it must use the same front-end.

A checkpoint has one output per kind of sound it was trained on (``whistle``; the ``harmonic``
arch adds ``hum``). Each kind gets its own contour tracker, so a hum never continues a whistle,
and its events carry ``kind``.

Frames go through the model once (``WhistleNet.forward_stream`` keeps the few past frames each
layer needs), so the cost per chunk does not grow with the model's receptive field.

Two backends run the same streaming step (``backend`` param):

* ``torch``: the checkpoint itself (``<name>.pt``) in PyTorch. The default.
* ``onnx``: its export (``<name>.onnx``, made by ``whistle-house export-onnx`` or the Models
  page) in ONNX Runtime. About twice as fast per chunk on the Pi for the ``small`` and
  ``harmonic`` models and needs no torch there. Same events and scores to float round-off.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from whistle_house.stages.base import StageConfig, StageContext, WhistleDetector, register
from whistle_house.stages.detectors.common import ContourTracker, refine_peak
from whistle_house.types import DetectorOutput, FrameBlock, WhistleEvent

log = logging.getLogger(__name__)


class _TorchBackend:
    """The checkpoint in PyTorch. Same interface as ``ml.export.OnnxStream``."""

    def __init__(self, path: Path):
        import torch  # deferred: torch is an optional extra

        from whistle_house.ml.model import load_checkpoint
        from whistle_house.threads import limit_torch

        limit_torch(torch)
        self._torch = torch
        self.model, ck = load_checkpoint(path)
        self.norm: dict = ck["norm"]
        self.frontend: dict = ck["frontend"]
        self.classes: tuple[str, ...] = tuple(self.model.cfg.classes)
        self.lookahead: int = self.model.cfg.lookahead
        self.n_bins: int = self.model.cfg.n_bins

    def init_state(self) -> list:
        return self.model.init_state()

    def stream(self, x: np.ndarray, state: list) -> tuple[np.ndarray, np.ndarray, list]:
        torch = self._torch
        with torch.no_grad():
            pres, sal, state = self.model.forward_stream(torch.from_numpy(x), state)
        return torch.sigmoid(pres)[0].numpy(), torch.sigmoid(sal)[0].numpy(), state


@register
class LearnedDetector(WhistleDetector):
    """WhistleNet (small causal CNN) presence + pitch, linked into contours."""

    name = "learned"

    class Config(StageConfig):
        checkpoint: str = Field(
            "whistlenet.pt", description="Checkpoint file in <data_dir>/models (or absolute path)"
        )
        backend: Literal["torch", "onnx"] = Field(
            "torch",
            description="What runs the model. torch: the .pt checkpoint. onnx: its export "
            "(<name>.onnx next to it; Models page: Export ONNX), about twice as fast on the Pi "
            "and needs no torch there",
        )
        threads: int = Field(
            1,
            ge=1,
            description="CPU threads for the onnx backend. More than 1 rarely pays for these "
            "small models",
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
        path = Path(config.checkpoint)
        if not path.is_absolute():
            path = (ctx.data_dir or Path(".")) / "models" / path
        if config.backend == "onnx" or path.suffix == ".onnx":
            from whistle_house.ml.export import OnnxStream

            checkpoint, path = path.with_suffix(".pt"), path.with_suffix(".onnx")
            if not path.exists():
                raise FileNotFoundError(
                    f"no ONNX export at {path}: run `whistle-house export-onnx {path.stem}` "
                    "(or Export ONNX on the Models page) where the checkpoint is, then copy "
                    "the .onnx file here"
                )
            self.backend = OnnxStream(path, config.threads)
            if self.backend.stale_against(checkpoint):
                log.warning(
                    "%s was not exported from the current %s: export it again",
                    path.name,
                    checkpoint.name,
                )
        else:
            if not path.exists():
                raise FileNotFoundError(f"learned detector checkpoint not found: {path}")
            self.backend = _TorchBackend(path)
        self.mean, self.std = self.backend.norm["mean"], self.backend.norm["std"]
        n_bins = self.backend.n_bins
        if ctx.axis is not None and ctx.axis.n_bins != n_bins:
            raise ValueError(
                f"checkpoint expects {n_bins} bins but the front-end produces {ctx.axis.n_bins}; "
                f"use the front-end it was trained with: {self.backend.frontend}"
            )
        self.classes: tuple[str, ...] = self.backend.classes
        unknown = [k for k in config.kinds if k not in self.classes]
        if unknown:
            raise ValueError(
                f"checkpoint {path.name} detects {list(self.classes)}, not {unknown}; "
                "fix the detector's `kinds`"
            )
        # (class index, kind) reported by this detector
        self.kinds = [(i, k) for i, k in enumerate(self.classes) if k in (config.kinds or [k])]
        self.lookahead = self.backend.lookahead
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
        self._state = self.backend.init_state()
        self._skip = self.lookahead  # causal outputs still to drop (they predate frame 0)
        self._times: list[float] = []  # times of frames whose prediction has not come out yet
        self._axis = self.ctx.axis

    def _norm(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)[None]

    def infer(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(T, F) -> presence prob (T, K), salience prob (T, K, F) for the model's K classes
        (``self.classes``), aligned with the input frames. Stateless (offline): one causal pass
        from a fresh state over the input plus ``lookahead`` frames of zeros (as
        ``WhistleNet.forward`` does), with the first ``lookahead`` outputs dropped."""
        la = self.lookahead
        x = np.pad(self._norm(values), ((0, 0), (0, la), (0, 0)))
        pres, sal, _ = self.backend.stream(x, self.backend.init_state())
        return pres[la:], sal[la:]

    def _advance(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Push new frames through the causal network; outputs lag by the model's lookahead."""
        pres, sal, self._state = self.backend.stream(self._norm(values), self._state)
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
            n_bins = self.backend.n_bins
            pad = np.full((len(self._times), n_bins), self.mean, np.float32)
            events = self._track(*self._advance(pad))
        return events + [e for tr in self.trackers.values() for e in tr.flush()]
