"""Stage interfaces + registry.

The pipeline has four swappable slots (``StageKind``). Each implementation:

* subclasses the matching base class below,
* declares a pydantic ``Config`` model (its tunable parameters - the manager renders a form
  from the model's JSON schema, so every field needs a sensible default and ideally a
  ``description``),
* is registered with ``@register`` under a unique ``name``.

All stages are *streaming*: they receive data incrementally and keep whatever state they need
between calls. ``reset()`` must return them to the freshly-constructed state. Offline analysis
simply feeds a whole recording through the same streaming code path in chunks, so live and
offline behaviour can never drift apart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from whistle_house.types import DetectorOutput, FrameBlock, FrequencyAxis, TuneMatch, WhistleEvent

if TYPE_CHECKING:
    from whistle_house.tunes import TuneLibrary


class StageKind(StrEnum):
    transform = "transform"
    filter = "filter"
    detector = "detector"
    recognizer = "recognizer"


class StageConfig(BaseModel):
    """Base for every stage's Config. Unknown keys are rejected so typos surface early."""

    model_config = ConfigDict(extra="forbid")


@dataclass
class StageContext:
    """Things stages may need at construction time beyond their own config."""

    sample_rate: int
    axis: FrequencyAxis | None = None  # set for filter/detector (output axis of the previous stage)
    tunes: TuneLibrary | None = None  # set for recognizers
    data_dir: Path | None = None  # root of the data directory (models, recordings, ...)
    extra: dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    kind: ClassVar[StageKind]
    name: ClassVar[str]
    description: ClassVar[str] = ""
    Config: ClassVar[type[StageConfig]] = StageConfig

    def __init__(self, config: StageConfig, ctx: StageContext):
        self.config = config
        self.ctx = ctx

    def reset(self) -> None:
        """Drop all streaming state."""


class Transform(Stage):
    """Raw mono float32 audio -> time/frequency frames."""

    kind = StageKind.transform

    @property
    @abstractmethod
    def axis(self) -> FrequencyAxis: ...

    @abstractmethod
    def push(self, samples: np.ndarray) -> FrameBlock:
        """Feed the next samples (any length). Returns frames completed by these samples.

        Frame times are on the stream clock: sample 0 of the stream is t=0.
        """


class Filter(Stage):
    """FrameBlock -> FrameBlock (denoising, whitening, band rejection, ...).

    Must preserve frame count and times. May change the axis (e.g. cropping bins); if so it
    must override ``output_axis``.
    """

    kind = StageKind.filter
    # False: output is in the input's units (e.g. absolute dB; band_reject, identity).
    # True: output is "dB above a floor", 0 = floor (noise_floor, spectral_gate, ...).
    # Only used to turn filter output back into audio (audio/render.py); no effect on processing.
    relative_output: ClassVar[bool] = False

    def output_axis(self, axis: FrequencyAxis) -> FrequencyAxis:
        return axis

    @abstractmethod
    def process(self, block: FrameBlock) -> FrameBlock: ...


class WhistleDetector(Stage):
    """FrameBlock -> whistle events (+ per-frame scores for visualisation)."""

    kind = StageKind.detector

    @abstractmethod
    def process(self, block: FrameBlock) -> DetectorOutput: ...

    def flush(self) -> list[WhistleEvent]:
        """End of stream: return any events still in progress (if long enough)."""
        return []


class TuneRecognizer(Stage):
    """Whistle events -> tune matches."""

    kind = StageKind.recognizer

    @abstractmethod
    def process(self, events: list[WhistleEvent], now: float) -> list[TuneMatch]:
        """Called once per block with the events that finished in it (possibly empty)."""


# --------------------------------------------------------------------------- registry

_REGISTRY: dict[StageKind, dict[str, type[Stage]]] = {k: {} for k in StageKind}


def register(cls: type[Stage]) -> type[Stage]:
    """Class decorator: make a stage implementation available to pipelines + the manager."""
    kind = cls.kind
    if cls.name in _REGISTRY[kind]:
        raise ValueError(f"duplicate {kind} stage name {cls.name!r}")
    _REGISTRY[kind][cls.name] = cls
    return cls


def get_stage_class(kind: StageKind | str, name: str) -> type[Stage]:
    _ensure_loaded()
    try:
        return _REGISTRY[StageKind(kind)][name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY[StageKind(kind)]))
        raise KeyError(f"unknown {kind} stage {name!r}; known: {known}") from None


def build_stage(kind: StageKind | str, name: str, params: dict, ctx: StageContext) -> Stage:
    cls = get_stage_class(kind, name)
    return cls(cls.Config.model_validate(params or {}), ctx)


def list_stages() -> dict[str, list[dict]]:
    """Registry description consumed by the manager (``GET /api/stages``)."""
    _ensure_loaded()
    return {
        kind.value: [
            {
                "name": cls.name,
                "description": cls.description or (cls.__doc__ or "").strip().split("\n")[0],
                "schema": cls.Config.model_json_schema(),
                "defaults": cls.Config().model_dump(mode="json"),
            }
            for cls in sorted(impls.values(), key=lambda c: c.name)
        ]
        for kind, impls in _REGISTRY.items()
    }


_loaded = False


def _ensure_loaded() -> None:
    """Import all built-in stage modules so their @register decorators run."""
    global _loaded
    if not _loaded:
        _loaded = True
        import whistle_house.stages.detectors
        import whistle_house.stages.filters
        import whistle_house.stages.recognizers
        import whistle_house.stages.transforms  # noqa: F401
