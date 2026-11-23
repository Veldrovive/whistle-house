"""Pipeline = transform -> [filters...] -> detector -> recognizer, built from a JSON spec.

A ``PipelineSpec`` names one registered stage per slot (plus any number of filters) with its
params. Specs are saved in ``<data_dir>/pipelines/<name>.json`` and edited in the manager, so
multiple pipelines can be compared on the same recordings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from whistle_house.stages.base import (
    Filter,
    StageContext,
    StageKind,
    Transform,
    TuneRecognizer,
    WhistleDetector,
    build_stage,
)
from whistle_house.tunes import TuneLibrary
from whistle_house.types import DetectorOutput, FrameBlock, FrequencyAxis, TuneMatch, WhistleEvent


class StageSpec(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)


class PipelineSpec(BaseModel):
    name: str = "default"
    description: str = ""
    sample_rate: int = 44100
    chunk_size: int = 1024
    transform: StageSpec = StageSpec(name="stft_mel")
    filters: list[StageSpec] = Field(default_factory=lambda: [StageSpec(name="noise_floor")])
    detector: StageSpec = StageSpec(name="peak_tracker")
    recognizer: StageSpec = StageSpec(name="note_template")

    def frontend_dict(self) -> dict:
        """The part of the spec that determines detector input features."""
        return {
            "sample_rate": self.sample_rate,
            "transform": self.transform.model_dump(),
            "filters": [f.model_dump() for f in self.filters],
        }


# --------------------------------------------------------------------------- runtime


class Frontend:
    """Transform + filter chain. Used by pipelines and by the training data generator."""

    def __init__(
        self,
        sample_rate: int,
        transform: StageSpec,
        filters: list[StageSpec],
        data_dir: Path | None = None,
    ):
        self.sample_rate = sample_rate
        self.transform: Transform = build_stage(  # type: ignore[assignment]
            StageKind.transform,
            transform.name,
            transform.params,
            StageContext(sample_rate, data_dir=data_dir),
        )
        self.filters: list[Filter] = []
        axis = self.transform.axis
        for i, f in enumerate(filters):
            # filters that depend on measured data (spectral_gate) check it was measured on
            # the same upstream chain
            upstream = {
                "sample_rate": sample_rate,
                "transform": transform.model_dump(),
                "filters": [x.model_dump() for x in filters[:i]],
            }
            flt: Filter = build_stage(  # type: ignore[assignment]
                StageKind.filter,
                f.name,
                f.params,
                StageContext(
                    sample_rate, axis=axis, data_dir=data_dir, extra={"upstream": upstream}
                ),
            )
            axis = flt.output_axis(axis)
            self.filters.append(flt)
        self.axis: FrequencyAxis = axis

    @staticmethod
    def from_dict(d: dict, data_dir: Path | None = None) -> Frontend:
        return Frontend(
            d["sample_rate"],
            StageSpec.model_validate(d["transform"]),
            [StageSpec.model_validate(f) for f in d["filters"]],
            data_dir,
        )

    def reset(self) -> None:
        self.transform.reset()
        for f in self.filters:
            f.reset()

    def push(self, samples: np.ndarray) -> tuple[FrameBlock, FrameBlock]:
        """-> (raw transform output, filtered output)."""
        raw = self.transform.push(samples)
        out = raw
        for f in self.filters:
            out = f.process(out)
        return raw, out

    def telemetry(self) -> list[dict]:
        """Filter telemetry for the last ``push`` ({index, name, ...} per filter that has some)."""
        out = []
        for i, f in enumerate(self.filters):
            t = f.telemetry()
            if t is not None:
                out.append({"index": i, "name": f.name, **t})
        return out

    def telemetry_info(self) -> list[dict]:
        """Static drawing info of the filters that report telemetry ({index, name, ...})."""
        out = []
        for i, f in enumerate(self.filters):
            t = f.telemetry_info()
            if t is not None:
                out.append({"index": i, "name": f.name, **t})
        return out

    def process_signal(self, samples: np.ndarray) -> tuple[FrameBlock, FrameBlock]:
        """Whole signal from a fresh state (identical to streaming it chunk by chunk)."""
        self.reset()
        return self.push(samples)


@dataclass
class StepResult:
    raw: FrameBlock
    filtered: FrameBlock
    detection: DetectorOutput
    matches: list[TuneMatch]
    # Frontend.telemetry(): per-filter state for these frames (e.g. adaptive_gate's posterior)
    filter_state: list[dict] = field(default_factory=list)


@dataclass
class AnalysisResult:
    raw: FrameBlock
    filtered: FrameBlock
    scores: np.ndarray
    events: list[WhistleEvent] = field(default_factory=list)
    matches: list[TuneMatch] = field(default_factory=list)
    kind_scores: dict[str, np.ndarray] = field(default_factory=dict)  # per-kind frame scores


class Pipeline:
    def __init__(
        self, spec: PipelineSpec, tunes: TuneLibrary | None = None, data_dir: Path | None = None
    ):
        self.spec = spec
        self.frontend = Frontend(spec.sample_rate, spec.transform, spec.filters, data_dir)
        ctx = StageContext(
            spec.sample_rate, axis=self.frontend.axis, tunes=tunes, data_dir=data_dir
        )
        self.detector: WhistleDetector = build_stage(  # type: ignore[assignment]
            StageKind.detector, spec.detector.name, spec.detector.params, ctx
        )
        self.recognizer: TuneRecognizer = build_stage(  # type: ignore[assignment]
            StageKind.recognizer, spec.recognizer.name, spec.recognizer.params, ctx
        )
        self._samples_seen = 0

    @property
    def axis(self) -> FrequencyAxis:
        return self.frontend.axis

    @property
    def now(self) -> float:
        return self._samples_seen / self.spec.sample_rate

    def reset(self) -> None:
        self.frontend.reset()
        self.detector.reset()
        self.recognizer.reset()
        self._samples_seen = 0

    def step(self, samples: np.ndarray) -> StepResult:
        self._samples_seen += len(samples)
        raw, filtered = self.frontend.push(samples)
        det = self.detector.process(filtered)
        matches = self.recognizer.process(det.events, self.now, det.active)
        return StepResult(raw, filtered, det, matches, self.frontend.telemetry())

    def flush(self) -> tuple[list[WhistleEvent], list[TuneMatch]]:
        events = self.detector.flush()
        matches = self.recognizer.process(events, self.now, [])
        return events, matches + self.recognizer.flush(self.now)

    def analyze(self, samples: np.ndarray) -> AnalysisResult:
        """Run a whole recording through the streaming path from a fresh state."""
        self.reset()
        raws, filts, scores, events, matches = [], [], [], [], []
        kind_scores: dict[str, list[np.ndarray]] = {}
        cs = self.spec.chunk_size
        for i in range(0, len(samples), cs):
            r = self.step(samples[i : i + cs])
            raws.append(r.raw)
            filts.append(r.filtered)
            s = r.detection.frame_scores
            scores.append(s if s is not None else np.zeros(r.filtered.n_frames, np.float32))
            for kind, ks in (r.detection.kind_scores or {}).items():
                kind_scores.setdefault(kind, []).append(ks)
            events += r.detection.events
            matches += r.matches
        ev, m = self.flush()
        events += ev
        matches += m
        return AnalysisResult(
            FrameBlock.concat(raws, self.frontend.transform.axis),
            FrameBlock.concat(filts, self.axis),
            np.concatenate(scores) if scores else np.zeros(0, np.float32),
            events,
            matches,
            {k: np.concatenate(v) for k, v in kind_scores.items()},
        )


# --------------------------------------------------------------------------- storage

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class PipelineStore:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        if not (self.dir / "default.json").exists():
            self.put(PipelineSpec())  # always keep a known-good baseline around

    def _path(self, name: str) -> Path:
        if not _NAME_RE.match(name):
            raise ValueError(f"invalid pipeline name {name!r}")
        return self.dir / f"{name}.json"

    def list(self) -> list[PipelineSpec]:
        return [
            PipelineSpec.model_validate_json(p.read_text()) for p in sorted(self.dir.glob("*.json"))
        ]

    def get(self, name: str) -> PipelineSpec:
        return PipelineSpec.model_validate_json(self._path(name).read_text())

    def put(self, spec: PipelineSpec) -> PipelineSpec:
        self._path(spec.name).write_text(json.dumps(spec.model_dump(mode="json"), indent=2))
        return spec

    def delete(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)
