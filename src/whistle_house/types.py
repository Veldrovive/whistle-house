"""Data types passed between pipeline stages.

Every stage boundary in the pipeline uses one of these types, so any implementation of a
stage can be swapped for another as long as it consumes/produces the same type:

    audio samples --Transform--> FrameBlock --Filter--> FrameBlock
        --WhistleDetector--> DetectorOutput (WhistleEvents) --TuneRecognizer--> TuneMatches
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FrequencyAxis:
    """Describes the frequency bins of a FrameBlock (bin index -> centre frequency in Hz)."""

    freqs: np.ndarray  # (n_bins,) centre frequency of each bin in Hz, ascending
    scale: str = (
        "mel"  # "mel" | "linear" | "log" | ... informational, used by the manager for labels
    )

    @property
    def n_bins(self) -> int:
        return int(self.freqs.shape[0])

    def bin_to_hz(self, b: float) -> float:
        """Fractional bin index -> Hz (linear interpolation between bin centres)."""
        return float(np.interp(b, np.arange(self.n_bins), self.freqs))

    def hz_to_bin(self, hz: float) -> float:
        return float(np.interp(hz, self.freqs, np.arange(self.n_bins)))

    def to_json(self) -> dict:
        return {"freqs": self.freqs.tolist(), "scale": self.scale}


@dataclass
class FrameBlock:
    """A block of consecutive time-frequency frames.

    values: (n_frames, n_bins) float32. Units are stage dependent (dB for the stock transform).
    times:  (n_frames,) float64 seconds, centre time of each frame on the stream clock.
    """

    values: np.ndarray
    times: np.ndarray
    axis: FrequencyAxis

    @property
    def n_frames(self) -> int:
        return int(self.values.shape[0])

    @staticmethod
    def empty(axis: FrequencyAxis) -> FrameBlock:
        return FrameBlock(np.zeros((0, axis.n_bins), np.float32), np.zeros((0,), np.float64), axis)

    @staticmethod
    def concat(blocks: list[FrameBlock], axis: FrequencyAxis) -> FrameBlock:
        if not blocks:
            return FrameBlock.empty(axis)
        return FrameBlock(
            np.concatenate([b.values for b in blocks], axis=0),
            np.concatenate([b.times for b in blocks], axis=0),
            axis,
        )


@dataclass
class ContourPoint:
    t: float  # seconds
    hz: float
    strength: float  # detector-specific confidence/amplitude for this point


@dataclass
class WhistleEvent:
    """A single continuous note (may glide in pitch). ``kind`` says how it was made:
    ``whistle`` or ``hum`` (for a hum the contour is the fundamental)."""

    start: float
    end: float
    contour: list[ContourPoint]
    confidence: float = 1.0
    meta: dict = field(default_factory=dict)
    kind: str = "whistle"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def median_hz(self) -> float:
        return float(np.median([p.hz for p in self.contour]))

    def to_json(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "contour": [[p.t, p.hz, p.strength] for p in self.contour],
            "meta": self.meta,
            "kind": self.kind,
        }

    @staticmethod
    def from_json(d: dict) -> WhistleEvent:
        return WhistleEvent(
            start=d["start"],
            end=d["end"],
            confidence=d.get("confidence", 1.0),
            contour=[ContourPoint(*p) for p in d["contour"]],
            meta=d.get("meta", {}),
            kind=d.get("kind", "whistle"),
        )


@dataclass
class DetectorOutput:
    """What a WhistleDetector returns for one FrameBlock.

    events:       note events (whistled or hummed) that *finished* during this block.
    frame_scores: optional (n_frames,) per-frame detection score in [0, 1] for visualisation /
                  evaluation (e.g. model probability, or normalised peak prominence). For a
                  detector with several kinds this is the highest of them.
    active:       events still in progress (for live display only; may change later).
    kind_scores:  optional per-kind (n_frames,) scores (``{"whistle": ..., "hum": ...}``) from
                  detectors that tell kinds apart.
    """

    events: list[WhistleEvent] = field(default_factory=list)
    frame_scores: np.ndarray | None = None
    active: list[WhistleEvent] = field(default_factory=list)
    kind_scores: dict[str, np.ndarray] | None = None


@dataclass
class TuneMatch:
    tune_id: str
    tune_name: str
    time: float  # stream time at which the match was confirmed
    score: float  # recognizer-specific, higher is better
    events: list[WhistleEvent] = field(default_factory=list)
    kind: str = "whistle"  # how the tune was performed: the kind of its events

    def to_json(self) -> dict:
        return {
            "tune_id": self.tune_id,
            "tune_name": self.tune_name,
            "time": self.time,
            "score": self.score,
            "kind": self.kind,
            "events": [e.to_json() for e in self.events],
        }
