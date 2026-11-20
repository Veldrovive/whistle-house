"""Heuristic baseline detector: spectral peaks linked over time.

Not expected to reject speech well - it is the baseline to beat, and is used to auto-label
pitch contours in clean whistle recordings for training the learned detector.

Every prominent peak becomes a candidate, so a whistle with visible overtones gives a second
(and third) contour at 2x / 3x its pitch. ``reject_harmonics`` drops those per frame (see
``common.drop_harmonics``). It is off by default so existing pipelines behave as before; the
auto-labeller turns it on.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field
from scipy.signal import find_peaks

from whistle_house.stages.base import StageConfig, StageContext, WhistleDetector, register
from whistle_house.stages.detectors.common import ContourTracker, drop_harmonics, refine_peak
from whistle_house.types import DetectorOutput, FrameBlock, WhistleEvent


@register
class PeakTracker(WhistleDetector):
    """Prominent spectral peaks linked into contours (heuristic baseline)."""

    name = "peak_tracker"

    class Config(StageConfig):
        min_prominence: float = Field(
            20.0, description="Peak prominence threshold (input units, dB)"
        )
        min_width_bins: float = Field(1.0, description="Minimum peak width in bins")
        max_peaks_per_frame: int = Field(3, description="Keep at most this many peaks per frame")
        max_jump_semitones: float = Field(1.0, description="Max pitch change between frames")
        max_gap_frames: int = Field(3, description="Frames a note may vanish before it ends")
        min_duration_s: float = Field(0.08, description="Shortest note reported")
        min_hz: float = Field(400.0, description="Ignore peaks below this")
        max_hz: float = Field(4000.0, description="Ignore peaks above this")
        max_strong_peaks: int = Field(
            0,
            description="Treat frames with more than this many qualifying peaks as non-tonal "
            "(speech/noise) and drop them. 0 disables.",
        )
        reject_harmonics: bool = Field(
            False,
            description="Drop a peak when a peak at least as loud sits at 1/2, 1/3, 1/4 or 1/5 "
            "of its frequency: it is an overtone of that note, not a note of its own",
        )
        harmonic_tolerance_semitones: float = Field(
            0.5, description="How far from an exact multiple a peak may be and still be an overtone"
        )

    def __init__(self, config: Config, ctx: StageContext):
        super().__init__(config, ctx)
        self.tracker = ContourTracker(
            config.max_jump_semitones, config.max_gap_frames, config.min_duration_s
        )

    def reset(self) -> None:
        self.tracker.reset()

    def process(self, block: FrameBlock) -> DetectorOutput:
        c = self.config
        axis = block.axis
        events: list[WhistleEvent] = []
        scores = np.zeros(block.n_frames, np.float32)
        for i, (t, frame) in enumerate(zip(block.times, block.values, strict=True)):
            peaks, props = find_peaks(frame, prominence=c.min_prominence, width=c.min_width_bins)
            hzs = [axis.bin_to_hz(refine_peak(frame, int(p))) for p in peaks]
            # Overtones are judged against every prominent peak, including fundamentals
            # outside [min_hz, max_hz].
            keep = (
                drop_harmonics(
                    [(hz, float(frame[p])) for hz, p in zip(hzs, peaks, strict=True)],
                    c.harmonic_tolerance_semitones,
                )
                if c.reject_harmonics
                else [True] * len(hzs)
            )
            cands = []
            for hz, prom, ok in zip(hzs, props["prominences"], keep, strict=True):
                if ok and c.min_hz <= hz <= c.max_hz:
                    cands.append((hz, float(prom)))
            if c.max_strong_peaks and len(cands) > c.max_strong_peaks:
                cands = []
            cands.sort(key=lambda x: -x[1])
            cands = cands[: c.max_peaks_per_frame]
            if cands:
                scores[i] = min(1.0, cands[0][1] / (2 * c.min_prominence))
            events.extend(self.tracker.step(float(t), cands))
        return DetectorOutput(events=events, frame_scores=scores, active=self.tracker.active())

    def flush(self) -> list[WhistleEvent]:
        return self.tracker.flush()
