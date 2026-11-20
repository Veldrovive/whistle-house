"""Shared building blocks for whistle detectors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from whistle_house.types import ContourPoint, FrequencyAxis, WhistleEvent


def semitones(a_hz: float, b_hz: float) -> float:
    return 12.0 * float(np.log2(max(a_hz, 1e-6) / max(b_hz, 1e-6)))


def refine_peak(values: np.ndarray, idx: int) -> float:
    """Parabolic interpolation of a spectral peak -> fractional bin index."""
    if idx <= 0 or idx >= len(values) - 1:
        return float(idx)
    a, b, c = values[idx - 1], values[idx], values[idx + 1]
    denom = a - 2 * b + c
    if denom == 0:
        return float(idx)
    return float(idx + 0.5 * (a - c) / denom)


def drop_harmonics(
    peaks: list[tuple[float, float]], tolerance_semitones: float = 0.5, max_harmonic: int = 5
) -> list[bool]:
    """Which spectral peaks of one frame are fundamentals (True) rather than harmonics (False).

    ``peaks`` is ``(hz, level)`` for every prominent peak in the frame. A peak is a harmonic
    when another peak sits at its frequency divided by 2..``max_harmonic`` (within the
    tolerance) and is at least as loud: a whistle's overtones are far weaker than its
    fundamental, so the quieter upper peak is the overtone. A louder upper peak is kept (it is
    a sound of its own, not an overtone of the quiet one below).
    """
    keep = []
    for hz, level in peaks:
        harmonic = any(
            lo_level >= level
            and any(
                abs(semitones(hz, k * lo_hz)) <= tolerance_semitones
                for k in range(2, max_harmonic + 1)
            )
            for lo_hz, lo_level in peaks
            if lo_hz < hz
        )
        keep.append(not harmonic)
    return keep


@dataclass
class _Track:
    points: list[ContourPoint] = field(default_factory=list)
    missed: int = 0

    @property
    def last_hz(self) -> float:
        return self.points[-1].hz


class ContourTracker:
    """Links per-frame pitch candidates into continuous contours (whistle notes).

    Each frame supplies a list of (hz, strength) candidates. Candidates are matched 1-to-1 to
    live tracks (closest pitch first, within ``max_jump_semitones``); unmatched candidates start
    new tracks; tracks unmatched for more than ``max_gap_frames`` end and become events if they
    lasted at least ``min_duration_s``.
    """

    def __init__(
        self,
        max_jump_semitones: float = 1.0,
        max_gap_frames: int = 3,
        min_duration_s: float = 0.08,
        min_points_active: int = 3,
        kind: str = "whistle",
    ):
        self.kind = kind  # stamped on the events this tracker produces
        self.max_jump = max_jump_semitones
        self.max_gap = max_gap_frames
        self.min_duration = min_duration_s
        self.min_points_active = min_points_active
        self.reset()

    def reset(self) -> None:
        self.tracks: list[_Track] = []

    def _finish(self, tr: _Track) -> WhistleEvent | None:
        if not tr.points or tr.points[-1].t - tr.points[0].t < self.min_duration:
            return None
        strengths = [p.strength for p in tr.points]
        return WhistleEvent(
            start=tr.points[0].t,
            end=tr.points[-1].t,
            contour=list(tr.points),
            confidence=float(np.mean(strengths)),
            kind=self.kind,
        )

    def step(self, t: float, candidates: list[tuple[float, float]]) -> list[WhistleEvent]:
        pairs = []
        for ci, (hz, _) in enumerate(candidates):
            for ti, tr in enumerate(self.tracks):
                d = abs(semitones(hz, tr.last_hz))
                if d <= self.max_jump:
                    pairs.append((d, ci, ti))
        pairs.sort()
        used_c, used_t = set(), set()
        for _, ci, ti in pairs:
            if ci in used_c or ti in used_t:
                continue
            used_c.add(ci)
            used_t.add(ti)
            hz, s = candidates[ci]
            self.tracks[ti].points.append(ContourPoint(t, hz, s))
            self.tracks[ti].missed = 0

        finished: list[WhistleEvent] = []
        alive: list[_Track] = []
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.missed += 1
                if tr.missed > self.max_gap:
                    ev = self._finish(tr)
                    if ev:
                        finished.append(ev)
                    continue
            alive.append(tr)
        for ci, (hz, s) in enumerate(candidates):
            if ci not in used_c:
                alive.append(_Track([ContourPoint(t, hz, s)]))
        self.tracks = alive
        return finished

    def flush(self) -> list[WhistleEvent]:
        out = [ev for tr in self.tracks if (ev := self._finish(tr))]
        self.tracks = []
        return out

    def active(self) -> list[WhistleEvent]:
        return [
            WhistleEvent(tr.points[0].t, tr.points[-1].t, list(tr.points), kind=self.kind)
            for tr in self.tracks
            if len(tr.points) >= self.min_points_active
        ]


def salience_candidates(
    frame: np.ndarray, axis: FrequencyAxis, threshold: float, max_candidates: int
) -> list[tuple[float, float]]:
    """Local maxima of a per-bin salience frame above ``threshold`` -> (hz, value) list."""
    if frame.size < 3:
        return []
    interior = (frame[1:-1] > frame[:-2]) & (frame[1:-1] >= frame[2:]) & (frame[1:-1] >= threshold)
    idx = np.flatnonzero(interior) + 1
    idx = idx[np.argsort(frame[idx])[::-1][:max_candidates]]
    return [(axis.bin_to_hz(refine_peak(frame, int(i))), float(frame[i])) for i in idx]
