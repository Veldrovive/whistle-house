"""Simulated performances of a written tune, for testing tune recognizers without audio.

``perform`` turns a ``Score`` into the pitch curve a person might whistle or hum: in another
key, at another tempo, with notes off pitch and off time, run together, or joined by slides.
``detect_curve`` turns that curve into note events exactly as the detectors do (one pitch
candidate per frame through a ``ContourTracker``), so slides merge into one event and leaps
split events just as they do live. ``render_audio`` makes the curve audible (a sine), for
tests through a whole pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from whistle_house.stages.detectors.common import ContourTracker
from whistle_house.tunes import Score
from whistle_house.types import WhistleEvent

FRAME_S = 512 / 44100  # the stock transforms' hop


@dataclass
class Style:
    """How a tune is performed. The defaults are a clean, detached, in-tempo performance."""

    tempo_scale: float = 1.0  # 2 = twice as fast as written
    transpose: float = 0.0  # semitones above the written pitch
    intonation_sd: float = 0.0  # each note's pitch error, semitones
    drift: float = 0.0  # how far the key has moved by the end of the tune, semitones
    timing_sd: float = 0.0  # each note's length error (sd of the log of the length)
    detach: float = 0.15  # share of each note's time left silent at its end
    legato: float = 0.0  # probability that a note runs into the next with no silence
    slide_s: float = 0.0  # how long a legato change of pitch takes (0 = a jump)
    hold_rests: bool = False  # sing through written rests
    scoop: float = 0.0  # semitones below the note that each detached note starts from
    jitter_sd: float = 0.03  # frame-to-frame pitch noise, semitones
    lead_s: float = 0.3  # silence before the first note
    tail_s: float = 0.3  # silence after the last note


def perform(
    score: Score, style: Style, rng: np.random.Generator, frame_s: float = FRAME_S
) -> tuple[np.ndarray, np.ndarray]:
    """-> (frame times, pitch on the MIDI scale with NaN where silent)."""
    spb = score.seconds_per_beat / style.tempo_scale
    total = sum(n.beats for n in score.notes) * spb
    # (start, end of sound, end of slot, pitch, runs into the next note)
    notes: list[list] = []
    t = style.lead_s
    for i, note in enumerate(score.notes):
        length = note.beats * spb * float(np.exp(rng.normal(0, style.timing_sd)))
        if note.pitch is None:
            if style.hold_rests and notes:
                notes[-1][1] = notes[-1][2] = t + length
            t += length
            continue
        pitch = note.pitch + style.transpose + rng.normal(0, style.intonation_sd)
        pitch += style.drift * (t - style.lead_s) / max(total, 1e-6)
        nxt = score.notes[i + 1] if i + 1 < len(score.notes) else None
        legato = nxt is not None and nxt.pitch is not None and rng.random() < style.legato
        sound = length if legato else length * (1 - style.detach)
        notes.append([t, t + sound, t + length, float(pitch), legato])
        t += length
    times = np.arange(int(np.ceil((t + style.tail_s) / frame_s))) * frame_s + frame_s / 2
    curve = np.full(len(times), np.nan)
    for i, (start, sound_end, _, pitch, legato) in enumerate(notes):
        on = (times >= start) & (times < sound_end)
        curve[on] = pitch
        after_legato = i > 0 and notes[i - 1][4]
        if style.scoop and not after_legato:
            curve[on] -= style.scoop * np.exp(-(times[on] - start) / 0.03)
    for i, (start, sound_end, _, pitch, legato) in enumerate(notes):
        if legato and style.slide_s > 0:  # slide across the join, half in each note
            half = min(style.slide_s, sound_end - start, notes[i + 1][1] - sound_end) / 2
            ramp = (times >= sound_end - half) & (times < sound_end + half)
            mix = (times[ramp] - (sound_end - half)) / max(2 * half, 1e-6)
            curve[ramp] = pitch + mix * (notes[i + 1][3] - pitch)
    return times, curve + rng.normal(0, style.jitter_sd, len(curve))


def detect_curve(
    times: np.ndarray,
    curve: np.ndarray,
    kind: str = "whistle",
    chunk_frames: int = 8,
    max_jump_semitones: float = 1.5,
    max_gap_frames: int = 2,
    min_duration_s: float = 0.08,
) -> Iterator[tuple[float, list[WhistleEvent], list[WhistleEvent]]]:
    """Run the curve through the detectors' contour tracker, ``chunk_frames`` at a time.
    Yields (stream time, events finished in the chunk, notes still sounding), then a last
    item with the events flushed at the end of the stream."""
    tracker = ContourTracker(max_jump_semitones, max_gap_frames, min_duration_s, kind=kind)
    now = 0.0
    for i in range(0, len(times), chunk_frames):
        finished: list[WhistleEvent] = []
        for t, p in zip(times[i : i + chunk_frames], curve[i : i + chunk_frames], strict=True):
            hz = 440.0 * 2.0 ** ((p - 69.0) / 12.0)
            finished += tracker.step(float(t), [] if np.isnan(p) else [(float(hz), 1.0)])
            now = float(t)
        yield now, finished, tracker.active()
    yield now, tracker.flush(), []


def curve_events(times: np.ndarray, curve: np.ndarray, kind: str = "whistle") -> list[WhistleEvent]:
    """All note events of a curve (offline)."""
    return [ev for _, finished, _ in detect_curve(times, curve, kind) for ev in finished]


def render_audio(times: np.ndarray, curve: np.ndarray, sample_rate: int) -> np.ndarray:
    """The curve as a sine wave (float32), faded in and out at every silence."""
    n = int(np.ceil((times[-1] + (times[1] - times[0])) * sample_rate))
    t = np.arange(n) / sample_rate
    voiced = ~np.isnan(curve)
    if not voiced.any():
        return np.zeros(n, np.float32)
    pitch = np.interp(t, times[voiced], curve[voiced])
    level = np.interp(t, times, voiced.astype(float))
    smooth = np.hanning(int(0.02 * sample_rate) | 1)
    level = np.convolve(level > 0.5, smooth / smooth.sum(), mode="same")
    phase = 2 * np.pi * np.cumsum(440.0 * 2.0 ** ((pitch - 69.0) / 12.0)) / sample_rate
    return (0.4 * level * np.sin(phase)).astype(np.float32)
