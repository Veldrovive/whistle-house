"""Tune definitions (what to listen for) and which outputs they trigger.

Stored as ``<data_dir>/tunes.json``. A tune is defined in one of two ways:

* written as sheet music (``Tune.score``: notes and rests with lengths in beats, and a base
  tempo), matched by the ``score_align`` recognizer, which is loose about key and tempo;
* learned from example recordings: the pipeline's front-end + detector turns each example into
  whistle events and a template is fitted to them (``fit_note_template``), matched by the
  ``note_template`` recognizer.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from whistle_house.stages.detectors.common import semitones
from whistle_house.types import WhistleEvent


class FeatureStat(BaseModel):
    mean: float
    std: float


class NoteTemplate(BaseModel):
    """Statistics for one note of a tune, relative to the tune's first note."""

    pitch: FeatureStat  # semitones relative to note 0 (note 0: 0)
    onset: FeatureStat  # onset as fraction of the tune span (note 0: 0, last: 1)
    log_duration: FeatureStat  # log(duration / span), or log(duration s) for 1-note tunes
    glide: FeatureStat  # semitones from start to end of the note


class TuneTemplate(BaseModel):
    notes: list[NoteTemplate]
    log_span: FeatureStat  # log(seconds from first onset to last onset); unused for 1 note
    n_examples: int


class ScoreNote(BaseModel):
    """One written note or rest. A quarter note is 1 beat."""

    pitch: int | None = Field(None, ge=0, le=127, description="MIDI note number; None = rest")
    beats: float = Field(1.0, gt=0, le=16, description="Length in quarter notes")
    flat: bool = Field(False, description="Display only: spell a black key as a flat")


class Score(BaseModel):
    """A tune as sheet music. Only the intervals and the relative note lengths are matched:
    the key it is performed in and the tempo are free (see ``score_align``)."""

    tempo_bpm: float = Field(100.0, ge=20, le=400, description="Quarter notes per minute")
    beats_per_bar: int = Field(4, ge=1, le=12, description="Display only: where bar lines go")
    notes: list[ScoreNote] = Field(default_factory=list)

    def sounding(self) -> list[ScoreNote]:
        return [n for n in self.notes if n.pitch is not None]

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.tempo_bpm


class Tune(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    description: str = ""
    enabled: bool = True
    examples: list[str] = Field(default_factory=list, description="recording ids")
    template: TuneTemplate | None = None
    score: Score | None = Field(None, description="sheet-music definition (score_align)")
    outputs: list[str] = Field(default_factory=list, description="output ids to fire on match")
    threshold: float | None = Field(None, description="per-tune override of recognizer threshold")


# --------------------------------------------------------------------------- features


def _edge_hz(ev: WhistleEvent, which: str) -> float:
    pts = ev.contour
    k = max(1, len(pts) // 5)
    sel = pts[:k] if which == "start" else pts[-k:]
    return float(np.median([p.hz for p in sel]))


def note_features(events: list[WhistleEvent]) -> tuple[np.ndarray, float]:
    """events (one tune instance) -> ((N, 4) [pitch, onset, log_dur, glide], log_span)."""
    ref = events[0].median_hz()
    t0 = events[0].start
    span = events[-1].start - t0
    feats = []
    for ev in events:
        onset = (ev.start - t0) / span if span > 0 else 0.0
        dur = max(ev.duration, 1e-3)
        log_dur = np.log(dur / span) if span > 0 else np.log(dur)
        glide = semitones(_edge_hz(ev, "end"), _edge_hz(ev, "start"))
        feats.append([semitones(ev.median_hz(), ref), onset, log_dur, glide])
    return np.asarray(feats, np.float64), float(np.log(max(span, 1e-3)))


# Minimum std per feature so a handful of near-identical examples doesn't make a tune
# impossible to match.
STD_FLOOR = {"pitch": 0.75, "onset": 0.08, "log_duration": 0.35, "glide": 1.5, "log_span": 0.25}


def split_by_silence(events: list[WhistleEvent], gap_s: float) -> list[list[WhistleEvent]]:
    """Split a stream of events into groups separated by >= gap_s of silence."""
    groups: list[list[WhistleEvent]] = []
    for ev in sorted(events, key=lambda e: e.start):
        if groups and ev.start - max(e.end for e in groups[-1]) < gap_s:
            groups[-1].append(ev)
        else:
            groups.append([ev])
    return groups


def fit_note_template(instances: list[list[WhistleEvent]]) -> TuneTemplate:
    """Fit per-note Gaussian stats over example instances (all with the modal note count)."""
    instances = [i for i in instances if i]
    if not instances:
        raise ValueError("no whistle events found in the examples")
    counts = [len(i) for i in instances]
    n = max(set(counts), key=counts.count)
    used = [i for i in instances if len(i) == n]
    feats, spans = zip(*(note_features(i) for i in used), strict=True)
    f = np.stack(feats)  # (E, N, 4)
    mean, std = f.mean(0), f.std(0)
    names = ["pitch", "onset", "log_duration", "glide"]
    notes = [
        NoteTemplate(
            **{
                nm: FeatureStat(mean=float(mean[j, k]), std=float(max(std[j, k], STD_FLOOR[nm])))
                for k, nm in enumerate(names)
            }
        )
        for j in range(n)
    ]
    sp = np.asarray(spans)
    return TuneTemplate(
        notes=notes,
        log_span=FeatureStat(
            mean=float(sp.mean()), std=float(max(sp.std(), STD_FLOOR["log_span"]))
        ),
        n_examples=len(used),
    )


# --------------------------------------------------------------------------- library


class TuneLibrary:
    """Thread-safe JSON-backed tune store."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._tunes: dict[str, Tune] = {}
        if path.exists():
            for d in json.loads(path.read_text()):
                t = Tune.model_validate(d)
                self._tunes[t.id] = t

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([t.model_dump(mode="json") for t in self._tunes.values()], indent=2)
        )
        tmp.replace(self.path)

    def list(self) -> list[Tune]:
        with self._lock:
            return list(self._tunes.values())

    def get(self, tune_id: str) -> Tune:
        with self._lock:
            return self._tunes[tune_id]

    def put(self, tune: Tune) -> Tune:
        with self._lock:
            self._tunes[tune.id] = tune
            self._save()
            return tune

    def delete(self, tune_id: str) -> None:
        with self._lock:
            self._tunes.pop(tune_id, None)
            self._save()
