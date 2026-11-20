"""Recording dataset: ``<data_dir>/recordings/<id>.wav`` + ``<id>.json`` metadata.

Clip-level ``label`` says what the clip mostly contains. Optional ``segments`` give finer
time-ranges (e.g. a whistle inside a mostly-speech clip). Tune examples are ``whistle`` clips
with ``tune_id`` set.

Recordings are made on the deployment mic (manager Live/Collect page) or uploaded. Labels that
matter for the learned detector:

* ``whistle``  - clean whistling (positives; pitch labels are derived automatically)
* ``hum``      - clean humming (a harmonic stack with a low fundamental). Positives for a model
  with a ``hum`` class (pitch labelled by pYIN, ``data/hum_labels.py``); a whistle-only model
  trains on them as hard negatives (the ``hum`` interferer group, with the external hums).
* ``speech``   - talking with no whistling (the v1 false-trigger problem - collect lots!)
* ``noise``    - TV, music, kitchen, door slams, ... anything that isn't whistling
* ``silence``  - room tone
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from whistle_house.audio.io import load_audio, save_wav

LABELS = ["whistle", "hum", "speech", "noise", "music", "silence", "other"]
NEGATIVE_LABELS = {"speech", "noise", "music", "silence"}

Split = Literal["train", "val", "test"]


def split_for(rec_id: str) -> Split:
    """Deterministic ~80/10/10 split from the id, so it never changes once assigned."""
    h = int(hashlib.sha1(rec_id.encode()).hexdigest()[:8], 16) % 10
    return "val" if h == 8 else "test" if h == 9 else "train"


class Segment(BaseModel):
    start: float
    end: float
    label: str


class Recording(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    label: str = "other"
    tune_id: str | None = None
    sample_rate: int
    duration: float
    source: str = "mic"  # mic | upload | synthetic
    device: str | None = None
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    split: Split | None = None


class RecordingUpdate(BaseModel):
    label: str | None = None
    tune_id: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    segments: list[Segment] | None = None
    split: Split | None = None


class RecordingStore:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def audio_path(self, rec_id: str) -> Path:
        return self.dir / f"{rec_id}.wav"

    def _meta_path(self, rec_id: str) -> Path:
        return self.dir / f"{rec_id}.json"

    def _write_meta(self, rec: Recording) -> None:
        self._meta_path(rec.id).write_text(json.dumps(rec.model_dump(mode="json"), indent=2))

    def add(self, samples: np.ndarray, sample_rate: int, **meta) -> Recording:
        rec = Recording(sample_rate=sample_rate, duration=len(samples) / sample_rate, **meta)
        if rec.split is None:
            rec.split = split_for(rec.id)
        with self._lock:
            save_wav(self.audio_path(rec.id), samples, sample_rate)
            self._write_meta(rec)
        return rec

    def get(self, rec_id: str) -> Recording:
        return Recording.model_validate_json(self._meta_path(rec_id).read_text())

    def list(self, label: str | None = None, tune_id: str | None = None) -> list[Recording]:
        recs = [Recording.model_validate_json(p.read_text()) for p in self.dir.glob("*.json")]
        if label:
            recs = [r for r in recs if r.label == label]
        if tune_id:
            recs = [r for r in recs if r.tune_id == tune_id]
        return sorted(recs, key=lambda r: r.created, reverse=True)

    def load(self, rec_id: str, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
        return load_audio(self.audio_path(rec_id), sample_rate)

    def update(self, rec_id: str, upd: RecordingUpdate) -> Recording:
        with self._lock:
            rec = self.get(rec_id)
            rec = rec.model_copy(update=upd.model_dump(exclude_unset=True))
            self._write_meta(rec)
            return rec

    def delete(self, rec_id: str) -> None:
        with self._lock:
            self.audio_path(rec_id).unlink(missing_ok=True)
            self._meta_path(rec_id).unlink(missing_ok=True)

    def stats(self) -> dict:
        out: dict[str, dict] = {}
        for r in self.list():
            s = out.setdefault(r.label, {"count": 0, "seconds": 0.0})
            s["count"] += 1
            s["seconds"] += r.duration
        return out
