"""Locations of runtime data. Everything mutable lives under one data directory.

    <data_dir>/
      recordings/<id>.wav + <id>.json   collected audio + labels (data/store.py)
      noise/**.wav|flac                 external background-noise / speech corpora for augmentation
      pipelines/<name>.json             saved pipeline specs (pipeline.py)
      models/<name>.pt (+ .json)        trained detector checkpoints (ml/train.py)
      tunes.json                        tune library (tunes.py)
      outputs.json                      output definitions (outputs/)
      state.json                        server state (active pipeline, ...)

Override with the WHISTLE_HOUSE_DATA env var; defaults to ./data in the repo root/cwd.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def recordings(self) -> Path:
        return self.root / "recordings"

    @property
    def noise(self) -> Path:
        return self.root / "noise"

    @property
    def pipelines(self) -> Path:
        return self.root / "pipelines"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def tunes(self) -> Path:
        return self.root / "tunes.json"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs.json"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    def ensure(self) -> Paths:
        for p in (self.recordings, self.noise, self.pipelines, self.models):
            p.mkdir(parents=True, exist_ok=True)
        return self


def default_paths() -> Paths:
    return Paths(Path(os.environ.get("WHISTLE_HOUSE_DATA", "data")).resolve())
