"""Settings: where runtime data and external datasets live.

Configuration comes from environment variables, which may be set in a ``.env`` file (see
``.env.example``). Real environment variables win over ``.env`` values.

    WHISTLE_HOUSE_DATA      runtime data dir (default ./data):
      recordings/<id>.wav + <id>.json   collected audio + labels (data/store.py)
      noise/**.wav|flac                 extra interferer audio for augmentation
      pipelines/<name>.json             saved pipeline specs (pipeline.py)
      models/<name>.pt (+ .json)        trained detector checkpoints (ml/train.py)
      cache/labels/                     cached pitch auto-labels for mic recordings
      tunes.json, outputs.json, state.json

    WHISTLE_HOUSE_EXTERNAL  external datasets root (no default - must be set explicitly so large
                            downloads never land somewhere by accident, e.g. on the Pi):
      raw/<dataset>/        downloads + extracted archives
      prepared/<dataset>/   audio/ (mono flac) + manifest.jsonl  (external/base.py)
      labels/               cached pitch auto-labels for external positives
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_env_loaded = False


def parse_env_file(text: str) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE lines, # comments, optional quotes, optional `export`."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ")
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        out[key.strip()] = os.path.expandvars(os.path.expanduser(value))
    return out


def load_env(path: Path | None = None, override: bool = False) -> Path | None:
    """Load ``.env`` from ``path``, else ./.env, else <repo>/.env. Returns the file used."""
    global _env_loaded
    candidates = [path] if path else [Path.cwd() / ".env", REPO_ROOT / ".env"]
    for p in candidates:
        if p and p.is_file():
            for k, v in parse_env_file(p.read_text()).items():
                if override or k not in os.environ:
                    os.environ[k] = v
            _env_loaded = True
            return p
    _env_loaded = True
    return None


def _ensure_env() -> None:
    if not _env_loaded:
        load_env()


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
    def label_cache(self) -> Path:
        return self.root / "cache" / "labels"

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
    _ensure_env()
    return Paths(Path(os.environ.get("WHISTLE_HOUSE_DATA", "data")).resolve())


class ExternalNotConfigured(RuntimeError):
    pass


def external_root(required: bool = True) -> Path | None:
    """Root of the external datasets (WHISTLE_HOUSE_EXTERNAL)."""
    _ensure_env()
    v = os.environ.get("WHISTLE_HOUSE_EXTERNAL")
    if not v:
        if required:
            raise ExternalNotConfigured(
                "WHISTLE_HOUSE_EXTERNAL is not set. Copy .env.example to .env and point it at a "
                "disk with room for the datasets (~40 GB for everything)."
            )
        return None
    return Path(v).resolve()


def is_raspberry_pi() -> bool:
    try:
        return "raspberry pi" in Path("/proc/device-tree/model").read_text().lower()
    except OSError:
        return False
