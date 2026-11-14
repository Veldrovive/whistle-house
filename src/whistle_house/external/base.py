"""External dataset framework: download -> prepare -> manifest.

Every external dataset is turned into the same on-disk form so training never needs to know
where data came from::

    <external_root>/raw/<name>/              whatever download() fetched / extracted
    <external_root>/prepared/<name>/
        audio/...                            mono FLAC, sample rate min(native, 22050)
        manifest.jsonl                       one ClipEntry per line
        info.json                            dataset metadata + prepare stats

``ClipEntry.label`` decides how training uses a clip:

* ``whistle``                     -> positive (pitch auto-labelled, like mic whistle clips)
* ``hum``                         -> hard negative (tonal, pitched, *not* whistling)
* ``speech`` / ``music`` / ``noise`` -> interferers / negatives
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import time
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import soundfile as sf

from whistle_house.audio.io import resample, to_mono_float

log = logging.getLogger(__name__)

MAX_SAMPLE_RATE = 22050  # whistles live in 0.4-4 kHz; no need to store more
POSITIVE_LABELS = {"whistle"}
NEGATIVE_LABELS = {"hum", "speech", "music", "noise"}
Split = Literal["train", "val", "test"]


@dataclass
class ClipEntry:
    path: str  # relative to the prepared dataset dir
    label: str
    split: Split
    duration: float
    sample_rate: int
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def hash_split(key: str, val: float = 0.05, test: float = 0.05) -> Split:
    """Deterministic split from a stable key (e.g. speaker id or relative file path)."""
    u = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if u < test else "val" if u < test + val else "train"


# --------------------------------------------------------------------------- dataset ABC


class ExternalDataset(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    homepage: ClassVar[str]
    license: ClassVar[str]
    download_size: ClassVar[str]  # human readable, informational
    raw_name: ClassVar[str | None] = None  # share a raw dir between datasets (MLEnd)

    def __init__(self, root: Path):
        self.root = root

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw" / (self.raw_name or self.name)

    @property
    def prepared_dir(self) -> Path:
        return self.root / "prepared" / self.name

    @property
    def manifest_path(self) -> Path:
        return self.prepared_dir / "manifest.jsonl"

    def is_prepared(self) -> bool:
        return self.manifest_path.exists()

    @abstractmethod
    def download(self) -> None:
        """Fetch into ``raw_dir``. Must be idempotent / resumable."""

    @abstractmethod
    def iter_sources(self) -> list[tuple[Path, str, Split, dict]]:
        """List (raw audio file, label, split, meta) to include. Called after download()."""

    def prepare(self, workers: int = 4) -> list[ClipEntry]:
        """Standardise every source file into prepared_dir and write the manifest."""
        sources = self.iter_sources()
        if not sources:
            raise RuntimeError(f"{self.name}: no source audio found in {self.raw_dir}")
        audio_dir = self.prepared_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        def one(item):
            src, label, split, meta = item
            rel = Path("audio") / src.relative_to(self.raw_dir).with_suffix(".flac")
            dst = self.prepared_dir / rel
            try:
                dur, sr = standardise_audio(src, dst, channel=meta.pop("_channel", None))
            except Exception as e:  # noqa: BLE001 - skip unreadable files, report count
                log.warning("%s: skipping %s: %s", self.name, src, e)
                return None
            return ClipEntry(str(rel), label, split, dur, sr, meta)

        t0 = time.time()
        with ThreadPoolExecutor(workers) as pool:
            entries = [e for e in pool.map(one, sources) if e is not None]
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text("".join(e.to_json() + "\n" for e in entries))
        tmp.replace(self.manifest_path)
        stats = summarize(entries)
        (self.prepared_dir / "info.json").write_text(
            json.dumps(
                {
                    "name": self.name,
                    "description": self.description,
                    "homepage": self.homepage,
                    "license": self.license,
                    "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prepare_seconds": round(time.time() - t0, 1),
                    "skipped": len(sources) - len(entries),
                    "stats": stats,
                },
                indent=2,
            )
        )
        log.info("%s: prepared %d clips (%s)", self.name, len(entries), stats)
        return entries

    def clean_raw(self) -> None:
        if self.raw_dir.exists():
            shutil.rmtree(self.raw_dir)


def summarize(entries: list[ClipEntry]) -> dict:
    out: dict[str, dict] = {}
    for e in entries:
        s = out.setdefault(e.label, {"clips": 0, "hours": 0.0})
        s["clips"] += 1
        s["hours"] += e.duration / 3600
    for s in out.values():
        s["hours"] = round(s["hours"], 2)
    return out


def load_manifest(prepared_dir: Path) -> list[ClipEntry]:
    path = prepared_dir / "manifest.jsonl"
    return [ClipEntry(**json.loads(line)) for line in path.read_text().splitlines() if line]


# --------------------------------------------------------------------------- registry

_DATASETS: dict[str, type[ExternalDataset]] = {}


def register_dataset(cls: type[ExternalDataset]) -> type[ExternalDataset]:
    _DATASETS[cls.name] = cls
    return cls


def dataset_names() -> list[str]:
    import whistle_house.external.datasets  # noqa: F401 - registers built-ins

    return list(_DATASETS)


def get_dataset(name: str, root: Path) -> ExternalDataset:
    dataset_names()
    if name not in _DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(_DATASETS)}")
    return _DATASETS[name](root)


def dataset_status(root: Path | None) -> list[dict]:
    """Status of every registered dataset (used by the CLI and the manager)."""
    out = []
    for name in dataset_names():
        cls = _DATASETS[name]
        row = {
            "name": name,
            "description": cls.description,
            "homepage": cls.homepage,
            "license": cls.license,
            "download_size": cls.download_size,
            "downloaded": False,
            "prepared": False,
            "stats": None,
        }
        if root is not None:
            ds = cls(root)
            row["downloaded"] = ds.raw_dir.exists() and any(ds.raw_dir.iterdir())
            info = ds.prepared_dir / "info.json"
            if ds.is_prepared() and info.exists():
                row["prepared"] = True
                row["stats"] = json.loads(info.read_text()).get("stats")
        out.append(row)
    return out


# --------------------------------------------------------------------------- helpers


def standardise_audio(src: Path, dst: Path, channel: int | None = None) -> tuple[float, int]:
    """Any libsndfile-readable file -> mono FLAC at min(native, 22050) Hz. Idempotent."""
    if dst.exists():
        info = sf.info(str(dst))
        return info.frames / info.samplerate, info.samplerate
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    x = data[:, channel] if channel is not None else to_mono_float(data)
    out_sr = min(sr, MAX_SAMPLE_RATE)
    x = resample(np.asarray(x, np.float32), sr, out_sr)
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.flac")
    sf.write(str(tmp), x, out_sr, format="FLAC", subtype="PCM_16")
    tmp.replace(dst)
    return len(x) / out_sr, out_sr


def download_file(url: str, dest: Path, retries: int = 5, quiet: bool = False) -> Path:
    """Resumable HTTP download (Range requests) with a simple progress line."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "whistle-house-dataset-fetch"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored Range: restart
                    have = 0
                total = int(r.headers.get("Content-Length") or 0) + have
                with part.open("ab" if have else "wb") as f:
                    done, last = have, 0.0
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if not quiet and time.time() - last > 2:
                            last = time.time()
                            pct = f"{100 * done / total:5.1f}%" if total else ""
                            print(
                                f"\r  {dest.name}: {done / 1e6:,.0f} MB {pct}", end="", flush=True
                            )
            if not quiet:
                print(f"\r  {dest.name}: done ({part.stat().st_size / 1e6:,.0f} MB)        ")
            part.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001 - network errors: retry with resume
            log.warning("download %s failed (attempt %d/%d): %s", url, attempt, retries, e)
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"failed to download {url}")


def download_many(items: list[tuple[str, Path]], workers: int | None = None) -> None:
    """Many small files in parallel (skips existing)."""
    workers = workers or int(os.environ.get("WHISTLE_HOUSE_DOWNLOAD_WORKERS", "8"))
    todo = [(u, p) for u, p in items if not p.exists()]
    print(f"  {len(items) - len(todo)} already present, downloading {len(todo)} files")
    done = 0

    def one(item):
        download_file(*item, quiet=True)

    with ThreadPoolExecutor(workers) as pool:
        for _ in pool.map(one, todo):
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"\r  {done}/{len(todo)} files", end="", flush=True)
    if todo:
        print()


def extract(archive: Path, into: Path) -> None:
    """Extract tar(.gz)/zip once (marker file makes it idempotent)."""
    marker = into / f".extracted-{archive.name}"
    if marker.exists():
        return
    into.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} ...")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            t.extractall(into, filter="data")
    marker.touch()
