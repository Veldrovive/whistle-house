"""Built-in external datasets. Add one by subclassing ExternalDataset + @register_dataset.

Sizes/licences checked 2026-11; verify licences before any non-personal use.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import ClassVar

from whistle_house.external.base import (
    ExternalDataset,
    Split,
    download_file,
    download_many,
    extract,
    hash_split,
    register_dataset,
)


def fetch_and_extract(
    url: str, raw_dir: Path, archive_name: str, keep_archive: bool = False
) -> None:
    """Download an archive into raw_dir, extract it there, then delete the archive."""
    marker = raw_dir / f".extracted-{archive_name}"
    if marker.exists():
        return
    archive = download_file(url, raw_dir / archive_name)
    extract(archive, raw_dir)
    if not keep_archive:
        archive.unlink()


# --------------------------------------------------------------------------- MLEnd


class _MLEnd(ExternalDataset):
    """MLEnd Hums and Whistles (Queen Mary University of London): 8 songs hummed/whistled by
    236 people, crowd-sourced recordings. 6611 wav clips: 1801 whistles, 4810 hums."""

    homepage = "https://mlenddatasets.github.io/hums_whistles"
    license = "not stated by the publisher - check before non-personal use"
    raw_name = "mlend"
    interpretation: str
    BASE = "https://raw.githubusercontent.com/MLEndDatasets/HumsAndWhistles/main/"
    ATTRS = "MLEndHWD_audio_attributes_benchmark.csv"

    def _rows(self) -> list[dict]:
        with (self.raw_dir / self.ATTRS).open() as f:
            return [r for r in csv.DictReader(f) if r["Interpretation"] == self.interpretation]

    def download(self) -> None:
        download_file(self.BASE + self.ATTRS, self.raw_dir / self.ATTRS)
        items = [
            (
                self.BASE + "MLEndHWD_audiofiles/" + r["filename"],
                self.raw_dir / "audio" / r["filename"],
            )
            for r in self._rows()
        ]
        download_many(items)

    def iter_sources(self) -> list[tuple[Path, str, Split, dict]]:
        out = []
        for r in self._rows():
            p = self.raw_dir / "audio" / r["filename"]
            if not p.exists():
                continue
            # Benchmark_A is speaker independent: keep its test speakers as our test split and
            # carve a speaker-disjoint val split out of its train speakers.
            split: Split = (
                "test"
                if r["Benchmark_A"] == "Test"
                else hash_split(r["Interpreter"], val=0.1, test=0)
            )
            label = "whistle" if self.interpretation == "Whistle" else "hum"
            meta = {"song": r["Song"], "speaker": r["Interpreter"]}
            out.append((p, label, split, meta))
        return out


@register_dataset
class MLEndWhistles(_MLEnd):
    name = "mlend_whistles"
    description = "MLEnd: 1801 whistled melodies from 236 people (positives)"
    download_size = "~4.7 GB"
    interpretation = "Whistle"


@register_dataset
class MLEndHums(_MLEnd):
    name = "mlend_hums"
    description = "MLEnd: 4810 hummed melodies (hum positives; negatives for whistle-only)"
    download_size = "~12.5 GB"
    interpretation = "Hum"


# --------------------------------------------------------------------------- MUSAN


@register_dataset
class Musan(ExternalDataset):
    name = "musan"
    description = "MUSAN: ~109 h of speech (12 languages), music and noise"
    homepage = "https://openslr.org/17"
    license = "CC BY 4.0 on OpenSLR; per-file licences in each subdirectory's LICENSE"
    download_size = "11 GB archive (deleted after extraction)"
    URL = "https://www.openslr.org/resources/17/musan.tar.gz"

    def download(self) -> None:
        fetch_and_extract(self.URL, self.raw_dir, "musan.tar.gz")

    def iter_sources(self) -> list[tuple[Path, str, Split, dict]]:
        base = self.raw_dir / "musan"
        out = []
        for label in ("speech", "music", "noise"):
            for p in sorted((base / label).rglob("*.wav")):
                rel = str(p.relative_to(base))
                out.append((p, label, hash_split(rel), {"subset": p.parent.name}))
        return out


# --------------------------------------------------------------------------- ESC-50


@register_dataset
class Esc50(ExternalDataset):
    name = "esc50"
    description = "ESC-50: 2000 5-second environmental sounds, 50 classes (incl. birds, crickets)"
    homepage = "https://github.com/karolpiczak/ESC-50"
    license = "CC BY-NC 3.0"
    download_size = "~600 MB"
    URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"

    def download(self) -> None:
        fetch_and_extract(self.URL, self.raw_dir, "ESC-50-master.zip")

    def iter_sources(self) -> list[tuple[Path, str, Split, dict]]:
        base = self.raw_dir / "ESC-50-master"
        out = []
        with (base / "meta" / "esc50.csv").open() as f:
            for r in csv.DictReader(f):
                split: Split = {"5": "test", "4": "val"}.get(r["fold"], "train")  # type: ignore[assignment]
                out.append(
                    (base / "audio" / r["filename"], "noise", split, {"category": r["category"]})
                )
        return out


# --------------------------------------------------------------------------- DEMAND


@register_dataset
class Demand(ExternalDataset):
    name = "demand"
    description = "DEMAND: long recordings of indoor background noise (kitchen, living room, ...)"
    homepage = "https://zenodo.org/records/1227121"
    license = "CC BY 4.0"
    download_size = "~700 MB (7 indoor environments, 16 kHz)"
    ENVS: ClassVar[list[str]] = [
        "DKITCHEN",
        "DLIVING",
        "DWASHING",
        "OOFFICE",
        "OMEETING",
        "PCAFETER",
        "PRESTO",
    ]
    URL = "https://zenodo.org/api/records/1227121/files/{env}_16k.zip/content"

    def download(self) -> None:
        for env in self.ENVS:
            fetch_and_extract(self.URL.format(env=env), self.raw_dir, f"{env}_16k.zip")

    def iter_sources(self) -> list[tuple[Path, str, Split, dict]]:
        out = []
        for env in self.ENVS:
            p = self.raw_dir / env / "ch01.wav"  # 16-channel array; one mic is enough
            if p.exists():
                out.append((p, "noise", hash_split(env, val=0.15, test=0), {"environment": env}))
        return out
