"""Noise profiles: per-bin statistics of the mic's background noise, for the gate filters.

``spectral_gate`` uses one profile, ``adaptive_gate`` several (one per background: quiet,
fridge, fan, ...) and tracks which one is on.

A profile is measured by running "silence" recordings (room tone, the mic's own hiss) through the
part of a pipeline *upstream* of where the filter sits (transform + the filters before it), then
taking each frequency bin's mean and standard deviation over time. Frames that are much louder
than typical (a door, a cough in the "silence" clip) are dropped first.

Profiles live in ``data/noise_profiles/<name>.json``. They are immutable once written: trained
checkpoints and label caches key on the frontend spec, which names the profile but does not hash
its contents, so a changed profile must get a new name.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from whistle_house.stages.base import StageKind, get_stage_class

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def canonical_stage(kind: StageKind | str, name: str, params: dict) -> dict:
    """{name, params} with params normalised through the stage's Config (defaults filled in)."""
    cls = get_stage_class(StageKind(kind), name)
    return {"name": name, "params": cls.Config.model_validate(params).model_dump(mode="json")}


def upstream_fingerprint(sample_rate: int, transform: dict, filters: list[dict]) -> dict:
    """What a profile depends on: everything before the filter that uses it."""
    return {
        "sample_rate": int(sample_rate),
        "transform": canonical_stage(StageKind.transform, transform["name"], transform["params"]),
        "filters": [canonical_stage(StageKind.filter, f["name"], f["params"]) for f in filters],
    }


@dataclass
class NoiseProfile:
    name: str
    freqs: list[float]  # bin centre frequencies (Hz) of the frames the stats were taken on
    mean: list[float]  # per-bin mean (input units, dB for the stock transform)
    std: list[float]  # per-bin standard deviation over time
    upstream: dict  # upstream_fingerprint(...) the stats were measured with
    sources: list[str] = field(default_factory=list)  # recording ids / file names
    n_frames: int = 0
    n_dropped: int = 0  # loud outlier frames excluded
    duration_s: float = 0.0
    created: str = ""

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.freqs, np.float64),
            np.asarray(self.mean, np.float32),
            np.asarray(self.std, np.float32),
        )


def compute_noise_profile(
    name: str,
    frames: list[np.ndarray],
    freqs: np.ndarray,
    upstream: dict,
    sources: list[str],
    duration_s: float,
    outlier_mad: float = 4.0,
    min_std: float = 0.5,
) -> NoiseProfile:
    """Per-bin mean/std over the frames of all silence clips, after dropping loud outliers.

    A frame is an outlier if its mean level is more than ``outlier_mad`` scaled MADs above the
    median frame level (transients in a "silence" recording should not inflate the profile).
    """
    if not NAME_RE.match(name):
        raise ValueError("profile name must be 1-64 chars of letters, digits, _ . -")
    v = np.concatenate([f for f in frames if f.size], axis=0) if frames else np.zeros((0, 0))
    if v.shape[0] < 20:
        raise ValueError(f"need at least 20 frames of silence, got {v.shape[0]}")
    level = v.mean(axis=1)
    med = np.median(level)
    mad = 1.4826 * np.median(np.abs(level - med)) + 1e-6
    keep = level <= med + outlier_mad * mad
    kept = v[keep]
    return NoiseProfile(
        name=name,
        freqs=[float(x) for x in freqs],
        mean=[round(float(x), 4) for x in kept.mean(axis=0)],
        std=[round(float(x), 4) for x in np.maximum(kept.std(axis=0), min_std)],
        upstream=upstream,
        sources=sources,
        n_frames=int(keep.sum()),
        n_dropped=int((~keep).sum()),
        duration_s=round(float(duration_s), 3),
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )


class NoiseProfileStore:
    """``data/noise_profiles/<name>.json``."""

    def __init__(self, directory: Path):
        self.dir = directory

    def path(self, name: str) -> Path:
        if not NAME_RE.match(name):
            raise ValueError(f"invalid noise profile name {name!r}")
        return self.dir / f"{name}.json"

    def list(self) -> list[NoiseProfile]:
        if not self.dir.exists():
            return []
        return [self._read(p) for p in sorted(self.dir.glob("*.json"))]

    def get(self, name: str) -> NoiseProfile:
        p = self.path(name)
        if not p.exists():
            raise KeyError(f"noise profile {name!r} not found in {self.dir}")
        return self._read(p)

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def save(self, profile: NoiseProfile) -> NoiseProfile:
        p = self.path(profile.name)
        if p.exists():
            raise FileExistsError(
                f"noise profile {profile.name!r} already exists; profiles are immutable "
                "(models and label caches refer to them by name), pick a new name"
            )
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(profile), indent=1))
        tmp.replace(p)
        return profile

    def delete(self, name: str) -> None:
        self.path(name).unlink(missing_ok=True)

    @staticmethod
    def _read(p: Path) -> NoiseProfile:
        return NoiseProfile(**json.loads(p.read_text()))


def profile_from_audio(
    name: str,
    spec_dict: dict,
    position: int,
    clips: list[tuple[str, np.ndarray]],
    data_dir: Path | None = None,
) -> NoiseProfile:
    """Measure a profile for a filter at ``filters[position]`` of a pipeline spec.

    ``spec_dict`` needs sample_rate/transform/filters (a PipelineSpec dump works). ``clips`` are
    (source id, mono float32 audio at the spec's sample rate).
    """
    from whistle_house.pipeline import Frontend, StageSpec  # avoid import cycle

    filters = spec_dict["filters"][:position]
    fe = Frontend(
        spec_dict["sample_rate"],
        StageSpec.model_validate(spec_dict["transform"]),
        [StageSpec.model_validate(f) for f in filters],
        data_dir,
    )
    frames, total = [], 0.0
    for _, audio in clips:
        _, out = fe.process_signal(audio)
        frames.append(out.values)
        total += len(audio) / spec_dict["sample_rate"]
    upstream = upstream_fingerprint(
        spec_dict["sample_rate"], spec_dict["transform"], [dict(f) for f in filters]
    )
    return compute_noise_profile(
        name, frames, fe.axis.freqs, upstream, [s for s, _ in clips], total
    )


GATE_FILTERS = ("spectral_gate", "adaptive_gate")  # filters that use noise profiles


def filter_profiles(f: dict) -> list[str]:
    """Noise profile names a filter spec ({name, params}) uses (none for other filters)."""
    params = f.get("params") or {}
    if f.get("name") == "spectral_gate":
        return [params["profile"]] if params.get("profile") else []
    if f.get("name") == "adaptive_gate":
        return [p for p in params.get("profiles") or [] if p]
    return []


def gate_position(spec_dict: dict) -> int:
    """Where a profile for this pipeline is measured: at its first gate filter
    (spectral_gate or adaptive_gate), else the end."""
    for i, f in enumerate(spec_dict["filters"]):
        if f["name"] in GATE_FILTERS:
            return i
    return len(spec_dict["filters"])


def profile_from_recordings(
    name: str,
    spec_dict: dict,
    recordings,  # RecordingStore (not imported: keeps this module light)
    recording_ids: list[str],
    position: int | None = None,
    data_dir: Path | None = None,
) -> NoiseProfile:
    """Measure a profile from stored (silence) recordings for ``spec_dict``'s gate position."""
    if not recording_ids:
        raise ValueError("choose at least one silence recording")
    pos = gate_position(spec_dict) if position is None else position
    if not 0 <= pos <= len(spec_dict["filters"]):
        raise ValueError(f"position {pos} out of range for {len(spec_dict['filters'])} filters")
    sr = spec_dict["sample_rate"]
    clips = [(rid, recordings.load(rid, sr)[0]) for rid in recording_ids]
    return profile_from_audio(name, spec_dict, pos, clips, data_dir)


def attach_gate(spec_dict: dict, profile: str, position: int | None = None) -> dict:
    """Copy of the spec with a gate using ``profile`` at ``position``.

    An existing spectral_gate at that position is pointed at the profile, an existing
    adaptive_gate gets it added to its list (other params kept); otherwise a spectral_gate is
    inserted.
    """
    spec = json.loads(json.dumps(spec_dict))
    filters = spec["filters"]
    pos = gate_position(spec) if position is None else position
    if pos < len(filters) and filters[pos]["name"] == "spectral_gate":
        filters[pos]["params"] = {**filters[pos].get("params", {}), "profile": profile}
    elif pos < len(filters) and filters[pos]["name"] == "adaptive_gate":
        params = filters[pos].get("params", {})
        names = list(params.get("profiles") or [])
        filters[pos]["params"] = {
            **params,
            "profiles": names if profile in names else [*names, profile],
        }
    else:
        filters.insert(pos, {"name": "spectral_gate", "params": {"profile": profile}})
    return spec
