"""Training example generation for frame-level whistle / hum detectors (numpy only, no torch).

An example is ``(features (T, F), f0 (T, K))``: features are the *pipeline front-end output*
(transform + filters, exactly what the detector sees live) and f0 holds, for each of the K
classes being trained (``GeneratorConfig.classes``: ``whistle``, optionally ``hum``), the
target pitch in Hz per frame (0 = that sound is absent). For a hum the pitch is the fundamental.

Data sources (``collect_sources`` gathers them into a picklable ``TrainingSources``):

* mic recordings (``data/recordings``): ``whistle`` clips are positives, ``speech``/``noise``/
  ``music``/``silence`` clips are interferers (group ``mic``);
* external datasets (``$WHISTLE_HOUSE_EXTERNAL/prepared/*``): ``whistle`` clips are positives,
  ``speech``/``music``/``noise`` are interferers;
* ``hum`` clips (mic and external): positives of the ``hum`` class when it is being trained;
  for a whistle-only model they are hard negatives (the ``hum`` interferer group);
* ``data/noise/**``: extra interferer audio;
* synthetic whistles, synthetic hums and synthetic pseudo-speech.

Pitch labels for whistle positives are produced by running the heuristic ``peak_tracker`` on
the clean clip. They are computed once per front-end and cached (``LabelCache``) as 100 Hz f0
tracks. Hum positives are labelled by pYIN on the waveform (``data/hum_labels.py``), which
does not depend on the front-end. External clips whose labels cover too little of the clip
(noisy/failed tracking) are dropped.

``GeneratorConfig.aux`` adds presence-only label columns after the f0 columns. The one that
exists is ``speech``: 1 in the frames where a speech interferer is sounding (``speech_activity``
on the speech as it sits in the mix), 0 elsewhere, -1 = "don't train on this frame" (speech too
far under the rest of the mix to say, or a source that may or may not count as speech).

Each example: pick a clean source by the mix probabilities (``GeneratorConfig``), mix in a
random interferer at a random SNR, randomise gain/EQ/reverb, run the front-end, and sample the
f0 labels at the frame times.

The ``*-gate`` augment presets also put a steady room background under the example (a mic
``silence`` take or ``synth_background``) and pass half of the examples through a spectral gate
with a noise profile measured on a stretch of that background, so the model sees the frames a
pipeline with ``spectral_gate`` / ``adaptive_gate`` gives it as well as ungated ones.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import multiprocessing as mp
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import soundfile as sf

from whistle_house.audio.io import resample, to_mono_float
from whistle_house.data.augment import (
    AUDIO_EXTS,
    active_level_db,
    envelope_clip,
    fit_length,
    mix_at_snr,
    random_channel,
    random_eq,
    random_gain,
    random_reverb,
    reverse_clip,
    rms,
    room_reverb,
    saturate,
    splice_notes,
    synth_background,
    synth_hum,
    synth_pseudo_speech,
    synth_whistle,
    warp_clip,
)
from whistle_house.data.store import NEGATIVE_LABELS, RecordingStore
from whistle_house.noise_profiles import filter_profiles
from whistle_house.pipeline import Frontend
from whistle_house.stages.base import StageContext, build_stage
from whistle_house.stages.filters.spectral_gate import SpectralGate
from whistle_house.threads import limit_worker_threads
from whistle_house.types import WhistleEvent

log = logging.getLogger(__name__)

LABEL_RATE = 100  # Hz, resolution of cached f0 tracks


def _mp_context():
    """forkserver: forking a process that already runs threads (torch, BLAS) can deadlock."""
    return mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")


# --------------------------------------------------------------------------- config


@dataclass
class GeneratorConfig:
    window_s: float = 3.0
    warmup_s: float = 1.0  # discarded lead-in so stateful filters (noise floor) are settled
    # What the model is trained to detect: one f0 label column per class, in this order.
    classes: tuple[str, ...] = ("whistle",)
    # Presence-only outputs (AUX_CLASSES: "speech"): one more label column each, after the f0
    # columns, holding 1 / 0 / -1 (= ignore) per frame. () = none, the labels are f0 only.
    aux: tuple[str, ...] = ()
    # Speech labels: a speech source counts where it is within ``speech_range_db`` of its own
    # loud parts (pauses shorter than ``speech_bridge_s`` are speech too). It is labelled when
    # its loud parts are no more than ``speech_audible_db[0]`` dB under the rest of the mix,
    # ignored down to ``[1]``, and below that the frames count as "no speech".
    speech_range_db: float = 30.0
    speech_bridge_s: float = 0.3
    speech_audible_db: tuple[float, float] = (10.0, 20.0)
    # Probability that an example's clean signal is a whistle from each source; the remainder
    # is "no whistle". Unavailable sources hand their share to synthetic whistles.
    p_mic_whistle: float = 0.45
    p_ext_whistle: float = 0.0
    p_synth_whistle: float = 0.25
    # The same for hums. Only used when "hum" is in ``classes``; a whistle-only model gets its
    # hums as interferers instead (the "hum" group below).
    p_mic_hum: float = 0.0
    p_ext_hum: float = 0.0
    p_synth_hum: float = 0.0
    # Relative weights when picking an interferer group (missing groups are skipped).
    interferer_weights: dict[str, float] = field(
        default_factory=lambda: {
            "mic": 3.0,  # negatives recorded on the deployment mic: most valuable
            "speech": 3.0,
            "hum": 2.0,
            "noise": 2.0,
            "music": 1.0,
            "noise_dir": 1.0,
            "synthetic": 1.0,
        }
    )
    # With fewer mic negative clips than this, the "mic" interferer weight is scaled down in
    # proportion, so one or two clips (e.g. a single silence take) don't crowd out speech.
    mic_full_weight_clips: int = 10
    # Extra augmentation of *mic* whistle clips (data/augment.py: splice_notes, warp_clip,
    # reverse_clip, envelope_clip). There are only ever a few of these clips, so this turns
    # each take into many melodies/pitches/tempos. Training examples only: ``eval_config()``
    # switches it off for validation, which must look like real whistling. Off by default; the
    # ``finetune`` mix enables it.
    p_mic_splice: float = 0.0  # re-order the clip's notes into a new melody
    p_mic_warp: float = 0.0  # change speed (pitch + tempo together), with slow drift
    p_mic_reverse: float = 0.0  # play it backwards
    p_mic_envelope: float = 0.0  # slow random loudness drift
    mic_warp_semitones: tuple[float, float] = (-4.0, 4.0)
    p_interferer: float = 0.85
    snr_db: tuple[float, float] = (-3.0, 25.0)
    p_reverb: float = 0.3
    p_eq: float = 0.5
    # Overtones. A whistle is nearly a pure tone, but loud or on a cheap mic it shows harmonics,
    # and a detector must not report those as notes. Two sources of harmonic-rich positives
    # whose label is still the fundamental: synthetic whistles drawn with overtones, and
    # soft-clipping the clean signal (whatever it is) as an overdriven mic would.
    p_whistle_overtones: float = 0.5
    p_saturate: float = 0.25
    # --- Channel and room (all off by default = the first recipe; see AUGMENT_PRESETS). ---
    # Without these the class could be read off things that have nothing to do with the sound:
    # hums were reverberated and speech never was, a lone interferer was up to 30 dB quieter
    # than a positive, and the spectral balance of each class was whatever its dataset's
    # microphones gave it. On another microphone those cues are simply wrong.
    # Random frequency response (augment.random_channel) on the clean signal and on each
    # interferer, drawn independently, and with half this probability on the whole mix.
    p_channel: float = 0.0
    # Reverb on the interferer too (the clean signal has p_reverb). With ``room_reverb`` both
    # use augment.room_reverb (near to far) instead of the nearly direct-less random_reverb.
    p_reverb_interferer: float = 0.0
    room_reverb: bool = False
    # A second interferer under the first, this much quieter (dB): speech in a noisy room.
    p_second_interferer: float = 0.0
    second_interferer_db: tuple[float, float] = (-25.0, 0.0)
    # Level of the loud parts of every example (dBFS, augment.active_level_db), whatever is in
    # it, instead of the +-gain around each source's own level. None = the old gain.
    level_dbfs: tuple[float, float] | None = None
    # --- Steady background and the noise gate (off by default; the ``*-gate`` presets). ---
    # A room is never silent: mains hum, a fan, a fridge. With this probability an example
    # gets a steady background under it (a `silence` take from the mic when there are any,
    # else augment.synth_background: coloured noise + steady tones) at an RMS level drawn
    # from ``background_dbfs`` and at least 10 dB under the example's loud parts. Ungated, the
    # model has to learn that a tone which never moves is not a hum.
    p_background: float = 0.0
    background_dbfs: tuple[float, float] = (-85.0, -45.0)
    # With this probability the example's frames go through a spectral gate whose noise
    # profile was measured on ``gate_profile_s`` of the same background (floor noise
    # included), as `spectral_gate` / `adaptive_gate` do with a profile of the room. So one
    # model works with and without a gate in its pipeline. The profile is off by a random
    # ``gate_offset_db`` (std, dB: gain drift since it was measured), and with
    # ``p_gate_mismatch`` it was measured with the background ``gate_mismatch_db`` quieter
    # than it is now, so part of it leaks through: no profile fits the room exactly.
    # Needs ``level_dbfs`` (the room recipes) and a front end without filters.
    p_gate: float = 0.0
    gate_profile_s: float = 3.0
    gate_offset_db: float = 1.5
    p_gate_mismatch: float = 0.15
    gate_mismatch_db: tuple[float, float] = (6.0, 20.0)
    gate_n_std: tuple[float, float] = (1.5, 3.0)  # the gate's threshold, drawn per example
    # Drop external positives whose auto-labels cover less of the clip (failed/noisy tracking).
    min_voiced_fraction: float = 0.2
    # peak_tracker params for auto-labelling. Stricter than live detection: on synthetic tests
    # this cut labels on white noise 29% -> 10% and on pseudo-speech 45% -> 5% while keeping
    # 99% whistle recall and raising pitch accuracy 86% -> 97%. `reject_harmonics` keeps a
    # whistle's overtones out of the labels (on synthetic whistles with strong overtones 22-31%
    # of labelled frames were an overtone without it, none with it).
    labeler_params: dict = field(
        default_factory=lambda: {
            "min_prominence": 25.0,
            "min_duration_s": 0.15,
            "reject_harmonics": True,
        }
    )

    def eval_config(self) -> GeneratorConfig:
        """Same mix without the training-only mic-clip augmentation (for validation sets)."""
        return replace(
            self, p_mic_splice=0.0, p_mic_warp=0.0, p_mic_reverse=0.0, p_mic_envelope=0.0
        )

    def __post_init__(self) -> None:
        self.classes = tuple(self.classes)
        self.aux = tuple(self.aux)
        self.speech_audible_db = tuple(self.speech_audible_db)  # type: ignore[assignment]
        if set(self.aux) - set(AUX_CLASSES) or len(set(self.aux)) != len(self.aux):
            raise ValueError(f"aux must be chosen from {AUX_CLASSES}, got {self.aux}")
        if self.level_dbfs is not None:
            self.level_dbfs = tuple(self.level_dbfs)  # type: ignore[assignment]
        elif self.p_background > 0 or self.p_gate > 0:
            raise ValueError("p_background / p_gate need level_dbfs (use a room augment)")
        unknown = set(self.classes) - set(CLASSES)
        if unknown or not self.classes:
            raise ValueError(f"classes must be chosen from {CLASSES}, got {self.classes}")


CLASSES = ("whistle", "hum")  # sounds a detector can be trained on
AUX_CLASSES = ("speech",)  # presence-only outputs (no pitch): see GeneratorConfig.aux

MIX_PRESETS: dict[str, dict] = {
    # only this mic + synthetic (no external data needed)
    "mic": {"p_mic_whistle": 0.45, "p_ext_whistle": 0.0, "p_synth_whistle": 0.25},
    # warm start: external + synthetic; mic data not required
    "pretrain": {"p_mic_whistle": 0.0, "p_ext_whistle": 0.5, "p_synth_whistle": 0.2},
    # adapt a pretrained model to the deployment mic. There are never many mic clips, so the
    # mix stays mostly external (generalisation) with a modest share of mic whistles.
    # Mic clips get extra training-only augmentation to stretch the few takes further.
    "finetune": {
        "p_mic_whistle": 0.15,
        "p_ext_whistle": 0.4,
        "p_synth_whistle": 0.15,
        "p_mic_splice": 0.5,
        "p_mic_warp": 0.8,
        "p_mic_reverse": 0.25,
        "p_mic_envelope": 0.5,
    },
}


# The same mixes for a model that also detects hums: the clean signal is a whistle, a hum or
# nothing, in roughly equal measure. Laid over MIX_PRESETS[mix] when "hum" is a class.
HUM_MIX_PRESETS: dict[str, dict] = {
    "mic": {
        "p_mic_whistle": 0.25,
        "p_synth_whistle": 0.15,
        "p_mic_hum": 0.25,
        "p_synth_hum": 0.15,
    },
    "pretrain": {
        "p_ext_whistle": 0.28,
        "p_synth_whistle": 0.12,
        "p_ext_hum": 0.28,
        "p_synth_hum": 0.12,
    },
    "finetune": {
        "p_mic_whistle": 0.1,
        "p_ext_whistle": 0.2,
        "p_synth_whistle": 0.1,
        "p_mic_hum": 0.1,
        "p_ext_hum": 0.2,
        "p_synth_hum": 0.1,
    },
}


# How sources are put through a channel and a room, laid over any mix (``--augment``).
#   v1       the first recipe: reverb on the clean signal only, mild EQ on the mix.
#   channel  v1 + a random frequency response per source (ablation: the response alone).
#   room     everything: response per source, the same room reverb on clean signal and
#            interferers, noise under speech, one level distribution for every example, and
#            more speech among the interferers (it is the failure that matters).
#   room-mic room, with the deployment mic's own negatives drawn three times as often (as
#            often as all external speech). For the fine-tune, once there are minutes of
#            speech from the room: external speech does not sound like the owner there.
_SPEECH_HEAVY = {
    "mic": 3.0,
    "speech": 6.0,
    "hum": 2.0,
    "noise": 2.0,
    "music": 1.0,
    "noise_dir": 1.0,
    "synthetic": 1.0,
}
AUGMENT_PRESETS: dict[str, dict] = {
    "v1": {},
    "channel": {"p_channel": 0.8},
    "room": {
        "p_channel": 0.8,
        "room_reverb": True,
        "p_reverb": 0.5,
        "p_reverb_interferer": 0.5,
        "p_second_interferer": 0.4,
        "level_dbfs": (-65.0, -10.0),
        "interferer_weights": _SPEECH_HEAVY,
    },
}
AUGMENT_PRESETS["room-mic"] = {
    **AUGMENT_PRESETS["room"],
    "interferer_weights": {**_SPEECH_HEAVY, "mic": 9.0},
}
#   room-gate / room-mic-gate  the same with a steady room background (mains hum, fan...)
#            under 60% of the examples, and half of all examples passed through a spectral
#            gate with a profile of their background: a model that can sit behind
#            `spectral_gate` / `adaptive_gate` or behind no filter at all.
_GATE = {"p_background": 0.6, "p_gate": 0.5}
AUGMENT_PRESETS["room-gate"] = {**AUGMENT_PRESETS["room"], **_GATE}
AUGMENT_PRESETS["room-mic-gate"] = {**AUGMENT_PRESETS["room-mic"], **_GATE}


def config_for_mix(
    mix: str, classes: tuple[str, ...] = ("whistle",), augment: str = "v1", **overrides
) -> GeneratorConfig:
    if mix not in MIX_PRESETS:
        raise KeyError(f"unknown mix {mix!r}; choose from {list(MIX_PRESETS)}")
    if augment not in AUGMENT_PRESETS:
        raise KeyError(f"unknown augment {augment!r}; choose from {list(AUGMENT_PRESETS)}")
    hum = HUM_MIX_PRESETS[mix] if "hum" in classes else {}
    aug = {k: (dict(v) if isinstance(v, dict) else v) for k, v in AUGMENT_PRESETS[augment].items()}
    return GeneratorConfig(
        **{**MIX_PRESETS[mix], **hum, **aug, "classes": tuple(classes), **overrides}
    )


# --------------------------------------------------------------------------- sources


@dataclass
class PositiveClip:
    key: str
    path: str
    f0: np.ndarray  # float16 f0 track at LABEL_RATE (0 = none)


@dataclass
class TrainingSources:
    """Everything a generator needs, picklable for DataLoader workers."""

    mic_positives: list[PositiveClip] = field(default_factory=list)  # whistles
    ext_positives: list[PositiveClip] = field(default_factory=list)
    mic_hums: list[PositiveClip] = field(default_factory=list)  # only when training "hum"
    ext_hums: list[PositiveClip] = field(default_factory=list)
    interferers: dict[str, list[str]] = field(default_factory=dict)  # group -> file paths
    # which interferer files are speech (mic ``speech`` clips, external ``speech`` entries):
    # the source of the ``speech`` aux labels
    speech_files: frozenset[str] = frozenset()
    # mic ``silence`` takes: the room's steady backgrounds (GeneratorConfig.p_background)
    backgrounds: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "mic_whistle_clips": len(self.mic_positives),
            "external_whistle_clips": len(self.ext_positives),
            "mic_hum_clips": len(self.mic_hums),
            "external_hum_clips": len(self.ext_hums),
            **{f"interferers_{k}": len(v) for k, v in self.interferers.items()},
            "speech_files": len(self.speech_files),
            "background_clips": len(self.backgrounds),
        }

    def has_positives(self) -> bool:
        return bool(self.mic_positives or self.ext_positives or self.mic_hums or self.ext_hums)


def frontend_hash(frontend_spec: dict, labeler_params: dict, data_dir: Path | None = None) -> str:
    """Cache key for auto-labels. Includes the contents of any noise profile the frontend uses."""
    blob = json.dumps({"f": frontend_spec, "l": labeler_params}, sort_keys=True)
    if data_dir is not None:
        for f in frontend_spec.get("filters", []):
            for prof in filter_profiles(f):
                p = data_dir / "noise_profiles" / f"{prof}.json"
                if p.exists():
                    blob += hashlib.sha1(p.read_bytes()).hexdigest()
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def events_to_f0(events: list[WhistleEvent], n_samples: int, sr: int) -> np.ndarray:
    """Per-sample f0 (Hz, 0 = none) from event contours (linear interpolation inside events).

    Where events overlap in time the more confident one wins (it is written last)."""
    f0 = np.zeros(n_samples, np.float32)
    for ev in sorted(events, key=lambda e: e.confidence):
        ts = np.array([p.t for p in ev.contour])
        hz = np.array([p.hz for p in ev.contour])
        s, e = max(0, int(ev.start * sr)), min(n_samples, int(ev.end * sr) + 1)
        if e > s:
            f0[s:e] = np.interp(np.arange(s, e) / sr, ts, hz)
    return f0


def auto_label(
    audio: np.ndarray,
    sr: int,
    frontend_spec: dict,
    labeler_params: dict,
    data_dir: Path | None = None,
) -> np.ndarray:
    """Clean audio -> f0 track at LABEL_RATE via the frontend + peak_tracker."""
    fe = Frontend.from_dict(frontend_spec, data_dir)
    if sr != fe.sample_rate:
        audio = resample(audio, sr, fe.sample_rate)
        sr = fe.sample_rate
    _, filt = fe.process_signal(audio)
    det = build_stage("detector", "peak_tracker", labeler_params, StageContext(sr, axis=fe.axis))
    events = det.process(filt).events + det.flush()
    n = int(len(audio) / sr * LABEL_RATE) + 1
    return events_to_f0(events, n, LABEL_RATE).astype(np.float16)


def _label_job(args) -> tuple[str, np.ndarray]:
    key, path, segments, frontend_spec, labeler_params, data_dir = args
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    f0 = auto_label(to_mono_float(audio), sr, frontend_spec, labeler_params, data_dir)
    if segments:  # restrict to annotated whistle segments
        mask = np.zeros(len(f0), bool)
        for s, e in segments:
            mask[int(s * LABEL_RATE) : int(e * LABEL_RATE) + 1] = True
        f0[~mask] = 0
    return key, f0


class LabelCache:
    """f0 tracks per clip key, one .npz per (namespace, frontend+labeler hash)."""

    def __init__(self, directory: Path, namespace: str, fhash: str):
        self.path = directory / f"{namespace}-{fhash}.npz"
        self.labels: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path) as z:
                self.labels = {k: z[k] for k in z.files}

    def ensure(
        self,
        items: list[tuple[str, str, list[tuple[float, float]]]],
        frontend_spec: dict | None = None,
        labeler_params: dict | None = None,
        workers: int = 4,
        data_dir: Path | None = None,
        job=None,
        job_args: tuple = (),
    ) -> dict[str, np.ndarray]:
        """Compute labels for (key, path, segments) items not yet cached.

        By default labels come from ``peak_tracker`` on the front-end (``_label_job``). Another
        labeller passes ``job`` (a picklable ``(key, path, segments, *job_args) -> (key, f0)``).
        """
        todo = [i for i in items if i[0] not in self.labels]
        if todo:
            log.info("auto-labelling %d clips -> %s", len(todo), self.path.name)
            if job is None:
                job, job_args = _label_job, (frontend_spec, labeler_params, data_dir)
            jobs = [(k, p, s, *job_args) for k, p, s in todo]
            if workers > 1:
                with ProcessPoolExecutor(
                    workers, mp_context=_mp_context(), initializer=limit_worker_threads
                ) as pool:
                    for n, (k, f0) in enumerate(pool.map(job, jobs, chunksize=4)):
                        self.labels[k] = f0
                        if (n + 1) % 200 == 0:
                            log.info("  labelled %d/%d", n + 1, len(jobs))
            else:
                for k, f0 in map(job, jobs):
                    self.labels[k] = f0
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp.npz")
            np.savez(tmp, **self.labels)
            tmp.replace(self.path)
        return {k: self.labels[k] for k, _, _ in items}


def collect_sources(
    frontend_spec: dict,
    splits: tuple[str, ...],
    store: RecordingStore | None = None,
    label_cache_dir: Path | None = None,
    external_root: Path | None = None,
    external_datasets: list[str] | None = None,
    noise_dirs: list[Path] | None = None,
    config: GeneratorConfig | None = None,
    workers: int = 4,
    data_dir: Path | None = None,
) -> TrainingSources:
    """Gather + auto-label every source for the given splits."""
    from whistle_house.data.hum_labels import HumLabelConfig, _hum_label_job
    from whistle_house.external.base import POSITIVE_LABELS, get_dataset, load_manifest

    cfg = config or GeneratorConfig()
    fh = frontend_hash(frontend_spec, cfg.labeler_params, data_dir)
    src = TrainingSources()
    speech_files: set[str] = set()
    hum_class = "hum" in cfg.classes  # hums are positives (pYIN-labelled), not interferers
    hum_cfg = HumLabelConfig()

    def hum_clips(cache: LabelCache, items: list) -> list[PositiveClip]:
        labels = cache.ensure(items, workers=workers, job=_hum_label_job, job_args=(hum_cfg,))
        return [PositiveClip(k, p, labels[k]) for k, p, _ in items]

    if store is not None:
        pos_items, hum_items, mic_neg = [], [], []
        for rec in store.list():
            if rec.split not in splits:
                continue
            path = str(store.audio_path(rec.id))
            if rec.label == "whistle":
                segs = [(s.start, s.end) for s in rec.segments if s.label == "whistle"]
                key = rec.id + (
                    ":" + hashlib.sha1(repr(segs).encode()).hexdigest()[:8] if segs else ""
                )
                pos_items.append((key, path, segs))
            elif rec.label == "hum" and hum_class:
                segs = [(s.start, s.end) for s in rec.segments if s.label == "hum"]
                key = rec.id + (
                    ":" + hashlib.sha1(repr(segs).encode()).hexdigest()[:8] if segs else ""
                )
                hum_items.append((key, path, segs))
            elif rec.label == "hum":  # whistle-only model: a hard negative, like external hums
                src.interferers.setdefault("hum", []).append(path)
            elif rec.label in NEGATIVE_LABELS:
                mic_neg.append(path)
                if rec.label == "speech":
                    speech_files.add(path)
                elif rec.label == "silence":
                    src.backgrounds.append(path)
        if pos_items:
            cache = LabelCache(label_cache_dir or Path("."), "mic", fh)
            labels = cache.ensure(pos_items, frontend_spec, cfg.labeler_params, workers, data_dir)
            src.mic_positives = [PositiveClip(k, p, labels[k]) for k, p, _ in pos_items]
        if hum_items:  # same cache (and keys) as the manager's hum pitch overlay
            cache = LabelCache(label_cache_dir or Path("."), "hum", hum_cfg.key())
            src.mic_hums = hum_clips(cache, hum_items)
        if mic_neg:
            src.interferers["mic"] = mic_neg

    if external_root is not None:
        for name in external_datasets or []:
            ds = get_dataset(name, external_root)
            if not ds.is_prepared():
                raise RuntimeError(
                    f"external dataset {name} is not prepared; run `whistle-house datasets fetch {name}`"
                )
            entries = [e for e in load_manifest(ds.prepared_dir) if e.split in splits]
            pos = [e for e in entries if e.label in POSITIVE_LABELS]
            if pos:
                items = [(f"{name}/{e.path}", str(ds.prepared_dir / e.path), []) for e in pos]
                cache = LabelCache(external_root / "labels", name, fh)
                labels = cache.ensure(items, frontend_spec, cfg.labeler_params, workers, data_dir)
                kept = [
                    PositiveClip(k, p, labels[k])
                    for k, p, _ in items
                    if float(np.mean(labels[k] > 0)) >= cfg.min_voiced_fraction
                ]
                log.info("%s: kept %d/%d whistle clips after label QA", name, len(kept), len(items))
                src.ext_positives += kept
            hums = [e for e in entries if e.label == "hum"] if hum_class else []
            if hums:
                items = [(f"{name}/{e.path}", str(ds.prepared_dir / e.path), []) for e in hums]
                cache = LabelCache(external_root / "labels", f"{name}-hum", hum_cfg.key())
                kept = [
                    c
                    for c in hum_clips(cache, items)
                    if float(np.mean(c.f0 > 0)) >= cfg.min_voiced_fraction
                ]
                log.info("%s: kept %d/%d hum clips after label QA", name, len(kept), len(items))
                src.ext_hums += kept
            for e in entries:
                if e.label not in POSITIVE_LABELS and not (hum_class and e.label == "hum"):
                    src.interferers.setdefault(e.label, []).append(str(ds.prepared_dir / e.path))
                    if e.label == "speech":
                        speech_files.add(str(ds.prepared_dir / e.path))

    files = [
        str(p)
        for d in noise_dirs or []
        if d.exists()
        for p in sorted(d.rglob("*"))
        if p.suffix.lower() in AUDIO_EXTS
    ]
    if files:
        src.interferers["noise_dir"] = files
    src.speech_files = frozenset(speech_files)
    return src


# --------------------------------------------------------------------------- audio access


def read_segment(path: str, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Random n-sample segment of a file at sample rate sr (reads only what it needs)."""
    info = sf.info(path)
    n_in = int(np.ceil(n * info.samplerate / sr)) + 16
    start = int(rng.integers(0, max(1, info.frames - n_in + 1)))
    data, _ = sf.read(path, frames=n_in, start=start, dtype="float32", always_2d=False)
    x = resample(to_mono_float(data), info.samplerate, sr)
    return fit_length(x, n, rng)


class _ClipCache:
    def __init__(self, sr: int, size: int = 64):
        self.sr, self.size = sr, size
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str) -> np.ndarray:
        if path in self._d:
            self._d.move_to_end(path)
            return self._d[path]
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        x = resample(to_mono_float(data), sr, self.sr)
        self._d[path] = x
        if len(self._d) > self.size:
            self._d.popitem(last=False)
        return x


# --------------------------------------------------------------------------- speech labels

SPEECH_FRAME_S = 0.02  # resolution of the speech activity labels
# A segment of a speech file quieter than this (dBFS, loud parts, as recorded) is a pause or
# room tone, not speech: its frames are ignored rather than labelled either way.
SPEECH_MIN_LEVEL_DB = -60.0


def _frame_db(x: np.ndarray, hop: int) -> np.ndarray:
    m = len(x) // hop
    e = np.mean(np.square(x[: m * hop].reshape(m, hop), dtype=np.float64), axis=1)
    return 10 * np.log10(e + 1e-14)


def speech_activity(
    x: np.ndarray, sr: int, range_db: float = 30.0, bridge_s: float = 0.3
) -> np.ndarray:
    """Where a speech signal is sounding: bool per ``SPEECH_FRAME_S`` frame.

    A frame is active when its level is within ``range_db`` of the signal's loud parts (95th
    percentile of the frame levels); gaps shorter than ``bridge_s`` between active frames
    (stop consonants, the breath between words) are filled, longer pauses are not speech."""
    db = _frame_db(x, max(1, int(SPEECH_FRAME_S * sr)))
    if not len(db):
        return np.zeros(0, bool)
    on = db >= np.percentile(db, 95) - range_db
    idx = np.flatnonzero(on)
    for a, b in itertools.pairwise(idx):
        if 1 < b - a <= round(bridge_s / SPEECH_FRAME_S):
            on[a:b] = True
    return on


def speech_labels(
    speech: np.ndarray, rest: np.ndarray, sr: int, cfg: GeneratorConfig
) -> np.ndarray:
    """Per-``SPEECH_FRAME_S``-frame speech target for a mix of ``speech`` + ``rest``:
    1 = speech sounding, 0 = not, -1 = ignore (see ``GeneratorConfig.speech_audible_db``)."""
    hop = max(1, int(SPEECH_FRAME_S * sr))
    on = speech_activity(speech, sr, cfg.speech_range_db, cfg.speech_bridge_s)
    y = np.zeros(len(on), np.float32)
    if not on.any():
        return y
    loud = float(np.percentile(_frame_db(speech, hop), 95))
    under = float(np.percentile(_frame_db(rest, hop), 95)) - loud  # dB the speech is under
    label, ignore = cfg.speech_audible_db
    if under <= label:
        y[on] = 1.0
    elif under <= ignore:
        y[on] = -1.0
    return y


# --------------------------------------------------------------------------- generator


class ExampleGenerator:
    def __init__(
        self,
        frontend_spec: dict,
        sources: TrainingSources,
        config: GeneratorConfig | None = None,
        data_dir: Path | None = None,
    ):
        self.cfg = config or GeneratorConfig()
        self.frontend = Frontend.from_dict(frontend_spec, data_dir)
        self.sr = self.frontend.sample_rate
        self.sources = sources
        if self.cfg.p_gate > 0 and self.frontend.filters:
            raise ValueError(
                "gated examples are made by the generator: train on the pipeline without its "
                "filters, then put the gate in the pipeline that uses the model"
            )
        self._clips = _ClipCache(self.sr)
        n_mic = len(sources.interferers.get("mic", []))
        mic_scale = min(1.0, n_mic / max(1, self.cfg.mic_full_weight_clips))
        groups = [
            (g, w * (mic_scale if g == "mic" else 1.0))
            for g, w in self.cfg.interferer_weights.items()
            if w > 0
            and (g == "synthetic" or sources.interferers.get(g))
            and not (g == "hum" and "hum" in self.cfg.classes)  # a positive there, never noise
        ]
        self._groups = [g for g, _ in groups]
        w = np.array([w for _, w in groups], np.float64)
        self._group_p = w / w.sum() if len(w) else w

    @property
    def axis(self):
        return self.frontend.axis

    def _augment_mic_clip(self, audio: np.ndarray, f0: np.ndarray, rng):
        """Training-only variations of a mic whistle/hum clip (no-op unless p_mic_* > 0)."""
        c, sr = self.cfg, self.sr
        if c.p_mic_splice > 0 and rng.random() < c.p_mic_splice:
            audio, f0 = splice_notes(audio, f0, sr, LABEL_RATE, rng)
        if c.p_mic_warp > 0 and rng.random() < c.p_mic_warp:
            freqs = self.frontend.axis.freqs  # keep labels on the frequency axis (5% margin)
            bounds = (float(freqs[0]) * 1.05, float(freqs[-1]) / 1.05)
            audio, f0 = warp_clip(
                audio, f0, sr, LABEL_RATE, rng, c.mic_warp_semitones, f0_bounds=bounds
            )
        if c.p_mic_reverse > 0 and rng.random() < c.p_mic_reverse:
            audio, f0 = reverse_clip(audio, f0)
        if c.p_mic_envelope > 0 and rng.random() < c.p_mic_envelope:
            audio = envelope_clip(audio, sr, rng)
        return audio, f0

    def _positive(
        self, clip: PositiveClip, n: int, rng, mic: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        audio = self._clips.get(clip.path)
        f0_full = clip.f0.astype(np.float32)
        if mic:
            audio, f0_full = self._augment_mic_clip(audio, f0_full, rng)
        # Pick a window, preferring ones that contain the labelled sound.
        L = len(audio)
        off = 0
        for _ in range(4):
            off = int(rng.integers(-n // 4, max(1, L - n + n // 4)))
            lo = max(0, off) * LABEL_RATE // self.sr
            hi = max(0, off + n) * LABEL_RATE // self.sr
            if np.any(f0_full[lo:hi] > 0):
                break
        clean = np.zeros(n, np.float32)
        s0, d0 = max(off, 0), max(-off, 0)
        seg = audio[s0 : s0 + n - d0]
        clean[d0 : d0 + len(seg)] = seg
        t = (np.arange(n) + off) / self.sr
        idx = np.clip((t * LABEL_RATE).astype(int), 0, len(f0_full) - 1)
        f0 = np.where((t >= 0) & (t < L / self.sr), f0_full[idx], 0).astype(np.float32)
        return clean, f0

    def _interferer(self, n: int, rng) -> tuple[np.ndarray, bool | None]:
        """(audio, is it speech). None = can't say: synthetic pseudo-speech (speech-shaped on
        purpose, but not speech), or a stretch of a speech file with nobody talking."""
        if not self._groups:
            return synth_pseudo_speech(self.sr, n / self.sr, rng), None
        g = self._groups[int(rng.choice(len(self._groups), p=self._group_p))]
        if g == "synthetic":
            return synth_pseudo_speech(self.sr, n / self.sr, rng), None
        files = self.sources.interferers[g]
        for _ in range(3):
            path = files[int(rng.integers(len(files)))]
            try:
                x = read_segment(path, n, self.sr, rng)
            except Exception as e:  # noqa: BLE001 - one bad file must not stop training
                log.warning("bad interferer file: %s", e)
                continue
            if rms(x) > 1e-5:
                if path not in self.sources.speech_files:
                    return x, False
                return x, (True if active_level_db(x, self.sr) >= SPEECH_MIN_LEVEL_DB else None)
        return synth_pseudo_speech(self.sr, n / self.sr, rng), None

    def _room_interferer(self, n: int, rng) -> tuple[np.ndarray, np.ndarray | None, bool]:
        """An interferer as it would reach the mic: optionally with a second, quieter one under
        it, reverberated and coloured like any other source (all off unless configured).

        Returns (audio, the part of it that is speech or None, whether some part may or may not
        be speech). The speech part is what the ``speech`` labels are made from."""
        c, sr = self.cfg, self.sr

        def one() -> tuple[np.ndarray, bool | None]:
            x, is_speech = self._interferer(n, rng)
            if c.p_reverb_interferer > 0 and rng.random() < c.p_reverb_interferer:
                x = (room_reverb if c.room_reverb else random_reverb)(x, sr, rng)
            if c.p_channel > 0 and rng.random() < c.p_channel:
                x = random_channel(x, sr, rng)
            return x, is_speech

        noise, is_speech = one()
        speech = noise if is_speech else None
        unsure = is_speech is None
        if c.p_second_interferer > 0 and rng.random() < c.p_second_interferer:
            other, is_speech = one()
            gain = 10 ** (rng.uniform(*c.second_interferer_db) / 20)
            other = other * gain * rms(noise) / rms(other)
            noise = noise + other
            if is_speech:
                speech = other if speech is None else speech + other
            unsure = unsure or is_speech is None
        return (
            noise.astype(np.float32),
            None if speech is None else speech.astype(np.float32),
            unsure,
        )

    def _steady(self, m: int, floor_db: float, level_db: float, rng) -> np.ndarray:
        """The steady part of the room for ``m`` samples, as (2, m): the mic's own noise at
        ``floor_db`` and, with ``p_background``, a room background (zeros without one)."""
        c, sr = self.cfg, self.sr
        out = np.zeros((2, m), np.float32)
        out[0] = 10 ** (floor_db / 20) * rng.standard_normal(m)
        if rng.random() < c.p_background:
            bg = None
            files = self.sources.backgrounds
            if files and rng.random() < 0.5:
                try:
                    bg = read_segment(files[int(rng.integers(len(files)))], m, sr, rng)
                except Exception as e:  # noqa: BLE001 - one bad file must not stop training
                    log.warning("bad background file: %s", e)
            if bg is None or rms(bg) < 1e-6:
                bg = synth_background(sr, m, rng)
            lo, hi = c.background_dbfs
            db = rng.uniform(lo, max(lo, min(hi, level_db - 10.0)))
            out[1] = bg * 10 ** (db / 20) / rms(bg)
        return out

    def _gate(self, block, steady: np.ndarray, rng) -> np.ndarray:
        """``block`` through a spectral gate whose profile is measured on ``steady`` (the
        stretch of the example's floor noise and background just before it)."""
        c = self.cfg
        floor, bg = steady
        if bg.any() and rng.random() < c.p_gate_mismatch:
            bg = bg * 10 ** (-rng.uniform(*c.gate_mismatch_db) / 20)
        _, prof = self.frontend.process_signal((floor + bg).astype(np.float32))
        v = prof.values[8:]  # the first frames still hold the transform's start-up
        gate = SpectralGate(
            SpectralGate.Config(n_std=float(rng.uniform(*c.gate_n_std))),
            StageContext(self.sr, axis=block.axis),
        )
        gate.set_profile(
            v.mean(axis=0) + rng.normal(0.0, c.gate_offset_db), np.maximum(v.std(axis=0), 0.5)
        )
        with np.errstate(over="ignore"):  # exp() of a bin far under the threshold: mask 0
            return gate.process(block).values

    def _clean(self, n: int, rng: np.random.Generator) -> tuple[str | None, np.ndarray, np.ndarray]:
        """Pick the example's clean signal: (class or None, audio (n,), per-sample f0 (n,))."""
        c, s = self.cfg, self.sources
        dur = c.window_s + c.warmup_s
        table = []  # (class, probability, clips, from mic, synthesiser)
        for kind, mic, ext, p, synth in (
            ("whistle", s.mic_positives, s.ext_positives, c.p_mic_whistle, synth_whistle),
            ("hum", s.mic_hums, s.ext_hums, c.p_mic_hum, synth_hum),
        ):
            if kind not in c.classes:
                continue
            p_ext, p_syn = (
                (c.p_ext_whistle, c.p_synth_whistle)
                if kind == "whistle"
                else (c.p_ext_hum, c.p_synth_hum)
            )
            # unavailable sources hand their share to the synthetic version of the same sound
            table += [
                (kind, p if mic else 0.0, mic, True, None),
                (kind, p_ext if ext else 0.0, ext, False, None),
                (kind, p_syn + (0.0 if mic else p) + (0.0 if ext else p_ext), None, False, synth),
            ]
        u = rng.random()
        for kind, p, clips, is_mic, synth in table:
            if u < p:
                if synth is synth_whistle:
                    sw = synth_whistle(
                        self.sr, dur, rng, overtones=rng.random() < c.p_whistle_overtones
                    )
                    return kind, sw.audio, sw.f0
                if synth is not None:
                    sw = synth(self.sr, dur, rng)
                    return kind, sw.audio, sw.f0
                clip = clips[int(rng.integers(len(clips)))]
                return (kind, *self._positive(clip, n, rng, mic=is_mic))
            u -= p
        return None, np.zeros(n, np.float32), np.zeros(n, np.float32)

    def example(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """(features (T, F), labels (T, K + A)): one f0 column per class in ``cfg.classes``,
        then one presence column (1 / 0 / -1 = ignore) per name in ``cfg.aux``."""
        c, sr = self.cfg, self.sr
        n = int((c.window_s + c.warmup_s) * sr)
        kind, clean, f0 = self._clean(n, rng)

        reverb = room_reverb if c.room_reverb else random_reverb
        if c.p_saturate > 0 and clean.any() and rng.random() < c.p_saturate:
            clean = saturate(clean, rng)
        if rng.random() < c.p_reverb and clean.any():
            clean = reverb(clean, sr, rng)
        if c.p_channel > 0 and rng.random() < c.p_channel:
            clean = random_channel(clean, sr, rng)
        has_signal = rms(clean) > 1e-4
        speech_y = None  # per SPEECH_FRAME_S frame; None = no speech anywhere in the example
        steady, n_prof = None, 0  # floor noise + room background (the *-gate recipes)
        if rng.random() < c.p_interferer or not has_signal:
            raw, speech, unsure = self._room_interferer(n, rng)
            noise = random_gain(raw, rng, -30, 0)
            mix = mix_at_snr(clean, noise, rng.uniform(*c.snr_db)) if has_signal else noise
            if "speech" in c.aux and unsure:
                speech_y = np.full(n // max(1, int(SPEECH_FRAME_S * sr)), -1.0, np.float32)
            elif "speech" in c.aux and speech is not None:
                # the interferer was scaled as a whole (gain, then to the SNR): by how much?
                in_mix = (mix - clean) if has_signal else mix
                k = float(np.dot(in_mix, raw) / max(float(np.dot(raw, raw)), 1e-20))
                speech_y = speech_labels(k * speech, mix - k * speech, sr, c)
        else:
            mix = clean
        if c.level_dbfs is None:
            mix = mix + 10 ** rng.uniform(-5, -3) * rng.standard_normal(n).astype(np.float32)
        if rng.random() < c.p_eq:
            mix = random_eq(mix, sr, rng)
        if c.p_channel > 0 and rng.random() < 0.5 * c.p_channel:
            mix = random_channel(mix, sr, rng)  # the microphone: the same for every source
        if c.level_dbfs is None:
            mix = np.clip(random_gain(mix, rng, -12, 6), -1, 1).astype(np.float32)
        else:
            # One level distribution whatever the example holds, then the mic's own noise:
            # at least 20 dB under the loud parts, so a labelled sound is never buried in it.
            level = rng.uniform(*c.level_dbfs)
            mix = mix * 10 ** ((level - active_level_db(mix, sr)) / 20)
            floor = rng.uniform(-95.0, min(-60.0, level - 20.0))
            if c.p_background > 0 or c.p_gate > 0:
                # the room's steady part, with a stretch before the example for the profile
                n_prof = int(c.gate_profile_s * sr)
                steady = self._steady(n_prof + n, floor, level, rng)
                mix = mix + steady[:, n_prof:].sum(axis=0)
            else:
                mix = mix + 10 ** (floor / 20) * rng.standard_normal(n)
            mix = np.clip(mix, -1, 1).astype(np.float32)

        _, block = self.frontend.process_signal(mix)
        values = block.values
        if steady is not None and rng.random() < c.p_gate:
            values = self._gate(block, steady[:, :n_prof], rng)
        keep = block.times >= c.warmup_s
        idx = np.clip(np.round(block.times[keep] * sr).astype(int), 0, n - 1)
        y = np.zeros((len(idx), len(c.classes) + len(c.aux)), np.float32)
        if kind is not None:
            y[:, c.classes.index(kind)] = f0[idx]
        if speech_y is not None:
            hop = max(1, int(SPEECH_FRAME_S * sr))
            y[:, len(c.classes) + c.aux.index("speech")] = speech_y[
                np.clip(idx // hop, 0, len(speech_y) - 1)
            ]
        return values[keep], y

    def batch(self, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = zip(*(self.example(rng) for _ in range(size)), strict=True)
        t = min(x.shape[0] for x in xs)
        return np.stack([x[:t] for x in xs]), np.stack([y[:t] for y in ys])


def f0_to_bins(f0: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Hz -> fractional bin index on the axis (NaN where f0 == 0 or out of range)."""
    b = np.interp(f0, freqs, np.arange(len(freqs)), left=np.nan, right=np.nan)
    b[f0 <= 0] = np.nan
    return b
