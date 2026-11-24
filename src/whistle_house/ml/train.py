"""Train WhistleNet (whistle, or whistle + hum) on generated (augmented) examples.

Training is meant to run on a workstation/GPU box (clone the repo + copy ``data/`` there), not
the Pi; the Pi only runs inference. ``--device auto`` picks CUDA/MPS when available.

Run via ``whistle-house train-detector`` (or ``whistle-house warmstart`` for the full
external-data recipe) or the manager's Models page. Progress is written to
``<data_dir>/models/<name>.json`` after every epoch so the manager can poll it.

Examples are generated on the fly (augmentation is random every time) by ``workers`` DataLoader
processes, each with its own ``ExampleGenerator``; the main process only runs the model.

Validation (once per epoch, so short epochs = frequent validation) uses fixed example sets:

* ``val``: generated from the *val* splits (speaker-disjoint external clips, held-out mic clips,
  synthetic). This is the generalisation number; the best checkpoint is chosen on its F1.
* ``mic_train``: fresh augmentations of the mic whistle clips being trained on. With only a
  handful of mic clips this keeps rising while ``val`` stalls or drops: that gap
  (``overfit_gap``) is the overfitting signal.
* ``mic_val``: held-out mic whistle clips only, when any exist.

A model with several classes (the ``harmonic`` arch: whistle + hum) reports every metric per
class (``whistle_f1``, ``hum_false_positive_rate``, ...); the unprefixed ones (``f1`` etc.) are
the mean over classes, so checkpoint selection and the overfitting signals work the same way.

``aux`` (``--aux speech``) adds presence-only outputs: a ``speech`` head trained on where the
speech interferers sound (``data/training.speech_labels``; frames labelled -1 are left out of
its loss). Its metrics are reported as ``speech_f1`` etc. and are not part of the mean that
picks the checkpoint. A fine-tune keeps the outputs of its checkpoint and can add the head to
one that has none (``model.add_aux``).

When fine-tuning (``init_from``) the starting checkpoint is validated first as epoch 0 and is
the initial "best", so a fine-tune can never return something worse on ``val`` than it started
from. ``patience`` stops a run once ``val`` F1 has not improved for that many validations.

Resuming: after every validation the full training state (model, optimiser, LR schedule,
epoch, best F1, patience counter, history) is written atomically to
``models/<name>.resume.ckpt``, next to the best checkpoint ``<name>.pt``. ``train(...,
resume=True)`` (CLI ``--resume``) continues from it after a crash or a stop, at the next
epoch, and refuses if the run's settings changed; a run that already finished is not
repeated. The example stream gets a new seed per resume so data is not replayed. Validation
sets are rebuilt from the seed, so they are the same as before as long as ``data/`` and the
external datasets have not changed.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from whistle_house.data.store import RecordingStore
from whistle_house.data.training import (
    ExampleGenerator,
    GeneratorConfig,
    TrainingSources,
    _mp_context,
    collect_sources,
    config_for_mix,
    f0_to_bins,
)
from whistle_house.ml.model import (
    ARCHS,
    WhistleNet,
    WhistleNetConfig,
    add_aux,
    config_from_dict,
    load_checkpoint,
    log_bins_per_octave,
    save_checkpoint,
)
from whistle_house.ml.tracking import Tracker
from whistle_house.pipeline import Frontend, PipelineSpec
from whistle_house.settings import Paths, external_root
from whistle_house.threads import limit_worker_threads

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 20
    steps_per_epoch: int = 100
    batch_size: int = 16
    lr: float = 2e-3
    val_examples: int = 64
    seed: int = 0
    channels: int = 16
    # ml/model.ARCHS: small (Pi model) | large (reference, with lookahead) | harmonic (whistle +
    # hum, needs a log-frequency front end)
    arch: str = "small"
    # sounds to detect (data/training.CLASSES); empty = the arch's own (whistle, or whistle+hum)
    classes: list[str] = field(default_factory=list)
    # presence-only outputs to add (data/training.AUX_CLASSES: speech); empty = none, or the
    # ones the init checkpoint has
    aux: list[str] = field(default_factory=list)
    aux_weight: float = 1.0
    salience_weight: float = 1.0
    device: str = "auto"  # auto | cpu | cuda | mps
    mix: str = "mic"  # data mix preset: mic | pretrain | finetune (data/training.MIX_PRESETS)
    # channel / room simulation laid over the mix: v1 | channel | room (AUGMENT_PRESETS)
    augment: str = "v1"
    external: list[str] = field(default_factory=list)  # external dataset names to use
    init_from: str | None = None  # checkpoint to fine-tune from (path or name in models/)
    patience: int = 0  # stop after this many validations without a better val F1 (0 = never)
    # Which epoch becomes <name>.pt: "best" (highest val F1) or "last". val is mostly external
    # data, so on a mic fine-tune "best" is often the init checkpoint or an epoch from the LR
    # warm-up that has hardly moved; "last" with a schedule that runs to its end (patience 0)
    # keeps what the fine-tune learned. Then judge it on recordings, not on val.
    keep: str = "best"
    workers: int = 4  # example-generation processes (0 = in the main process)
    wandb_project: str | None = None  # log to this Weights & Biases project (ml/tracking.py)
    wandb_group: str | None = None  # wandb group (warmstart: pretrain + fine-tune together)
    log_every: int = 10  # wandb: log the training loss every this many steps


def _window_seconds(
    net: WhistleNetConfig, spec: PipelineSpec, data_dir: Path, default: float
) -> float:
    """Length of the training examples for a model: the mix's own (3 s) unless the model's
    receptive field needs more.

    A window starts from silence, like a stream, so only frames further in than the receptive
    field are trained with their whole context. Windows are at least 1.5x the receptive field
    (to the next half second). Every arch up to ``harmonic_deep`` (1.45 s) keeps 3 s."""
    fe = Frontend(spec.sample_rate, spec.transform, spec.filters, data_dir)
    times = fe.transform.push(np.zeros(spec.sample_rate, np.float32)).times
    period = float(np.median(np.diff(times)))
    need = 1.5 * net.receptive_field * period
    return max(default, float(np.ceil(need * 2) / 2))


class _ExampleStream(torch.utils.data.IterableDataset):
    """Endless stream of generated batches; every worker gets its own RNG + generator."""

    def __init__(
        self,
        frontend: dict,
        sources: TrainingSources,
        gcfg: GeneratorConfig,
        bs: int,
        seed: int,
        data_dir: Path | None = None,
    ):
        super().__init__()
        self.args = (frontend, sources, gcfg, data_dir)
        self.bs, self.seed = bs, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        gen = ExampleGenerator(*self.args)
        rng = np.random.default_rng([self.seed, wid])
        while True:
            x, f0 = gen.batch(rng, self.bs)
            yield x.astype(np.float32), f0.astype(np.float32)


def _targets(f0: np.ndarray, freqs: np.ndarray, sigma_bins: float = 1.0):
    """f0 (B, T, K) -> presence (B, T, K) {0,1}, salience (B, T, K, F) gaussian bumps."""
    bins = f0_to_bins(f0, freqs)
    pres = (~np.isnan(bins)).astype(np.float32)
    grid = np.arange(len(freqs), dtype=np.float32)
    centre = np.nan_to_num(bins, nan=-1e3).astype(np.float32)[..., None]
    sal = np.exp(-0.5 * ((grid - centre) / sigma_bins) ** 2)
    return pres, (sal * pres[..., None]).astype(np.float32)


def _pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _class_metrics(p: np.ndarray, est_hz: np.ndarray, f0: np.ndarray) -> dict:
    """Frame metrics for one class: p (B, T) predicted present, est_hz / f0 (B, T) pitches."""
    t = f0 > 0
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    both = p & t
    st = 12 * np.log2(est_hz[both] / f0[both]) if both.any() else np.array([])
    pitch_ok = np.abs(st) <= 1.0
    # reported pitch is the 2nd-4th harmonic of the labelled one (within 0.7 semitone)
    overtone = sum(np.abs(st - 12 * np.log2(k)) <= 0.7 for k in (2, 3, 4)) > 0
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
        "false_positive_rate": float(fp / max((~t).sum(), 1)),
        "pitch_acc_1st": float(pitch_ok.mean()) if pitch_ok.size else 0.0,
        "overtone_rate": float(np.mean(overtone)) if st.size else 0.0,
    }


def _evaluate(
    model: WhistleNet, x: np.ndarray, f0: np.ndarray, freqs: np.ndarray, norm, dev
) -> dict:
    """x (B, T, F), f0 (B, T, K + A) -> metrics. With one class: that class's metrics. With
    several: the mean over classes under the plain names, plus ``<class>_<metric>`` for each.
    The A presence-only outputs (``cfg.aux``; labels 1 / 0 / -1 = ignore) add
    ``<name>_precision`` / ``_recall`` / ``_f1`` / ``_false_positive_rate``, outside the mean."""
    model.eval()
    pres_parts, bin_parts = [], []
    with torch.no_grad():
        for i in range(0, len(x), 32):  # chunked: a big val set must not blow up GPU memory
            pres, sal = model(torch.from_numpy((x[i : i + 32] - norm[0]) / norm[1]).to(dev))
            pres_parts.append(torch.sigmoid(pres).cpu().numpy())
            bin_parts.append(sal.argmax(-1).cpu().numpy())
    model.train()
    p = np.concatenate(pres_parts) >= 0.5  # (B, T, K)
    est_hz = freqs[np.concatenate(bin_parts)]
    classes = model.cfg.classes
    per = [_class_metrics(p[..., k], est_hz[..., k], f0[..., k]) for k in range(len(classes))]
    if len(classes) == 1:
        out = per[0]
    else:
        out = {key: float(np.mean([m[key] for m in per])) for key in per[0]}
        for name, m in zip(classes, per, strict=True):
            out.update({f"{name}_{key}": v for key, v in m.items()})
    for a, name in enumerate(model.cfg.aux, start=len(classes)):
        out.update(
            {f"{name}_{key}": v for key, v in _presence_metrics(p[..., a], f0[..., a]).items()}
        )
    return out


def _presence_metrics(p: np.ndarray, target: np.ndarray) -> dict:
    """Frame metrics of a presence-only output: p predicted, target 1 / 0 / -1 (= ignored)."""
    ok, t = target >= 0, target > 0.5
    tp, fp, fn = (p & t & ok).sum(), (p & ~t & ok).sum(), (~p & t & ok).sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
        "false_positive_rate": float(fp / max((~t & ok).sum(), 1)),
    }


def _resolve_checkpoint(ref: str, paths: Paths) -> Path:
    p = Path(ref)
    if p.exists():
        return p
    p = paths.models / (ref if ref.endswith(".pt") else f"{ref}.pt")
    if not p.exists():
        raise FileNotFoundError(f"checkpoint {ref!r} not found")
    return p


# TrainConfig fields that may differ when a run is resumed (they don't change what is learned)
_RESUME_FREE = {"device", "workers", "patience", "wandb_project", "wandb_group", "log_every"}


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load_resume(path: Path, frontend: dict, cfg: TrainConfig) -> dict:
    rs = torch.load(path, map_location="cpu", weights_only=False)
    if rs["frontend"] != frontend:
        raise ValueError(f"{path}: the run was started on a different front-end; can't resume")
    # a state written before a setting existed ran with that setting's default
    old, new = {**asdict(TrainConfig()), **rs["status"]["config"]}, asdict(cfg)
    changed = sorted(k for k in new if k not in _RESUME_FREE and old.get(k) != new[k])
    if changed:
        diff = ", ".join(f"{k}: {old.get(k)!r} -> {new[k]!r}" for k in changed)
        raise ValueError(f"{path}: settings differ from the interrupted run ({diff})")
    return rs


def train(
    spec: PipelineSpec, paths: Paths, name: str, cfg: TrainConfig, resume: bool = False
) -> Path:
    """Train ``name`` and return its best checkpoint. ``resume``: continue an interrupted run
    from its resume state if there is one (else start fresh); return at once if it finished."""
    if cfg.keep not in ("best", "last"):
        raise ValueError(f"keep must be 'best' or 'last', got {cfg.keep!r}")
    torch.manual_seed(cfg.seed)
    status_path = paths.models / f"{name}.json"
    ckpt_path = paths.models / f"{name}.pt"
    state_path = paths.resume_state(name)
    paths.models.mkdir(parents=True, exist_ok=True)
    frontend = spec.frontend_dict()
    rs = _load_resume(state_path, frontend, cfg) if resume and state_path.exists() else None
    if rs is not None and rs["done"]:
        log.info("%s already finished (%s); nothing to resume", name, state_path)
        return ckpt_path
    if rs is not None:
        status = rs["status"]
        status["state"] = "preparing"
        status["config"] = asdict(cfg)
        status.setdefault("resumed", []).append({"epoch": rs["epoch"], "time": time.time()})
        status.pop("error", None)
        log.info("resuming %s after epoch %d (best f1 %.4f)", name, rs["epoch"], rs["best"])
    else:
        state_path.unlink(missing_ok=True)  # a fresh run: an older run's state must not be resumed
        status = {
            "name": name,
            "state": "preparing",
            "pipeline": spec.name,
            "frontend": frontend,
            "config": asdict(cfg),
            "history": [],
            "started": time.time(),
        }

    def write_status() -> None:
        status_path.write_text(json.dumps(status, indent=2))

    write_status()
    tracker = Tracker()  # wandb, once the model is built (off without cfg.wandb_project)
    try:
        init = None  # (path, model, checkpoint) when fine-tuning
        if cfg.init_from:
            init_path = _resolve_checkpoint(cfg.init_from, paths)
            init = (init_path, *load_checkpoint(init_path))
            if init[2]["frontend"] != frontend:
                raise ValueError(
                    f"{init_path} was trained on a different front-end; fine-tune with the "
                    f"pipeline it came from ({init[2]['frontend']})"
                )
            classes = tuple(init[1].cfg.classes)  # a fine-tune keeps the checkpoint's outputs
            if cfg.aux:  # ... and may add presence-only ones to a checkpoint without any
                init = (init[0], add_aux(init[1], tuple(cfg.aux)), init[2])
            aux = tuple(init[1].cfg.aux)
        else:
            if cfg.arch not in ARCHS:
                raise ValueError(f"unknown arch {cfg.arch!r}; choose from {sorted(ARCHS)}")
            classes = tuple(cfg.classes) or tuple(ARCHS[cfg.arch].get("classes", ("whistle",)))
            aux = tuple(cfg.aux)
        status["classes"] = list(classes)
        status["aux"] = list(aux)
        net_cfg = (
            init[1].cfg
            if init is not None
            else WhistleNetConfig(**{"channels": cfg.channels, **ARCHS[cfg.arch]})
        )
        gcfg = config_for_mix(cfg.mix, classes, cfg.augment, aux=aux)
        window_s = _window_seconds(net_cfg, spec, paths.root, gcfg.window_s)
        gcfg = replace(gcfg, window_s=window_s)
        status["window_s"] = window_s
        common = {
            "store": RecordingStore(paths.recordings),
            "label_cache_dir": paths.label_cache,
            "external_root": external_root(required=bool(cfg.external)),
            "external_datasets": cfg.external,
            "noise_dirs": [paths.noise],
            "config": gcfg,
            "workers": max(1, cfg.workers),
            "data_dir": paths.root,
        }
        train_src = collect_sources(frontend, ("train",), **common)
        val_src = collect_sources(frontend, ("val",), **common)
        if not val_src.has_positives():
            val_src = train_src  # early days: validate on fresh draws from train sources
        status["data"] = {"train": train_src.summary(), "val": val_src.summary()}
        write_status()

        gen = ExampleGenerator(frontend, train_src, gcfg, paths.root)
        freqs = gen.axis.freqs
        ecfg = gcfg.eval_config()  # validation: no training-only mic-clip augmentation
        vx, vf0 = ExampleGenerator(frontend, val_src, ecfg, paths.root).batch(
            np.random.default_rng(cfg.seed + 1), cfg.val_examples
        )
        vx = vx.astype(np.float32)
        # Probe sets made only of mic whistle/hum clips (+ the usual interferers/augmentation):
        # the clips being trained on, and held-out mic clips when there are any.
        probes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if gcfg.p_mic_whistle > 0 or gcfg.p_mic_hum > 0:
            n_probe = max(4, cfg.val_examples // 2)
            for key, psrc in (("mic_train", train_src), ("mic_val", val_src)):
                if key == "mic_val" and val_src is train_src:
                    continue
                kinds = [
                    k
                    for k, clips in (("whistle", psrc.mic_positives), ("hum", psrc.mic_hums))
                    if clips and k in classes
                ]
                if not kinds:
                    continue
                share = 0.7 / len(kinds)
                probe_cfg = replace(
                    ecfg,
                    p_mic_whistle=share if "whistle" in kinds else 0.0,
                    p_mic_hum=share if "hum" in kinds else 0.0,
                    p_ext_whistle=0.0,
                    p_synth_whistle=0.0,
                    p_ext_hum=0.0,
                    p_synth_hum=0.0,
                )
                px, pf0 = ExampleGenerator(frontend, psrc, probe_cfg, paths.root).batch(
                    np.random.default_rng(cfg.seed + 2), n_probe
                )
                probes[key] = (px.astype(np.float32), pf0)

        def validate() -> dict:
            m = _evaluate(model, vx, vf0, freqs, norm, dev)
            for key, (px, pf0) in probes.items():
                pm = _evaluate(model, px, pf0, freqs, norm, dev)
                m.update({f"{key}_{k}": v for k, v in pm.items()})
            if "mic_train_f1" in m:
                m["overfit_gap"] = m["mic_train_f1"] - m["f1"]
            return m

        dev = _pick_device(cfg.device)
        status["device"] = str(dev)
        if rs is not None:  # same architecture and normalisation as the interrupted run
            model = WhistleNet(config_from_dict(rs["arch"]))
            model.load_state_dict(rs["model"])
            norm = tuple(rs["norm"])
        elif init is not None:
            init_path, model, ck = init
            norm = (ck["norm"]["mean"], ck["norm"]["std"])
            status["init_from"] = str(init_path)
        else:
            xs, _ = gen.batch(np.random.default_rng(cfg.seed), 16)
            norm = (float(xs.mean()), float(xs.std() + 1e-6))
            arch = {"channels": cfg.channels, **ARCHS[cfg.arch], "classes": classes, "aux": aux}
            if arch.get("harmonics"):  # harmonic input: multiples are bin shifts on a log axis
                arch["bins_per_octave"] = log_bins_per_octave(freqs)
            model = WhistleNet(WhistleNetConfig(n_bins=len(freqs), **arch))
        model = model.to(dev)
        model.train()
        start_epoch = rs["epoch"] if rs is not None else 0

        # a resumed run draws new examples rather than replaying the ones it already saw
        stream_seed = cfg.seed + 1_000_003 * start_epoch
        stream = _ExampleStream(frontend, train_src, gcfg, cfg.batch_size, stream_seed, paths.root)
        loader = torch.utils.data.DataLoader(
            stream,
            batch_size=None,
            num_workers=cfg.workers,
            prefetch_factor=4 if cfg.workers else None,
            persistent_workers=cfg.workers > 0,
            multiprocessing_context=_mp_context() if cfg.workers else None,
            worker_init_fn=limit_worker_threads if cfg.workers else None,
        )
        batches = iter(loader)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=cfg.epochs * cfg.steps_per_epoch
        )
        best, stale = -1.0, 0
        if rs is not None:
            opt.load_state_dict(rs["opt"])
            sched.load_state_dict(rs["sched"])
            best, stale = rs["best"], rs["stale"]
            torch.set_rng_state(rs["torch_rng"])
        status["state"] = "training"
        status["params"] = sum(p.numel() for p in model.parameters())
        prev = status.get("wandb") or {}
        tracker = Tracker.start(
            cfg.wandb_project,
            name,
            cfg.wandb_group,
            {
                **asdict(cfg),
                "pipeline": spec.name,
                "frontend": frontend,
                "classes": list(classes),
                "aux": list(aux),
                "params": status["params"],
                "data": status["data"],
            },
            run_id=prev.get("id") if prev.get("project") == cfg.wandb_project else None,
        )
        status["wandb"] = tracker.info or prev or None
        write_status()
        spe = cfg.steps_per_epoch

        def log_validation(metrics: dict) -> None:
            e = metrics["epoch"]
            tracker.log(e * spe, {k: v for k, v in metrics.items() if k != "epoch"}, "val/")
            tracker.log(e * spe, {"epoch": e})
            if status.get("best") is metrics:
                tracker.summary({"best_f1": best, "best_epoch": e})

        def save_state(epoch: int, done: bool = False) -> None:
            """Everything needed to continue after ``epoch`` (atomic: a crash mid-write keeps
            the previous state)."""
            _atomic_save(
                {
                    "epoch": epoch,
                    "done": done,
                    "arch": asdict(model.cfg),
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "best": best,
                    "stale": stale,
                    "norm": list(norm),
                    "frontend": frontend,
                    "torch_rng": torch.get_rng_state(),
                    "status": status,
                },
                state_path,
            )

        def save_model(metrics: dict) -> None:
            meta = {
                "metrics": metrics,
                "mix": cfg.mix,
                "augment": cfg.augment,
                "external": cfg.external,
                "data": status["data"],
            }
            save_checkpoint(ckpt_path, model.cpu(), norm, frontend, meta)
            model.to(dev)

        def keep_if_best(metrics: dict) -> bool:
            nonlocal best
            if metrics["f1"] <= best:
                return False
            best = metrics["f1"]
            status["best"] = metrics
            save_model(metrics)
            return True

        if cfg.init_from and rs is None:  # baseline: the checkpoint we start from, untouched
            metrics = validate()
            metrics.update(epoch=0, loss=None, seconds=0.0)
            status["history"].append(metrics)
            log.info("epoch 0 (init checkpoint) %s", metrics)
            keep_if_best(metrics)
            log_validation(metrics)
            write_status()
            save_state(0)
        stopped = cfg.patience and stale >= cfg.patience  # resumed after its last epoch
        for epoch in range(start_epoch, 0 if stopped else cfg.epochs):
            t0, losses = time.time(), []
            for i in range(cfg.steps_per_epoch):
                x, f0 = next(batches)
                x, f0 = x.numpy(), f0.numpy()
                nk = len(classes)
                pres_t, sal_t = _targets(f0[..., :nk], freqs)
                x_t = torch.from_numpy((x - norm[0]) / norm[1]).float().to(dev)
                pres, sal = model(x_t)
                pres_t, sal_t = torch.from_numpy(pres_t).to(dev), torch.from_numpy(sal_t).to(dev)
                # every class: presence everywhere, pitch only in the frames where it sounds
                loss_p = F.binary_cross_entropy_with_logits(pres[..., :nk], pres_t)
                mask = pres_t[..., None].expand_as(sal)
                loss_s = (
                    F.binary_cross_entropy_with_logits(sal, sal_t, reduction="none") * mask
                ).sum() / (mask.sum() + 1)
                loss = loss_p + cfg.salience_weight * loss_s
                loss_a = None
                if aux:  # presence-only outputs: every frame whose label is not "ignore" (-1)
                    aux_t = torch.from_numpy(f0[..., nk:]).to(dev)
                    known = (aux_t >= 0).float()
                    loss_a = (
                        F.binary_cross_entropy_with_logits(
                            pres[..., nk:], aux_t.clamp(min=0), reduction="none"
                        )
                        * known
                    ).sum() / (known.sum() + 1)
                    loss = loss + cfg.aux_weight * loss_a
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                losses.append(loss.item())
                gstep = epoch * spe + i + 1
                if gstep % cfg.log_every == 0:
                    tracker.log(
                        gstep,
                        {
                            "loss": np.mean(losses[-cfg.log_every :]),
                            "loss_presence": loss_p.item(),
                            "loss_salience": loss_s.item(),
                            **({"loss_aux": loss_a.item()} if loss_a is not None else {}),
                            "lr": sched.get_last_lr()[0],
                        },
                        "train/",
                    )
            metrics = validate()
            metrics.update(epoch=epoch + 1, loss=float(np.mean(losses)), seconds=time.time() - t0)
            status["history"].append(metrics)
            log.info("epoch %d %s", epoch + 1, metrics)
            stale = 0 if keep_if_best(metrics) else stale + 1
            log_validation(metrics)
            if cfg.patience and stale >= cfg.patience:
                status["stopped_early"] = epoch + 1
                log.info(
                    "stopping: val f1 has not improved for %d validations (best: epoch %d)",
                    stale,
                    status["best"]["epoch"],
                )
                break
            write_status()
            save_state(epoch + 1)
        if cfg.keep == "last" and status["history"]:  # <name>.pt = where training ended
            save_model(status["history"][-1])
            status["kept"] = {"epoch": status["history"][-1]["epoch"], "which": "last"}
            log.info("kept the last epoch (%d) as %s", status["kept"]["epoch"], ckpt_path.name)
        status["state"] = "done"
        status["finished"] = time.time()
        write_status()
        save_state(status.get("stopped_early", cfg.epochs), done=True)
        tracker.finish()
    except BaseException as e:  # incl. Ctrl-C: the wandb run is marked failed, resume continues it
        tracker.finish(failed=True)
        if isinstance(e, Exception):
            status["state"] = "failed"
            status["error"] = repr(e)
            write_status()
        raise
    return ckpt_path
