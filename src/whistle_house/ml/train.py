"""Train WhistleNet on generated (augmented) examples.

Training is meant to run on a workstation/GPU box (clone the repo + copy ``data/`` there), not
the Pi; the Pi only runs inference. ``--device auto`` picks CUDA/MPS when available.

Run via ``whistle-house train-detector`` (or ``whistle-house warmstart`` for the full
external-data recipe) or the manager's Models page. Progress is written to
``<data_dir>/models/<name>.json`` after every epoch so the manager can poll it.

Examples are generated on the fly (augmentation is random every time) by ``workers`` DataLoader
processes, each with its own ``ExampleGenerator``; the main process only runs the model.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
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
from whistle_house.ml.model import WhistleNet, WhistleNetConfig, load_checkpoint, save_checkpoint
from whistle_house.pipeline import PipelineSpec
from whistle_house.settings import Paths, external_root

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
    salience_weight: float = 1.0
    device: str = "auto"  # auto | cpu | cuda | mps
    mix: str = "mic"  # data mix preset: mic | pretrain | finetune (data/training.MIX_PRESETS)
    external: list[str] = field(default_factory=list)  # external dataset names to use
    init_from: str | None = None  # checkpoint to fine-tune from (path or name in models/)
    workers: int = 4  # example-generation processes (0 = in the main process)


class _ExampleStream(torch.utils.data.IterableDataset):
    """Endless stream of generated batches; every worker gets its own RNG + generator."""

    def __init__(
        self, frontend: dict, sources: TrainingSources, gcfg: GeneratorConfig, bs: int, seed: int
    ):
        super().__init__()
        self.args = (frontend, sources, gcfg)
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
    """f0 (B, T) -> presence (B, T) {0,1}, salience (B, T, F) gaussian bumps."""
    bins = f0_to_bins(f0, freqs)
    pres = (~np.isnan(bins)).astype(np.float32)
    grid = np.arange(len(freqs))[None, None, :]
    sal = np.exp(-0.5 * ((grid - np.nan_to_num(bins, nan=-1e3)[..., None]) / sigma_bins) ** 2)
    return pres, (sal * pres[..., None]).astype(np.float32)


def _pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate(
    model: WhistleNet, x: np.ndarray, f0: np.ndarray, freqs: np.ndarray, norm, dev
) -> dict:
    model.eval()
    with torch.no_grad():
        pres, sal = model(torch.from_numpy((x - norm[0]) / norm[1]).to(dev))
    pres, sal = pres.cpu(), sal.cpu()
    p = torch.sigmoid(pres).numpy() >= 0.5
    t = f0 > 0
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    pitch_bins = sal.numpy().argmax(-1)
    est_hz = freqs[pitch_bins]
    both = p & t
    pitch_ok = np.abs(12 * np.log2(est_hz[both] / f0[both])) <= 1.0 if both.any() else np.array([])
    model.train()
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
        "false_positive_rate": float(fp / max((~t).sum(), 1)),
        "pitch_acc_1st": float(pitch_ok.mean()) if pitch_ok.size else 0.0,
    }


def _resolve_checkpoint(ref: str, paths: Paths) -> Path:
    p = Path(ref)
    if p.exists():
        return p
    p = paths.models / (ref if ref.endswith(".pt") else f"{ref}.pt")
    if not p.exists():
        raise FileNotFoundError(f"checkpoint {ref!r} not found")
    return p


def train(spec: PipelineSpec, paths: Paths, name: str, cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    status_path = paths.models / f"{name}.json"
    ckpt_path = paths.models / f"{name}.pt"
    paths.models.mkdir(parents=True, exist_ok=True)
    frontend = spec.frontend_dict()
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
    try:
        gcfg = config_for_mix(cfg.mix)
        common = {
            "store": RecordingStore(paths.recordings),
            "label_cache_dir": paths.label_cache,
            "external_root": external_root(required=bool(cfg.external)),
            "external_datasets": cfg.external,
            "noise_dirs": [paths.noise],
            "config": gcfg,
            "workers": max(1, cfg.workers),
        }
        train_src = collect_sources(frontend, ("train",), **common)
        val_src = collect_sources(frontend, ("val",), **common)
        if not (val_src.mic_positives or val_src.ext_positives):
            val_src = train_src  # early days: validate on fresh draws from train sources
        status["data"] = {"train": train_src.summary(), "val": val_src.summary()}
        write_status()

        gen = ExampleGenerator(frontend, train_src, gcfg)
        freqs = gen.axis.freqs
        vx, vf0 = ExampleGenerator(frontend, val_src, gcfg).batch(
            np.random.default_rng(cfg.seed + 1), cfg.val_examples
        )
        vx = vx.astype(np.float32)

        dev = _pick_device(cfg.device)
        status["device"] = str(dev)
        if cfg.init_from:
            init_path = _resolve_checkpoint(cfg.init_from, paths)
            model, ck = load_checkpoint(init_path)
            if ck["frontend"] != frontend:
                raise ValueError(
                    f"{init_path} was trained on a different front-end; fine-tune with the "
                    f"pipeline it came from ({ck['frontend']})"
                )
            norm = (ck["norm"]["mean"], ck["norm"]["std"])
            status["init_from"] = str(init_path)
        else:
            xs, _ = gen.batch(np.random.default_rng(cfg.seed), 16)
            norm = (float(xs.mean()), float(xs.std() + 1e-6))
            model = WhistleNet(WhistleNetConfig(n_bins=len(freqs), channels=cfg.channels))
        model = model.to(dev)
        model.train()

        stream = _ExampleStream(frontend, train_src, gcfg, cfg.batch_size, cfg.seed)
        loader = torch.utils.data.DataLoader(
            stream,
            batch_size=None,
            num_workers=cfg.workers,
            prefetch_factor=4 if cfg.workers else None,
            persistent_workers=cfg.workers > 0,
            multiprocessing_context=_mp_context() if cfg.workers else None,
        )
        batches = iter(loader)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=cfg.epochs * cfg.steps_per_epoch
        )
        status["state"] = "training"
        status["params"] = sum(p.numel() for p in model.parameters())
        write_status()
        best = -1.0
        for epoch in range(cfg.epochs):
            t0, losses = time.time(), []
            for _ in range(cfg.steps_per_epoch):
                x, f0 = next(batches)
                x, f0 = x.numpy(), f0.numpy()
                pres_t, sal_t = _targets(f0, freqs)
                x_t = torch.from_numpy((x - norm[0]) / norm[1]).float().to(dev)
                pres, sal = model(x_t)
                pres_t, sal_t = torch.from_numpy(pres_t).to(dev), torch.from_numpy(sal_t).to(dev)
                loss_p = F.binary_cross_entropy_with_logits(pres, pres_t)
                mask = pres_t[..., None].expand_as(sal)
                loss_s = (
                    F.binary_cross_entropy_with_logits(sal, sal_t, reduction="none") * mask
                ).sum() / (mask.sum() + 1)
                loss = loss_p + cfg.salience_weight * loss_s
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                losses.append(loss.item())
            metrics = _evaluate(model, vx, vf0, freqs, norm, dev)
            metrics.update(epoch=epoch + 1, loss=float(np.mean(losses)), seconds=time.time() - t0)
            status["history"].append(metrics)
            log.info("epoch %d %s", epoch + 1, metrics)
            if metrics["f1"] > best:
                best = metrics["f1"]
                status["best"] = metrics
                meta = {
                    "metrics": metrics,
                    "mix": cfg.mix,
                    "external": cfg.external,
                    "data": status["data"],
                }
                save_checkpoint(ckpt_path, model.cpu(), norm, frontend, meta)
                model.to(dev)
            write_status()
        status["state"] = "done"
    except Exception as e:
        status["state"] = "failed"
        status["error"] = repr(e)
        write_status()
        raise
    status["finished"] = time.time()
    write_status()
    return ckpt_path
