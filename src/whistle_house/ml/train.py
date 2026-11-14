"""Train WhistleNet on generated (augmented) examples.

Training is meant to run on a workstation/GPU box (clone the repo + copy ``data/`` there), not
the Pi; the Pi only runs inference. ``--device auto`` picks CUDA/MPS when available.

Run via ``whistle-house train-detector --pipeline default --name whistlenet`` or the manager's
Models page (which launches this as a subprocess). Progress is written to
``<data_dir>/models/<name>.json`` after every epoch so the manager can poll it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from whistle_house.data.store import RecordingStore
from whistle_house.data.training import ExampleGenerator, f0_to_bins
from whistle_house.ml.model import WhistleNet, WhistleNetConfig, save_checkpoint
from whistle_house.pipeline import PipelineSpec
from whistle_house.settings import Paths

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


def train(spec: PipelineSpec, paths: Paths, name: str, cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    status_path = paths.models / f"{name}.json"
    ckpt_path = paths.models / f"{name}.pt"
    paths.models.mkdir(parents=True, exist_ok=True)
    status = {
        "name": name,
        "state": "preparing",
        "pipeline": spec.name,
        "frontend": spec.frontend_dict(),
        "config": asdict(cfg),
        "history": [],
        "started": time.time(),
    }

    def write_status() -> None:
        status_path.write_text(json.dumps(status, indent=2))

    write_status()
    try:
        store = RecordingStore(paths.recordings)
        noise_dirs = [paths.noise]
        gen = ExampleGenerator(
            spec.frontend_dict(), store, noise_dirs, ("train",), data_dir=paths.root
        )
        val_gen = ExampleGenerator(
            spec.frontend_dict(), store, noise_dirs, ("val",), data_dir=paths.root
        )
        if not val_gen.positives:
            val_gen = gen  # early days: no val recordings yet, validate on fresh train draws
        freqs = gen.axis.freqs
        status["data"] = {
            "train_whistle_recordings": len(gen.positives),
            "val_whistle_recordings": len(val_gen.positives) if val_gen is not gen else 0,
            "interferer_files": len(gen.noise.files),
        }

        rng = np.random.default_rng(cfg.seed)
        xs, _ = gen.batch(rng, 16)
        norm = (float(xs.mean()), float(xs.std() + 1e-6))
        vx, vf0 = val_gen.batch(np.random.default_rng(cfg.seed + 1), cfg.val_examples)
        vx = vx.astype(np.float32)

        dev = _pick_device(cfg.device)
        status["device"] = str(dev)
        model = WhistleNet(WhistleNetConfig(n_bins=len(freqs), channels=cfg.channels)).to(dev)
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
                x, f0 = gen.batch(rng, cfg.batch_size)
                pres_t, sal_t = _targets(f0, freqs)
                x_t = torch.from_numpy(((x - norm[0]) / norm[1]).astype(np.float32)).to(dev)
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
                save_checkpoint(
                    ckpt_path, model.cpu(), norm, spec.frontend_dict(), {"metrics": metrics}
                )
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
