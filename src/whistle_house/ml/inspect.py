"""Layer-by-layer report of a whistle + hum checkpoint (``harmonic`` arch) on the mic recordings.

``build_report`` runs the checkpoint over every recording in the store and writes a static,
self-contained report directory (``whistle-house inspect-model``, or the Models page):

* ``index.html``  the viewer (``inspect_page.html`` next to this module; plain JS, no build);
* ``data.json``   per-recording scores, the decision space (presence logits per frame), input
                  comb profiles, input attributions, pitch-error histograms, block-1 weights and
                  the checkpoint's training history;
* ``clips/<id>.json``  for a sample of recordings (all hums and whistles, some of the rest, cut
                  to ``max_clip_s``): every stage as an image (grayscale PNG data URIs): the
                  spectrogram, both salience maps, each block's activity, the last block's
                  features in PCA colour, and each presence-head input's push on the logit.

What it answers: does the harmonic stack show a comb at the right pitch, does the network
read it, which recordings fire which class, and why speech frames fire as hum.

Reference pitch comes from the training labellers: pYIN for ``hum`` recordings
(``data/hum_labels.py``), the strict ``peak_tracker`` for ``whistle`` recordings. The run goes
through the real streaming ``Pipeline`` for events, and a mirror of ``WhistleNet.forward``
that keeps every block's output (``_internals``; tests pin it to the model's own output).
"""

from __future__ import annotations

import base64
import json
import shutil
import struct
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from whistle_house.data.hum_labels import hum_f0
from whistle_house.data.store import Recording, RecordingStore
from whistle_house.data.training import LABEL_RATE, GeneratorConfig, auto_label
from whistle_house.ml.model import WhistleNet, load_checkpoint
from whistle_house.pipeline import Pipeline, PipelineSpec, StageSpec

PAGE = Path(__file__).with_name("inspect_page.html")
SCATTER_PER_CLIP = 260  # frames per recording in the decision-space scatter
PCA_SAMPLES_PER_CLIP = 3000


def report_pipeline(checkpoint: Path, base: PipelineSpec | None = None) -> PipelineSpec:
    """The pipeline a report runs: ``base``'s detector params (thresholds) if given, else the
    defaults, on the checkpoint's own front end, with the detector pointed at ``checkpoint``."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fe = ck["frontend"]
    params = dict(base.detector.params) if base and base.detector.name == "learned" else {}
    params["checkpoint"] = str(checkpoint.resolve())
    return PipelineSpec(
        name=base.name if base else f"{checkpoint.stem} (report)",
        sample_rate=fe["sample_rate"],
        chunk_size=base.chunk_size if base else 4096,
        transform=StageSpec(**fe["transform"]),
        filters=[StageSpec(**f) for f in fe["filters"]],
        detector=StageSpec(name="learned", params=params),
        recognizer=base.recognizer if base else StageSpec(name="score_align"),
    )


# ----------------------------------------------------------------------------- png helpers


def _png(img: np.ndarray) -> str:
    """uint8 (H, W) gray or (H, W, 3) RGB -> PNG data URI. Row 0 = top."""
    img = np.ascontiguousarray(img, np.uint8)
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[i].tobytes() for i in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        crc = zlib.crc32(t + d) & 0xFFFFFFFF
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", crc)

    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0 if img.ndim == 2 else 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _q8(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / (hi - lo) * 255 + 0.5, 0, 255).astype(np.uint8)


def _fmap(x: np.ndarray, lo: float, hi: float, levels: int = 256) -> dict:
    """(T, F) -> image with frequency up (row 0 = highest bin), time across. ``levels`` < 256
    coarsens the quantisation (much smaller PNGs for maps that are only looked at)."""
    q = _q8(x, lo, hi)
    if levels < 256:
        step = 256 // levels
        q = (q // step * step + step // 2).astype(np.uint8)
    return {"png": _png(q.T[::-1]), "lo": lo, "hi": hi}


# ----------------------------------------------------------------------------- model internals


def _internals(model: WhistleNet, x: np.ndarray) -> dict:
    """Offline pass over normalised frames (T, F) that keeps every block's output.

    Mirrors ``WhistleNet.forward`` (pads ``lookahead`` frames at the end, drops the first
    outputs), so ``logits`` equals ``model(x)[0]`` exactly."""
    la = model.cfg.lookahead
    with torch.no_grad():
        stack = model._embed(F.pad(torch.from_numpy(x)[None], (0, 0, 0, la)))
        h, blocks, ctxs = stack, [], []
        for blk in model.blocks:
            padded = F.pad(h, (0, 0, blk.pad_t, 0))
            y = F.gelu(blk.norm(blk.conv(F.pad(padded, (blk.pad_f, blk.pad_f, 0, 0)))))
            c = torch.zeros_like(y)
            if blk.ctx is not None:
                c = blk.ctx(torch.cat([y.mean(-1), y.amax(-1)], dim=1)).unsqueeze(-1).expand_as(y)
            h = y + c + blk.skip(h)
            blocks.append(h[0, :, la:].numpy())  # (C, T, F)
            ctxs.append(c[0, :, la:, 0].numpy())  # (C, T)
        pooled = torch.cat([h.mean(-1), h.amax(-1)], dim=1)[0, :, la:].numpy()  # (2C, T)
        pres, sal = model._heads(h)
    return {
        "blocks": blocks,
        "ctx": ctxs,
        "pooled": pooled,
        "logits": pres[0, la:].numpy(),  # (T, K)
        "sal": torch.sigmoid(sal[0, la:]).numpy(),  # (T, K, F)
        "stack": stack[0, :, la:].numpy(),  # (H, T, F), normalised units
    }


def _attribution(model: WhistleNet, x: np.ndarray, k: int, mask: np.ndarray) -> np.ndarray:
    """grad x input of class ``k``'s presence logit, summed over the masked frames, w.r.t. each
    harmonic view and pitch candidate of the stack -> (H, F). Baseline: each frame's floor
    (what off-axis positions hold), so it measures what energy above the floor contributes."""
    la = model.cfg.lookahead
    stack = model._embed(F.pad(torch.from_numpy(x)[None], (0, 0, 0, la))).detach()
    stack.requires_grad_(True)
    pres, _ = model._heads(model.blocks(stack))
    (pres[0, la:, k] * torch.from_numpy(mask.astype(np.float32))).sum().backward()
    floor = stack.detach().amin(dim=(1, 3), keepdim=True)
    return (stack.grad * (stack.detach() - floor))[0].sum(dim=1).numpy()


def _comb_at(stack_db: np.ndarray, sel: np.ndarray, bins: np.ndarray, ch1: int) -> np.ndarray:
    """(H, T, F) stack in dB, frame mask, a bin per selected frame -> (n, H) levels above each
    frame's floor (the floor of the 1x view, i.e. of the spectrum)."""
    t = np.flatnonzero(sel)
    return (stack_db[:, t, bins] - stack_db[ch1, t].min(-1)[None]).T


def _reference(
    rec: Recording, audio: np.ndarray, sr: int, times: np.ndarray, frontend: dict, data_dir: Path
) -> tuple[str | None, np.ndarray]:
    """Reference pitch per frame (Hz, 0 = none) and its kind, from the training labellers."""
    if rec.label == "hum":
        f0, kind = hum_f0(audio, sr), "hum"
    elif rec.label == "whistle":
        f0 = auto_label(audio, sr, frontend, GeneratorConfig().labeler_params, data_dir)
        kind = "whistle"
    else:
        return None, np.zeros(len(times), np.float32)
    f0 = np.asarray(f0, np.float32)
    idx = np.clip(np.round(times * LABEL_RATE).astype(int), 0, len(f0) - 1)
    return kind, f0[idx]


def _detail_ids(recs: list[Recording]) -> set[str]:
    """Recordings that get the full per-stage view: every hum and whistle, the held-out and
    four more speech clips, three of each other label."""
    out: set[str] = set()
    by_label: dict[str, list[Recording]] = {}
    for r in recs:
        by_label.setdefault(r.label, []).append(r)
    for lab, rs in by_label.items():
        if lab in ("hum", "whistle"):
            out |= {r.id for r in rs}
        elif lab == "speech":
            out |= {r.id for r in rs if r.split != "train"} | {r.id for r in rs[:4]}
        else:
            out |= {r.id for r in rs[:3]}
    return out


def _clip_view(
    r: Recording,
    frames: np.ndarray,
    times: np.ndarray,
    out: dict,
    probs: np.ndarray,
    ref: np.ndarray,
    kind: str | None,
    events: list,
    classes: list[str],
    cap: int,
) -> dict:
    """The per-clip view (first ``cap`` frames), except the parts that need every clip first
    (``pca``, ``contrib``: ``build_report`` adds them)."""
    blocks = []
    for b, c in zip(out["blocks"], out["ctx"], strict=True):
        bb = b[:, :cap]
        dev = bb - np.median(bb, axis=1, keepdims=True)  # drop the static per-bin pattern
        rms = np.sqrt((dev**2).mean(0))  # (T, F)
        hi_v = float(np.percentile(rms, 99.5)) + 1e-6
        ctx = np.abs(c[:, :cap]).mean(0)
        blocks.append({**_fmap(rms, 0.0, hi_v, 32), "ctx": ctx.round(3).tolist()})
    return {
        "id": r.id,
        "label": r.label,
        "split": r.split,
        "times": [float(times[0]), float(times[1] - times[0])],
        "T": int(cap),
        "spec": _fmap(frames[:cap], -100.0, 0.0),  # dB
        "sal": {k: _fmap(out["sal"][:cap, i], 0.0, 1.0) for i, k in enumerate(classes)},
        "prob": {k: probs[:cap, i].round(3).tolist() for i, k in enumerate(classes)},
        "ref": {"kind": kind, "hz": ref[:cap].round(1).tolist()},
        "events": [
            {
                "kind": e.kind,
                "start": e.start,
                "end": e.end,
                "t": [round(p.t, 3) for p in e.contour],
                "hz": [round(p.hz, 1) for p in e.contour],
            }
            for e in events
            if e.start < times[cap - 1]
        ],
        "blocks": blocks,
    }


# ----------------------------------------------------------------------------- report


def build_report(
    checkpoint: Path,
    store: RecordingStore,
    out_dir: Path,
    data_dir: Path,
    base: PipelineSpec | None = None,
    max_clip_s: float = 20.0,
    log: Callable[[str], None] = print,
) -> dict:
    """Run ``checkpoint`` over every recording in ``store`` and write the report to
    ``out_dir`` (replacing a previous one). Returns ``data.json``'s content."""
    model, ck = load_checkpoint(checkpoint)
    cfg = model.cfg
    if not cfg.harmonics or set(cfg.classes) != {"whistle", "hum"}:
        raise ValueError(
            f"{checkpoint.name} is not a whistle + hum (harmonic arch) model; the report "
            "explains the harmonic stack and both classes"
        )
    torch.manual_seed(0)
    spec = report_pipeline(checkpoint, base)
    pipe = Pipeline(spec, data_dir=data_dir)
    det = pipe.detector
    mean, std = ck["norm"]["mean"], ck["norm"]["std"]
    classes = list(cfg.classes)
    hi, wi = classes.index("hum"), classes.index("whistle")
    ch1 = list(cfg.harmonics).index(1.0)
    freqs = np.asarray(pipe.axis.freqs, np.float64)
    on_thr = det.config.on_threshold
    recs = sorted(store.list(), key=lambda r: (r.label, r.split or "", r.id))
    if not recs:
        raise ValueError("no recordings to run the model on")
    detail_ids = _detail_ids(recs)
    rng = np.random.default_rng(0)

    summary, scatter = [], []
    hist_err: dict[str, list[float]] = {k: [] for k in classes}
    n_bins, n_h = cfg.n_bins, len(cfg.harmonics)
    attr = {f"{k}|{g}": np.zeros((n_h, n_bins)) for k in classes for g in ("pos", "neg")}
    attr_n = dict.fromkeys(attr, 0)
    comb: dict[str, list[np.ndarray]] = {
        g: [] for g in ("hum@f0", "whistle@f0", "speech false hum@argmax", "speech other@argmax")
    }
    pca_samples: list[np.ndarray] = []
    silence_pooled: list[np.ndarray] = []
    kept: dict[str, tuple] = {}
    fps = 0.0

    for n, r in enumerate(recs):
        audio, sr = store.load(r.id, spec.sample_rate)
        res = pipe.analyze(audio)
        frames = res.raw.values.astype(np.float32)
        times = np.asarray(res.raw.times, np.float64)
        T = len(times)
        if T < 2:
            continue
        fps = 1.0 / float(times[1] - times[0])
        x = ((frames - mean) / std).astype(np.float32)
        out = _internals(model, x)
        probs = 1 / (1 + np.exp(-out["logits"]))
        kind, ref = _reference(r, audio, sr, times, ck["frontend"], data_dir)
        row: dict = {"id": r.id, "label": r.label, "split": r.split, "frames": T}
        row["dur"] = round(T / fps, 1)
        for i, k in enumerate(classes):
            evs = [e for e in res.events if e.kind == k]
            row[f"{k}_active"] = float((probs[:, i] >= on_thr).mean())
            row[f"{k}_events"] = len(evs)
            row[f"{k}_event_s"] = round(sum(e.duration for e in evs), 2)
        stack_db = out["stack"] * std + mean
        if kind is not None:
            ki = classes.index(kind)
            voiced = ref > 0
            row["labelled"] = float(voiced.mean())
            if voiced.any():
                row["recall"] = float((probs[voiced, ki] >= on_thr).mean())
                row["wrong_kind"] = float((probs[voiced, 1 - ki] >= on_thr).mean())
                fb = np.interp(ref[voiced], freqs, np.arange(n_bins))
                comb[f"{kind}@f0"].append(_comb_at(stack_db, voiced, np.round(fb).astype(int), ch1))
                attr[f"{kind}|pos"] += _attribution(model, x, ki, voiced)
                attr_n[f"{kind}|pos"] += int(voiced.sum())
            if (~voiced).any():
                row["outside"] = float((probs[~voiced, ki] >= on_thr).mean())
            sel = voiced & (probs[:, ki] >= 0.5)
            if sel.any():
                err = 1200 * np.log2(freqs[out["sal"][sel, ki].argmax(-1)] / ref[sel])
                hist_err[kind].extend(err.tolist())
                row["pitch_ok"] = float((np.abs(err) <= 50).mean())
                row["octave_err"] = float((np.abs(np.abs(err) - 1200) <= 60).mean())
        else:
            for i, k in enumerate(classes):  # what drives false detections on negatives
                fp = probs[:, i] >= on_thr
                if fp.any():
                    attr[f"{k}|neg"] += _attribution(model, x, i, fp)
                    attr_n[f"{k}|neg"] += int(fp.sum())
        if r.label == "speech":
            am = out["sal"][:, hi].argmax(-1)
            fp = probs[:, hi] >= on_thr
            loud = frames.max(-1) > np.percentile(frames.max(-1), 50)  # skip pauses
            if fp.any():
                comb["speech false hum@argmax"].append(_comb_at(stack_db, fp, am[fp], ch1))
            if (~fp & loud).any():
                rest = ~fp & loud
                comb["speech other@argmax"].append(_comb_at(stack_db, rest, am[rest], ch1))
        if r.label == "silence":
            silence_pooled.append(out["pooled"])
        summary.append(row)

        for t in np.sort(rng.choice(T, size=min(T, SCATTER_PER_CLIP), replace=False)):
            lab = r.label if kind is None or ref[t] > 0 else f"{r.label} (gap)"
            scatter.append(
                [round(float(out["logits"][t, wi]), 2), round(float(out["logits"][t, hi]), 2)]
                + [lab, r.id, int(t)]
            )
        if r.id in detail_ids:
            last = out["blocks"][-1]
            flat = last.reshape(last.shape[0], -1).T
            take = rng.choice(len(flat), size=min(len(flat), PCA_SAMPLES_PER_CLIP), replace=False)
            pca_samples.append(flat[take])
            cap = min(T, int(max_clip_s * fps))
            clip = _clip_view(r, frames, times, out, probs, ref, kind, res.events, classes, cap)
            # only what still needs the global PCA / silence baseline; the rest is done
            kept[r.id] = (clip, last[:, :cap].astype(np.float16), out["pooled"][:, :cap].copy())
        del out
        log(f"[{n + 1}/{len(recs)}] {r.id} {r.label}")

    # ------------------------------------------------------------------ per-clip files
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "clips").mkdir(parents=True)
    X = np.concatenate(pca_samples)
    pca_mu = X.mean(0)
    pcs = np.linalg.svd(X - pca_mu, full_matrices=False)[2][:3]
    proj = (X - pca_mu) @ pcs.T
    pca_lo, pca_hi = np.percentile(proj, 1, 0), np.percentile(proj, 99, 0)
    w_pres = model.presence.weight.detach().numpy()  # (K, 2C)
    b_pres = model.presence.bias.detach().numpy()
    base_pooled = (
        np.concatenate(silence_pooled, 1).mean(1) if silence_pooled else np.zeros(w_pres.shape[1])
    )
    clip_index = []
    for cid, (clip, last, pooled) in kept.items():
        last = last.astype(np.float32)
        rgb = np.stack(
            [
                _q8(np.tensordot(pcs[j], last - pca_mu[:, None, None], 1), pca_lo[j], pca_hi[j])
                // 8
                * 8
                for j in range(3)
            ],
            -1,
        )  # (T, F, 3)
        clip["pca"] = {"png": _png(rgb.transpose(1, 0, 2)[::-1])}
        clip["contrib"] = {}
        for i, k in enumerate(classes):
            c = (w_pres[i][:, None] * (pooled - base_pooled[:, None])).T  # push vs silence
            lim = float(np.percentile(np.abs(c), 99.5)) + 1e-6
            clip["contrib"][k] = _fmap(c, -lim, lim)
        (out_dir / "clips" / f"{cid}.json").write_text(json.dumps(clip, separators=(",", ":")))
        clip_index.append({k: clip[k] for k in ("id", "label", "split", "T")})

    # ------------------------------------------------------------------ global data
    blocks_info, rf = [], 1
    for d in cfg.dilations:
        rf += (cfg.kernel_t - 1) * d
        blocks_info.append({"dilation": d, "rf_frames": rf})
    hist = {}
    for k, errs in hist_err.items():
        e = np.clip(np.asarray(errs, np.float64), -2450, 2450)
        counts = np.histogram(e, bins=np.arange(-2450, 2451, 50))[0]
        hist[k] = {"counts": counts.tolist(), "lo": -2450, "step": 50, "n": len(errs)}

    def stats(v: list[np.ndarray], q: float) -> list[float]:
        return np.percentile(np.concatenate(v), q, 0).round(2).tolist() if v else []

    status = checkpoint.with_suffix(".json")
    w0 = model.blocks[0].conv.weight.detach().numpy()  # (C, H, kt, kf)
    data = {
        "checkpoint": checkpoint.name,
        "pipeline": spec.name,
        "arch": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()},
        "params": int(sum(p.numel() for p in model.parameters())),
        "freqs": freqs.round(2).tolist(),
        "norm": {"mean": mean, "std": std},
        "fps": fps,
        "on_threshold": on_thr,
        "off_threshold": det.config.off_threshold,
        "summary": summary,
        "scatter": scatter,
        "hist": hist,
        "attr": {k: (v / max(1, attr_n[k])).round(4).tolist() for k, v in attr.items()},
        "attr_n": attr_n,
        "comb": {
            g: {
                "n": int(sum(len(a) for a in v)),
                "median": stats(v, 50),
                "p25": stats(v, 25),
                "p75": stats(v, 75),
            }
            for g, v in comb.items()
        },
        "w0": w0.sum(axis=(2, 3)).round(4).tolist(),  # (C, H): each view's total weight
        "blocks": blocks_info,
        "silence_logit": dict(zip(classes, (w_pres @ base_pooled + b_pres).tolist(), strict=True)),
        "clips": clip_index,
        "history": json.loads(status.read_text()).get("history", []) if status.exists() else [],
    }
    (out_dir / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    shutil.copyfile(PAGE, out_dir / "index.html")
    log(f"report: {out_dir / 'index.html'} ({len(clip_index)} clips in detail)")
    return data
