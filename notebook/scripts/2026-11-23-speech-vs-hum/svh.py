"""Speech-vs-hum measurement harness (investigation 2026-11-23).

A *job* is one stretch of audio (file, start, duration, optional waveform ops). For each job
every (model, front-end filters) *variant* is run through the real streaming pipeline code
(Frontend.push -> LearnedDetector.process -> score_align), in 4096-sample chunks like dirk.
Per job and variant we keep: per-frame hum / whistle presence probabilities (aligned to the
frames: the detector reports them `lookahead` frames late), the events, and tune matches for
each recognizer config. pYIN gives which frames are voiced and their pitch.

Run from the repo root on lgn9:  uv run python scratch/svh/<experiment>.py
"""

from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
DATA = Path("data").resolve()
EXT = Path("/z/home/adempst/data/whistle-house")
SR = 44100
CHUNK = 4096

RECOGNIZERS = {
    # what dirk ran during the conversation (threshold lowered to see every score)
    "dirk": {"settle_s": 0.5, "threshold": 0.05, "off_pitch_cost": 0.75, "median_weight": 1},
    # score_align defaults (+ settle for the lookahead model)
    "strict": {"settle_s": 0.5, "threshold": 0.05},
}


def _limit_threads():
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


# ----------------------------------------------------------------------------- audio ops


def load(path: str, start: float = 0.0, dur: float | None = None) -> np.ndarray:
    from whistle_house.audio.io import resample, to_mono_float

    info = sf.info(path)
    s = int(start * info.samplerate)
    n = -1 if dur is None else int(dur * info.samplerate)
    x, sr = sf.read(path, start=s, frames=n, dtype="float32", always_2d=False)
    return resample(to_mono_float(x), sr, SR).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


def apply_ops(x: np.ndarray, ops: list, seed: int) -> np.ndarray:
    from scipy.signal import butter, fftconvolve, sosfilt

    rng = np.random.default_rng(seed)
    for op in ops or []:
        kind = op[0]
        if kind == "gain":
            x = x * 10 ** (op[1] / 20)
        elif kind == "rms_to":  # scale to a target RMS in dBFS
            x = x * 10 ** ((op[1] - rms_db(x)) / 20)
        elif kind == "reverb":  # synthetic room: rt60 seconds, direct-to-reverb ratio in dB
            rt60, drr = op[1], op[2]
            n = int(rt60 * SR)
            t = np.arange(n) / SR
            ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
            ir[0] = 0.0
            ir *= 10 ** (-drr / 20) / np.sqrt(np.sum(ir**2))
            ir[0] = 1.0
            level = rms_db(x)
            x = fftconvolve(x, ir)[: len(x)]
            x = x * 10 ** ((level - rms_db(x)) / 20)
        elif kind == "add":  # add another file (looped, random offset) at a gain in dB
            y = load(op[1])
            off = int(rng.integers(0, max(1, len(y))))
            y = np.roll(y, -off)
            y = np.tile(y, int(np.ceil(len(x) / len(y))))[: len(x)]
            x = x + y * 10 ** (op[2] / 20)
        elif kind == "lowpass":
            x = sosfilt(butter(4, op[1], "low", fs=SR, output="sos"), x)
        elif kind == "highpass":
            x = sosfilt(butter(2, op[1], "high", fs=SR, output="sos"), x)
        elif kind == "eqcurve":  # arbitrary magnitude response: (freqs Hz, gains dB), linear phase
            from scipy.signal import firwin2

            fr = np.concatenate([[0.0], np.asarray(op[1], float), [SR / 2]])
            gd = np.concatenate([[op[2][0]], np.asarray(op[2], float), [op[2][-1]]])
            taps = firwin2(8191, fr, 10 ** (gd / 20), fs=SR)
            level = rms_db(x)
            x = fftconvolve(x, taps, mode="same")
            if len(op) > 3 and op[3] == "keep_rms":
                x = x * 10 ** ((level - rms_db(x)) / 20)
        elif kind == "tilt":  # dB per octave around a pivot frequency (negative = darker)
            from scipy.signal import firwin2

            fr = np.geomspace(30, SR / 2 * 0.999, 200)
            gd = np.clip(op[1] * np.log2(fr / op[2]), -40, 40)
            taps = firwin2(8191, np.concatenate([[0], fr, [SR / 2]]), 10 ** (np.concatenate([[gd[0]], gd, [gd[-1]]]) / 20), fs=SR)
            level = rms_db(x)
            x = fftconvolve(x, taps, mode="same")
            x = x * 10 ** ((level - rms_db(x)) / 20)
        elif kind == "white":  # white noise at a level in dBFS
            x = x + 10 ** (op[1] / 20) * rng.standard_normal(len(x))
        else:
            raise ValueError(kind)
    return np.clip(x, -1, 1).astype(np.float32)


# ----------------------------------------------------------------------------- worker

_cache: dict = {}


def _tunes():
    from whistle_house.tunes import TuneLibrary

    if "tunes" not in _cache:
        _cache["tunes"] = TuneLibrary(HERE / "tunes-dirk.json")
    return _cache["tunes"]


def _frontend(filters: tuple):
    from whistle_house.pipeline import Frontend

    key = ("fe", filters)
    if key not in _cache:
        spec = {
            "sample_rate": SR,
            "transform": {"name": "stft_log", "params": {}},
            "filters": [json.loads(f) for f in filters],
        }
        _cache[key] = Frontend.from_dict(spec, DATA)
    fe = _cache[key]
    fe.reset()
    return fe


def _detector(ckpt: str, axis, params: dict | None = None):
    from whistle_house.stages.base import StageContext, build_stage

    key = ("det", ckpt, json.dumps(params or {}, sort_keys=True))
    if key not in _cache:
        _cache[key] = build_stage(
            "detector",
            "learned",
            {"checkpoint": ckpt, **(params or {})},
            StageContext(SR, axis=axis, data_dir=DATA),
        )
    det = _cache[key]
    det.reset()
    return det


def _recognizers(axis):
    from whistle_house.stages.base import StageContext, build_stage

    ctx = StageContext(SR, axis=axis, tunes=_tunes(), data_dir=DATA)
    return {k: build_stage("recognizer", "score_align", p, ctx) for k, p in RECOGNIZERS.items()}


def run_job(job: dict) -> dict:
    """job: {id, group, path, start, dur, ops, seed, variants: [(tag, ckpt, filters[, detector
    params])], pyin}. A checkpoint with a speech output also gives ``p_speech``."""
    _limit_threads()
    from whistle_house.data.hum_labels import hum_f0
    from whistle_house.data.training import LABEL_RATE

    x = load(job["path"], job.get("start", 0.0), job.get("dur"))
    x = apply_ops(x, job.get("ops"), job.get("seed", 0))
    out = {k: job[k] for k in ("id", "group") if k in job}
    out["meta"] = job.get("meta", {})
    out["rms_db"] = rms_db(x)
    out["seconds"] = len(x) / SR
    f0 = hum_f0(x, SR).astype(np.float32) if job.get("pyin", True) else None
    out["variants"] = {}
    by_fe: dict[tuple, list] = {}
    for tag, ckpt, filters, *rest in job["variants"]:
        by_fe.setdefault(tuple(json.dumps(f, sort_keys=True) for f in filters), []).append(
            (tag, ckpt, rest[0] if rest else None)
        )
    for filters, vs in by_fe.items():
        fe = _frontend(filters)
        blocks = [fe.push(x[i : i + CHUNK])[1] for i in range(0, len(x), CHUNK)]
        times = np.concatenate([b.times for b in blocks])
        if "times" not in out:
            out["times"] = times.astype(np.float32)
            if f0 is not None:
                idx = np.clip(np.round(times * LABEL_RATE).astype(int), 0, len(f0) - 1)
                out["f0"] = f0[idx]
            # frame level: loudest bin (dB) as a cheap activity measure
            out["frame_max_db"] = np.concatenate([b.values.max(1) for b in blocks]).astype(
                np.float32
            )
        for tag, ckpt, params in vs:
            det = _detector(ckpt, fe.axis, params)
            recs = _recognizers(fe.axis)
            scores = {k: [] for k in ("whistle", "hum")}
            if det._speech is not None:
                scores["speech"] = []
            events, matches = [], {k: [] for k in recs}
            n = 0
            for i, b in enumerate(blocks):
                n = min(len(x), (i + 1) * CHUNK)
                d = det.process(b)
                for k in scores:
                    if d.kind_scores:  # None for a block without frames
                        scores[k].append(d.kind_scores[k])
                events += d.events
                for rk, r in recs.items():
                    matches[rk] += r.process(d.events, n / SR, d.active)
            tail = det.flush()
            events += tail
            for rk, r in recs.items():
                matches[rk] += r.process(tail, n / SR, []) + r.flush(n / SR)
            la = det.lookahead
            res = {}
            for k, v in scores.items():
                v = np.concatenate(v)
                res[f"p_{k}"] = np.concatenate([v[la:], np.zeros(la, np.float32)])[
                    : len(v)  # a clip shorter than the lookahead
                ].astype(np.float16)
            res["events"] = [
                (
                    e.kind,
                    float(e.start),
                    float(e.end),
                    float(np.median([p.hz for p in e.contour])),
                    float(e.confidence),
                )
                for e in events
            ]
            res["matches"] = {
                rk: [(m.kind, m.tune_name, float(m.time), float(m.score)) for m in ms]
                for rk, ms in matches.items()
            }
            out["variants"][tag] = res
    return out


def run(jobs: list[dict], out: str | Path, workers: int = 32) -> list[dict]:
    out = HERE / "results" / out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return pickle.loads(out.read_bytes())
    import multiprocessing as mp

    res = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("forkserver")) as pool:
        for i, r in enumerate(pool.map(run_job, jobs, chunksize=1)):
            res.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    out.write_bytes(pickle.dumps(res))
    return res


# ----------------------------------------------------------------------------- job sets


def manifest(ds: str) -> list[dict]:
    with (EXT / "prepared" / ds / "manifest.jsonl").open() as f:
        return [json.loads(line) for line in f]


def external_jobs(
    ds: str, label: str, split: str, n: int, dur: float = 30.0, seed: int = 0, group: str = ""
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = [r for r in manifest(ds) if r["label"] == label and r["split"] == split]
    # n segments, drawn over files in proportion to their length (a file can give several)
    w = np.array([max(r["duration"], 1.0) for r in rows])
    picks = rng.choice(len(rows), size=n, replace=n > len(rows), p=w / w.sum())
    jobs = []
    for k, i in enumerate(picks):
        r = rows[int(i)]
        start = float(rng.uniform(0, max(0.0, r["duration"] - dur)))
        jobs.append(
            {
                "id": f"{r['path']}@{start:.1f}#{k}",
                "group": group or f"{ds}:{label}:{split}",
                "path": str(EXT / "prepared" / ds / r["path"]),
                "start": start,
                "dur": min(dur, r["duration"]),
                "meta": {**r.get("meta", {}), "file": r["path"]},
            }
        )
    return jobs


def mic_jobs(label: str | None = None) -> list[dict]:
    from whistle_house.data.store import RecordingStore

    store = RecordingStore(DATA / "recordings")
    jobs = []
    for r in store.list():
        if label and r.label != label:
            continue
        mic = "fifine" if (r.device or "").startswith("USB PnP") else "mac"
        jobs.append(
            {
                "id": r.id,
                "group": f"{mic}:{r.label}",
                "path": str(store.audio_path(r.id)),
                "meta": {"split": r.split, "device": r.device, "notes": r.notes, "tune": r.tune_id},
            }
        )
    return jobs


def with_variants(jobs: list[dict], variants: list[tuple], **extra) -> list[dict]:
    return [{**j, "variants": variants, **extra} for j in jobs]


# ----------------------------------------------------------------------------- summaries


def summarize(results: list[dict], tag: str, thr: float = 0.5) -> dict:
    """Pooled over the given results for one variant."""
    frames = voiced = hum = hum_v = wh = sp = sp_v = 0
    has_sp = False
    secs = 0.0
    n_ev = {"hum": 0, "whistle": 0}
    ev_s = {"hum": 0.0, "whistle": 0.0}
    m = {rk: [] for rk in RECOGNIZERS}
    for r in results:
        v = r["variants"][tag]
        ph, pw = v["p_hum"].astype(np.float32), v["p_whistle"].astype(np.float32)
        vo = r["f0"] > 0 if "f0" in r else np.zeros(len(ph), bool)
        frames += len(ph)
        voiced += int(vo.sum())
        hum += int((ph >= thr).sum())
        hum_v += int(((ph >= thr) & vo).sum())
        wh += int((pw >= thr).sum())
        if "p_speech" in v:
            has_sp = True
            ps = v["p_speech"].astype(np.float32)
            sp += int((ps >= thr).sum())
            sp_v += int(((ps >= thr) & vo).sum())
        secs += r["seconds"]
        for kind, s, e, _, _ in v["events"]:
            n_ev[kind] += 1
            ev_s[kind] += e - s
        for rk in m:
            m[rk] += [x[3] for x in v["matches"][rk]]
    mins = secs / 60
    return {
        "clips": len(results),
        "minutes": round(mins, 2),
        "voiced%": round(100 * voiced / max(frames, 1), 1),
        "hum%": round(100 * hum / max(frames, 1), 2),
        "hum%_of_voiced": round(100 * hum_v / max(voiced, 1), 2),
        "whistle%": round(100 * wh / max(frames, 1), 2),
        **(
            {
                "speech%": round(100 * sp / max(frames, 1), 2),
                "speech%_of_voiced": round(100 * sp_v / max(voiced, 1), 2),
            }
            if has_sp
            else {}
        ),
        "hum_ev/min": round(n_ev["hum"] / max(mins, 1e-9), 2),
        "hum_ev_s/min": round(ev_s["hum"] / max(mins, 1e-9), 2),
        "wh_ev/min": round(n_ev["whistle"] / max(mins, 1e-9), 2),
        **{
            f"match/min {rk}>={t}": round(sum(s >= t for s in ms) / max(mins, 1e-9), 2)
            for rk, ms in m.items()
            for t in ((0.1, 0.5) if rk == "dirk" else (0.5,))
        },
    }


def table(rows: dict[str, dict]) -> str:
    cols = list(next(iter(rows.values())).keys())
    w0 = max(len(k) for k in rows) + 1
    w = max(14, *(len(str(r[c])) + 2 for r in rows.values() for c in cols))
    lines = [" " * w0 + " ".join(f"{c:>{w}}" for c in cols)]
    for k, r in rows.items():
        lines.append(f"{k:<{w0}}" + " ".join(f"{r[c]!s:>{w}}" for c in cols))
    return "\n".join(lines)


def groups(results: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = {}
    for r in results:
        g.setdefault(r["group"], []).append(r)
    return g
