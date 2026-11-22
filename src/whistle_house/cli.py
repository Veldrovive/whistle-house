"""Command line entry point: ``whistle-house <command>`` (or ``python -m whistle_house.cli``)."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def _paths(args):
    from whistle_house.settings import default_paths

    if args.data:
        os.environ["WHISTLE_HOUSE_DATA"] = str(Path(args.data).resolve())
    return default_paths().ensure()


def cmd_serve(args) -> None:
    from whistle_house.threads import limit_inference_threads

    limit_inference_threads()  # before anything imports numpy

    import uvicorn

    from whistle_house.server.app import create_app

    app = create_app(_paths(args), device=args.device, live=not args.no_live, replay=args.replay)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_devices(args) -> None:
    from whistle_house.audio.sources import list_input_devices

    for d in list_input_devices():
        print(f"{d['index']:3d}  {d['name']}  ({d['default_samplerate']:.0f} Hz)")


def cmd_analyze(args) -> None:
    from whistle_house.audio.io import load_audio
    from whistle_house.pipeline import Pipeline, PipelineStore
    from whistle_house.tunes import TuneLibrary

    paths = _paths(args)
    spec = PipelineStore(paths.pipelines).get(args.pipeline)
    pipe = Pipeline(spec, tunes=TuneLibrary(paths.tunes), data_dir=paths.root)
    audio, _ = load_audio(args.file, spec.sample_rate)
    res = pipe.analyze(audio)
    print(f"{len(res.events)} events")
    for e in res.events:
        print(
            f"  {e.start:7.2f}-{e.end:7.2f}s  {e.kind:8s} {e.median_hz():7.1f} Hz  "
            f"conf={e.confidence:.2f}"
        )
    print(f"{len(res.matches)} tune matches")
    for m in res.matches:
        print(f"  {m.time:7.2f}s  {m.tune_name} ({m.kind})  score={m.score:.2f}")


def cmd_import(args) -> None:
    from whistle_house.audio.io import load_audio
    from whistle_house.data.store import LABELS, RecordingStore

    if args.label not in LABELS:
        raise SystemExit(f"label must be one of {LABELS}")
    store = RecordingStore(_paths(args).recordings)
    for f in args.files:
        audio, sr = load_audio(f)
        rec = store.add(
            audio, sr, label=args.label, tune_id=args.tune, source="upload", device=Path(f).name
        )
        print(f"{rec.id}  {rec.label}  {rec.duration:.1f}s  <- {f}")


def _external_list(value: str | None) -> list[str]:
    """'all' -> every prepared dataset; 'a,b' -> [a, b]; None/'' -> []."""
    from whistle_house.external import dataset_status
    from whistle_house.settings import external_root

    if not value:
        return []
    if value == "all":
        return [d["name"] for d in dataset_status(external_root()) if d["prepared"]]
    return [v.strip() for v in value.split(",") if v.strip()]


def _classes(value: str | None) -> list[str]:
    """'whistle,hum' -> ['whistle', 'hum']; None/'' -> [] (the arch's own classes)."""
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _train_config(args, **overrides):
    from whistle_house.ml.train import TrainConfig

    return TrainConfig(
        **{
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "device": args.device,
            "workers": args.workers,
            "val_examples": args.val_examples,
            "patience": args.patience,
            "arch": args.arch,
            "classes": _classes(args.classes),
            "wandb_project": args.wandb_project,
            "wandb_group": getattr(args, "wandb_group", None),
            **overrides,
        }
    )


def cmd_train(args) -> None:
    from whistle_house.ml.train import train
    from whistle_house.pipeline import PipelineStore

    paths = _paths(args)
    spec = PipelineStore(paths.pipelines).get(args.pipeline)
    cfg = _train_config(
        args, mix=args.mix, external=_external_list(args.external), init_from=args.init_from
    )
    path = train(spec, paths, args.name, cfg, resume=args.resume)
    print(f"saved {path}")


def cmd_datasets_list(args) -> None:
    from whistle_house.external import dataset_status
    from whistle_house.settings import external_root

    root = external_root(required=False)
    print(f"external root: {root or '(WHISTLE_HOUSE_EXTERNAL not set)'}\n")
    for d in dataset_status(root):
        state = "prepared" if d["prepared"] else "downloaded" if d["downloaded"] else "-"
        print(f"{d['name']:16s} {state:10s} {d['download_size']:>40s}  {d['description']}")
        if d["stats"]:
            print(
                " " * 28
                + ", ".join(
                    f"{k}: {v['clips']} clips / {v['hours']} h" for k, v in d["stats"].items()
                )
            )
        print(" " * 28 + f"licence: {d['license']}")


def _fetch(names: list[str], clean_raw: bool, allow_pi: bool, workers: int) -> None:
    from whistle_house.external import dataset_names, get_dataset
    from whistle_house.settings import external_root, is_raspberry_pi

    if is_raspberry_pi() and not allow_pi:
        raise SystemExit(
            "Refusing to fetch datasets on a Raspberry Pi (they are tens of GB). Run this on the "
            "training machine, or pass --allow-pi if you really mean it."
        )
    root = external_root()
    if names == ["all"]:
        names = dataset_names()
    for name in names:
        ds = get_dataset(name, root)
        print(f"== {name}: {ds.description}\n   licence: {ds.license}")
        if ds.is_prepared():
            print("   already prepared (delete its prepared/ dir to redo)")
            continue
        ds.download()
        print("   preparing ...")
        ds.prepare(workers=workers)
        if clean_raw:
            ds.clean_raw()


def cmd_datasets_fetch(args) -> None:
    _fetch(args.names, args.clean_raw, args.allow_pi, args.workers)


def cmd_datasets_labels(args) -> None:
    """Pre-compute pitch auto-labels (otherwise done at the start of training)."""
    from whistle_house.data.store import RecordingStore
    from whistle_house.data.training import GeneratorConfig, collect_sources
    from whistle_house.pipeline import PipelineStore
    from whistle_house.settings import external_root

    paths = _paths(args)
    spec = PipelineStore(paths.pipelines).get(args.pipeline)
    ext = _external_list(args.external)
    src = collect_sources(
        spec.frontend_dict(),
        ("train", "val", "test"),
        store=RecordingStore(paths.recordings),
        label_cache_dir=paths.label_cache,
        external_root=external_root(required=bool(ext)),
        external_datasets=ext,
        noise_dirs=[paths.noise],
        config=GeneratorConfig(classes=tuple(_classes(args.classes) or ["whistle"])),
        workers=args.workers,
        data_dir=paths.root,
    )
    print(src.summary())


def cmd_warmstart(args) -> None:
    """fetch external datasets -> pretrain on them -> fine-tune on this mic's recordings."""
    import shutil

    from whistle_house.data.store import RecordingStore
    from whistle_house.ml.model import ARCHS
    from whistle_house.ml.train import train
    from whistle_house.pipeline import PipelineStore

    paths = _paths(args)
    if not args.skip_fetch:
        _fetch(args.datasets.split(","), args.clean_raw, args.allow_pi, args.workers)
    spec = PipelineStore(paths.pipelines).get(args.pipeline)
    ext = _external_list(args.datasets if args.datasets != "all" else "all")

    pre_name = f"{args.name}-pretrain"
    print(f"\n== pretraining {pre_name} on {ext} + synthetic ({args.pretrain_epochs} epochs)")
    train(
        spec,
        paths,
        pre_name,
        _train_config(
            args,
            mix="pretrain",
            external=ext,
            epochs=args.pretrain_epochs,
            patience=args.patience,
            wandb_group=args.name,
        ),
        resume=args.resume,
    )

    classes = _classes(args.classes) or list(ARCHS[args.arch].get("classes", ("whistle",)))
    has_mic = any(r.label in classes for r in RecordingStore(paths.recordings).list())
    if not has_mic:
        shutil.copy(paths.models / f"{pre_name}.pt", paths.models / f"{args.name}.pt")
        print(
            f"\nNo mic {'/'.join(classes)} recordings in {paths.recordings}: skipped fine-tuning, saved the "
            f"pretrained model as {args.name}.pt. Collect data on the Pi, copy data/ here and run "
            f"`whistle-house train-detector --mix finetune --init-from {pre_name} "
            f"--external all --name {args.name}`."
        )
        return
    print(
        f"\n== fine-tuning {args.name} on mic recordings (up to {args.finetune_epochs} epochs of "
        f"{args.finetune_steps_per_epoch} steps, patience {args.finetune_patience})"
    )
    train(
        spec,
        paths,
        args.name,
        _train_config(
            args,
            mix="finetune",
            external=ext,
            init_from=pre_name,
            epochs=args.finetune_epochs,
            steps_per_epoch=args.finetune_steps_per_epoch,
            patience=args.finetune_patience,
            lr=args.lr / 4,
            wandb_group=args.name,
        ),
        resume=args.resume,
    )
    print(f"\nDone: {paths.models / args.name}.pt (+ .json). Copy both to the Pi's data/models/.")


def cmd_noise_list(args) -> None:
    from whistle_house.noise_profiles import NoiseProfileStore

    for p in NoiseProfileStore(_paths(args).noise_profiles).list():
        _, mu, sd = p.arrays()
        print(
            f"{p.name:20s} {p.duration_s:6.1f}s  {p.n_frames} frames ({p.n_dropped} dropped)  "
            f"noise {mu.min():.0f}..{mu.max():.0f} dB ±{sd.mean():.1f}  "
            f"after {[f['name'] for f in p.upstream['filters']] or 'transform'}  {p.created}"
        )


def cmd_noise_create(args) -> None:
    from whistle_house.data.store import RecordingStore
    from whistle_house.noise_profiles import (
        NoiseProfileStore,
        attach_gate,
        gate_position,
        profile_from_recordings,
    )
    from whistle_house.pipeline import PipelineSpec, PipelineStore

    paths = _paths(args)
    recs = RecordingStore(paths.recordings)
    pipes = PipelineStore(paths.pipelines)
    spec = pipes.get(args.pipeline).model_dump()
    ids = args.recordings.split(",") if args.recordings else [r.id for r in recs.list("silence")]
    pos = gate_position(spec) if args.position is None else args.position
    store = NoiseProfileStore(paths.noise_profiles)
    try:
        if store.exists(args.name):
            raise FileExistsError(f"noise profile {args.name!r} already exists (pick a new name)")
        prof = profile_from_recordings(args.name, spec, recs, ids, pos, paths.root)
        store.save(prof)
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        raise SystemExit(f"error: {e}") from None
    print(
        f"saved {paths.noise_profiles / args.name}.json from {len(ids)} recording(s), "
        f"{prof.n_frames} frames ({prof.n_dropped} loud frames dropped), measured after "
        f"{[f['name'] for f in spec['filters'][:pos]] or 'the transform'}"
    )
    if args.add:
        new = PipelineSpec.model_validate(attach_gate(spec, args.name, pos))
        pipes.put(new)
        print(f"pipeline {new.name}: filters now {[f.name for f in new.filters]}")
    else:
        print(f"use it with a spectral_gate filter at position {pos} (or re-run with --add)")


def cmd_noise_delete(args) -> None:
    from whistle_house.noise_profiles import NoiseProfileStore

    NoiseProfileStore(_paths(args).noise_profiles).delete(args.name)


def cmd_hum_labels(args) -> None:
    """pYIN pitch labels for hum recordings: voiced %, pitch range, harmonic profile."""
    import numpy as np

    from whistle_house.audio.io import to_mono_float
    from whistle_house.data.hum_labels import ensure_hum_labels, harmonic_levels, summarize
    from whistle_house.data.store import RecordingStore

    paths = _paths(args)
    store = RecordingStore(paths.recordings)
    ids = args.recordings.split(",") if args.recordings else None
    labels = ensure_hum_labels(store, paths.label_cache, ids, workers=args.workers)
    if not labels:
        raise SystemExit("no hum recordings (record some with label 'hum', or pass --recordings)")
    print(
        f"{'id':14s} {'split':5s} {'dur':>5s} {'voiced':>6s} {'f0 p5..p50..p95 (Hz)':>22s}  "
        f"harmonics h1..h{args.harmonics} (dB vs strongest)"
    )
    for rid, f0 in labels.items():
        rec = store.get(rid)
        s = summarize(f0)
        rng = (
            f"{s['f0_p5']:5.0f}..{s['f0_median']:4.0f}..{s['f0_p95']:4.0f}"
            if "f0_median" in s
            else "-"
        )
        audio, sr = store.load(rid)
        h = harmonic_levels(to_mono_float(audio), sr, f0, args.harmonics)
        hs = " ".join("  ." if np.isnan(v) else f"{v:3.0f}" for v in h)
        print(
            f"{rid:14s} {rec.split:5s} {rec.duration:5.1f} {s['voiced_fraction']:6.0%} "
            f"{rng:>22s}  {hs}"
        )


def cmd_inspect_model(args) -> None:
    """Layer-by-layer report of a whistle + hum checkpoint on the mic recordings."""
    from whistle_house.data.store import RecordingStore
    from whistle_house.ml.inspect import build_report
    from whistle_house.pipeline import PipelineStore

    paths = _paths(args)
    ck = Path(args.checkpoint)
    if not ck.exists():
        ck = paths.models / (ck.name if ck.suffix == ".pt" else f"{ck.name}.pt")
    if not ck.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    base = PipelineStore(paths.pipelines).get(args.pipeline) if args.pipeline else None
    out = Path(args.out) if args.out else paths.reports / ck.stem
    build_report(
        ck, RecordingStore(paths.recordings), out, paths.root, base, args.max_clip_s, print
    )


def cmd_export_onnx(args) -> None:
    """Export checkpoints to ONNX for the `learned` detector's onnx backend."""
    from whistle_house.ml.export import export_onnx

    paths = _paths(args)
    names = args.checkpoints
    if names == ["all"]:
        names = sorted(p.stem for p in paths.models.glob("*.pt"))
    for name in names:
        ck = Path(name)
        if not ck.exists():
            ck = paths.models / (ck.name if ck.suffix == ".pt" else f"{ck.name}.pt")
        if not ck.exists():
            raise SystemExit(f"checkpoint not found: {name}")
        out = export_onnx(ck)
        print(f"{ck.name} -> {out}  ({out.stat().st_size / 1024:.0f} kB, matches PyTorch)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="whistle-house")
    ap.add_argument("--data", help="data directory (default: $WHISTLE_HOUSE_DATA or ./data)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the live engine + manager API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", help="input device name substring")
    p.add_argument("--no-live", action="store_true", help="don't open the microphone")
    p.add_argument("--replay", help="loop this audio file in real time instead of using the mic")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("devices", help="list audio input devices")
    p.set_defaults(fn=cmd_devices)

    p = sub.add_parser("analyze", help="run a saved pipeline over an audio file")
    p.add_argument("file")
    p.add_argument("--pipeline", default="default")
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("import", help="add audio files to the recording dataset")
    p.add_argument("files", nargs="+")
    p.add_argument("--label", required=True)
    p.add_argument("--tune", help="tune id (for tune examples)")
    p.set_defaults(fn=cmd_import)

    def train_args(p, epochs=20, steps=100, batch=16, val=64):
        p.add_argument("--pipeline", default="default", help="pipeline whose front-end to train on")
        p.add_argument("--name", default="whistlenet")
        p.add_argument("--epochs", type=int, default=epochs)
        p.add_argument("--steps-per-epoch", type=int, default=steps)
        p.add_argument("--batch-size", type=int, default=batch)
        p.add_argument("--lr", type=float, default=2e-3)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
        p.add_argument("--workers", type=int, default=4, help="example-generation processes")
        p.add_argument(
            "--arch",
            default="small",
            help="model: small (whistle-only Pi model) | large (whistle-only reference model "
            "with lookahead) | harmonic (whistles + hums; needs a log-frequency pipeline such "
            "as stft_log) | harmonic_deep (harmonic with 10 blocks: 1.45 s of context instead "
            "of 0.73 s). Ignored with --init-from, which keeps the checkpoint's architecture",
        )
        p.add_argument(
            "--classes",
            help="sounds to detect, comma list of whistle,hum (default: the arch's own)",
        )
        p.add_argument(
            "--val-examples", type=int, default=val, help="size of the fixed validation set"
        )
        p.add_argument(
            "--patience",
            type=int,
            default=0,
            help="stop after this many validations (epochs) without a better val F1; 0 = never",
        )
        p.add_argument(
            "--wandb-project",
            default=os.environ.get("WHISTLE_HOUSE_WANDB_PROJECT") or None,
            help="log metrics to this Weights & Biases project (needs the wandb extra; default: "
            "$WHISTLE_HOUSE_WANDB_PROJECT, else off)",
        )
        p.add_argument(
            "--resume",
            action="store_true",
            help="continue an interrupted run (same command, same settings) from "
            "models/<name>.resume.ckpt; a stage that already finished is skipped",
        )

    p = sub.add_parser("train-detector", help="train the learned detector (needs ml extra)")
    train_args(p)
    p.add_argument("--mix", default="mic", help="data mix preset: mic | pretrain | finetune")
    p.add_argument("--external", help="external datasets to use: 'all' or comma list")
    p.add_argument("--init-from", help="checkpoint (path or model name) to fine-tune from")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("datasets", help="external datasets (training machine only)")
    dsub = p.add_subparsers(dest="dcmd", required=True)
    q = dsub.add_parser("list", help="show datasets and their status")
    q.set_defaults(fn=cmd_datasets_list)
    q = dsub.add_parser("fetch", help="download + prepare datasets into $WHISTLE_HOUSE_EXTERNAL")
    q.add_argument("names", nargs="+", help="dataset names, or 'all'")
    q.add_argument("--clean-raw", action="store_true", help="delete raw downloads after prepare")
    q.add_argument("--allow-pi", action="store_true", help="allow running on a Raspberry Pi")
    q.add_argument("--workers", type=int, default=8)
    q.set_defaults(fn=cmd_datasets_fetch)
    q = dsub.add_parser("labels", help="pre-compute pitch auto-labels for a pipeline's front-end")
    q.add_argument("--pipeline", default="default")
    q.add_argument("--external", default="all")
    q.add_argument(
        "--classes",
        help="comma list of whistle,hum: with hum, also pYIN-label the hum clips "
        "(default: whistle)",
    )
    q.add_argument("--workers", type=int, default=8)
    q.set_defaults(fn=cmd_datasets_labels)

    p = sub.add_parser("warmstart", help="fetch datasets, pretrain on them, fine-tune on mic data")
    train_args(p, epochs=0, steps=500, batch=32, val=256)
    p.add_argument("--datasets", default="all", help="'all' or comma list of dataset names")
    p.add_argument("--skip-fetch", action="store_true", help="datasets are already prepared")
    p.add_argument("--clean-raw", action="store_true")
    p.add_argument("--allow-pi", action="store_true")
    p.add_argument("--pretrain-epochs", type=int, default=40)
    p.add_argument(
        "--finetune-epochs", type=int, default=60, help="max epochs; --patience usually ends it"
    )
    p.add_argument(
        "--finetune-steps-per-epoch",
        type=int,
        default=50,
        help="short epochs = frequent validation, to catch where overfitting starts",
    )
    p.add_argument(
        "--finetune-patience", type=int, default=15, help="--patience for the fine-tune stage"
    )
    p.set_defaults(fn=cmd_warmstart)

    p = sub.add_parser("noise-profile", help="noise profiles for the spectral_gate filter")
    nsub = p.add_subparsers(dest="ncmd", required=True)
    q = nsub.add_parser("list", help="show saved noise profiles")
    q.set_defaults(fn=cmd_noise_list)
    q = nsub.add_parser("create", help="measure a noise profile from silence recordings")
    q.add_argument("name")
    q.add_argument("--pipeline", default="default", help="pipeline the profile is for")
    q.add_argument(
        "--recordings", help="comma-separated recording ids (default: every 'silence' recording)"
    )
    q.add_argument(
        "--position",
        type=int,
        help="filter index the gate sits at (default: the pipeline's spectral_gate, else the end)",
    )
    q.add_argument("--add", action="store_true", help="also add/point a spectral_gate at it")
    q.set_defaults(fn=cmd_noise_create)
    q = nsub.add_parser("delete", help="delete a noise profile")
    q.add_argument("name")
    q.set_defaults(fn=cmd_noise_delete)

    p = sub.add_parser("hum-labels", help="pYIN pitch labels for hum recordings (needs ml extra)")
    p.add_argument("--recordings", help="comma-separated recording ids (default: every 'hum')")
    p.add_argument("--harmonics", type=int, default=8, help="harmonics to profile")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(fn=cmd_hum_labels)

    p = sub.add_parser(
        "inspect-model",
        help="layer-by-layer report of a whistle + hum checkpoint on the recordings (ml extra)",
    )
    p.add_argument("checkpoint", help="model name or .pt path (default dir: <data>/models)")
    p.add_argument(
        "--pipeline",
        help="saved pipeline whose learned-detector params (thresholds) to use; the front end "
        "always comes from the checkpoint",
    )
    p.add_argument("--out", help="output directory (default: <data>/reports/<model>)")
    p.add_argument("--max-clip-s", type=float, default=20.0, help="seconds shown per clip")
    p.set_defaults(fn=cmd_inspect_model)

    p = sub.add_parser(
        "export-onnx",
        help="export checkpoints to ONNX (<name>.onnx next to <name>.pt) for the learned "
        "detector's onnx backend (ml extra)",
    )
    p.add_argument("checkpoints", nargs="+", help="model names or .pt paths, or 'all'")
    p.set_defaults(fn=cmd_export_onnx)

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.fn(args)


if __name__ == "__main__":
    main()
