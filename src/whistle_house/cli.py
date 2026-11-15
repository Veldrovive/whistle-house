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
    print(f"{len(res.events)} whistle events")
    for e in res.events:
        print(f"  {e.start:7.2f}-{e.end:7.2f}s  {e.median_hz():7.1f} Hz  conf={e.confidence:.2f}")
    print(f"{len(res.matches)} tune matches")
    for m in res.matches:
        print(f"  {m.time:7.2f}s  {m.tune_name}  score={m.score:.2f}")


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
    path = train(spec, paths, args.name, cfg)
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
    from whistle_house.data.training import collect_sources
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
        workers=args.workers,
    )
    print(src.summary())


def cmd_warmstart(args) -> None:
    """fetch external datasets -> pretrain on them -> fine-tune on this mic's recordings."""
    import shutil

    from whistle_house.data.store import RecordingStore
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
        _train_config(args, mix="pretrain", external=ext, epochs=args.pretrain_epochs),
    )

    has_mic = any(r.label == "whistle" for r in RecordingStore(paths.recordings).list())
    if not has_mic:
        shutil.copy(paths.models / f"{pre_name}.pt", paths.models / f"{args.name}.pt")
        print(
            f"\nNo mic whistle recordings in {paths.recordings}: skipped fine-tuning, saved the "
            f"pretrained model as {args.name}.pt. Collect data on the Pi, copy data/ here and run "
            f"`whistle-house train-detector --mix finetune --init-from {pre_name} "
            f"--external all --name {args.name}`."
        )
        return
    print(f"\n== fine-tuning {args.name} on mic recordings ({args.finetune_epochs} epochs)")
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
            lr=args.lr / 4,
        ),
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

    def train_args(p, epochs=20, steps=100, batch=16):
        p.add_argument("--pipeline", default="default", help="pipeline whose front-end to train on")
        p.add_argument("--name", default="whistlenet")
        p.add_argument("--epochs", type=int, default=epochs)
        p.add_argument("--steps-per-epoch", type=int, default=steps)
        p.add_argument("--batch-size", type=int, default=batch)
        p.add_argument("--lr", type=float, default=2e-3)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
        p.add_argument("--workers", type=int, default=4, help="example-generation processes")

    p = sub.add_parser("train-detector", help="train the learned whistle detector (needs ml extra)")
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
    q.add_argument("--workers", type=int, default=8)
    q.set_defaults(fn=cmd_datasets_labels)

    p = sub.add_parser("warmstart", help="fetch datasets, pretrain on them, fine-tune on mic data")
    train_args(p, epochs=0, steps=500, batch=32)
    p.add_argument("--datasets", default="all", help="'all' or comma list of dataset names")
    p.add_argument("--skip-fetch", action="store_true", help="datasets are already prepared")
    p.add_argument("--clean-raw", action="store_true")
    p.add_argument("--allow-pi", action="store_true")
    p.add_argument("--pretrain-epochs", type=int, default=40)
    p.add_argument("--finetune-epochs", type=int, default=15)
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

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.fn(args)


if __name__ == "__main__":
    main()
