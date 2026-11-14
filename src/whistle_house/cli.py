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


def cmd_train(args) -> None:
    from whistle_house.ml.train import TrainConfig, train
    from whistle_house.pipeline import PipelineStore

    paths = _paths(args)
    spec = PipelineStore(paths.pipelines).get(args.pipeline)
    cfg = TrainConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    path = train(spec, paths, args.name, cfg)
    print(f"saved {path}")


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

    p = sub.add_parser("train-detector", help="train the learned whistle detector (needs ml extra)")
    p.add_argument("--pipeline", default="default", help="pipeline whose front-end to train on")
    p.add_argument("--name", default="whistlenet")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.set_defaults(fn=cmd_train)

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.fn(args)


if __name__ == "__main__":
    main()
