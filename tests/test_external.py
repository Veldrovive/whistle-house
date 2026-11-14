import csv
import json

import numpy as np
import pytest
import soundfile as sf

from whistle_house.data.augment import synth_pseudo_speech, synth_whistle
from whistle_house.data.store import RecordingStore
from whistle_house.data.training import (
    ExampleGenerator,
    LabelCache,
    collect_sources,
    config_for_mix,
    frontend_hash,
)
from whistle_house.external import dataset_status, get_dataset, load_manifest
from whistle_house.pipeline import PipelineSpec
from whistle_house.settings import (
    ExternalNotConfigured,
    Paths,
    external_root,
    load_env,
    parse_env_file,
)

FE = PipelineSpec().frontend_dict()


def test_parse_env_file(monkeypatch):
    monkeypatch.setenv("HOME", "/home/x")
    env = parse_env_file(
        "# comment\nexport A=1\nB = 'two words'\nC=\"q\"\nD=~/data  # trailing\n\nnot a line\n"
    )
    assert env == {"A": "1", "B": "two words", "C": "q", "D": "/home/x/data"}


def test_load_env_sets_external_root(tmp_path, monkeypatch):
    monkeypatch.delenv("WHISTLE_HOUSE_EXTERNAL", raising=False)
    with pytest.raises(ExternalNotConfigured):
        external_root()
    (tmp_path / ".env").write_text(f"WHISTLE_HOUSE_EXTERNAL={tmp_path}/ext\n")
    load_env(tmp_path / ".env")
    assert external_root() == (tmp_path / "ext").resolve()
    monkeypatch.delenv("WHISTLE_HOUSE_EXTERNAL")


def _fake_mlend(root, n_whistle=4, n_hum=3, sr=44100):
    """Raw MLEnd layout (as download() leaves it) filled with synthetic audio."""
    raw = root / "raw" / "mlend"
    (raw / "audio").mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n_whistle + n_hum):
        whistle = i < n_whistle
        fn = f"{i:04d}.wav"
        if whistle and i == 0:
            audio = 0.001 * rng.standard_normal(sr * 3)  # a "whistle" clip with no whistling
        elif whistle:
            audio = synth_whistle(sr, 3.0, rng, n_notes=4).audio
        else:
            audio = synth_pseudo_speech(sr, 3.0, rng)
        stereo = np.stack([audio, audio], 1)  # exercise down-mixing
        sf.write(raw / "audio" / fn, stereo, sr)
        rows.append(
            {
                "filename": fn,
                "Interpreter": str(i % 3),
                "Song": "Potter",
                "Interpretation": "Whistle" if whistle else "Hum",
                "Benchmark_A": "Train",
                "Benchmark_B": "Train",
            }
        )
    with (raw / "MLEndHWD_audio_attributes_benchmark.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fake_esc50(root):
    base = root / "raw" / "esc50" / "ESC-50-master"
    (base / "audio").mkdir(parents=True)
    (base / "meta").mkdir()
    rng = np.random.default_rng(1)
    with (base / "meta" / "esc50.csv").open("w") as f:
        f.write("filename,fold,target,category,esc10,src_file,take\n")
        for i, fold in enumerate([1, 4, 5]):
            fn = f"{fold}-x-{i}.wav"
            sf.write(base / "audio" / fn, 0.1 * rng.standard_normal(44100 * 5), 44100)
            f.write(f"{fn},{fold},0,dog,False,x,A\n")


def test_prepare_and_manifest(tmp_path):
    _fake_mlend(tmp_path)
    _fake_esc50(tmp_path)
    for name in ("mlend_whistles", "mlend_hums", "esc50"):
        get_dataset(name, tmp_path).prepare(workers=2)

    w = load_manifest(tmp_path / "prepared" / "mlend_whistles")
    h = load_manifest(tmp_path / "prepared" / "mlend_hums")
    assert [e.label for e in w] == ["whistle"] * 4 and [e.label for e in h] == ["hum"] * 3
    assert all(e.sample_rate == 22050 and abs(e.duration - 3.0) < 0.01 for e in w)
    info = sf.info(str(tmp_path / "prepared" / "mlend_whistles" / w[0].path))
    assert info.channels == 1 and info.format == "FLAC"

    e = load_manifest(tmp_path / "prepared" / "esc50")
    assert sorted(x.split for x in e) == ["test", "train", "val"]

    status = {d["name"]: d for d in dataset_status(tmp_path)}
    assert status["mlend_whistles"]["prepared"] and not status["musan"]["prepared"]
    assert status["esc50"]["stats"]["noise"]["clips"] == 3


def test_collect_sources_labels_and_qa(tmp_path):
    _fake_mlend(tmp_path)
    for name in ("mlend_whistles", "mlend_hums"):
        get_dataset(name, tmp_path).prepare()
    cfg = config_for_mix("pretrain")
    # MLEnd has no val speakers with these ids; collect every split for the test
    src = collect_sources(
        FE,
        ("train", "val", "test"),
        external_root=tmp_path,
        external_datasets=["mlend_whistles", "mlend_hums"],
        config=cfg,
        workers=2,
    )
    assert len(src.ext_positives) == 3  # the silent "whistle" clip failed label QA
    assert len(src.interferers["hum"]) == 3
    cache = LabelCache(tmp_path / "labels", "mlend_whistles", frontend_hash(FE, cfg.labeler_params))
    assert len(cache.labels) == 4  # cached (including the rejected one)

    gen = ExampleGenerator(FE, src, cfg)
    rng = np.random.default_rng(0)
    xs, f0s = gen.batch(rng, 8)
    assert xs.shape[2] == 128 and (f0s > 0).any()


def test_train_pretrain_then_finetune(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from whistle_house.ml.train import TrainConfig, train

    ext = tmp_path / "ext"
    _fake_mlend(ext)
    get_dataset("mlend_whistles", ext).prepare()
    monkeypatch.setenv("WHISTLE_HOUSE_EXTERNAL", str(ext))
    paths = Paths(tmp_path / "data").ensure()
    store = RecordingStore(paths.recordings)
    for i in range(2):
        store.add(
            synth_whistle(44100, 3.0, np.random.default_rng(10 + i)).audio,
            44100,
            label="whistle",
            split="train",
        )
    store.add(
        synth_pseudo_speech(44100, 3.0, np.random.default_rng(3)),
        44100,
        label="speech",
        split="train",
    )

    small = {"epochs": 1, "steps_per_epoch": 2, "batch_size": 2, "val_examples": 4, "device": "cpu"}
    spec = PipelineSpec()
    train(
        spec,
        paths,
        "pre",
        TrainConfig(**small, mix="pretrain", external=["mlend_whistles"], workers=0),
    )
    status = json.loads((paths.models / "pre.json").read_text())
    assert status["state"] == "done" and status["data"]["train"]["external_whistle_clips"] == 3

    train(
        spec,
        paths,
        "ft",
        TrainConfig(
            **small, mix="finetune", external=["mlend_whistles"], init_from="pre", workers=2
        ),
    )
    status = json.loads((paths.models / "ft.json").read_text())
    assert status["state"] == "done" and status["init_from"].endswith("pre.pt")
    assert (paths.models / "ft.pt").exists()

    other = PipelineSpec(transform={"name": "stft_mel", "params": {"n_mels": 64}})
    with pytest.raises(ValueError, match="different front-end"):
        train(other, paths, "bad", TrainConfig(**small, init_from="pre", workers=0))
