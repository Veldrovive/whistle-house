"""Resuming an interrupted training run (``train(..., resume=True)`` / ``--resume``)."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")

from test_harmonic import LOG_SPEC, _store_with_hums

from whistle_house.ml import train as train_mod
from whistle_house.ml.train import TrainConfig, train

TINY = {
    "epochs": 3,
    "steps_per_epoch": 2,
    "batch_size": 2,
    "val_examples": 4,
    "device": "cpu",
    "workers": 0,
    "arch": "harmonic",
}


class Crash(RuntimeError):
    pass


def _crash_at_step(monkeypatch, n: int) -> None:
    """Make the n-th training step (1-based) raise, as a crash would."""
    calls = {"n": 0}
    real = train_mod._targets

    def targets(*a, **kw):
        calls["n"] += 1
        if calls["n"] == n:
            raise Crash("boom")
        return real(*a, **kw)

    monkeypatch.setattr(train_mod, "_targets", targets)


def test_resume_continues_after_the_last_finished_epoch(tmp_path, monkeypatch):
    paths, _ = _store_with_hums(tmp_path)
    cfg = TrainConfig(**TINY)
    state = paths.resume_state("r")

    with monkeypatch.context() as m:
        _crash_at_step(m, 5)  # first step of epoch 3
        with pytest.raises(Crash):
            train(LOG_SPEC, paths, "r", cfg)
    status = json.loads((paths.models / "r.json").read_text())
    assert status["state"] == "failed" and [h["epoch"] for h in status["history"]] == [1, 2]
    saved = torch.load(state, weights_only=False)
    assert saved["epoch"] == 2 and not saved["done"]
    assert saved["sched"]["last_epoch"] == 4  # 2 epochs x 2 steps

    # other settings are refused; device / workers / patience may change
    with pytest.raises(ValueError, match="epochs"):
        train(LOG_SPEC, paths, "r", TrainConfig(**{**TINY, "epochs": 5}), resume=True)

    train(LOG_SPEC, paths, "r", TrainConfig(**{**TINY, "patience": 9}), resume=True)
    status = json.loads((paths.models / "r.json").read_text())
    assert status["state"] == "done" and "error" not in status
    assert [h["epoch"] for h in status["history"]] == [1, 2, 3]
    assert [r["epoch"] for r in status["resumed"]] == [2]
    saved = torch.load(state, weights_only=False)
    assert saved["done"] and saved["epoch"] == 3 and saved["sched"]["last_epoch"] == 6
    assert (paths.models / "r.pt").exists()

    # resuming a finished run does nothing
    before = (paths.models / "r.pt").stat().st_mtime_ns, (paths.models / "r.json").read_text()
    train(LOG_SPEC, paths, "r", cfg, resume=True)
    assert ((paths.models / "r.pt").stat().st_mtime_ns, (paths.models / "r.json").read_text()) == (
        before
    )

    # a fresh run (no resume) forgets the old state at once, so a later --resume can't pick it up
    with monkeypatch.context() as m:
        _crash_at_step(m, 1)
        with pytest.raises(Crash):
            train(LOG_SPEC, paths, "r", cfg)
    assert not state.exists()


def test_resumed_finetune_skips_the_init_validation(tmp_path, monkeypatch):
    paths, _ = _store_with_hums(tmp_path)
    train(LOG_SPEC, paths, "pre", TrainConfig(**{**TINY, "epochs": 1}))
    ft = TrainConfig(**{**TINY, "init_from": "pre", "mix": "finetune"})
    with monkeypatch.context() as m:
        _crash_at_step(m, 3)  # epoch 2
        with pytest.raises(Crash):
            train(LOG_SPEC, paths, "ft", ft)
    train(LOG_SPEC, paths, "ft", ft, resume=True)
    status = json.loads((paths.models / "ft.json").read_text())
    assert [h["epoch"] for h in status["history"]] == [0, 1, 2, 3]


def test_wandb_run_is_continued_on_resume(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    paths, _ = _store_with_hums(tmp_path)
    cfg = TrainConfig(**{**TINY, "log_every": 1}, wandb_project="test", wandb_group="g")
    with monkeypatch.context() as m:
        _crash_at_step(m, 5)
        with pytest.raises(Crash):
            train(LOG_SPEC, paths, "w", cfg)
    first = json.loads((paths.models / "w.json").read_text())["wandb"]
    assert first["project"] == "test" and first["id"]
    train(LOG_SPEC, paths, "w", cfg, resume=True)
    status = json.loads((paths.models / "w.json").read_text())
    assert status["state"] == "done" and status["wandb"]["id"] == first["id"]
    runs = list((tmp_path / "wandb").glob(f"offline-run-*-{first['id']}"))
    assert len(runs) == 2  # the crashed and the resumed process, one wandb run id
