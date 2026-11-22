"""Model report (ml/inspect.py): the internals mirror the model, and a report builds + serves."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from whistle_house.data.augment import synth_hum, synth_pseudo_speech, synth_whistle
from whistle_house.data.store import RecordingStore
from whistle_house.pipeline import Frontend, PipelineSpec, StageSpec
from whistle_house.settings import Paths

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")

SR = 44100
FE = PipelineSpec(transform=StageSpec(name="stft_log"), filters=[]).frontend_dict()


def _checkpoint(path, arch: str = "harmonic"):
    """An untrained checkpoint of ``arch`` on the stft_log front end."""
    from whistle_house.ml.model import (
        ARCHS,
        WhistleNet,
        WhistleNetConfig,
        log_bins_per_octave,
        save_checkpoint,
    )

    axis = Frontend.from_dict(FE).axis
    kw = dict(ARCHS[arch])
    if kw.get("harmonics"):
        kw["bins_per_octave"] = log_bins_per_octave(axis.freqs)
    torch.manual_seed(0)
    model = WhistleNet(WhistleNetConfig(n_bins=axis.n_bins, **kw)).eval()
    save_checkpoint(path, model, (-40.0, 20.0), FE, {})
    return model


def test_internals_match_model(tmp_path):
    from whistle_house.ml.inspect import _internals

    model = _checkpoint(tmp_path / "m.pt")
    x = np.random.default_rng(0).normal(size=(80, model.cfg.n_bins)).astype(np.float32)
    out = _internals(model, x)
    with torch.no_grad():
        pres, sal = model(torch.from_numpy(x)[None])
    np.testing.assert_allclose(out["logits"], pres[0].numpy(), atol=1e-5)
    np.testing.assert_allclose(out["sal"], torch.sigmoid(sal[0]).numpy(), atol=1e-5)
    assert len(out["blocks"]) == len(model.cfg.dilations)
    assert out["stack"].shape == (len(model.cfg.harmonics), 80, model.cfg.n_bins)


def _store(root) -> RecordingStore:
    rng = np.random.default_rng(3)
    store = RecordingStore(root / "recordings")
    store.add(synth_hum(SR, 2.0, rng, n_notes=2).audio, SR, label="hum")
    store.add(synth_whistle(SR, 2.0, rng, n_notes=2).audio, SR, label="whistle")
    store.add(synth_pseudo_speech(SR, 2.0, rng), SR, label="speech")
    store.add((0.003 * rng.standard_normal(SR)).astype(np.float32), SR, label="silence")
    return store


def test_build_report(tmp_path):
    from whistle_house.ml.inspect import build_report

    paths = Paths(tmp_path).ensure()
    _checkpoint(paths.models / "hn.pt")
    store = _store(tmp_path)
    out = paths.reports / "hn"
    data = build_report(paths.models / "hn.pt", store, out, tmp_path, log=lambda _: None)

    assert (out / "index.html").read_text().startswith("<meta charset")
    assert json.loads((out / "data.json").read_text())["checkpoint"] == "hn.pt"
    assert {r["label"] for r in data["summary"]} == {"hum", "whistle", "speech", "silence"}
    assert len(data["clips"]) == 4 and all(
        (out / "clips" / f"{c['id']}.json").exists() for c in data["clips"]
    )
    hum = next(r for r in data["summary"] if r["label"] == "hum")
    assert hum["labelled"] > 0.5  # pYIN reference on the synthetic hum
    assert len(data["comb"]["hum@f0"]["median"]) == len(data["arch"]["harmonics"])
    clip = json.loads((out / "clips" / f"{data['clips'][0]['id']}.json").read_text())
    assert clip["spec"]["png"].startswith("data:image/png;base64,")
    assert len(clip["prob"]["hum"]) == clip["T"]


def test_report_needs_harmonic_model(tmp_path):
    from whistle_house.ml.inspect import build_report

    paths = Paths(tmp_path).ensure()
    _checkpoint(paths.models / "w.pt", arch="small")
    with pytest.raises(ValueError, match="harmonic"):
        build_report(paths.models / "w.pt", _store(tmp_path), paths.reports / "w", tmp_path)


def test_report_routes(tmp_path):
    from whistle_house.server.app import create_app

    paths = Paths(tmp_path).ensure()
    (paths.models / "hn.json").write_text(json.dumps({"name": "hn", "history": []}))
    (paths.reports / "hn" / "clips").mkdir(parents=True)
    (paths.reports / "hn" / "index.html").write_text("<p>report</p>")
    (paths.reports / "hn" / "data.json").write_text("{}")
    (tmp_path / "secret.txt").write_text("no")
    with TestClient(create_app(paths, live=False)) as c:
        st = c.get("/api/models").json()[0]["report"]
        assert st["exists"] and not st["running"] and st["built"]
        assert c.get("/api/models/hn/report/index.html").text == "<p>report</p>"
        assert c.get("/api/models/hn/report/data.json").json() == {}
        assert c.get("/api/models/hn/report/..%2F..%2Fsecret.txt").status_code == 404
        assert c.post("/api/models/hn/report").status_code == 404  # no checkpoint
