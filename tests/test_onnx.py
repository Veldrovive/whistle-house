"""ONNX export: the learned detector's `onnx` backend gives what the `torch` backend gives."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from whistle_house.data.augment import synth_hum, synth_whistle
from whistle_house.pipeline import Frontend, Pipeline, PipelineSpec, StageSpec
from whistle_house.server.app import create_app
from whistle_house.settings import Paths

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

SR = 44100


def _checkpoint(tmp_path, transform="stft_mel", name="m", **arch):
    """An untrained net saved as models/<name>.pt + the pipeline spec that uses it."""
    from whistle_house.ml.model import (
        WhistleNet,
        WhistleNetConfig,
        log_bins_per_octave,
        save_checkpoint,
    )

    spec = PipelineSpec(
        transform=StageSpec(name=transform),
        detector=StageSpec(name="learned", params={"checkpoint": f"{name}.pt"}),
    )
    axis = Frontend(spec.sample_rate, spec.transform, spec.filters).axis
    if arch.get("harmonics"):
        arch["bins_per_octave"] = log_bins_per_octave(axis.freqs)
    torch.manual_seed(0)
    model = WhistleNet(WhistleNetConfig(n_bins=axis.n_bins, **arch))
    # bias the presence head so every class reports notes and the trackers have work
    torch.nn.init.constant_(model.presence.bias, 0.5)
    model.eval()
    path = tmp_path / "models" / f"{name}.pt"
    save_checkpoint(path, model, (-40.0, 20.0), spec.frontend_dict(), {})
    return path, spec


def _with(spec, **params):
    det = StageSpec(name="learned", params={**spec.detector.params, **params})
    return spec.model_copy(update={"detector": det})


def _events(result):
    return [(e.kind, round(e.start, 4), round(e.end, 4), round(e.median_hz(), 1)) for e in result]


CASES = {
    "causal": ("stft_mel", {"channels": 4, "dilations": (1, 2)}),
    "lookahead": ("stft_mel", {"channels": 4, "dilations": (1, 2, 4), "lookahead": 5}),
    "freq_dilation": ("stft_mel", {"channels": 4, "dilations": (1, 2), "dilations_f": (1, 2)}),
    "harmonic": (
        "stft_log",
        {
            "channels": 6,
            "dilations": (1, 2, 4),
            "lookahead": 3,
            "classes": ("whistle", "hum"),
            "harmonics": (1 / 3, 0.5, 1.0, 1.5, 2.0, 3.0),
            "freq_context": True,
        },
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_onnx_backend_matches_torch(tmp_path, case):
    from whistle_house.ml.export import OnnxStream, export_onnx, file_sha256

    transform, arch = CASES[case]
    path, spec = _checkpoint(tmp_path, transform, **arch)
    out = export_onnx(path)
    assert out == path.with_suffix(".onnx") and not out.with_name(out.name + ".tmp").exists()

    # the file carries everything the detector needs
    exported = OnnxStream(out)
    assert exported.classes == tuple(arch.get("classes", ("whistle",)))
    assert exported.lookahead == arch.get("lookahead", 0)
    assert exported.frontend == spec.frontend_dict()
    assert exported.source == {"name": "m.pt", "sha256": file_sha256(path)}

    rng = np.random.default_rng(0)
    audio = np.concatenate(
        [synth_whistle(SR, 1.5, rng, n_notes=3).audio, synth_hum(SR, 1.5, rng).audio]
    ).astype(np.float32)

    def run(backend, chunk):
        s = _with(spec, backend=backend).model_copy(update={"chunk_size": chunk})
        p = Pipeline(s, data_dir=tmp_path)
        r = p.analyze(audio)
        return r, p.detector.infer(r.filtered.values)

    (t, t_off), (o, o_off), (o2, _) = run("torch", 4096), run("onnx", 4096), run("onnx", 700)
    assert _events(t.events) and _events(t.events) == _events(o.events) == _events(o2.events)
    assert set(t.kind_scores) == set(o.kind_scores)
    for kind in t.kind_scores:
        np.testing.assert_allclose(o.kind_scores[kind], t.kind_scores[kind], atol=1e-4)
        np.testing.assert_allclose(o2.kind_scores[kind], t.kind_scores[kind], atol=1e-4)
    for a, b in zip(o_off, t_off, strict=True):  # offline (input-aligned) inference too
        np.testing.assert_allclose(a, b, atol=1e-4)


def test_onnx_backend_without_the_checkpoint(tmp_path, caplog):
    """The Pi has only the .onnx file; a changed checkpoint next to it is reported as stale."""
    from whistle_house.ml.export import export_onnx
    from whistle_house.ml.model import load_checkpoint, save_checkpoint

    path, spec = _checkpoint(tmp_path, channels=4, dilations=(1, 2))
    # an old checkpoint: saved before the architecture had classes / lookahead / harmonics
    ck = torch.load(path, weights_only=False)
    ck["arch"] = {k: ck["arch"][k] for k in ("n_bins", "channels", "dilations", "kernel_t")}
    torch.save(ck, path)
    onnx_spec = _with(spec, backend="onnx")
    with pytest.raises(FileNotFoundError, match="export-onnx m"):
        Pipeline(onnx_spec, data_dir=tmp_path)
    export_onnx(path)

    # retrained since the export: still runs, but says so
    model, ck = load_checkpoint(path)
    torch.nn.init.constant_(model.presence.bias, -3.0)
    original = path.read_bytes()
    save_checkpoint(path, model, (ck["norm"]["mean"], ck["norm"]["std"]), ck["frontend"], {})
    with caplog.at_level("WARNING"):
        Pipeline(onnx_spec, data_dir=tmp_path)
    assert "export it again" in caplog.text
    path.write_bytes(original)

    caplog.clear()
    path.unlink()
    audio = synth_whistle(SR, 1.0, np.random.default_rng(0), n_notes=2).audio.astype(np.float32)
    # naming the .onnx file works as well as backend="onnx"
    by_name = _with(spec, checkpoint="m.onnx")
    a = Pipeline(onnx_spec, data_dir=tmp_path).analyze(audio)
    b = Pipeline(by_name, data_dir=tmp_path).analyze(audio)
    assert _events(a.events) and _events(a.events) == _events(b.events)
    assert "export it again" not in caplog.text

    # a front end the model was not trained on is still refused
    wrong = onnx_spec.model_copy(update={"transform": StageSpec(name="stft_log")})
    with pytest.raises(ValueError, match="front-end"):
        Pipeline(wrong, data_dir=tmp_path)


def test_export_route_and_status(tmp_path):
    paths = Paths(tmp_path)
    path, _ = _checkpoint(tmp_path, name="hn", channels=4, dilations=(1, 2))
    status = {"name": "hn", "state": "done", "pipeline": "default", "config": {}, "history": []}
    (paths.models / "hn.json").write_text(json.dumps(status))
    with TestClient(create_app(paths, live=False)) as c:
        assert c.get("/api/models").json()[0]["onnx"] == {
            "exists": False,
            "exported": None,
            "stale": False,
        }
        assert c.post("/api/models/nope/export-onnx").status_code == 404
        r = c.post("/api/models/hn/export-onnx")
        assert r.status_code == 200 and r.json()["exists"] and not r.json()["stale"]
        assert (paths.models / "hn.onnx").exists()
        # the checkpoint is rewritten later: the export is out of date
        later = (paths.models / "hn.onnx").stat().st_mtime + 10
        os.utime(path, (later, later))
        assert c.get("/api/models").json()[0]["onnx"]["stale"]


def test_thread_limit_is_for_this_process_only():
    """`serve` runs BLAS and torch with one thread but leaves the environment as it was, so
    the training runs and reports it starts keep every core."""
    code = (
        "import os, sys\n"
        "from whistle_house.threads import limit_inference_threads, limit_torch\n"
        "os.environ['MKL_NUM_THREADS'] = '3'\n"
        "limit_inference_threads()\n"
        "assert 'numpy' in sys.modules\n"
        "assert 'OPENBLAS_NUM_THREADS' not in os.environ and 'OMP_NUM_THREADS' not in os.environ\n"
        "assert os.environ['MKL_NUM_THREADS'] == '3'\n"
        "import torch\n"
        "limit_torch(torch)\n"
        "assert torch.get_num_threads() == 1\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS")}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
