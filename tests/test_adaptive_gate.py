"""adaptive_gate: several noise profiles, a Bayes filter picks the current one, then gates."""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from whistle_house.data.augment import synth_pseudo_speech, synth_whistle
from whistle_house.noise_profiles import (
    NoiseProfileStore,
    attach_gate,
    filter_profiles,
    gate_position,
    profile_from_audio,
)
from whistle_house.pipeline import Frontend, Pipeline, PipelineSpec, StageSpec
from whistle_house.server.app import create_app
from whistle_house.server.engine import merge_filter_state
from whistle_house.settings import Paths

SR = 44100
TRANSFORM = StageSpec(name="stft_log")


def _noise(seconds: float, seed: int, kind: str) -> np.ndarray:
    """Stationary noises with clearly different spectra (stand-ins for quiet / fridge / fan)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = np.fft.rfft(white)
    if kind in ("quiet", "fridge"):  # low tilted hiss (fridge: + a hum with harmonics, below)
        spec /= np.sqrt(1 + f / 300)
        level = 0.002
    else:  # "fan": loud broadband rumble
        spec /= np.sqrt(1 + f / 1500)
        level = 0.02
    x = np.fft.irfft(spec, n)
    x = level * x / np.std(x)
    if kind == "fridge":
        t = np.arange(n) / SR
        x += sum(0.01 / h * np.sin(2 * np.pi * 120 * h * t) for h in range(1, 5))
    return x.astype(np.float32)


KINDS = ("quiet", "fridge", "fan")


def _wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV")
    return buf.getvalue()


@pytest.fixture
def data_dir(tmp_path):
    spec = PipelineSpec(transform=TRANSFORM, filters=[]).model_dump()
    store = NoiseProfileStore(tmp_path / "noise_profiles")
    for i, k in enumerate(KINDS):
        store.save(profile_from_audio(k, spec, 0, [(k, _noise(6.0, i, k))], tmp_path))
    return tmp_path


def _frontend(data_dir, profiles=KINDS, **params) -> Frontend:
    return Frontend(
        SR,
        TRANSFORM,
        [StageSpec(name="adaptive_gate", params={"profiles": list(profiles), **params})],
        data_dir,
    )


def _posteriors(fe: Frontend, audio: np.ndarray, chunk: int = 4096):
    fe.reset()
    probs, fg = [], []
    for i in range(0, len(audio), chunk):
        fe.push(audio[i : i + chunk])
        (t,) = fe.telemetry()
        probs += t["series"]["probs"]
        fg += t["series"]["foreground"]
    return np.array(probs), np.array(fg)


def test_streaming_matches_offline(data_dir):
    audio = np.concatenate([_noise(2.0, 10, "quiet"), _noise(2.0, 11, "fan")])
    audio[SR : SR + len(synth_whistle(SR, 1.0, np.random.default_rng(1)).audio)] += (
        0.05 * synth_whistle(SR, 1.0, np.random.default_rng(1)).audio
    )
    fe = _frontend(data_dir, freq_smooth_bins=3)
    _, whole = fe.process_signal(audio)
    p_whole, fg_whole = _posteriors(fe, audio, chunk=len(audio))
    for chunk in (333, 1024, 5000):
        fe.reset()
        parts = [fe.push(audio[i : i + chunk])[1].values for i in range(0, len(audio), chunk)]
        np.testing.assert_allclose(np.concatenate(parts), whole.values, atol=1e-5)
        p, fg = _posteriors(fe, audio, chunk)
        np.testing.assert_allclose(p, p_whole, atol=2e-3)  # telemetry is rounded to 3 places
        np.testing.assert_allclose(fg, fg_whole, atol=2e-3)


@pytest.mark.parametrize("params", [{}, {"freq_smooth_bins": 3, "smooth_output": False}])
def test_one_profile_equals_spectral_gate(data_dir, params):
    sw = synth_whistle(SR, 3.0, np.random.default_rng(2), n_notes=2)
    audio = _noise(3.0, 20, "fridge") + 0.05 * sw.audio[: 3 * SR]
    sg = Frontend(
        SR,
        TRANSFORM,
        [StageSpec(name="spectral_gate", params={"profile": "fridge", **params})],
        data_dir,
    )
    ag = _frontend(data_dir, ["fridge"], **params)
    _, a = sg.process_signal(audio)
    _, b = ag.process_signal(audio)
    np.testing.assert_allclose(b.values, a.values, atol=1e-4, rtol=1e-5)


def test_picks_the_right_profile_and_follows_a_switch(data_dir):
    fe = _frontend(data_dir, dwell_s=2.0)
    for j, k in enumerate(KINDS):
        p, _ = _posteriors(fe, _noise(3.0, 30 + j, k))
        assert (p[100:].argmax(1) == j).mean() > 0.95, k
    # quiet -> fan -> fridge, unseen noise
    audio = np.concatenate(
        [_noise(4.0, 40, "quiet"), _noise(4.0, 41, "fan"), _noise(4.0, 42, "fridge")]
    )
    p, _ = _posteriors(fe, audio)
    fps = len(p) / 12.0
    for seg, j in ((0, 0), (1, 2), (2, 1)):
        lo, hi = int((seg * 4 + 1.5) * fps), int((seg + 1) * 4 * fps)
        assert (p[lo:hi].argmax(1) == j).mean() > 0.95, (seg, j)


@pytest.mark.parametrize("burst", ["speech", "tone"])
def test_foreground_burst_holds_the_profile(data_dir, burst):
    fe = _frontend(data_dir)
    noise = _noise(6.0, 50, "fridge")
    n = int(1.5 * SR)
    if burst == "speech":  # broadband harmonic stacks: called foreground
        fg_audio = 0.1 * synth_pseudo_speech(SR, 1.5, np.random.default_rng(5))[:n]
    else:  # a loud whistle-like note: only a few bins rise, explained as background + rise
        t = np.arange(n) / SR
        fg_audio = 0.1 * np.sin(2 * np.pi * (1200 + 300 * np.sin(2 * np.pi * 0.7 * t)) * t)
    audio = noise.copy()
    audio[3 * SR : 3 * SR + n] += fg_audio
    p, fg = _posteriors(fe, audio)
    fps = len(p) / 6.0
    before = slice(int(1.5 * fps), int(2.9 * fps))
    during = slice(int(3.3 * fps), int(4.4 * fps))
    after = slice(int(5.5 * fps), len(p))
    for part in (before, during, after):  # the belief over profiles holds throughout
        assert (p[part].argmax(1) == 1).all()
        assert p[part, 1].mean() > 0.95
    assert fg[before].mean() < 0.1
    if burst == "speech":  # syllables come and go: foreground in the voiced parts
        assert fg[during].max() > 0.99 and fg[during].mean() > 0.2
        assert fg[after].mean() < 0.1


def test_telemetry_shapes_and_reset(data_dir):
    fe = _frontend(data_dir)
    raw, _ = fe.push(_noise(1.0, 60, "quiet"))
    (t,) = fe.telemetry()
    assert t["index"] == 0 and t["name"] == "adaptive_gate"
    assert t["profiles"] == list(KINDS)
    probs = np.array(t["series"]["probs"])
    assert probs.shape == (raw.n_frames, 3)
    np.testing.assert_allclose(probs.sum(1), 1.0, atol=3e-3)
    assert len(t["series"]["foreground"]) == raw.n_frames
    assert len(t["mu"]) == len(t["input"]) == raw.axis.n_bins
    (info,) = fe.telemetry_info()
    assert np.array(info["means"]).shape == (3, raw.axis.n_bins)
    # a push with no complete frame still reports (empty series, last mu kept)
    fe.push(np.zeros(10, np.float32))
    (t2,) = fe.telemetry()
    assert t2["series"]["probs"] == [] and t2["mu"] == t["mu"]
    fe.reset()
    assert fe.telemetry() == []
    # no profiles: pass-through, no telemetry
    fe = _frontend(data_dir, [])
    a, b = fe.process_signal(_noise(0.5, 61, "quiet"))
    np.testing.assert_array_equal(a.values, b.values)
    assert fe.telemetry() == [] and fe.telemetry_info() == []


def test_validates_profiles(data_dir):
    with pytest.raises(ValueError, match="different transform/filter chain"):
        Frontend(
            SR,
            TRANSFORM,
            [
                StageSpec(name="frame_median"),
                StageSpec(name="adaptive_gate", params={"profiles": ["quiet"]}),
            ],
            data_dir,
        )
    with pytest.raises(ValueError, match="bins"):
        Frontend(
            SR,
            StageSpec(name="stft_mel"),
            [StageSpec(name="adaptive_gate", params={"profiles": ["quiet"]})],
            data_dir,
        )
    with pytest.raises(ValueError, match="once"):
        _frontend(data_dir, ["quiet", "quiet"])
    with pytest.raises(KeyError):
        _frontend(data_dir, ["nope"])


def test_profile_helpers_know_the_adaptive_gate():
    spec = {
        "filters": [
            {"name": "identity", "params": {}},
            {"name": "adaptive_gate", "params": {"profiles": ["a"]}},
        ]
    }
    assert gate_position(spec) == 1
    new = attach_gate(spec, "b")
    assert new["filters"][1]["params"]["profiles"] == ["a", "b"]
    assert attach_gate(new, "b")["filters"][1]["params"]["profiles"] == ["a", "b"]
    assert filter_profiles(new["filters"][1]) == ["a", "b"]
    assert filter_profiles({"name": "spectral_gate", "params": {"profile": "x"}}) == ["x"]
    assert filter_profiles({"name": "spectral_gate", "params": {}}) == []


def test_pipeline_step_carries_filter_state(data_dir):
    spec = PipelineSpec(
        transform=TRANSFORM,
        filters=[StageSpec(name="adaptive_gate", params={"profiles": ["quiet", "fan"]})],
    )
    pipe = Pipeline(spec, data_dir=data_dir)
    r = pipe.step(_noise(0.5, 70, "fan"))
    (st,) = r.filter_state
    assert st["profiles"] == ["quiet", "fan"] and len(st["series"]["probs"]) == r.filtered.n_frames


def test_engine_accumulates_filter_state_between_publishes():
    acc: dict = {}
    step = {"index": 1, "name": "adaptive_gate", "profiles": ["a", "b"], "mu": [1.0]}
    merge_filter_state(acc, [{**step, "series": {"probs": [[1, 0]], "foreground": [0.1]}}])
    merge_filter_state(
        acc, [{**step, "series": {"probs": [[0, 1]] * 2, "foreground": [0.2] * 2}, "mu": [2.0]}]
    )
    (st,) = acc.values()
    assert st["series"]["probs"] == [[1, 0], [0, 1], [0, 1]]
    assert st["series"]["foreground"] == [0.1, 0.2, 0.2] and st["mu"] == [2.0]


def test_server_measures_at_an_adaptive_gate_and_guards_its_profiles(tmp_path):
    with TestClient(create_app(Paths(tmp_path), live=False)) as c:
        ids = []
        for i, k in enumerate(("quiet", "fan")):
            r = c.post(
                "/api/recordings/upload",
                files={"file": (f"{k}.wav", _wav(_noise(3.0, 80 + i, k)), "audio/wav")},
                data={"label": "silence"},
            )
            ids.append(r.json()["id"])
        spec = c.get("/api/pipelines").json()["pipelines"][0]
        spec["transform"] = {"name": "stft_log", "params": {}}
        spec["filters"] = [{"name": "adaptive_gate", "params": {"profiles": []}}]
        for k, rid in zip(("quiet", "fan"), ids, strict=True):
            body = {"name": k, "recording_ids": [rid], "spec": spec, "position": 0}
            assert c.post("/api/noise-profiles", json=body).status_code == 200
        spec["filters"][0]["params"]["profiles"] = ["quiet", "fan"]
        assert c.put("/api/pipelines/adaptive", json=spec).status_code == 200
        r = c.post("/api/analyze", json={"recording_id": ids[1], "spec": spec})
        assert r.status_code == 200
        assert c.delete("/api/noise-profiles/fan").status_code == 400  # in use
        c.delete("/api/pipelines/adaptive")
        assert c.delete("/api/noise-profiles/fan").json() == {"ok": True}
