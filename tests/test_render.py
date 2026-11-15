"""Resynthesising filtered frames back to audio (audio/render.py)."""

import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from whistle_house.audio.render import render_processed
from whistle_house.data.augment import synth_whistle
from whistle_house.noise_profiles import NoiseProfileStore, profile_from_audio
from whistle_house.pipeline import PipelineSpec, StageSpec
from whistle_house.server.app import create_app
from whistle_house.settings import Paths

SR = 44100
T = np.arange(2 * SR) / SR


def _tone(hz: float, amp: float = 0.3) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * hz * T)).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _mid(x: np.ndarray) -> np.ndarray:
    return x[SR // 4 : -SR // 4]  # skip edges (zero-history start, partial last frame)


def test_no_filters_passes_transform_band_only():
    spec = PipelineSpec(filters=[])
    inband = _tone(1000)
    out = render_processed(inband, spec)
    assert out.shape == inband.shape
    assert abs(_rms(_mid(out)) / _rms(_mid(inband)) - 1) < 0.05
    assert np.corrcoef(_mid(out), _mid(inband))[0, 1] > 0.99
    # 100 Hz is below the default 300 Hz transform floor: the pipeline never sees it
    assert _rms(_mid(render_processed(_tone(100), spec))) < 0.02 * _rms(_tone(100))


def test_band_reject_removes_its_band():
    spec = PipelineSpec(filters=[StageSpec(name="band_reject", params={"bands": [[900, 1100]]})])
    out = render_processed(_tone(1000) + _tone(2000), spec)
    keep = render_processed(_tone(2000), PipelineSpec(filters=[]))
    resid = _mid(out) - _mid(keep)
    assert _rms(resid) < 0.1 * _rms(_tone(1000))


def test_gate_removes_noise_keeps_whistle(tmp_path):
    rng = np.random.default_rng(0)
    noise = lambda n: (0.02 * rng.standard_normal(n)).astype(np.float32)
    base = PipelineSpec(filters=[])
    prof = profile_from_audio("room", base.model_dump(), 0, [("s", noise(4 * SR))], tmp_path)
    NoiseProfileStore(tmp_path / "noise_profiles").save(prof)
    spec = PipelineSpec(filters=[StageSpec(name="spectral_gate", params={"profile": "room"})])

    n = noise(3 * SR)
    sw = synth_whistle(SR, 3.0, np.random.default_rng(1), n_notes=2)
    # keep the noise at the profiled level (mix_at_snr would rescale it); whistle 5 dB above
    mix = sw.audio * (_rms(n) * 10 ** (5 / 20) / _rms(sw.audio)) + n
    out = render_processed(mix, spec, tmp_path)
    clean = render_processed(mix - n, base)  # whistle alone, band-limited the same way
    voiced = sw.f0 > 0
    # noise-only = at least 0.1 s from any note (the gate's release keeps note tails on purpose)
    near = np.convolve(voiced.astype(float), np.ones(SR // 5), mode="same") > 0
    quiet = ~near
    quiet[: SR // 4] = False
    # noise-only parts: almost all removed (vs. ungated band-limited noise)
    ungated = render_processed(mix, base)
    assert _rms(out[quiet]) < 0.3 * _rms(ungated[quiet])  # >10 dB less noise (~14 dB measured)
    # whistle parts: most of the whistle survives
    assert np.corrcoef(out[voiced], clean[voiced])[0, 1] > 0.8


def test_render_endpoint(tmp_path):
    buf = io.BytesIO()
    sf.write(buf, _tone(1000), SR, format="WAV")
    with TestClient(create_app(Paths(tmp_path), live=False)) as c:
        rec = c.post(
            "/api/recordings/upload",
            files={"file": ("t.wav", buf.getvalue(), "audio/wav")},
            data={"label": "whistle"},
        ).json()
        r = c.post("/api/render", json={"recording_id": rec["id"]})
        assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
        audio, sr = sf.read(io.BytesIO(r.content))
        assert sr == SR and len(audio) == len(T)
        assert c.post("/api/render", json={"recording_id": "nope"}).status_code == 404
