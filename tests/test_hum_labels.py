"""pYIN hum labeller: follows a harmonic stack's fundamental, caches, serves via the API."""

import io

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")

from whistle_house.data.hum_labels import (
    ensure_hum_labels,
    harmonic_levels,
    hum_f0,
    summarize,
)
from whistle_house.data.store import RecordingStore
from whistle_house.data.training import LABEL_RATE

SR = 44100


def _hum(f0_hz: np.ndarray, levels_db=(0, -6, -12, -25, -30)) -> np.ndarray:
    """Harmonic stack following a per-sample f0, harmonic amplitudes from levels_db."""
    phase = 2 * np.pi * np.cumsum(f0_hz) / SR
    x = sum(10 ** (db / 20) * np.sin((h + 1) * phase) for h, db in enumerate(levels_db))
    return (0.2 * x / np.max(np.abs(x))).astype(np.float32)


def _two_notes() -> tuple[np.ndarray, np.ndarray]:
    """1 s at 110 Hz, 0.5 s silence, 1 s at 165 Hz (+ a little noise)."""
    n = SR
    f0 = np.concatenate([np.full(n, 110.0), np.zeros(n // 2), np.full(n, 165.0)])
    x = _hum(f0) * (f0 > 0)
    x += 1e-3 * np.random.default_rng(0).standard_normal(x.size).astype(np.float32)
    return x.astype(np.float32), f0


def test_hum_f0_tracks_fundamental_and_gaps():
    x, _truth = _two_notes()
    f0 = hum_f0(x, SR).astype(np.float32)
    assert len(f0) == int(len(x) / SR * LABEL_RATE) + 1
    t = np.arange(len(f0)) / LABEL_RATE
    first = (t > 0.1) & (t < 0.9)
    second = (t > 1.6) & (t < 2.4)
    gap = (t > 1.1) & (t < 1.4)
    assert np.mean(np.abs(12 * np.log2(f0[first] / 110)) < 0.3) > 0.95
    assert np.mean(np.abs(12 * np.log2(f0[second] / 165)) < 0.3) > 0.95
    assert np.all(f0[gap] == 0)


def test_hum_f0_weak_fundamental_is_not_octave_doubled():
    """h2 louder than h1 (as on 1 of the 6 mic clips): the label is still the fundamental."""
    x = _hum(np.full(SR * 2, 120.0), levels_db=(-6, 0, -10, -25))
    f0 = hum_f0(x, SR).astype(np.float32)
    assert abs(np.median(f0[f0 > 0]) - 120) < 3


def test_harmonic_levels_and_summary():
    x = _hum(np.full(SR * 2, 130.0), levels_db=(0, -6, -12, -25, -30))
    f0 = np.where(np.arange(2 * LABEL_RATE + 1) < 2 * LABEL_RATE, 130.0, 0).astype(np.float16)
    h = harmonic_levels(x, SR, f0, n_harmonics=5)
    np.testing.assert_allclose(h, [0, -6, -12, -25, -30], atol=1.5)
    s = summarize(f0)
    assert s["voiced_fraction"] > 0.99 and s["f0_median"] == pytest.approx(130, abs=0.5)


def test_ensure_hum_labels_caches(tmp_path):
    store = RecordingStore(tmp_path / "recordings")
    x, _ = _two_notes()
    rec = store.add(x, SR, label="hum")
    store.add(x, SR, label="speech")
    labels = ensure_hum_labels(store, tmp_path / "cache")
    assert list(labels) == [rec.id]
    cached = list((tmp_path / "cache").glob("hum-*.npz"))
    assert len(cached) == 1
    again = ensure_hum_labels(store, tmp_path / "cache")
    np.testing.assert_array_equal(again[rec.id], labels[rec.id])


def test_hum_pitch_api(tmp_path):
    from fastapi.testclient import TestClient

    from whistle_house.server.app import create_app
    from whistle_house.settings import Paths

    x, _ = _two_notes()
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV")
    with TestClient(create_app(Paths(tmp_path), live=False)) as c:
        rec = c.post(
            "/api/recordings/upload",
            files={"file": ("h.wav", buf.getvalue(), "audio/wav")},
            data={"label": "hum"},
        ).json()
        res = c.get(f"/api/recordings/{rec['id']}/hum-pitch").json()
        assert res["rate"] == LABEL_RATE and 0.6 < res["voiced_fraction"] < 0.9
        assert 100 < res["f0_p5"] and res["f0_p95"] < 175
        assert c.get("/api/recordings/nope/hum-pitch").status_code == 404
