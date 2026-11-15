"""Noise profiles + the spectral_gate filter."""

import numpy as np
import pytest

from whistle_house.data.augment import mix_at_snr, synth_whistle
from whistle_house.data.training import frontend_hash
from whistle_house.noise_profiles import NoiseProfileStore, profile_from_audio
from whistle_house.pipeline import Frontend, Pipeline, PipelineSpec, StageSpec

SR = 44100


def _mic_noise(seconds: float, seed: int) -> np.ndarray:
    """Stationary coloured noise (more energy at low frequencies, like a cheap USB mic)."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(int(seconds * SR))
    spec = np.fft.rfft(white) / np.sqrt(1 + np.fft.rfftfreq(white.size, 1 / SR) / 300)
    x = np.fft.irfft(spec, white.size)
    return (0.02 * x / np.std(x)).astype(np.float32)


def _gate_spec(**params) -> PipelineSpec:
    return PipelineSpec(
        filters=[StageSpec(name="spectral_gate", params={"profile": "room", **params})]
    )


@pytest.fixture
def data_dir(tmp_path):
    spec = PipelineSpec(filters=[])
    prof = profile_from_audio("room", spec.model_dump(), 0, [("sil", _mic_noise(6.0, 0))], tmp_path)
    NoiseProfileStore(tmp_path / "noise_profiles").save(prof)
    return tmp_path


def test_profile_stats_and_outlier_rejection(tmp_path):
    noise = _mic_noise(6.0, 1)
    noise[SR : SR + SR // 4] += 0.5 * np.sin(np.arange(SR // 4) * 2 * np.pi * 1000 / SR)  # a bang
    p = profile_from_audio("x", PipelineSpec().model_dump(), 0, [("a", noise)], tmp_path)
    assert p.n_dropped > 0 and p.n_frames > 400
    freqs, mu, sd = p.arrays()
    assert mu.shape == sd.shape == freqs.shape and np.all(sd > 0)
    assert mu[0] > mu[-1]  # coloured noise: louder at low frequencies
    assert p.upstream["transform"]["name"] == "stft_mel"
    assert p.upstream["transform"]["params"]  # defaults filled in (canonical)


def test_profiles_are_immutable_and_named_safely(data_dir):
    store = NoiseProfileStore(data_dir / "noise_profiles")
    with pytest.raises(FileExistsError):
        store.save(store.get("room"))
    with pytest.raises(ValueError):
        store.path("../evil")
    assert [p.name for p in store.list()] == ["room"]


def test_gate_silences_mic_noise_and_keeps_whistles(data_dir):
    noise = _mic_noise(4.0, 7)  # unseen noise, same distribution
    sw = synth_whistle(SR, 4.0, np.random.default_rng(3), n_notes=3)
    gated = Pipeline(_gate_spec(), data_dir=data_dir)

    fe = gated.frontend
    _, out = fe.process_signal(noise)
    assert (out.values > 3.0).mean() < 0.01  # noise is gated to ~0 dB (soft mask, not exact 0)
    assert out.values.mean() < 0.5
    assert len(gated.analyze(noise).events) == 0

    mix = mix_at_snr(sw.audio, noise, 0.0)
    events = gated.analyze(mix).events
    assert len(events) == 3
    for ev in events:
        truth = np.median(sw.f0[int(ev.start * SR) : int(ev.end * SR)])
        assert abs(12 * np.log2(ev.median_hz() / truth)) < 1.0


def test_gate_streaming_matches_offline(data_dir):
    sw = synth_whistle(SR, 3.0, np.random.default_rng(4), n_notes=2)
    audio = mix_at_snr(sw.audio, _mic_noise(3.0, 9), 5.0)
    fe = Pipeline(_gate_spec(freq_smooth_bins=3), data_dir=data_dir).frontend
    _, whole = fe.process_signal(audio)
    fe.reset()
    parts = [fe.push(audio[i : i + 777])[1].values for i in range(0, len(audio), 777)]
    np.testing.assert_allclose(np.concatenate(parts), whole.values, atol=1e-5)


def test_gate_rejects_profile_from_a_different_chain(data_dir):
    # profile was measured directly on stft_mel; putting a filter in front invalidates it
    spec = PipelineSpec(
        filters=[
            StageSpec(name="frame_median"),
            StageSpec(name="spectral_gate", params={"profile": "room"}),
        ]
    )
    with pytest.raises(ValueError, match="different transform/filter chain"):
        Pipeline(spec, data_dir=data_dir)
    # ...and so does changing a transform parameter
    spec = _gate_spec()
    spec.transform.params = {"top_db": 60.0}
    with pytest.raises(ValueError, match="different transform/filter chain"):
        Pipeline(spec, data_dir=data_dir)
    # explicitly writing a default value is the same chain
    spec = _gate_spec()
    spec.transform.params = {"top_db": 80.0}
    Pipeline(spec, data_dir=data_dir)


def test_gate_without_profile_is_identity(tmp_path):
    fe = Frontend(SR, StageSpec(name="stft_mel"), [StageSpec(name="spectral_gate")], tmp_path)
    raw, out = fe.process_signal(_mic_noise(1.0, 2))
    np.testing.assert_array_equal(raw.values, out.values)


def test_label_cache_key_tracks_profile_contents(data_dir):
    fspec = _gate_spec().frontend_dict()
    h1 = frontend_hash(fspec, {}, data_dir)
    store = NoiseProfileStore(data_dir / "noise_profiles")
    prof = store.get("room")
    store.delete("room")
    prof.mean = [m + 1 for m in prof.mean]
    store.save(prof)
    assert frontend_hash(fspec, {}, data_dir) != h1
