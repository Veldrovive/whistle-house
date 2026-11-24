"""Steady room backgrounds and gated training examples (the ``*-gate`` augment presets)."""

from __future__ import annotations

import numpy as np
import pytest

from whistle_house.data.augment import rms, synth_background
from whistle_house.data.training import (
    AUGMENT_PRESETS,
    ExampleGenerator,
    GeneratorConfig,
    TrainingSources,
    config_for_mix,
)
from whistle_house.noise_profiles import NoiseProfileStore, compute_noise_profile
from whistle_house.pipeline import Frontend
from whistle_house.stages.base import StageContext
from whistle_house.stages.filters.spectral_gate import SpectralGate

SR = 44100
SPEC = {"sample_rate": SR, "transform": {"name": "stft_log", "params": {}}, "filters": []}


def test_synth_background_is_steady_and_seeded():
    a = synth_background(SR, 4 * SR, np.random.default_rng(3))
    b = synth_background(SR, 4 * SR, np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert a.dtype == np.float32 and abs(rms(a) - 1.0) < 1e-3
    # nothing in it comes and goes: every second is as loud as every other
    per_s = [20 * np.log10(rms(a[i * SR : (i + 1) * SR])) for i in range(4)]
    assert max(per_s) - min(per_s) < 3.0
    # over many draws some have a tone standing well clear of the noise, and it does not move
    fe = Frontend.from_dict(SPEC)
    tonal, still = 0, 0
    for seed in range(12):
        x = synth_background(SR, 3 * SR, np.random.default_rng(seed))
        _, block = fe.process_signal(x)
        v = block.values[8:]
        mean = v.mean(axis=0)
        i = int(np.argmax(mean - np.median(mean)))
        if mean[i] - np.median(mean) > 20:
            tonal += 1
            still += v[:, i].std() < 3.0
    assert tonal >= 3 and still >= 0.75 * tonal


def test_set_profile_equals_a_stored_profile(tmp_path):
    rng = np.random.default_rng(0)
    fe = Frontend.from_dict(SPEC)
    _, noise = fe.process_signal(0.01 * rng.standard_normal(3 * SR).astype(np.float32))
    prof = compute_noise_profile("p", [noise.values], noise.axis.freqs, {}, [], 3.0)
    NoiseProfileStore(tmp_path / "noise_profiles").save(prof)
    _, block = fe.process_signal(
        (
            0.01 * rng.standard_normal(2 * SR)
            + 0.1 * np.sin(2 * np.pi * 440 * np.arange(2 * SR) / SR)
        ).astype(np.float32)
    )
    stored = SpectralGate(
        SpectralGate.Config(profile="p"), StageContext(SR, axis=block.axis, data_dir=tmp_path)
    )
    direct = SpectralGate(SpectralGate.Config(), StageContext(SR, axis=block.axis))
    _, mu, sigma = prof.arrays()
    direct.set_profile(mu, sigma)
    assert np.array_equal(stored.process(block).values, direct.process(block).values)


def _gen(**kw) -> ExampleGenerator:
    cfg = config_for_mix("mic", ("whistle", "hum"), "room-gate", **kw)
    return ExampleGenerator(SPEC, TrainingSources(), cfg)


def test_presets_and_config_checks():
    for name in ("room-gate", "room-mic-gate"):
        assert AUGMENT_PRESETS[name]["p_gate"] > 0 and AUGMENT_PRESETS[name]["p_background"] > 0
    for name in ("v1", "channel", "room", "room-mic"):  # the old recipes have neither
        assert "p_gate" not in AUGMENT_PRESETS[name]
        assert "p_background" not in AUGMENT_PRESETS[name]
    with pytest.raises(ValueError, match="level_dbfs"):
        GeneratorConfig(p_gate=0.5)
    gated = {**SPEC, "filters": [{"name": "noise_floor", "params": {}}]}
    with pytest.raises(ValueError, match="without its"):
        ExampleGenerator(gated, TrainingSources(), config_for_mix("mic", augment="room-gate"))


def test_gated_examples_have_a_floor_of_zero_and_keep_their_labels():
    same = {"p_background": 1.0, "p_gate_mismatch": 0.0, "gate_offset_db": 0.0}
    gated = _gen(p_gate=1.0, **same)
    plain = _gen(p_gate=0.0, **same)
    n_zero = n = 0
    hit = []
    for seed in range(10):
        xg, yg = gated.example(np.random.default_rng(seed))
        xp, yp = plain.example(np.random.default_rng(seed))
        # the gate is the last draw of an example: the audio and the labels are the same
        assert np.array_equal(yg, yp)
        assert xg.shape == xp.shape and xg.min() >= 0.0 and xp.max() < 60.0
        assert not np.array_equal(xg, xp)
        n_zero += int((xg < 1.0).sum())
        n += xg.size
        t, k = np.nonzero(yg[:, :2] > 0)
        freqs = gated.axis.freqs
        hit += [
            xg[ti, np.argmin(np.abs(freqs - yg[ti, ki]))] > 3.0 for ti, ki in zip(t, k, strict=True)
        ]
    assert n_zero / n > 0.2  # the floor between the sounds is gated to 0
    # labelled notes are still there after the gate (not all: some hums are drawn without
    # their fundamental, some notes sit under the interferer)
    assert len(hit) > 200 and np.mean(hit) > 0.6


def test_gate_removes_a_background_it_has_a_profile_of():
    """Nothing but the room: gated, almost every bin is 0; a profile measured with the
    background much quieter (the mismatch case) lets it through."""
    only_room = {
        "p_mic_whistle": 0.0,
        "p_synth_whistle": 0.0,
        "p_mic_hum": 0.0,
        "p_synth_hum": 0.0,
        "p_interferer": 0.0,
        "p_background": 1.0,
        "p_gate": 1.0,
        "gate_offset_db": 0.0,
        "background_dbfs": (-50.0, -45.0),
        "level_dbfs": (-30.0, -30.0),
    }

    class Quiet(ExampleGenerator):  # no interferer either: silence + the room
        def _room_interferer(self, n, rng):
            return np.zeros(n, np.float32), None, False

    cfg = config_for_mix("mic", ("whistle", "hum"), "room-gate", **only_room)
    fit = Quiet(
        SPEC, TrainingSources(), GeneratorConfig(**{**cfg.__dict__, "p_gate_mismatch": 0.0})
    )
    off = Quiet(
        SPEC,
        TrainingSources(),
        GeneratorConfig(
            **{**cfg.__dict__, "p_gate_mismatch": 1.0, "gate_mismatch_db": (20.0, 20.0)}
        ),
    )
    passed_fit, passed_off = [], []
    for seed in range(8):
        passed_fit.append(float((fit.example(np.random.default_rng(seed))[0] > 3.0).mean()))
        passed_off.append(float((off.example(np.random.default_rng(seed))[0] > 3.0).mean()))
    assert np.mean(passed_fit) < 0.02
    assert np.mean(passed_off) > 5 * np.mean(passed_fit) and np.mean(passed_off) > 0.1


def test_backgrounds_come_from_silence_takes(tmp_path):
    import soundfile as sf

    path = tmp_path / "room.wav"
    t = np.arange(12 * SR) / SR
    sf.write(path, (0.05 * np.sin(2 * np.pi * 120.0 * t)).astype(np.float32), SR)
    cfg = config_for_mix("mic", ("whistle", "hum"), "room-gate", p_background=1.0)
    gen = ExampleGenerator(SPEC, TrainingSources(backgrounds=[str(path)]), cfg)
    rng = np.random.default_rng(1)
    m = 5 * SR
    seen = 0
    for _ in range(12):
        floor, bg = gen._steady(m, -90.0, -20.0, rng)
        assert floor.shape == bg.shape == (m,) and bg.any()
        spec = np.abs(np.fft.rfft(bg))
        if abs(np.fft.rfftfreq(m, 1 / SR)[int(np.argmax(spec))] - 120.0) < 1.0:
            seen += spec.max() ** 2 > 0.99 * (spec**2).sum()  # the file: a pure 120 Hz tone
        assert -85.0 - 0.1 <= 20 * np.log10(rms(bg)) <= -30.0 + 0.1
    assert 2 <= seen <= 10  # about half real takes, half synthetic
