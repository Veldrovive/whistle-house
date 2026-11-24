"""Mic-clip augmentations: the f0 labels must follow what is done to the audio."""

from dataclasses import replace

import numpy as np

from whistle_house.data.augment import (
    envelope_clip,
    reverse_clip,
    splice_notes,
    synth_whistle,
    warp_clip,
)
from whistle_house.data.training import (
    LABEL_RATE,
    ExampleGenerator,
    PositiveClip,
    TrainingSources,
    config_for_mix,
)
from whistle_house.pipeline import PipelineSpec

SR = 44100


def _notes(freqs=(1000.0, 1500.0, 2000.0), note_s=0.4, gap_s=0.2):
    """Clean tones separated by silence, with an exact f0 track at LABEL_RATE."""
    audio, f0 = [], []
    for hz in freqs:
        t = np.arange(int(note_s * SR)) / SR
        audio += [0.3 * np.sin(2 * np.pi * hz * t), np.zeros(int(gap_s * SR))]
        f0 += [np.full(int(note_s * LABEL_RATE), hz), np.zeros(int(gap_s * LABEL_RATE))]
    audio, f0 = np.concatenate(audio).astype(np.float32), np.concatenate(f0).astype(np.float32)
    return audio, np.concatenate([f0, [0.0]]).astype(np.float32)


def _peak_hz(x):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / SR)[spec.argmax()]


def _check_labels_match_audio(audio, f0, tol=0.03):
    """Wherever a label is well inside a note, the audio there has that pitch."""
    v = f0 > 0
    inside = v & np.roll(v, 4) & np.roll(v, -4)
    idx = np.flatnonzero(inside)[5:-5:7]
    assert len(idx) > 3
    w = 1024
    for i in idx:
        c = i * SR // LABEL_RATE
        seg = audio[max(0, c - w) : c + w]
        assert abs(_peak_hz(seg) / f0[i] - 1) < tol + (SR / len(seg)) / f0[i]


def test_warp_shifts_pitch_and_tempo_together():
    audio, f0 = _notes()
    rng = np.random.default_rng(0)
    out, f0w = warp_clip(audio, f0, SR, LABEL_RATE, rng, semitones=(3.0, 3.0), wobble=0.0)
    r = 2 ** (3 / 12)
    assert abs(len(out) / len(audio) - 1 / r) < 0.01
    assert np.allclose(np.unique(f0w[f0w > 0]), np.array([1000.0, 1500.0, 2000.0]) * r, rtol=1e-4)
    assert abs((f0w > 0).mean() - (f0 > 0).mean()) < 0.02
    _check_labels_match_audio(out, f0w)
    out, f0w = warp_clip(audio, f0, SR, LABEL_RATE, np.random.default_rng(1))  # with wobble
    _check_labels_match_audio(out, f0w, tol=0.05)


def test_warp_keeps_labels_inside_bounds():
    audio, f0 = _notes()
    for seed in range(20):
        _, f0w = warp_clip(
            audio,
            f0,
            SR,
            LABEL_RATE,
            np.random.default_rng(seed),
            semitones=(-12.0, 12.0),
            f0_bounds=(900.0, 2300.0),
        )
        v = f0w[f0w > 0]
        assert v.min() >= 900.0 * 0.96 and v.max() <= 2300.0 * 1.04  # bounds +- wobble


def test_splice_reorders_whole_notes():
    audio, f0 = _notes()
    seen = set()
    for seed in range(12):
        out, f0s = splice_notes(audio, f0, SR, LABEL_RATE, np.random.default_rng(seed))
        assert len(f0s) == int(len(out) / SR * LABEL_RATE) + 1
        assert set(np.unique(f0s[f0s > 0])) <= {1000.0, 1500.0, 2000.0}
        _check_labels_match_audio(out, f0s)
        seen.add(tuple(f0s[np.flatnonzero(np.diff(f0s) > 0) + 1]))  # the note sequence
    assert len(seen) > 4  # different melodies from the same take
    one = _notes(freqs=(1200.0,))
    assert splice_notes(*one, SR, LABEL_RATE, np.random.default_rng(0))[0] is one[0]


def test_reverse_and_envelope():
    audio, f0 = _notes()
    out, f0r = reverse_clip(audio, f0)
    assert f0r[f0r > 0][0] == 2000.0 and len(out) == len(audio)
    env = envelope_clip(audio, SR, np.random.default_rng(0))
    gain = np.abs(env[audio != 0] / audio[audio != 0])
    assert gain.min() >= 10 ** (-6.01 / 20) and gain.max() <= 10 ** (6.01 / 20) and gain.std() > 0


def test_finetune_mix_augments_training_but_not_validation(tmp_path):
    import soundfile as sf

    sw = synth_whistle(SR, 4.0, np.random.default_rng(5))
    path = tmp_path / "mic.wav"
    sf.write(path, sw.audio, SR)
    f0 = sw.f0[:: SR // LABEL_RATE].astype(np.float32)
    src = TrainingSources(mic_positives=[PositiveClip("mic", str(path), f0)])
    fe = PipelineSpec().frontend_dict()
    cfg = config_for_mix("finetune")
    assert cfg.p_mic_warp > 0 and cfg.p_mic_splice > 0
    ev = cfg.eval_config()
    assert (ev.p_mic_warp, ev.p_mic_splice, ev.p_mic_reverse, ev.p_mic_envelope) == (0, 0, 0, 0)
    assert ev.p_mic_whistle == cfg.p_mic_whistle  # same mix otherwise

    gen = ExampleGenerator(fe, src, cfg)
    x, y = gen.batch(np.random.default_rng(0), 16)
    assert y.shape == (*x.shape[:2], 1) and (y > 0).any()  # one label column: whistle
    freqs = gen.axis.freqs
    clip = src.mic_positives[0]
    for seed in range(30):  # augmented mic labels always stay on the frequency axis
        f = gen._positive(clip, 3 * SR, np.random.default_rng(seed), mic=True)[1]
        if (f > 0).any():
            assert f[f > 0].min() >= freqs[0] and f[f > 0].max() <= freqs[-1]
    # Warped pitches are not in the original take; unaugmented ones all are.
    orig = set(np.round(f0[f0 > 0], 1))
    raw = ExampleGenerator(fe, src, ev)._positive(
        src.mic_positives[0], 3 * SR, np.random.default_rng(1), mic=True
    )[1]
    always = ExampleGenerator(fe, src, replace(cfg, p_mic_warp=1.0))
    aug = always._positive(src.mic_positives[0], 3 * SR, np.random.default_rng(1), mic=True)[1]
    assert set(np.round(raw[raw > 0], 1)) <= orig
    assert not set(np.round(aug[aug > 0], 1)) <= orig


# --------------------------------------------------------------------------- channel and room


def test_random_channel_changes_balance_not_pitch_or_loudness():
    from whistle_house.data.augment import channel_response_db, random_channel, rms

    t = np.arange(SR) / SR
    x = (0.2 * np.sin(2 * np.pi * 150 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
    ratios = []
    for seed in range(20):
        y = random_channel(x, SR, np.random.default_rng(seed))
        assert len(y) == len(x) and abs(rms(y) / rms(x) - 1) < 1e-3
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        ratios.append(20 * np.log10(spec[1200] / spec[150]))
        for k in (150, 1200):  # still the same two tones
            assert spec[k] > 10 * max(spec[k - 8], spec[k + 8])
    # the balance between a fundamental and the mid band moves by tens of dB across draws
    assert max(ratios) - min(ratios) > 25
    g = channel_response_db(np.geomspace(60, 4000, 50), np.random.default_rng(0))
    assert np.abs(np.diff(g)).max() < 6  # smooth: no narrow notches


def test_room_reverb_has_the_requested_direct_to_reverb_ratio():
    from whistle_house.data.augment import room_reverb

    x = np.zeros(SR, np.float32)
    x[0] = 1.0
    for drr in (-6.0, 6.0, 18.0):
        y = room_reverb(x, SR, np.random.default_rng(1), rt60_s=(0.5, 0.5), drr_db=(drr, drr))
        got = 10 * np.log10(y[0] ** 2 / np.sum(y[1:] ** 2))
        assert abs(got - drr) < 0.5


def test_room_preset_keeps_labels_and_sets_one_level_range():
    """`--augment room`: labels still follow the audio, and a lone interferer is as loud as an
    example with a whistle in it (level must not tell the classes apart)."""
    from whistle_house.data.augment import active_level_db

    spec = PipelineSpec(
        transform={"name": "stft_log", "params": {}}, filters=[], detector={"name": "peak_tracker"}
    )
    cfg = config_for_mix("mic", ("whistle", "hum"), "room")
    assert cfg.p_channel > 0 and cfg.room_reverb and cfg.level_dbfs == (-65.0, -10.0)
    assert config_for_mix("mic", ("whistle", "hum")).p_channel == 0  # v1 stays the old recipe
    gen = ExampleGenerator(spec.frontend_dict(), TrainingSources(), cfg)
    rng = np.random.default_rng(0)
    for _ in range(6):
        x, f0 = gen.example(rng)
        assert np.isfinite(x).all() and x.shape[0] == f0.shape[0] and f0.shape[1] == 2

    # a clean synthetic whistle through reverb + channel: the loudest bin is still the label
    clean = replace(
        cfg,
        p_interferer=0.0,
        p_synth_whistle=1.0,
        p_synth_hum=0.0,
        p_mic_whistle=0.0,
        p_mic_hum=0.0,
        p_whistle_overtones=0.0,
        p_saturate=0.0,
    )
    gen = ExampleGenerator(spec.frontend_dict(), TrainingSources(), clean)
    freqs = gen.axis.freqs
    ok = n = 0
    for _ in range(8):
        x, f0 = gen.example(rng)
        on = f0[:, 0] > 0
        est = freqs[x[on].argmax(1)]
        ok += int((np.abs(12 * np.log2(est / f0[on, 0])) < 1.0).sum())
        n += int(on.sum())
    assert n > 200 and ok / n > 0.8

    # level: the example's loud parts land in the configured range, with or without a positive
    class Probe(ExampleGenerator):
        def example_audio(self, rng):
            self.frontend.process_signal = lambda mix: (_ for _ in ()).throw(_Stop(mix))
            try:
                self.example(rng)
            except _Stop as s:
                return s.args[0]

    class _Stop(Exception):
        pass

    for share in (0.0, 1.0):
        c = replace(cfg, p_synth_whistle=share, p_synth_hum=0.0, p_mic_whistle=0.0, p_mic_hum=0.0)
        probe = Probe(spec.frontend_dict(), TrainingSources(), c)
        levels = [active_level_db(probe.example_audio(rng), SR) for _ in range(12)]
        assert -68 < min(levels) and max(levels) < -8
