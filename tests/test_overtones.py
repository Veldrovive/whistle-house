"""Overtones of a whistle are not notes: labeller check, label merging, harmonic-rich positives."""

import numpy as np

from whistle_house.data.augment import rms, saturate, synth_whistle
from whistle_house.data.training import (
    LABEL_RATE,
    ExampleGenerator,
    GeneratorConfig,
    TrainingSources,
    auto_label,
    events_to_f0,
)
from whistle_house.pipeline import Pipeline, PipelineSpec, StageSpec
from whistle_house.stages.detectors.common import drop_harmonics
from whistle_house.types import ContourPoint, WhistleEvent

SR = 44100
FRONTENDS = [
    PipelineSpec().frontend_dict(),  # stft_mel + noise_floor
    PipelineSpec(transform=StageSpec(name="stft_log"), filters=[]).frontend_dict(),
]


def _tone(hz: float, seconds: float = 1.0, overtones=((2, -15.0), (3, -25.0))) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = np.sin(2 * np.pi * hz * t)
    for k, db in overtones:
        x += 10 ** (db / 20) * np.sin(2 * np.pi * k * hz * t + k)
    pad = np.zeros(int(0.2 * SR))
    return np.concatenate([pad, 0.4 * x, pad]).astype(np.float32)


def _spectrum_db(x: np.ndarray, hz: float) -> float:
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    k = round(hz * len(x) / SR)
    return float(20 * np.log10(mag[k - 3 : k + 4].max() + 1e-12))


def test_drop_harmonics_keeps_fundamentals_and_louder_upper_tones():
    # 1000 Hz with overtones at x2 and x3, plus an unrelated tone
    peaks = [(1000.0, 60.0), (2004.0, 30.0), (2990.0, 20.0), (1370.0, 40.0)]
    assert drop_harmonics(peaks) == [True, False, False, True]
    # an octave above but louder: a sound of its own, not an overtone of the quiet one below
    assert drop_harmonics([(1000.0, 30.0), (2000.0, 60.0)]) == [True, True]
    # a fifth above (x1.5) is a different note
    assert drop_harmonics([(1000.0, 60.0), (1500.0, 30.0)]) == [True, True]
    # overtone of an overtone is still dropped (x4 of the fundamental)
    assert drop_harmonics([(500.0, 60.0), (1000.0, 40.0), (2000.0, 25.0)]) == [True, False, False]


def test_peak_tracker_reports_overtones_unless_told_to_reject_them():
    audio = _tone(900.0)

    def events(**params):
        spec = PipelineSpec(detector=StageSpec(name="peak_tracker", params=params))
        return sorted(e.median_hz() for e in Pipeline(spec).analyze(audio).events)

    plain = events()
    assert len(plain) >= 2 and abs(plain[1] / plain[0] - 2) < 0.05  # the x2 overtone is a "note"
    clean = events(reject_harmonics=True)
    assert len(clean) == 1 and abs(clean[0] / 900 - 1) < 0.02


def test_auto_labels_follow_the_fundamental_of_an_overtone_rich_whistle():
    cfg = GeneratorConfig()
    assert cfg.labeler_params["reject_harmonics"] is True
    rng = np.random.default_rng(0)
    for fe in FRONTENDS:
        for _ in range(3):
            sw = synth_whistle(SR, 3.0, rng, n_notes=3, overtones=True)
            truth = sw.f0[:: SR // LABEL_RATE]
            label = auto_label(sw.audio, SR, fe, cfg.labeler_params).astype(np.float32)
            n = min(len(label), len(truth))
            both = (label[:n] > 0) & (truth[:n] > 0)
            st = 12 * np.log2(label[:n][both] / truth[:n][both])
            assert both.sum() > 20
            assert np.mean(np.abs(st) < 1.0) > 0.97  # never an octave (or more) above


def test_overlapping_label_events_resolve_to_the_more_confident_one():
    def ev(hz, conf, start=0.0, end=1.0):
        pts = [ContourPoint(start, hz, conf), ContourPoint(end, hz, conf)]
        return WhistleEvent(start, end, pts, confidence=conf)

    strong, weak = ev(1000.0, 40.0), ev(2000.0, 26.0, 0.2, 0.8)
    for order in ([strong, weak], [weak, strong]):
        f0 = events_to_f0(order, 100, 100)
        assert set(np.unique(f0[5:95])) == {1000.0}


def test_synth_whistle_overtones_are_optional_and_labelled_at_the_fundamental():
    plain = synth_whistle(SR, 2.0, np.random.default_rng(3), n_notes=1)
    rich = synth_whistle(SR, 2.0, np.random.default_rng(3), n_notes=1, overtones=True)
    for sw, lo, hi in ((plain, -200.0, -20.0), (rich, -42.0, -8.0)):
        v = np.flatnonzero(sw.f0 > 0)
        mid = v[len(v) // 2]
        seg = sw.audio[mid - 2048 : mid + 2048]
        hz = float(sw.f0[mid])
        h1, h2 = _spectrum_db(seg, hz), _spectrum_db(seg, 2 * hz)
        assert lo <= h2 - h1 <= hi, h2 - h1
        assert h1 > h2  # the label (f0) is the strongest line


def test_saturate_adds_overtones_without_moving_the_pitch():
    t = np.arange(SR) / SR
    x = (0.3 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    assert _spectrum_db(x, 3000.0) - _spectrum_db(x, 1000.0) < -80
    levels = []
    for seed in range(8):
        y = saturate(x, np.random.default_rng(seed))
        assert abs(rms(y) / rms(x) - 1) < 1e-3
        h1 = _spectrum_db(y, 1000.0)
        assert all(_spectrum_db(y, k * 1000.0) < h1 - 6 for k in (2, 3, 4))  # still the fundamental
        levels.append(max(_spectrum_db(y, 2000.0), _spectrum_db(y, 3000.0)) - h1)
    assert max(levels) > -25 and min(levels) < -30  # from barely to clearly overdriven
    assert saturate(np.zeros(100, np.float32), np.random.default_rng(0)).sum() == 0


def test_generator_makes_harmonic_rich_positives_with_fundamental_labels():
    fe = FRONTENDS[1]
    cfg = GeneratorConfig(
        window_s=1.5,
        p_mic_whistle=0.0,
        p_synth_whistle=1.0,
        p_whistle_overtones=1.0,
        p_saturate=1.0,
        p_interferer=0.0,
        p_reverb=0.0,
        p_eq=0.0,
    )
    gen = ExampleGenerator(fe, TrainingSources(), cfg)
    x, f0 = gen.batch(np.random.default_rng(0), 6)
    freqs = gen.axis.freqs
    voiced = f0[..., 0] > 0
    assert voiced.any()
    # the labelled bin is the loudest one in almost every labelled frame, although the audio
    # has overtones above it
    loudest = freqs[x.argmax(-1)]
    st = 12 * np.log2(loudest[voiced] / f0[..., 0][voiced])
    assert np.mean(np.abs(st) < 1.0) > 0.9
    b2 = np.clip(np.searchsorted(freqs, 2 * f0[..., 0]), 0, len(freqs) - 1)
    floor = np.median(x, axis=-1)
    lift = np.take_along_axis(x, b2[..., None], -1)[..., 0] - floor
    assert np.median(lift[voiced & (2 * f0[..., 0] < freqs[-1])]) > 10  # the x2 line is there
