import librosa
import numpy as np
import pytest

from whistle_house.stages.base import StageContext, build_stage
from whistle_house.stages.transforms.stft import mel_filterbank, mel_frequencies

SR = 44100


def _signal(seconds=1.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 1500 * t) + 0.05 * rng.standard_normal(t.size)).astype(
        np.float32
    )


def test_mel_filterbank_matches_librosa():
    ours = mel_filterbank(SR, 2048, 128, 300, 4000)
    ref = librosa.filters.mel(sr=SR, n_fft=2048, n_mels=128, fmin=300, fmax=4000)
    np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(
        mel_frequencies(130, 300, 4000), librosa.mel_frequencies(130, fmin=300, fmax=4000)
    )


@pytest.mark.parametrize("name", ["stft_mel", "stft_linear", "stft_log"])
@pytest.mark.parametrize("chunk", [1, 333, 512, 1024, 4096])
def test_streaming_equals_offline(name, chunk):
    x = _signal()
    offline = build_stage("transform", name, {}, StageContext(SR)).push(x)
    tr = build_stage("transform", name, {}, StageContext(SR))
    blocks = [tr.push(x[i : i + chunk]) for i in range(0, len(x), chunk)]
    vals = np.concatenate([b.values for b in blocks])
    times = np.concatenate([b.times for b in blocks])
    np.testing.assert_allclose(vals, offline.values, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(times, offline.times)


def _v1_frames(x, n_fft=2048, hop=512, n_mels=128, fmin=300, fmax=4000):
    """Verbatim maths from v1 audio_lib._process_chunk (the reference we ported)."""
    basis = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    buf = np.zeros(n_fft, np.float32)
    win = np.hanning(n_fft)
    out = []
    for i in range(len(x) // hop):
        buf[:-hop] = buf[hop:]
        buf[-hop:] = x[i * hop : (i + 1) * hop]
        power = np.abs(np.fft.rfft(buf * win, n=n_fft)) ** 2
        out.append(librosa.power_to_db(basis @ power, ref=1))
    return np.array(out)


def test_matches_v1_transform():
    x = _signal(1.0)
    ours = build_stage("transform", "stft_mel", {}, StageContext(SR)).push(x)
    np.testing.assert_allclose(ours.values, _v1_frames(x), atol=1e-3)
    # v1 frame time convention: centre of the window
    assert ours.times[0] == pytest.approx((512 - 1024) / SR)


def test_log_axis_and_tone_level():
    tr = build_stage("transform", "stft_log", {}, StageContext(SR))
    f = tr.axis.freqs
    assert f[0] == pytest.approx(60.0) and f[-1] <= 4000.0
    np.testing.assert_allclose(f[1:] / f[:-1], 2 ** (1 / 24))
    # a pure tone peaks in the bin nearest its frequency, at both ends of the axis
    t = np.arange(SR) / SR
    for hz in (100.0, 130.8, 2000.0):
        x = (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
        frames = tr.push(x).values[10:]
        tr.reset()
        peak = np.bincount(frames.argmax(axis=1)).argmax()
        assert abs(np.log2(f[peak] / hz) * 12) <= 0.5


def test_log_harmonics_are_a_fixed_bin_offset():
    """h2 is always +24 bins (one octave) above h1, at any pitch."""
    tr = build_stage("transform", "stft_log", {}, StageContext(SR))
    t = np.arange(SR) / SR
    for f0 in (90.0, 150.0, 220.0):
        x = (0.3 * np.sin(2 * np.pi * f0 * t) + 0.15 * np.sin(2 * np.pi * 2 * f0 * t)).astype(
            np.float32
        )
        v = tr.push(x).values[10:].mean(axis=0)
        tr.reset()
        b1 = round(tr.axis.hz_to_bin(f0))
        b2 = round(tr.axis.hz_to_bin(2 * f0))
        assert b2 - b1 == 24
        assert v[b1] > v[b1 + 12] + 10 and v[b2] > v[b2 - 12] + 10
