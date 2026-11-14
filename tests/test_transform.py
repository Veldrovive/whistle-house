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


@pytest.mark.parametrize("name", ["stft_mel", "stft_linear"])
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
