"""MicSource's resampling path: 48 kHz device -> 44.1 kHz pipeline chunks."""

import numpy as np

from whistle_house.audio.sources import StreamResampler


def _run(x: np.ndarray, cuts: list[int]) -> np.ndarray:
    r = StreamResampler(48000, 44100, 4096)
    out, i = [], 0
    for n in cuts:
        out += r.push(x[i : i + n])
        i += n
    assert all(len(c) == 4096 for c in out)
    return np.concatenate(out)


def test_chunking_does_not_change_output():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(48000 * 2).astype(np.float32) * 0.1
    a = _run(x, [len(x)])
    cuts = list(rng.integers(100, 6000, 200))
    b = _run(x, cuts)
    n = min(len(a), len(b))
    assert n > 40000
    np.testing.assert_allclose(a[:n], b[:n], atol=1e-5)


def test_pitch_is_kept():
    sr_in, f = 48000, 1500.0
    t = np.arange(sr_in * 2) / sr_in
    y = _run(np.sin(2 * np.pi * f * t).astype(np.float32), [4459] * 22)
    spec = np.abs(np.fft.rfft(y[4096:] * np.hanning(len(y) - 4096)))
    peak = np.argmax(spec) * 44100 / (len(y) - 4096)
    assert abs(peak - f) < 2
