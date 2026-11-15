"""Turn a pipeline's filtered frames back into audio: "what the detector hears".

Filters work on transform frames (mel dB etc.), not on audio, so there is no processed signal
to play directly. This module derives a time-frequency gain from what the filter chain kept and
applies it to the original signal's STFT, then resynthesises by weighted overlap-add. The STFT
uses exactly the transform's frames (same n_fft, hop, window, alignment), so gains line up.

Gain per transform bin (power), then interpolated onto the FFT bins:

* no filters: 1 (you hear the transform's frequency range only);
* chain contains a ``relative_output`` filter (output = dB above a floor, 0 = floor): the
  spectral-subtraction gain ``1 - 10^(-out/10)``. If out = 10*log10((S+N)/N), that is
  S/(S+N), i.e. the noise power the filter removed is taken out of the signal;
* otherwise (absolute dB in, absolute dB out, e.g. band_reject): ``10^(min(out - raw, 0)/10)``.

Bins outside the transform's range get gain 0: the pipeline never sees them. Expect some
"musical noise" warble where a mask flickers; that is honest about what passes the filters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from whistle_house.pipeline import Frontend, PipelineSpec


class RenderUnsupported(ValueError):
    """The pipeline's transform has no STFT to invert."""


def frame_gains(frontend: Frontend, raw: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    """(n_frames, n_bins) power gain in [0, 1] from raw vs filtered transform frames."""
    if not frontend.filters:
        return np.ones_like(raw)
    if raw.shape != filtered.shape:
        raise RenderUnsupported("filters changed the number of bins; can't map gains back")
    power_units = getattr(frontend.transform.config, "output", "db") == "power"
    if any(f.relative_output for f in frontend.filters):
        if power_units:
            raise RenderUnsupported("relative filters on a power-output transform")
        g = 1.0 - 10.0 ** (-np.maximum(filtered, 0.0) / 10.0)
    elif power_units:
        g = filtered / np.maximum(raw, 1e-12)
    else:
        g = 10.0 ** (np.minimum(filtered - raw, 0.0) / 10.0)
    return np.clip(g, 0.0, 1.0).astype(np.float32)


def render_processed(
    audio: np.ndarray, spec: PipelineSpec, data_dir: Path | None = None
) -> np.ndarray:
    """Mono float32 audio at ``spec.sample_rate`` -> processed audio of the same length."""
    fe = Frontend(spec.sample_rate, spec.transform, spec.filters, data_dir)
    stft = getattr(fe.transform, "stft", None)
    if stft is None:
        raise RenderUnsupported(f"transform {spec.transform.name!r} can't be resynthesised")
    raw, filt = fe.process_signal(audio)
    gains = frame_gains(fe, raw.values, filt.values)

    n_fft, hop, win = stft.n_fft, stft.hop, stft.window.astype(np.float64)
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / spec.sample_rate)
    bin_freqs = raw.axis.freqs
    # transform-bin power gain -> FFT-bin amplitude gain (0 outside the transform's range)
    amp = (
        np.sqrt(np.stack([np.interp(fft_freqs, bin_freqs, g, left=0.0, right=0.0) for g in gains]))
        if len(gains)
        else np.zeros((0, fft_freqs.size))
    )

    # same framing as StreamingSTFT: zero history of n_fft, frame k starts at (k+1)*hop
    x = np.concatenate([np.zeros(n_fft), audio.astype(np.float64), np.zeros(n_fft)])
    n = amp.shape[0]
    frames = sliding_window_view(x, n_fft)[hop::hop][:n]
    spec_c = np.fft.rfft(frames * win, axis=-1) * amp
    y_frames = np.fft.irfft(spec_c, n=n_fft, axis=-1) * win

    y = np.zeros_like(x)
    norm = np.zeros_like(x)
    for k in range(n):
        s = (k + 1) * hop
        y[s : s + n_fft] += y_frames[k]
        norm[s : s + n_fft] += win**2
    ok = norm > 1e-3 * max(norm.max(), 1e-12)
    y[ok] /= norm[ok]
    y[~ok] = 0.0
    out = y[n_fft : n_fft + audio.size]
    return np.clip(out, -1.0, 1.0).astype(np.float32)
