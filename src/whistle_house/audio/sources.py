"""Audio sources: anything that yields mono float32 chunks at a fixed sample rate.

A microphone that cannot run at the pipeline's rate (the Fifine K669B on the Pi offers 48 kHz
only, and bare ALSA ``hw:`` devices refuse other rates) is opened at its own rate and
resampled with a stateful stream (``StreamResampler``), so chunk boundaries leave no trace.
"""

from __future__ import annotations

import logging
import queue
from collections.abc import Iterator

import numpy as np

log = logging.getLogger(__name__)


def list_input_devices() -> list[dict]:
    import sounddevice as sd

    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append(
                {"index": i, "name": d["name"], "default_samplerate": d["default_samplerate"]}
            )
    return out


class ArraySource:
    """Replays an in-memory signal in fixed-size chunks (offline analysis, tests)."""

    def __init__(self, samples: np.ndarray, sample_rate: int, chunk_size: int = 1024):
        self.samples = np.asarray(samples, np.float32)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(0, len(self.samples), self.chunk_size):
            yield self.samples[i : i + self.chunk_size]


class ReplaySource:
    """Plays a signal in real time (looping), standing in for a microphone.

    ``whistle-house serve --replay file.wav`` uses this to exercise the live path without a mic
    (or while another process owns it).
    """

    def __init__(
        self, samples: np.ndarray, sample_rate: int, chunk_size: int = 1024, loop: bool = True
    ):
        self.samples = np.asarray(samples, np.float32)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.loop = loop
        self._closed = False

    def __iter__(self) -> Iterator[np.ndarray]:
        import time

        period = self.chunk_size / self.sample_rate
        next_t = time.monotonic()
        while not self._closed:
            for i in range(0, len(self.samples) - self.chunk_size + 1, self.chunk_size):
                if self._closed:
                    return
                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))
                yield self.samples[i : i + self.chunk_size]
            if not self.loop:
                return

    def close(self) -> None:
        self._closed = True


class StreamResampler:
    """Streaming resampler + re-blocker: raw chunks at ``rate_in`` in, chunks of exactly
    ``chunk_size`` samples at ``rate_out`` out. Output does not depend on how input is cut."""

    def __init__(self, rate_in: float, rate_out: int, chunk_size: int):
        import soxr

        self.rate_in, self.rate_out, self.chunk_size = rate_in, rate_out, chunk_size
        self._stream = soxr.ResampleStream(rate_in, rate_out, 1, dtype="float32", quality="HQ")
        self._buf = np.zeros(0, np.float32)

    def push(self, x: np.ndarray) -> list[np.ndarray]:
        y = self._stream.resample_chunk(np.asarray(x, np.float32))
        self._buf = np.concatenate([self._buf, y])
        n = len(self._buf) // self.chunk_size
        out = [self._buf[i * self.chunk_size : (i + 1) * self.chunk_size] for i in range(n)]
        self._buf = self._buf[n * self.chunk_size :]
        return out


def refresh_devices() -> None:
    """Reinitialise PortAudio so it sees devices plugged in since it started.

    Only call with no stream open (the live engine's watchdog does, before reopening the mic).
    """
    import sounddevice as sd

    sd._terminate()
    sd._initialize()


def pick_input_rate(sample_rate: int, device: int | None) -> int:
    """``sample_rate`` if the device accepts it, else the device's default (native) rate."""
    import sounddevice as sd

    try:
        sd.check_input_settings(device=device, samplerate=sample_rate, channels=1, dtype="float32")
        return sample_rate
    except Exception:  # noqa: BLE001 - PortAudio raises various errors for unsupported rates
        rate = int(sd.query_devices(device, "input")["default_samplerate"])
        log.info(
            "input device does not accept %s Hz; opening at %s Hz and resampling", sample_rate, rate
        )
        return rate


class MicSource:
    """Live microphone via sounddevice. Iterate to receive chunks; ``close()`` to stop.

    ``device`` is a case-insensitive substring of the device name (or None for default).
    Chunks are ``chunk_size`` samples at ``sample_rate`` whatever rate the device runs at
    (``device_rate``).
    """

    def __init__(self, sample_rate: int, chunk_size: int = 1024, device: str | None = None):
        import sounddevice as sd

        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._closed = False
        index = None
        if device:
            for d in list_input_devices():
                if device.lower() in d["name"].lower():
                    index = d["index"]
                    break
            else:
                log.warning("input device %r not found; using default", device)
        self.device_index = index
        self.device_rate = pick_input_rate(sample_rate, index)
        self._resampler = (
            StreamResampler(self.device_rate, sample_rate, chunk_size)
            if self.device_rate != sample_rate
            else None
        )
        self._stream = sd.InputStream(
            samplerate=self.device_rate,
            # Resampling re-blocks anyway, so let PortAudio pick the block: on a bare ALSA hw:
            # device an odd block (1024 at 44.1 kHz = 1115 at 48 kHz) lost ~40% of the audio
            # to overflows on the Pi, while blocksize 0 ran clean.
            blocksize=chunk_size if self._resampler is None else 0,
            channels=1,
            dtype="float32",
            device=index,
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            "mic stream started: %s Hz (device %s Hz), device=%s",
            sample_rate,
            self.device_rate,
            index,
        )

    def _callback(self, indata, frames, time, status) -> None:
        if status:
            log.warning("audio status: %s", status)
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            log.warning("audio queue full; dropping chunk")

    def __iter__(self) -> Iterator[np.ndarray]:
        while not self._closed:
            try:
                chunk = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._resampler is None:
                yield chunk
            else:
                yield from self._resampler.push(chunk)

    def close(self) -> None:
        self._closed = True
        self._stream.stop()
        self._stream.close()
