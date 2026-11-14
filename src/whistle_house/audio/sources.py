"""Audio sources: anything that yields mono float32 chunks at a fixed sample rate."""

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


class MicSource:
    """Live microphone via sounddevice. Iterate to receive chunks; ``close()`` to stop.

    ``device`` is a case-insensitive substring of the device name (or None for default).
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
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            blocksize=chunk_size,
            channels=1,
            dtype="float32",
            device=index,
            callback=self._callback,
        )
        self._stream.start()
        log.info("mic stream started: %s Hz, device=%s", sample_rate, index)

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
                yield self._q.get(timeout=0.5)
            except queue.Empty:
                continue

    def close(self) -> None:
        self._closed = True
        self._stream.stop()
        self._stream.close()
