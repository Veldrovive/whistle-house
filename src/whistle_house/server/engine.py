"""Live engine: microphone -> active pipeline -> subscribers + outputs, in a background thread.

The mic can only be opened once, so the engine also owns data collection: it keeps a rolling
buffer of recent audio (``capture``: "save the last N seconds") and can record between
``start_recording``/``stop_recording``.

Unattended runs (the systemd service on the Pi) need the mic to come back on its own, so a
watchdog thread reopens it while the engine is meant to run (``start`` was called and ``stop``
was not): after an error (mic missing at boot, PortAudio failure) and after a stall (no audio
for ``STALL_S``, e.g. the USB mic was unplugged). Retries back off to ``RETRY_MAX_S``. Before
a reopen PortAudio is reinitialised, since it only sees devices present when it started.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from whistle_house.audio.io import load_audio
from whistle_house.audio.sources import MicSource, ReplaySource, refresh_devices
from whistle_house.data.store import Recording, RecordingStore
from whistle_house.outputs import OutputStore
from whistle_house.pipeline import Pipeline, PipelineSpec
from whistle_house.settings import Paths
from whistle_house.tunes import TuneLibrary
from whistle_house.types import FrameBlock, TuneMatch

log = logging.getLogger(__name__)


def encode_block(block: FrameBlock) -> dict:
    """FrameBlock -> JSON-able dict; values as base64 little-endian float32 (row-major T x F)."""
    return {
        "values": base64.b64encode(np.ascontiguousarray(block.values, "<f4").tobytes()).decode(),
        "n_frames": block.n_frames,
        "n_bins": block.axis.n_bins,
        "times": block.times.tolist(),
    }


def merge_filter_state(acc: dict[int, dict], states: list[dict]) -> None:
    """Accumulate filter telemetry between publishes: per-frame ``series`` lists are
    concatenated, every other key keeps its latest value (see ``Filter.telemetry``)."""
    for st in states:
        prev = acc.get(st["index"])
        series = dict(st.get("series") or {})
        if prev is not None and prev.get("name") == st["name"]:
            for k, v in series.items():
                series[k] = prev["series"].get(k, []) + v
        acc[st["index"]] = {**st, "series": series}


class LiveEngine:
    RING_SECONDS = 60
    PUBLISH_INTERVAL_S = 0.1
    WATCHDOG_INTERVAL_S = 1.0  # how often the watchdog looks
    STALL_S = 10.0  # no audio for this long while running = stalled mic
    RETRY_MIN_S = 5.0  # first reopen delay; doubles per failed attempt
    RETRY_MAX_S = 60.0

    def __init__(
        self,
        paths: Paths,
        tunes: TuneLibrary,
        outputs: OutputStore,
        recordings: RecordingStore,
        device: str | None = None,
        replay: str | None = None,
    ):
        self.replay = replay  # audio file to loop instead of opening the mic
        self.paths = paths
        self.tunes = tunes
        self.outputs = outputs
        self.recordings = recordings
        self.device = device
        self.spec: PipelineSpec | None = None
        self.pipeline: Pipeline | None = None
        self.state = "stopped"  # stopped | running | error
        self.error: str | None = None
        self.matches: deque[dict] = deque(maxlen=200)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._source: MicSource | ReplaySource | None = None
        self._ring: deque[np.ndarray] = deque()
        self._ring_len = 0
        self._recording: list[np.ndarray] | None = None
        self._recording_meta: dict = {}
        self._subscribers: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()
        self._pending: list[dict] = []
        self.on_match: Callable[[TuneMatch], None] | None = None
        self._control = threading.RLock()  # start/stop/reopen: API threads + watchdog
        self._want_running = False  # start() called and stop() not since
        self._last_audio = time.monotonic()
        self._retry_s = self.RETRY_MIN_S
        self._next_retry = 0.0
        self._watchdog: threading.Thread | None = None

    # ------------------------------------------------------------------ control

    def set_pipeline(self, spec: PipelineSpec) -> None:
        """Swap the active pipeline. Restarts the mic if the sample rate/chunk size changed."""
        pipeline = Pipeline(spec, tunes=self.tunes, data_dir=self.paths.root)  # may raise
        restart = (
            self.state == "running"
            and self.spec is not None
            and (
                spec.sample_rate != self.spec.sample_rate or spec.chunk_size != self.spec.chunk_size
            )
        )
        with self._lock:
            self.spec, self.pipeline = spec, pipeline
        self._broadcast({"type": "config", **self.config_payload()})
        if restart:
            self.stop()
            self.start()

    def config_payload(self) -> dict:
        p = self.pipeline
        return {
            "pipeline": self.spec.model_dump(mode="json") if self.spec else None,
            "raw_axis": p.frontend.transform.axis.to_json() if p else None,
            "axis": p.axis.to_json() if p else None,
            # static info of filters that report live state (adaptive_gate: profile means)
            "filter_info": p.frontend.telemetry_info() if p else [],
        }

    def start(self) -> None:
        with self._control:
            self._want_running = True
            self._retry_s = self.RETRY_MIN_S
            self._open()
            if self._watchdog is None and not self.replay:
                self._watchdog = threading.Thread(
                    target=self._watch, name="live-watchdog", daemon=True
                )
                self._watchdog.start()

    def stop(self) -> None:
        with self._control:
            self._want_running = False
            self._close()

    def _open(self) -> None:
        if self.state == "running" or self.spec is None:
            return
        try:
            if self.replay:
                audio, _ = load_audio(self.replay, self.spec.sample_rate)
                self._source = ReplaySource(audio, self.spec.sample_rate, self.spec.chunk_size)
            else:
                self._source = MicSource(self.spec.sample_rate, self.spec.chunk_size, self.device)
        except Exception as e:  # noqa: BLE001 - report to UI instead of crashing
            self.state, self.error = "error", f"could not open microphone: {e}"
            log.error(self.error)
            return
        self.state, self.error = "running", None
        self._last_audio = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="live-engine", daemon=True)
        self._thread.start()

    def _close(self) -> None:
        self._stop.set()
        if self._source:
            try:
                self._source.close()
            except Exception as e:  # noqa: BLE001 - a dead stream may fail to close
                log.warning("closing audio source failed: %s", e)
        if self._thread:
            self._thread.join(timeout=2)
        self._source = self._thread = None
        if self.state == "running":
            self.state = "stopped"

    def _watch(self) -> None:
        """Reopen the mic after an error or a stall while the engine is meant to run."""
        while True:
            time.sleep(self.WATCHDOG_INTERVAL_S)
            with self._control:
                if not self._want_running:
                    continue
                now = time.monotonic()
                stalled = self.state == "running" and now - self._last_audio > self.STALL_S
                if stalled:
                    log.warning("no audio for %.0f s; reopening the microphone", self.STALL_S)
                    self.state, self.error = "error", f"no audio for {self.STALL_S:.0f} s"
                    self._next_retry = 0.0  # first reopen after a stall: right away
                if self.state != "error" or now < self._next_retry:
                    continue
                self._close()
                try:
                    refresh_devices()
                except Exception as e:  # noqa: BLE001 - next attempt tries again
                    log.warning("could not reinitialise PortAudio: %s", e)
                log.info("reopening the microphone")
                self._open()
                if self.state == "running":
                    self._retry_s = self.RETRY_MIN_S
                else:
                    self._next_retry = now + self._retry_s
                    self._retry_s = min(self._retry_s * 2, self.RETRY_MAX_S)

    def status(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "device": f"replay:{self.replay}" if self.replay else self.device,
            "pipeline": self.spec.name if self.spec else None,
            "sample_rate": self.spec.sample_rate if self.spec else None,
            "recording": self._recording is not None,
            "recording_seconds": (
                sum(len(c) for c in self._recording) / self.spec.sample_rate
                if self._recording is not None and self.spec
                else 0.0
            ),
            "buffered_seconds": self._ring_len / self.spec.sample_rate if self.spec else 0.0,
        }

    # ------------------------------------------------------------------ data collection

    def start_recording(self, **meta) -> None:
        if self.state != "running":
            raise RuntimeError("live engine is not running (no microphone)")
        with self._lock:
            self._recording = []
            self._recording_meta = meta

    def stop_recording(self) -> Recording | None:
        with self._lock:
            chunks, meta = self._recording, self._recording_meta
            self._recording = None
        if not chunks:
            return None
        return self.recordings.add(
            np.concatenate(chunks), self.spec.sample_rate, source="mic", device=self.device, **meta
        )

    def cancel_recording(self) -> None:
        with self._lock:
            self._recording = None

    def capture(self, seconds: float, **meta) -> Recording:
        """Save the last ``seconds`` of audio from the rolling buffer."""
        with self._lock:
            audio = np.concatenate(list(self._ring)) if self._ring else np.zeros(0, np.float32)
        n = int(seconds * self.spec.sample_rate)
        if len(audio) == 0:
            raise RuntimeError("no audio buffered yet")
        return self.recordings.add(
            audio[-n:], self.spec.sample_rate, source="mic", device=self.device, **meta
        )

    # ------------------------------------------------------------------ pub/sub

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers = {s for s in self._subscribers if s[1] is not q}

    def _broadcast(self, msg: dict) -> None:
        for loop, q in list(self._subscribers):

            def put(q=q):
                if q.full():
                    q.get_nowait()  # drop oldest for slow clients
                q.put_nowait(msg)

            loop.call_soon_threadsafe(put)

    # ------------------------------------------------------------------ loop

    def _run(self) -> None:
        last_pub = time.monotonic()
        raw_blocks: list[FrameBlock] = []
        filt_blocks: list[FrameBlock] = []
        scores: list[np.ndarray] = []
        kind_scores: dict[str, list[np.ndarray]] = {}
        events: list[dict] = []
        active: list[dict] = []
        filter_state: dict[int, dict] = {}  # filter index -> accumulated telemetry
        assert self._source is not None
        try:
            for chunk in self._source:
                if self._stop.is_set():
                    break
                self._last_audio = time.monotonic()
                with self._lock:
                    self._ring.append(chunk)
                    self._ring_len += len(chunk)
                    while (
                        self._ring_len - len(self._ring[0])
                        > self.RING_SECONDS * self.spec.sample_rate
                    ):
                        self._ring_len -= len(self._ring.popleft())
                    if self._recording is not None:
                        self._recording.append(chunk)
                    pipeline = self.pipeline
                r = pipeline.step(chunk)
                raw_blocks.append(r.raw)
                filt_blocks.append(r.filtered)
                fs = r.detection.frame_scores
                scores.append(fs if fs is not None else np.zeros(r.filtered.n_frames, np.float32))
                for kind, ks in (r.detection.kind_scores or {}).items():
                    kind_scores.setdefault(kind, []).append(ks)
                events += [e.to_json() for e in r.detection.events]
                active = [e.to_json() for e in r.detection.active]
                merge_filter_state(filter_state, r.filter_state)
                for m in r.matches:
                    self._handle_match(m)
                now = time.monotonic()
                if now - last_pub >= self.PUBLISH_INTERVAL_S and self._subscribers:
                    self._broadcast(
                        {
                            "type": "frames",
                            "raw": encode_block(
                                FrameBlock.concat(raw_blocks, pipeline.frontend.transform.axis)
                            ),
                            "filtered": encode_block(FrameBlock.concat(filt_blocks, pipeline.axis)),
                            "scores": np.concatenate(scores).tolist(),
                            "kind_scores": {
                                k: np.concatenate(v).tolist() for k, v in kind_scores.items()
                            },
                            "events": events,
                            "active": active,
                            "filter_state": list(filter_state.values()),
                            "now": pipeline.now,
                        }
                    )
                if now - last_pub >= self.PUBLISH_INTERVAL_S:
                    raw_blocks, filt_blocks, scores, events = [], [], [], []
                    kind_scores = {}
                    filter_state = {}
                    last_pub = now
        except Exception as e:
            log.exception("live engine crashed")
            self.state, self.error = "error", repr(e)

    def _handle_match(self, m: TuneMatch) -> None:
        payload = {**m.to_json(), "wall_time": time.time()}
        self.matches.appendleft(payload)
        self._broadcast({"type": "match", "match": payload})
        try:
            tune = self.tunes.get(m.tune_id)
            self.outputs.fire(tune.outputs, m)
        except KeyError:
            pass
        if self.on_match:
            self.on_match(m)
