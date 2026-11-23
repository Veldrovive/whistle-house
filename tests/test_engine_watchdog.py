"""The live engine reopens the mic by itself: missing at start, and after a stall."""

import threading
import time

import numpy as np

from whistle_house.data.store import RecordingStore
from whistle_house.outputs import OutputStore
from whistle_house.pipeline import PipelineSpec
from whistle_house.server import engine as engine_mod
from whistle_house.settings import Paths
from whistle_house.tunes import TuneLibrary


class FakeMic:
    """Fails to open while ``fail_opens`` > 0; yields silence until ``stall`` is set."""

    fail_opens = 0
    opened = 0
    stall = threading.Event()

    def __init__(self, sample_rate, chunk_size, device=None):
        if FakeMic.fail_opens > 0:
            FakeMic.fail_opens -= 1
            raise RuntimeError("no such device")
        FakeMic.opened += 1
        self.chunk_size = chunk_size
        self._closed = False

    def __iter__(self):
        while not self._closed:
            time.sleep(0.005)
            if not FakeMic.stall.is_set():
                yield np.zeros(self.chunk_size, np.float32)

    def close(self):
        self._closed = True


def _engine(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_mod, "MicSource", FakeMic)
    monkeypatch.setattr(engine_mod, "refresh_devices", lambda: None)
    for name, v in [("WATCHDOG_INTERVAL_S", 0.02), ("STALL_S", 0.2), ("RETRY_MIN_S", 0.05)]:
        monkeypatch.setattr(engine_mod.LiveEngine, name, v)
    paths = Paths(tmp_path).ensure()
    eng = engine_mod.LiveEngine(
        paths,
        TuneLibrary(paths.tunes),
        OutputStore(paths.outputs),
        RecordingStore(paths.recordings),
    )
    eng.set_pipeline(PipelineSpec())
    return eng


def _wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_reopens_missing_mic_and_stalled_mic(tmp_path, monkeypatch):
    FakeMic.fail_opens, FakeMic.opened = 2, 0
    FakeMic.stall.clear()
    eng = _engine(tmp_path, monkeypatch)
    try:
        eng.start()
        assert eng.state == "error"  # mic missing at start
        assert _wait(lambda: eng.state == "running")
        assert FakeMic.opened == 1

        FakeMic.stall.set()  # audio stops arriving
        assert _wait(lambda: FakeMic.opened >= 2)
        FakeMic.stall.clear()
        assert _wait(lambda: eng.state == "running")
    finally:
        eng.stop()


def test_stop_is_respected(tmp_path, monkeypatch):
    FakeMic.fail_opens, FakeMic.opened = 0, 0
    FakeMic.stall.clear()
    eng = _engine(tmp_path, monkeypatch)
    eng.start()
    eng.stop()
    assert eng.state == "stopped"
    time.sleep(0.3)
    assert eng.state == "stopped" and FakeMic.opened == 1
