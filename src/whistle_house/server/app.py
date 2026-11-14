"""FastAPI backend for the manager (``manager/``). All routes are under /api and /ws.

Keep this file and ``manager/src/api.ts`` in sync (see AGENTS.md).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from whistle_house.audio.io import to_mono_float
from whistle_house.audio.sources import list_input_devices
from whistle_house.data.store import LABELS, Recording, RecordingStore, RecordingUpdate
from whistle_house.outputs import OutputInstance, OutputStore, list_output_types
from whistle_house.pipeline import Pipeline, PipelineSpec, PipelineStore
from whistle_house.server.engine import LiveEngine, encode_block
from whistle_house.settings import Paths, default_paths
from whistle_house.stages.base import list_stages
from whistle_house.tunes import Tune, TuneLibrary, fit_note_template, split_by_silence
from whistle_house.types import TuneMatch

log = logging.getLogger(__name__)

MANAGER_DIST = Path(__file__).resolve().parents[3] / "manager" / "dist"


# --------------------------------------------------------------------------- request bodies


class DeviceBody(BaseModel):
    device: str | None


class RecordBody(BaseModel):
    label: str = "whistle"
    tune_id: str | None = None
    notes: str = ""
    tags: list[str] = Field(default_factory=list)


class CaptureBody(RecordBody):
    seconds: float = 5.0


class AnalyzeBody(BaseModel):
    recording_id: str
    pipeline: str | None = None  # saved pipeline name...
    spec: PipelineSpec | None = None  # ...or an unsaved spec from the editor


class FitBody(BaseModel):
    pipeline: str | None = None
    split_gap_s: float = Field(
        1.0, description="Silence that separates repetitions in one recording"
    )


class TrainBody(BaseModel):
    name: str = "whistlenet"
    pipeline: str | None = None
    epochs: int = 20
    steps_per_epoch: int = 100
    batch_size: int = 16


class AppState:
    def __init__(self, paths: Paths, device: str | None, live: bool):
        self.paths = paths.ensure()
        self.recordings = RecordingStore(paths.recordings)
        self.tunes = TuneLibrary(paths.tunes)
        self.outputs = OutputStore(paths.outputs)
        self.pipelines = PipelineStore(paths.pipelines)
        self.state = json.loads(paths.state.read_text()) if paths.state.exists() else {}
        self.device = device or self.state.get("device")
        self.engine = LiveEngine(paths, self.tunes, self.outputs, self.recordings, self.device)
        self.live = live
        self.training: dict[str, subprocess.Popen] = {}

    def save_state(self) -> None:
        self.paths.state.write_text(json.dumps(self.state, indent=2))

    def active_spec(self) -> PipelineSpec:
        name = self.state.get("active_pipeline", "default")
        try:
            return self.pipelines.get(name)
        except FileNotFoundError:
            return self.pipelines.list()[0]


def create_app(paths: Paths | None = None, device: str | None = None, live: bool = True) -> FastAPI:
    S = AppState(paths or default_paths(), device, live)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            S.engine.set_pipeline(S.active_spec())
        except Exception as e:  # noqa: BLE001 - surface any device error to the UI
            log.error("active pipeline failed to build, falling back to default: %s", e)
            S.engine.set_pipeline(PipelineSpec())
        if S.live:
            S.engine.start()
        yield
        S.engine.stop()
        for p in S.training.values():
            p.terminate()

    app = FastAPI(title="whistle-house", lifespan=lifespan)
    app.state.S = S

    # ------------------------------------------------------------------ status / registry

    @app.get("/api/status")
    def status():
        return {"engine": S.engine.status(), "labels": LABELS, "data_dir": str(S.paths.root)}

    @app.get("/api/devices")
    def devices():
        try:
            return list_input_devices()
        except Exception as e:
            raise HTTPException(500, f"cannot query audio devices: {e}") from e

    @app.post("/api/live/device")
    def set_device(body: DeviceBody):
        S.state["device"] = S.device = S.engine.device = body.device
        S.save_state()
        S.engine.stop()
        S.engine.start()
        return S.engine.status()

    @app.post("/api/live/start")
    def live_start():
        S.engine.start()
        return S.engine.status()

    @app.post("/api/live/stop")
    def live_stop():
        S.engine.stop()
        return S.engine.status()

    @app.get("/api/live/config")
    def live_config():
        return S.engine.config_payload()

    @app.get("/api/stages")
    def stages():
        return list_stages()

    # ------------------------------------------------------------------ pipelines

    @app.get("/api/pipelines")
    def pipelines():
        return {
            "active": S.engine.spec.name if S.engine.spec else None,
            "pipelines": S.pipelines.list(),
        }

    @app.put("/api/pipelines/{name}")
    def put_pipeline(name: str, spec: PipelineSpec):
        spec.name = name
        try:
            Pipeline(spec, tunes=S.tunes, data_dir=S.paths.root)  # validate it builds
            S.pipelines.put(spec)
        except Exception as e:
            raise HTTPException(400, str(e)) from e
        if S.engine.spec and S.engine.spec.name == name:
            S.engine.set_pipeline(spec)
        return spec

    @app.delete("/api/pipelines/{name}")
    def delete_pipeline(name: str):
        if S.engine.spec and S.engine.spec.name == name:
            raise HTTPException(400, "cannot delete the active pipeline")
        S.pipelines.delete(name)
        return {"ok": True}

    @app.post("/api/pipelines/{name}/activate")
    def activate_pipeline(name: str):
        try:
            S.engine.set_pipeline(S.pipelines.get(name))
        except Exception as e:
            raise HTTPException(400, str(e)) from e
        S.state["active_pipeline"] = name
        S.save_state()
        return S.engine.status()

    # ------------------------------------------------------------------ recordings

    @app.get("/api/recordings")
    def recordings(label: str | None = None, tune_id: str | None = None) -> list[Recording]:
        return S.recordings.list(label, tune_id)

    @app.get("/api/recordings/stats")
    def recording_stats():
        return S.recordings.stats()

    @app.get("/api/recordings/{rec_id}")
    def recording(rec_id: str) -> Recording:
        try:
            return S.recordings.get(rec_id)
        except FileNotFoundError as e:
            raise HTTPException(404, "no such recording") from e

    @app.get("/api/recordings/{rec_id}/audio")
    def recording_audio(rec_id: str):
        p = S.recordings.audio_path(rec_id)
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="audio/wav")

    @app.patch("/api/recordings/{rec_id}")
    def update_recording(rec_id: str, upd: RecordingUpdate) -> Recording:
        return S.recordings.update(rec_id, upd)

    @app.delete("/api/recordings/{rec_id}")
    def delete_recording(rec_id: str):
        S.recordings.delete(rec_id)
        return {"ok": True}

    @app.post("/api/recordings/upload")
    async def upload_recording(
        file: UploadFile = File(...),  # noqa: B008 - FastAPI idiom
        label: str = Form("other"),
        tune_id: str | None = Form(None),
        notes: str = Form(""),
    ) -> Recording:
        import soundfile as sf

        try:
            data, sr = sf.read(io.BytesIO(await file.read()), dtype="float32")
        except Exception as e:
            raise HTTPException(400, f"unsupported audio file (use wav/flac/ogg): {e}") from e
        return S.recordings.add(
            to_mono_float(data),
            sr,
            label=label,
            tune_id=tune_id or None,
            notes=notes,
            source="upload",
            device=file.filename,
        )

    @app.post("/api/record/start")
    def record_start(body: RecordBody):
        try:
            S.engine.start_recording(**body.model_dump())
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return S.engine.status()

    @app.post("/api/record/stop")
    def record_stop() -> Recording | None:
        return S.engine.stop_recording()

    @app.post("/api/record/cancel")
    def record_cancel():
        S.engine.cancel_recording()
        return S.engine.status()

    @app.post("/api/record/capture")
    def record_capture(body: CaptureBody) -> Recording:
        meta = body.model_dump(exclude={"seconds"})
        try:
            return S.engine.capture(body.seconds, **meta)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    # ------------------------------------------------------------------ analysis

    def _analyze(spec: PipelineSpec, rec_id: str):
        pipe = Pipeline(spec, tunes=S.tunes, data_dir=S.paths.root)
        audio, _ = S.recordings.load(rec_id, spec.sample_rate)
        return pipe, pipe.analyze(audio)

    @app.post("/api/analyze")
    async def analyze(body: AnalyzeBody):
        spec = body.spec or (S.pipelines.get(body.pipeline) if body.pipeline else S.active_spec())
        try:
            pipe, res = await asyncio.to_thread(_analyze, spec, body.recording_id)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}") from e
        return {
            "pipeline": spec,
            "raw_axis": pipe.frontend.transform.axis.to_json(),
            "axis": pipe.axis.to_json(),
            "raw": encode_block(res.raw),
            "filtered": encode_block(res.filtered),
            "scores": res.scores.tolist(),
            "events": [e.to_json() for e in res.events],
            "matches": [m.to_json() for m in res.matches],
        }

    # ------------------------------------------------------------------ tunes

    @app.get("/api/tunes")
    def tunes() -> list[Tune]:
        return S.tunes.list()

    @app.post("/api/tunes")
    def create_tune(tune: Tune) -> Tune:
        return S.tunes.put(tune)

    @app.put("/api/tunes/{tune_id}")
    def update_tune(tune_id: str, tune: Tune) -> Tune:
        tune.id = tune_id
        return S.tunes.put(tune)

    @app.delete("/api/tunes/{tune_id}")
    def delete_tune(tune_id: str):
        S.tunes.delete(tune_id)
        return {"ok": True}

    @app.post("/api/tunes/{tune_id}/fit")
    async def fit_tune(tune_id: str, body: FitBody):
        tune = S.tunes.get(tune_id)
        rec_ids = list(
            dict.fromkeys(tune.examples + [r.id for r in S.recordings.list(tune_id=tune_id)])
        )
        if not rec_ids:
            raise HTTPException(400, "tune has no example recordings")
        spec = S.pipelines.get(body.pipeline) if body.pipeline else S.active_spec()

        def work():
            instances, per_rec = [], {}
            for rid in rec_ids:
                _, res = _analyze(spec, rid)
                groups = split_by_silence(res.events, body.split_gap_s)
                per_rec[rid] = [len(g) for g in groups]
                instances += groups
            return fit_note_template(instances), per_rec

        try:
            template, per_rec = await asyncio.to_thread(work)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        tune.template = template
        tune.examples = rec_ids
        S.tunes.put(tune)
        return {"tune": tune, "notes_per_instance": per_rec}

    # ------------------------------------------------------------------ outputs

    @app.get("/api/output-types")
    def output_types():
        return list_output_types()

    @app.get("/api/outputs")
    def outputs() -> list[OutputInstance]:
        return S.outputs.list()

    @app.post("/api/outputs")
    def create_output(o: OutputInstance) -> OutputInstance:
        try:
            return S.outputs.put(o)
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.put("/api/outputs/{oid}")
    def update_output(oid: str, o: OutputInstance) -> OutputInstance:
        o.id = oid
        try:
            return S.outputs.put(o)
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.delete("/api/outputs/{oid}")
    def delete_output(oid: str):
        S.outputs.delete(oid)
        return {"ok": True}

    @app.post("/api/outputs/{oid}/test")
    async def test_output(oid: str):
        inst = S.outputs.get(oid)
        match = TuneMatch("test", "test", 0.0, 1.0)
        try:
            await asyncio.to_thread(inst.build().fire, match)
        except Exception as e:
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e
        return {"ok": True}

    @app.get("/api/matches")
    def matches():
        return list(S.engine.matches)

    # ------------------------------------------------------------------ models

    @app.get("/api/models")
    def models():
        out = []
        for p in sorted(S.paths.models.glob("*.json")):
            st = json.loads(p.read_text())
            proc = S.training.get(st["name"])
            st["running"] = proc is not None and proc.poll() is None
            st["has_checkpoint"] = (S.paths.models / f"{st['name']}.pt").exists()
            st.pop("frontend", None)
            out.append(st)
        return out

    @app.post("/api/models/train")
    def train_model(body: TrainBody):
        proc = S.training.get(body.name)
        if proc is not None and proc.poll() is None:
            raise HTTPException(409, "already training")
        spec_name = body.pipeline or S.active_spec().name
        cmd = [
            sys.executable,
            "-m",
            "whistle_house.cli",
            "--data",
            str(S.paths.root),
            "train-detector",
            "--pipeline",
            spec_name,
            "--name",
            body.name,
            "--epochs",
            str(body.epochs),
            "--steps-per-epoch",
            str(body.steps_per_epoch),
            "--batch-size",
            str(body.batch_size),
        ]
        log_path = S.paths.models / f"{body.name}.log"
        S.training[body.name] = subprocess.Popen(
            cmd, stdout=log_path.open("w"), stderr=subprocess.STDOUT
        )
        return {"ok": True, "cmd": cmd}

    @app.post("/api/models/{name}/stop")
    def stop_training(name: str):
        proc = S.training.get(name)
        if proc and proc.poll() is None:
            proc.terminate()
        return {"ok": True}

    @app.get("/api/models/{name}/log")
    def model_log(name: str):
        p = S.paths.models / f"{name}.log"
        return {"log": p.read_text()[-20000:] if p.exists() else ""}

    # ------------------------------------------------------------------ live websocket

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        q = S.engine.subscribe()
        try:
            await ws.send_json({"type": "config", **S.engine.config_payload()})
            while True:
                await ws.send_json(await q.get())
        except WebSocketDisconnect:
            pass
        finally:
            S.engine.unsubscribe(q)

    # ------------------------------------------------------------------ static manager

    if MANAGER_DIST.exists():
        app.mount("/", StaticFiles(directory=MANAGER_DIST, html=True), name="manager")

    return app
