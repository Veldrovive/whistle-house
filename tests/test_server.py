import base64
import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from whistle_house.data.augment import synth_whistle
from whistle_house.server.app import create_app
from whistle_house.settings import Paths

SR = 44100


def _wav(audio):
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV")
    return buf.getvalue()


def test_api_roundtrip(tmp_path):
    app = create_app(Paths(tmp_path), live=False)
    with TestClient(app) as c:
        assert c.get("/api/status").json()["engine"]["state"] == "stopped"
        stages = c.get("/api/stages").json()
        assert "peak_tracker" in [s["name"] for s in stages["detector"]]

        # upload + analyze
        audio = synth_whistle(SR, 4.0, np.random.default_rng(1), n_notes=3).audio
        rec = c.post(
            "/api/recordings/upload",
            files={"file": ("a.wav", _wav(audio), "audio/wav")},
            data={"label": "whistle"},
        ).json()
        assert rec["label"] == "whistle"
        res = c.post("/api/analyze", json={"recording_id": rec["id"]}).json()
        vals = np.frombuffer(base64.b64decode(res["filtered"]["values"]), "<f4")
        assert vals.size == res["filtered"]["n_frames"] * res["filtered"]["n_bins"]
        assert len(res["events"]) == 3

        # pipelines: save a variant, bad stage name rejected
        spec = c.get("/api/pipelines").json()["pipelines"][0]
        spec["filters"] = [{"name": "frame_median", "params": {}}]
        assert c.put("/api/pipelines/median", json=spec).status_code == 200
        spec["detector"] = {"name": "nope", "params": {}}
        assert c.put("/api/pipelines/bad", json=spec).status_code == 400

        # tune from the example recording
        tune = c.post("/api/tunes", json={"name": "t", "examples": [rec["id"]]}).json()
        fit = c.post(f"/api/tunes/{tune['id']}/fit", json={}).json()
        assert len(fit["tune"]["template"]["notes"]) == 3
        res = c.post("/api/analyze", json={"recording_id": rec["id"]}).json()
        assert [m["tune_name"] for m in res["matches"]] == ["t"]

        # outputs
        types = [t["type"] for t in c.get("/api/output-types").json()]
        assert {"log", "webhook", "home_assistant_event"} <= set(types)
        out = c.post("/api/outputs", json={"name": "log", "type": "log"}).json()
        assert c.post(f"/api/outputs/{out['id']}/test").json() == {"ok": True}

        # recording without a mic is a clean 409
        assert c.post("/api/record/start", json={}).status_code == 409


def test_noise_profile_endpoints(tmp_path):
    rng = np.random.default_rng(0)
    with TestClient(create_app(Paths(tmp_path), live=False)) as c:
        sil = c.post(
            "/api/recordings/upload",
            files={"file": ("s.wav", _wav(0.02 * rng.standard_normal(SR * 3)), "audio/wav")},
            data={"label": "silence"},
        ).json()
        spec = c.get("/api/pipelines").json()["pipelines"][0]
        spec["filters"] = []
        body = {"name": "room", "recording_ids": [sil["id"]], "spec": spec}
        prof = c.post("/api/noise-profiles", json=body).json()
        assert prof["name"] == "room" and len(prof["mean"]) == len(prof["freqs"])
        assert c.post("/api/noise-profiles", json=body).status_code == 400  # immutable
        assert [p["name"] for p in c.get("/api/noise-profiles").json()] == ["room"]

        spec["filters"] = [{"name": "spectral_gate", "params": {"profile": "room"}}]
        assert c.put("/api/pipelines/gated", json=spec).status_code == 200
        # a filter in front of the gate invalidates the profile
        bad = {**spec, "filters": [{"name": "frame_median", "params": {}}, *spec["filters"]]}
        r = c.put("/api/pipelines/bad", json=bad)
        assert r.status_code == 400 and "rebuild" in r.json()["detail"]

        assert c.delete("/api/noise-profiles/room").status_code == 400  # in use
        c.delete("/api/pipelines/gated")
        assert c.delete("/api/noise-profiles/room").json() == {"ok": True}


def test_datasets_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("WHISTLE_HOUSE_EXTERNAL", raising=False)
    with TestClient(create_app(Paths(tmp_path), live=False)) as c:
        info = c.get("/api/datasets").json()
        assert info["root"] is None
        assert {d["name"] for d in info["datasets"]} >= {"mlend_whistles", "musan", "esc50"}
        assert set(info["mixes"]) == {"mic", "pretrain", "finetune"}


def test_written_tune_roundtrip_and_match(tmp_path):
    from whistle_house.data.perform import Style, perform, render_audio
    from whistle_house.tunes import Score

    notes = [{"pitch": p, "beats": b} for p, b in [(60, 1), (64, 1), (67, 1), (None, 1), (72, 2)]]
    with TestClient(create_app(Paths(tmp_path), live=False)) as c:
        tune = c.post("/api/tunes", json={"name": "call", "score": {"notes": notes}}).json()
        assert tune["score"]["tempo_bpm"] == 100 and tune["score"]["notes"][3]["pitch"] is None
        tune["score"]["tempo_bpm"] = 120
        tune["score"]["notes"][0]["flat"] = True
        saved = c.put(f"/api/tunes/{tune['id']}", json=tune).json()
        assert saved["score"]["tempo_bpm"] == 120 and c.get("/api/tunes").json() == [saved]
        bad = {**tune, "score": {"notes": [{"pitch": 60, "beats": 0}]}}
        assert c.put(f"/api/tunes/{tune['id']}", json=bad).status_code == 422

        # the recognizer is a stage like any other: listed with its form, usable in a pipeline
        stages = c.get("/api/stages").json()["recognizer"]
        assert "quiet_before_s" in next(s for s in stages if s["name"] == "score_align")["defaults"]
        spec = c.get("/api/pipelines").json()["pipelines"][0]
        spec["recognizer"] = {"name": "score_align", "params": {}}
        assert c.put("/api/pipelines/written", json=spec).status_code == 200

        style = Style(transpose=24, tempo_scale=0.9, legato=1.0, slide_s=0.08)
        times, curve = perform(
            Score.model_validate(saved["score"]), style, np.random.default_rng(0)
        )
        rec = c.post(
            "/api/recordings/upload",
            files={"file": ("a.wav", _wav(render_audio(times, curve, SR)), "audio/wav")},
            data={"label": "whistle"},
        ).json()
        res = c.post("/api/analyze", json={"recording_id": rec["id"], "pipeline": "written"}).json()
        (match,) = res["matches"]
        assert match["tune_name"] == "call" and len(match["meta"]["notes"]) == 4
        assert abs(match["meta"]["key"] - 24) < 0.5
