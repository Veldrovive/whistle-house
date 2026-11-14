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
