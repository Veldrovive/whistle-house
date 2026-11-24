"""Offline check on the Pi: events + ms per 4096-sample chunk, old vs new pipeline, real tunes."""
from whistle_house.threads import limit_inference_threads

limit_inference_threads()
import json
import sys
import time
from pathlib import Path

import numpy as np

from whistle_house.audio.io import load_audio
from whistle_house.pipeline import Pipeline, PipelineStore
from whistle_house.tunes import TuneLibrary

data = Path("data")
store = PipelineStore(data / "pipelines")
tunes = TuneLibrary(data / "tunes.json")
clips = sys.argv[1:]
for name in ("humnet-long-room2", "humnet-long-gate-ft"):
    spec = store.get(name)
    times = []
    print("==", name)
    for cid in clips:
        meta = json.loads((data / "recordings" / f"{cid}.json").read_text())
        audio, _ = load_audio(data / "recordings" / f"{cid}.wav", spec.sample_rate)
        pipe = Pipeline(spec, tunes=tunes, data_dir=data)
        ev, ma = [], []
        for i in range(0, len(audio), spec.chunk_size):
            t = time.perf_counter()
            r = pipe.step(audio[i : i + spec.chunk_size])
            times.append((time.perf_counter() - t) * 1e3)
            ev += r.detection.events
            ma += r.matches
        e2, m2 = pipe.flush()
        ev += e2
        ma += m2
        kinds = {}
        for e in ev:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        print(f"  {cid} {meta.get('label'):8} {len(audio) / spec.sample_rate:5.1f} s  events {kinds}  "
              f"matches {[(getattr(m, 'tune_name', None) or getattr(m, 'tune_id', '?'), round(m.score, 2)) for m in ma]}")
    t = np.array(times)
    print(f"  ms per chunk: median {np.median(t):.1f}  p95 {np.percentile(t, 95):.1f}  worst {t.max():.1f}  (chunk = 92.9 ms of audio)")
