"""Share of frames with each speech label in a mix (sanity check of data/training.speech_labels)."""
import sys
from pathlib import Path
import numpy as np
from whistle_house.data.store import RecordingStore
from whistle_house.data.training import ExampleGenerator, collect_sources, config_for_mix
from whistle_house.pipeline import PipelineStore
from whistle_house.settings import external_root

if __name__ == "__main__":
    data = Path(sys.argv[1]); mix, aug = sys.argv[2], sys.argv[3]
    spec = PipelineStore(data / "pipelines").get("hum-log")
    fe = spec.frontend_dict()
    cfg = config_for_mix(mix, ("whistle", "hum"), aug, aux=("speech",), window_s=4.5)
    src = collect_sources(fe, ("train",), store=RecordingStore(data / "recordings"), label_cache_dir=data / "cache" / "labels", external_root=external_root(required=True), external_datasets=["mlend_whistles", "mlend_hums", "musan", "esc50", "demand"], config=cfg, workers=4, data_dir=data)
    gen = ExampleGenerator(fe, src, cfg, data)
    x, y = gen.batch(np.random.default_rng(3), 96)
    sp = y[..., 2]
    per = [(float((s == 1).mean()), float((s == -1).mean())) for s in sp]
    print(mix, aug, "frames: speech %.3f ignore %.3f none %.3f" % ((sp == 1).mean(), (sp == -1).mean(), (sp == 0).mean()))
    print("examples with any speech label: %d/96, all-ignore: %d, with hum+speech: %d, whistle+speech: %d" % (sum(p[0] > 0 for p in per), sum(p[1] == 1 for p in per), int(((y[..., 1] > 0).any(1) & (sp == 1).any(1)).sum()), int(((y[..., 0] > 0).any(1) & (sp == 1).any(1)).sum())))
    on = [p[0] for p in per if p[0] > 0]
    print("active share within speech examples: mean %.2f min %.2f max %.2f" % (np.mean(on), np.min(on), np.max(on)))
