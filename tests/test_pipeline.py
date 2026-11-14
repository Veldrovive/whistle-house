import numpy as np
import pytest

from whistle_house.data.augment import mix_at_snr, rms, synth_pseudo_speech, synth_whistle
from whistle_house.data.store import RecordingStore
from whistle_house.data.training import ExampleGenerator, GeneratorConfig
from whistle_house.pipeline import Pipeline, PipelineSpec, PipelineStore, StageSpec
from whistle_house.stages.base import list_stages
from whistle_house.tunes import Tune, TuneLibrary, fit_note_template, split_by_silence

SR = 44100


def test_registry_describes_every_stage():
    stages = list_stages()
    assert set(stages) == {"transform", "filter", "detector", "recognizer"}
    for impls in stages.values():
        for s in impls:
            assert s["schema"]["type"] == "object"


def test_peak_tracker_finds_synthetic_notes():
    sw = synth_whistle(SR, 4.0, np.random.default_rng(1), n_notes=3)
    events = Pipeline(PipelineSpec()).analyze(sw.audio).events
    assert len(events) == 3
    for ev in events:
        truth = np.median(sw.f0[int(ev.start * SR) : int(ev.end * SR)])
        assert abs(12 * np.log2(ev.median_hz() / truth)) < 1.0


def test_streaming_matches_offline_pipeline():
    sw = synth_whistle(SR, 3.0, np.random.default_rng(2), n_notes=2)
    p = Pipeline(
        PipelineSpec(filters=[StageSpec(name="noise_floor"), StageSpec(name="frame_median")])
    )
    a = p.analyze(sw.audio)
    p.reset()
    events = []
    for i in range(0, len(sw.audio), 777):
        events += p.step(sw.audio[i : i + 777]).detection.events
    events += p.flush()[0]
    assert [round(e.start, 3) for e in events] == [round(e.start, 3) for e in a.events]


def test_tune_fit_and_match(tmp_path):
    lib = TuneLibrary(tmp_path / "tunes.json")
    spec = PipelineSpec()

    def phrase(base, scale=1.0):
        """3 notes: base, +4 st, +7 st, separated by short gaps."""
        out, t = np.zeros(int(SR * 2.5), np.float32), 0.1
        for st in (0, 4, 7):
            n = np.arange(int(0.25 * scale * SR))
            s = int(t * SR)
            out[s : s + len(n)] = 0.5 * np.sin(2 * np.pi * base * 2 ** (st / 12) * n / SR)
            t += 0.4 * scale
        return out

    examples = [Pipeline(spec).analyze(phrase(b)).events for b in (900, 1000, 1100, 1200)]
    tpl = fit_note_template(examples)
    assert len(tpl.notes) == 3
    lib.put(Tune(name="arpeggio", template=tpl))

    hit = Pipeline(spec, tunes=lib).analyze(phrase(1400, scale=1.1)).matches
    assert [m.tune_name for m in hit] == ["arpeggio"]

    # descending version must not match
    rev = phrase(1400)[::-1].copy()
    assert Pipeline(spec, tunes=lib).analyze(rev).matches == []


def test_split_by_silence():
    sw = synth_whistle(SR, 4.0, np.random.default_rng(1), n_notes=3)
    events = Pipeline(PipelineSpec()).analyze(sw.audio).events
    assert len(split_by_silence(events, 10.0)) == 1
    assert len(split_by_silence(events, 0.0)) == len(events)


def test_mix_at_snr():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(SR).astype(np.float32)
    n = synth_pseudo_speech(SR, 1.0, rng)
    mixed = mix_at_snr(s, n, 6.0)
    noise_part = mixed - s
    assert 20 * np.log10(rms(s) / rms(noise_part)) == pytest.approx(6.0, abs=0.01)


def test_example_generator_shapes(tmp_path):
    store = RecordingStore(tmp_path / "rec")
    sw = synth_whistle(SR, 3.0, np.random.default_rng(4), n_notes=3)
    rec = store.add(sw.audio, SR, label="whistle", split="train")
    assert store.get(rec.id).label == "whistle"
    gen = ExampleGenerator(
        PipelineSpec().frontend_dict(), store, [], config=GeneratorConfig(window_s=1.0)
    )
    assert len(gen.positives) == 1
    x, f0 = gen.batch(np.random.default_rng(0), 4)
    assert x.shape[0] == 4 and x.shape[2] == 128 and f0.shape == x.shape[:2]


def test_pipeline_store_default(tmp_path):
    store = PipelineStore(tmp_path)
    assert [s.name for s in store.list()] == ["default"]
    with pytest.raises(ValueError):
        store.get("../etc")
