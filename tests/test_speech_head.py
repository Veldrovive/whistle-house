"""Speech output: labels from the speech interferers, the presence-only head, and the learned
detector holding hums back while and after someone talks."""

import json
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from whistle_house.data.augment import synth_pseudo_speech
from whistle_house.data.training import (
    SPEECH_FRAME_S,
    ExampleGenerator,
    GeneratorConfig,
    TrainingSources,
    collect_sources,
    config_for_mix,
    speech_activity,
    speech_labels,
)
from whistle_house.pipeline import Pipeline, PipelineSpec, StageSpec

SR = 44100
LOG_SPEC = PipelineSpec(transform=StageSpec(name="stft_log"), filters=[])
FE = LOG_SPEC.frontend_dict()


def _bursts(pattern: list[tuple[float, bool]], seed: int = 0) -> np.ndarray:
    """Noise at -26 dBFS where on, near silence (-80 dBFS) where off."""
    rng = np.random.default_rng(seed)
    parts = [
        (0.05 if on else 1e-4) * rng.standard_normal(int(s * SR)).astype(np.float32)
        for s, on in pattern
    ]
    return np.concatenate(parts)


def test_speech_activity_fills_short_gaps_only():
    x = _bursts([(1.0, True), (0.16, False), (1.0, True), (1.0, False), (0.5, True)])
    on = speech_activity(x, SR)
    t = (np.arange(len(on)) + 0.5) * SPEECH_FRAME_S
    assert on[t < 2.1].all()  # the 0.16 s gap is part of the phrase
    assert not on[(t > 2.25) & (t < 3.1)].any()  # the 1 s pause is not
    assert on[t > 3.25].all()


def test_speech_labels_depend_on_how_far_under_the_mix_the_speech_is():
    cfg = GeneratorConfig(aux=("speech",))
    speech = _bursts([(1.0, True), (1.0, False), (1.0, True)])
    rest = 0.05 * np.random.default_rng(1).standard_normal(len(speech)).astype(np.float32)
    on = speech_activity(speech, SR)
    as_loud = speech_labels(speech, rest, SR, cfg)
    assert (as_loud[on] == 1).all() and (as_loud[~on] == 0).all() and 0.5 < on.mean() < 0.8
    under15 = speech_labels(speech, rest * 10 ** (15 / 20), SR, cfg)
    assert (under15[on] == -1).all() and (under15[~on] == 0).all()  # can't say: ignored
    assert (speech_labels(speech, rest * 10 ** (30 / 20), SR, cfg) == 0).all()  # inaudible


def test_examples_carry_a_speech_column_only_when_asked(tmp_path):
    path = str(tmp_path / "talk.wav")
    sf.write(path, _bursts([(0.7, True), (0.7, False)] * 8), SR)
    alone = {  # the interferer on its own: no whistle or hum on top
        "p_interferer": 1.0,
        "p_mic_whistle": 0.0,
        "p_synth_whistle": 0.0,
        "p_mic_hum": 0.0,
        "p_synth_hum": 0.0,
        "window_s": 2.0,
    }
    cfg = config_for_mix("mic", ("whistle", "hum"), "room", aux=("speech",), **alone)
    speech = TrainingSources(interferers={"speech": [path]}, speech_files=frozenset([path]))
    gen = ExampleGenerator(FE, speech, replace(cfg, interferer_weights={"speech": 1.0}))
    rng = np.random.default_rng(0)
    x, y = gen.batch(rng, 6)
    assert y.shape == (*x.shape[:2], 3) and not y[..., :2].any()
    sp = y[..., 2]
    assert set(np.unique(sp)) == {0.0, 1.0} and 0.3 < sp.mean() < 0.9
    assert x[sp == 1].mean() > x[sp == 0].mean() + 5  # labelled frames are the loud ones (dB)

    # the same file not known as speech -> no speech; pseudo-speech -> "can't say" everywhere
    other = TrainingSources(interferers={"speech": [path]})
    _, y = ExampleGenerator(FE, other, replace(cfg, interferer_weights={"speech": 1.0})).batch(
        rng, 3
    )
    assert not y[..., 2].any()
    _, y = ExampleGenerator(FE, TrainingSources(), cfg).batch(rng, 3)
    assert (y[..., 2] == -1).all()

    # without aux the labels are the f0 columns alone, as before
    _, y = ExampleGenerator(FE, speech, replace(cfg, aux=())).batch(rng, 2)
    assert y.shape[-1] == 2
    with pytest.raises(ValueError, match="aux"):
        GeneratorConfig(aux=("laughter",))


torch = pytest.importorskip("torch")


def test_aux_head_adds_presence_columns_and_leaves_the_rest_alone():
    from whistle_house.ml.model import WhistleNet, WhistleNetConfig, add_aux

    torch.manual_seed(0)
    base = WhistleNet(
        WhistleNetConfig(n_bins=40, channels=4, dilations=(1, 2), classes=("whistle", "hum"))
    ).eval()
    model = add_aux(base, ("speech",))
    assert model.cfg.aux == ("speech",) and add_aux(model, ("speech",)) is model
    x = torch.randn(2, 30, 40)
    with torch.no_grad():
        pres0, sal0 = base(x)
        pres, sal = model(x)
        # streaming in pieces == the causal pass, aux column included
        state, parts = model.init_state(2), []
        for chunk in (x[:, :7], x[:, 7:8], x[:, 8:]):
            p, _, state = model.forward_stream(chunk, state)
            parts.append(p)
        causal, _ = model.forward_causal(x)
    assert pres.shape == (2, 30, 3) and sal.shape == (2, 30, 2, 40)
    assert torch.equal(pres[..., :2], pres0) and torch.equal(sal, sal0)
    assert torch.allclose(torch.cat(parts, 1), causal, atol=1e-6)
    with pytest.raises(ValueError, match="not classes"):
        WhistleNet(WhistleNetConfig(n_bins=40, classes=("whistle",), aux=("whistle",)))


def test_train_with_a_speech_head_and_export_it(tmp_path):
    pytest.importorskip("librosa")
    from test_harmonic import _store_with_hums

    from whistle_house.ml.export import OnnxStream, export_onnx
    from whistle_house.ml.model import load_checkpoint
    from whistle_house.ml.train import TrainConfig, train

    paths, store = _store_with_hums(tmp_path)
    src = collect_sources(FE, ("train",), store=store, label_cache_dir=paths.label_cache)
    assert len(src.speech_files) == 1 and src.summary()["speech_files"] == 1
    tiny = {
        "epochs": 1,
        "steps_per_epoch": 2,
        "batch_size": 2,
        "val_examples": 4,
        "device": "cpu",
        "workers": 0,
    }
    train(LOG_SPEC, paths, "sp", TrainConfig(**tiny, arch="harmonic", aux=["speech"]))
    status = json.loads((paths.models / "sp.json").read_text())
    assert status["state"] == "done" and status["aux"] == ["speech"]
    last = status["history"][-1]
    assert {"speech_f1", "speech_recall", "speech_false_positive_rate", "hum_f1"} <= set(last)
    model, _ = load_checkpoint(paths.models / "sp.pt")
    assert model.cfg.aux == ("speech",)

    # a fine-tune keeps the head; one from a checkpoint without it can add it
    train(LOG_SPEC, paths, "sp2", TrainConfig(**tiny, init_from="sp", mix="finetune"))
    assert json.loads((paths.models / "sp2.json").read_text())["aux"] == ["speech"]
    train(LOG_SPEC, paths, "plain", TrainConfig(**tiny, arch="harmonic"))
    assert load_checkpoint(paths.models / "plain.pt")[0].cfg.aux == ()
    train(LOG_SPEC, paths, "added", TrainConfig(**tiny, init_from="plain", aux=["speech"]))
    assert load_checkpoint(paths.models / "added.pt")[0].cfg.aux == ("speech",)

    pytest.importorskip("onnxruntime")
    exported = OnnxStream(export_onnx(paths.models / "sp.pt"))  # checked against torch inside
    assert exported.aux == ("speech",) and exported.classes == ("whistle", "hum")


def _detector(tmp_path, **params):
    """A learned detector on an (untrained) whistle + hum + speech checkpoint."""
    from whistle_house.ml.model import WhistleNet, WhistleNetConfig, save_checkpoint
    from whistle_house.pipeline import Frontend

    axis = Frontend(LOG_SPEC.sample_rate, LOG_SPEC.transform, LOG_SPEC.filters).axis
    torch.manual_seed(0)
    cfg = WhistleNetConfig(
        n_bins=axis.n_bins,
        channels=4,
        dilations=(1, 2),
        classes=("whistle", "hum"),
        aux=params.pop("aux", ("speech",)),
    )
    path = tmp_path / "models" / "m.pt"
    save_checkpoint(path, WhistleNet(cfg).eval(), (-40.0, 20.0), FE, {})
    det = StageSpec(name="learned", params={"checkpoint": "m.pt", **params})
    return Pipeline(LOG_SPEC.model_copy(update={"detector": det}), data_dir=tmp_path).detector


def _held_note_with_speech(det, speech_frames: range, n: int = 400):
    """Model outputs for a hum and a whistle held throughout, with speech on some frames."""
    period = 512 / SR
    pres = np.zeros((n, 3), np.float32)
    pres[:, :2] = 0.9
    pres[list(speech_frames), 2] = 0.9
    sal = np.zeros((n, 2, det.backend.n_bins), np.float32)
    sal[:, 0, 100] = 1.0
    sal[:, 1, 30] = 1.0
    det.reset()
    det._times.extend(np.arange(n) * period)
    events = det._track(pres, sal) + [e for tr in det.trackers.values() for e in tr.flush()]
    return {k: [e for e in events if e.kind == k] for k in ("whistle", "hum")}, period


def test_speech_holds_hums_back_but_not_whistles(tmp_path):
    det = _detector(tmp_path, speech_hold_s=1.0, speech_min_s=0.1)
    ev, period = _held_note_with_speech(det, range(100, 150))
    # the whistle runs through; the hum that was sounding when speech was confirmed is dropped
    # and hums come back one hold after the last speech frame
    assert len(ev["whistle"]) == 1 and ev["whistle"][0].duration > 399 * period - 0.01
    assert len(ev["hum"]) == 1
    assert ev["hum"][0].start == pytest.approx(149 * period + 1.0, abs=2 * period)

    # speech shorter than speech_min_s is a blip: nothing is held
    ev, _ = _held_note_with_speech(det, range(100, 105))
    assert len(ev["hum"]) == 1 and ev["hum"][0].duration > 399 * period - 0.01

    # off by default (the speech score is still reported); and it needs a speech output
    off = _detector(tmp_path)
    ev, _ = _held_note_with_speech(off, range(100, 150))
    assert len(ev["hum"]) == 1 and ev["hum"][0].duration > 399 * period - 0.01
    with pytest.raises(ValueError, match="speech output"):
        _detector(tmp_path, aux=(), speech_hold_s=1.0)


def test_speech_score_is_reported_and_streaming_is_chunk_invariant(tmp_path):
    from whistle_house.ml.model import load_checkpoint, save_checkpoint

    _detector(tmp_path)  # writes models/m.pt
    model, ck = load_checkpoint(tmp_path / "models" / "m.pt")
    torch.nn.init.constant_(model.presence.bias, 0.6)  # notes to track
    save_checkpoint(tmp_path / "models" / "m.pt", model, (-40.0, 20.0), ck["frontend"], {})
    audio = synth_pseudo_speech(SR, 3.0, np.random.default_rng(2))

    def run(chunk):
        det = StageSpec(
            name="learned",
            params={"checkpoint": "m.pt", "speech_hold_s": 0.3, "speech_threshold": 0.45},
        )
        spec = LOG_SPEC.model_copy(update={"chunk_size": chunk, "detector": det})
        return Pipeline(spec, data_dir=tmp_path).analyze(audio)

    a, b = run(4096), run(700)
    assert set(a.kind_scores) == {"whistle", "hum", "speech"}
    np.testing.assert_allclose(a.kind_scores["speech"], b.kind_scores["speech"], atol=1e-5)
    np.testing.assert_allclose(a.scores, np.maximum(a.kind_scores["whistle"], a.kind_scores["hum"]))

    def key(r):
        return [(e.kind, round(e.start, 4), round(e.end, 4)) for e in r.events]

    assert key(a) == key(b)
