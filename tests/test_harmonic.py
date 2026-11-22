"""Whistle + hum model: harmonic-aligned input, streaming, hum training data, event kinds."""

import json

import numpy as np
import pytest

from whistle_house.data.augment import synth_hum, synth_pseudo_speech, synth_whistle
from whistle_house.data.store import RecordingStore
from whistle_house.data.training import (
    ExampleGenerator,
    GeneratorConfig,
    collect_sources,
    config_for_mix,
)
from whistle_house.pipeline import Frontend, Pipeline, PipelineSpec, StageSpec
from whistle_house.settings import Paths
from whistle_house.stages.recognizers.note_template import NoteTemplateRecognizer
from whistle_house.tunes import Tune, TuneLibrary, fit_note_template
from whistle_house.types import ContourPoint, WhistleEvent

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")

SR = 44100
# No noise_floor filter (the PipelineSpec default): it adapts to held notes and swallows hums.
LOG_SPEC = PipelineSpec(transform=StageSpec(name="stft_log"), filters=[])
FE = LOG_SPEC.frontend_dict()


def _stack_at(audio: np.ndarray, hz: float) -> dict[float, float]:
    """Median level (dB, relative to the strongest multiple) of each harmonic channel at the
    pitch bin nearest ``hz``, for a steady sound through the real stft_log front end."""
    from whistle_house.ml.model import HARMONIC_GRID, HarmonicStack, log_bins_per_octave

    fe = Frontend.from_dict(FE)
    _, block = fe.process_signal(audio)
    freqs = fe.axis.freqs
    stack = HarmonicStack(len(freqs), log_bins_per_octave(freqs), HARMONIC_GRID)
    b = int(np.argmin(np.abs(freqs - hz)))
    v = stack(torch.from_numpy(block.values[20:-20])[None])[0, :, :, b].median(dim=1).values
    return dict(zip(HARMONIC_GRID, (v - v.max()).tolist(), strict=True))


def _steady_hum(f0: float, seconds: float = 1.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    levels = (0, -6, -12, -20, -24, -28, -30, -32)
    x = sum(10 ** (db / 20) * np.sin(2 * np.pi * (k + 1) * f0 * t) for k, db in enumerate(levels))
    return (0.2 * x / np.abs(x).max()).astype(np.float32)


@pytest.mark.parametrize("f0", [90.0, 131.0, 205.0, 330.0])
def test_harmonic_stack_puts_harmonics_in_the_same_channels_at_every_pitch(f0):
    """At the fundamental's bin, the integer channels hold the harmonics and the half-integer
    ones the gaps between them, whatever the pitch (the physics the model relies on)."""
    at_f0 = _stack_at(_steady_hum(f0), f0)
    for h in range(1, 9):
        if (h + 0.5) * f0 > 3900:
            break
        assert at_f0[float(h)] > at_f0[h - 0.5] + 10, (h, at_f0)
        assert h == 8 or at_f0[float(h)] > at_f0[h + 0.5] + 10, (h, at_f0)
    assert at_f0[1.0] == 0.0 and at_f0[0.5] < -25 and at_f0[1 / 3] < -25
    # The wrong candidates give themselves away through the sub-harmonic channels...
    assert _stack_at(_steady_hum(f0), 2 * f0)[0.5] > -3  # at h2, half the frequency is filled
    assert _stack_at(_steady_hum(f0), 3 * f0)[1 / 3] > -3  # at h3, a third of it is
    # ...and one octave below, only every other integer channel is filled.
    below = _stack_at(_steady_hum(f0), f0 / 2) if f0 / 2 > 65 else None
    if below:
        assert below[2.0] > below[1.0] + 20 and below[2.0] > below[3.0] + 20


def test_whistle_is_a_single_channel():
    t = np.arange(int(1.5 * SR)) / SR
    tone = (0.3 * np.sin(2 * np.pi * 1200.0 * t)).astype(np.float32)
    at = _stack_at(tone, 1200.0)
    assert at[1.0] == 0.0 and max(v for m, v in at.items() if m != 1.0) < -30


def test_harmonic_model_streams_and_keeps_old_models_unchanged():
    from whistle_house.ml.model import (
        ARCHS,
        WhistleNet,
        WhistleNetConfig,
        log_bins_per_octave,
    )
    from whistle_house.stages.transforms.stft import log_frequencies, mel_frequencies

    freqs = log_frequencies(60, 4000, 24)
    cfg = WhistleNetConfig(
        n_bins=len(freqs), bins_per_octave=log_bins_per_octave(freqs), **ARCHS["harmonic"]
    )
    torch.manual_seed(0)
    model = WhistleNet(cfg).eval()
    for blk in model.blocks:  # the frequency context starts at zero; make it do something
        torch.nn.init.normal_(blk.ctx.weight, std=0.2)
    assert cfg.classes == ("whistle", "hum") and model.receptive_field == 63
    x = torch.randn(1, 90, len(freqs))
    with torch.no_grad():
        pres, sal = model.forward_causal(x)
        state, parts = model.init_state(), []
        for i in range(0, 90, 7):
            p, s, state = model.forward_stream(x[:, i : i + 7], state)
            parts.append((p, s))
    assert pres.shape == (1, 90, 2) and sal.shape == (1, 90, 2, len(freqs))
    assert torch.allclose(torch.cat([p for p, _ in parts], 1), pres, atol=1e-5)
    assert torch.allclose(torch.cat([s for _, s in parts], 1), sal, atol=1e-5)

    # the harmonic input only makes sense on a log axis
    with pytest.raises(ValueError, match="log-frequency"):
        log_bins_per_octave(mel_frequencies(130, 300, 4000)[1:-1])
    # A default (whistle-only) model has exactly the parameters it always had, so checkpoints
    # trained before the hum work still load.
    old = WhistleNet(WhistleNetConfig())
    assert sum(p.numel() for p in old.parameters()) == 7330
    assert {k.split(".")[0] for k in old.state_dict()} == {"blocks", "salience", "presence"}


def test_deep_harmonic_model_streams_with_twice_the_context():
    from whistle_house.ml.model import ARCHS, WhistleNet, WhistleNetConfig, log_bins_per_octave
    from whistle_house.stages.transforms.stft import log_frequencies

    freqs = log_frequencies(60, 4000, 24)
    bpo = log_bins_per_octave(freqs)
    deep = WhistleNet(
        WhistleNetConfig(n_bins=len(freqs), bins_per_octave=bpo, **ARCHS["harmonic_deep"])
    )
    base = WhistleNet(WhistleNetConfig(n_bins=len(freqs), bins_per_octave=bpo, **ARCHS["harmonic"]))
    assert len(deep.blocks) == 2 * len(base.blocks) and deep.receptive_field == 125
    n_deep, n_base = (sum(p.numel() for p in m.parameters()) for m in (deep, base))
    assert 1.8 * n_base < n_deep < 2.1 * n_base  # ~60k vs ~31k
    torch.manual_seed(1)
    deep.eval()
    for blk in deep.blocks:
        torch.nn.init.normal_(blk.ctx.weight, std=0.2)
    x = torch.randn(1, 160, len(freqs))  # longer than the receptive field
    with torch.no_grad():
        pres, sal = deep.forward_causal(x)
        state, parts = deep.init_state(), []
        for i in range(0, 160, 11):
            p, s, state = deep.forward_stream(x[:, i : i + 11], state)
            parts.append((p, s))
    assert torch.allclose(torch.cat([p for p, _ in parts], 1), pres, atol=1e-5)
    assert torch.allclose(torch.cat([s for _, s in parts], 1), sal, atol=1e-5)


def test_synth_hum_is_a_labelled_harmonic_stack():
    from whistle_house.data.hum_labels import harmonic_levels, hum_f0

    for seed in range(3):
        h = synth_hum(SR, 3.0, np.random.default_rng(seed))
        v = h.f0 > 0
        assert 0.2 < v.mean() < 1.0 and 75 < h.f0[v].min() and h.f0[v].max() < 450
        label = h.f0[:: SR // 100]
        levels = harmonic_levels(h.audio, SR, label, 6)
        assert levels[3] < -8 and levels[5] < -15  # steep roll-off, like a real hum
        est = hum_f0(h.audio, SR).astype(np.float32)[: len(label)]
        both = (est > 0) & (label[: len(est)] > 0)
        # pYIN (the labeller for real hums) agrees with the synthetic label: it is the fundamental
        assert np.median(np.abs(12 * np.log2(est[both] / label[: len(est)][both]))) < 0.5


def _store_with_hums(tmp_path) -> tuple[Paths, RecordingStore]:
    paths = Paths(tmp_path / "data").ensure()
    store = RecordingStore(paths.recordings)
    for i in range(2):
        store.add(
            synth_hum(SR, 3.0, np.random.default_rng(i)).audio, SR, label="hum", split="train"
        )
    store.add(
        synth_whistle(SR, 3.0, np.random.default_rng(5), n_notes=3).audio,
        SR,
        label="whistle",
        split="train",
    )
    store.add(
        synth_pseudo_speech(SR, 3.0, np.random.default_rng(3)), SR, label="speech", split="train"
    )
    return paths, store


def test_hums_are_positives_only_when_the_hum_class_is_trained(tmp_path):
    paths, store = _store_with_hums(tmp_path)
    common = {"store": store, "label_cache_dir": paths.label_cache, "workers": 1}

    plain = collect_sources(FE, ("train",), config=GeneratorConfig(window_s=1.0), **common)
    assert len(plain.interferers["hum"]) == 2 and not plain.mic_hums

    cfg = config_for_mix("mic", ("whistle", "hum"), window_s=1.0)
    assert cfg.p_mic_hum > 0 and cfg.p_synth_hum > 0
    src = collect_sources(FE, ("train",), config=cfg, **common)
    assert len(src.mic_hums) == 2 and "hum" not in src.interferers
    assert all((c.f0 > 0).mean() > 0.2 for c in src.mic_hums)  # pYIN found the humming
    gen = ExampleGenerator(FE, src, cfg)
    assert "hum" not in gen._groups  # never mixed in as unlabelled noise
    x, f0 = gen.batch(np.random.default_rng(0), 24)
    assert f0.shape == (*x.shape[:2], 2)
    whistle, hum = f0[..., 0], f0[..., 1]
    assert (whistle > 0).any() and (hum > 0).any()
    assert not ((whistle > 0) & (hum > 0)).any()  # one clean source per example
    assert hum[hum > 0].max() < 500 < whistle[whistle > 0].min()


def test_train_harmonic_and_detect_both_kinds(tmp_path):
    from whistle_house.ml.train import TrainConfig, train

    paths, _ = _store_with_hums(tmp_path)
    tiny = {"epochs": 1, "steps_per_epoch": 2, "batch_size": 2, "val_examples": 4, "device": "cpu"}
    train(LOG_SPEC, paths, "hn", TrainConfig(**tiny, arch="harmonic", workers=0))
    status = json.loads((paths.models / "hn.json").read_text())
    assert status["state"] == "done" and status["classes"] == ["whistle", "hum"]
    assert status["data"]["train"]["mic_hum_clips"] == 2
    last = status["history"][-1]
    assert {"f1", "whistle_f1", "hum_f1", "hum_false_positive_rate", "mic_train_f1"} <= set(last)

    # a fine-tune keeps the checkpoint's classes; the mel front end is refused for this arch
    train(LOG_SPEC, paths, "hn2", TrainConfig(**tiny, init_from="hn", mix="finetune", workers=0))
    assert json.loads((paths.models / "hn2.json").read_text())["classes"] == ["whistle", "hum"]
    with pytest.raises(ValueError, match="log-frequency"):
        train(PipelineSpec(), paths, "bad", TrainConfig(**tiny, arch="harmonic", workers=0))

    # Detection: bias the (barely trained) presence outputs so both kinds report, then check
    # that streaming in any chunk size gives the same events and scores, lookahead included.
    from whistle_house.ml.model import load_checkpoint, save_checkpoint

    model, ck = load_checkpoint(paths.models / "hn.pt")
    torch.nn.init.constant_(model.presence.bias, 0.6)
    norm = (ck["norm"]["mean"], ck["norm"]["std"])
    save_checkpoint(paths.models / "hn.pt", model, norm, ck["frontend"], {})
    spec = LOG_SPEC.model_copy(
        update={"detector": StageSpec(name="learned", params={"checkpoint": "hn.pt"})}
    )
    audio = synth_hum(SR, 2.0, np.random.default_rng(1)).audio

    def run(chunk, **params):
        det = StageSpec(name="learned", params={"checkpoint": "hn.pt", **params})
        p = Pipeline(
            spec.model_copy(update={"chunk_size": chunk, "detector": det}), data_dir=paths.root
        )
        return p, p.analyze(audio)

    (pa, a), (_, b) = run(4096), run(700)

    def key(r):
        return [
            (e.kind, round(e.start, 4), round(e.end, 4), round(e.median_hz(), 1)) for e in r.events
        ]

    assert key(a) == key(b) and {e.kind for e in a.events} == {"whistle", "hum"}
    la = pa.detector.lookahead
    assert la == 12 and set(a.kind_scores) == {"whistle", "hum"}
    offline, _ = pa.detector.infer(a.filtered.values)
    for i, kind in enumerate(pa.detector.classes):
        np.testing.assert_allclose(
            a.kind_scores[kind][la:], offline[: len(offline) - la, i], atol=1e-4
        )
        np.testing.assert_allclose(
            b.kind_scores[kind][la:], offline[: len(offline) - la, i], atol=1e-4
        )
    np.testing.assert_allclose(a.scores, np.maximum(a.kind_scores["whistle"], a.kind_scores["hum"]))

    # `kinds` narrows what a pipeline reports; asking for a kind the model lacks is an error
    _, only = run(4096, kinds=["hum"])
    assert {e.kind for e in only.events} == {"hum"} and set(only.kind_scores) == {"hum"}
    with pytest.raises(ValueError, match="detects"):
        run(4096, kinds=["song"])


def _phrase(base_hz: float, kind: str, t0: float = 0.0) -> list[WhistleEvent]:
    events = []
    for i, st in enumerate((0, 4, 7)):
        hz = base_hz * 2 ** (st / 12)
        s = t0 + 0.4 * i
        pts = [ContourPoint(s + 0.01 * j, hz, 1.0) for j in range(25)]
        events.append(WhistleEvent(s, s + 0.25, pts, kind=kind))
    return events


def test_tunes_match_hummed_or_whistled_but_not_mixed(tmp_path):
    lib = TuneLibrary(tmp_path / "tunes.json")
    lib.put(
        Tune(
            name="arp",
            template=fit_note_template([_phrase(b, "whistle") for b in (900, 1000, 1100)]),
        )
    )
    from whistle_house.stages.base import StageContext

    def recognizer():
        return NoteTemplateRecognizer(NoteTemplateRecognizer.Config(), StageContext(SR, tunes=lib))

    # hummed three octaves lower: same tune (templates are key-invariant), reported as a hum
    hit = recognizer().process(_phrase(125.0, "hum"), now=2.0)
    assert [(m.tune_name, m.kind) for m in hit] == [("arp", "hum")]
    assert hit[0].to_json()["kind"] == "hum"
    # two hummed notes + a whistled third: neither kind has the whole tune
    mixed = _phrase(125.0, "hum")
    mixed[2] = _phrase(1000.0, "whistle")[2]
    assert recognizer().process(mixed, now=2.0) == []
    # a stray whistle event in the middle of a hummed tune does not break the hum match
    stray = [*_phrase(125.0, "hum"), _phrase(1000.0, "whistle")[1]]
    assert [m.kind for m in recognizer().process(stray, now=2.0)] == ["hum"]


def test_event_kind_round_trips_and_defaults_to_whistle():
    ev = _phrase(200.0, "hum")[0]
    assert WhistleEvent.from_json(ev.to_json()).kind == "hum"
    legacy = {k: v for k, v in ev.to_json().items() if k != "kind"}
    assert WhistleEvent.from_json(legacy).kind == "whistle"
