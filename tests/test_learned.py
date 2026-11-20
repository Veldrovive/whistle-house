"""Learned detector: lookahead models are delayed when streaming but give the offline result."""

import numpy as np
import pytest

from whistle_house.data.augment import synth_whistle
from whistle_house.pipeline import Pipeline, PipelineSpec

torch = pytest.importorskip("torch")


def _checkpoint(tmp_path, **arch):
    from whistle_house.ml.model import WhistleNet, WhistleNetConfig, save_checkpoint

    torch.manual_seed(0)
    model = WhistleNet(WhistleNetConfig(n_bins=128, **arch))
    # An untrained net; bias the presence head so it reports notes and the tracker has work.
    torch.nn.init.constant_(model.presence.bias, 0.5)
    model.eval()
    spec = PipelineSpec(detector={"name": "learned", "params": {"checkpoint": "m.pt"}})
    save_checkpoint(tmp_path / "models" / "m.pt", model, (-40.0, 20.0), spec.frontend_dict(), {})
    return model, spec


def test_lookahead_forward_is_the_causal_pass_shifted():
    from whistle_house.ml.model import ARCHS, WhistleNet, WhistleNetConfig

    model = WhistleNet(WhistleNetConfig(n_bins=32, **ARCHS["large"])).eval()
    la = model.cfg.lookahead
    assert la > 0 and sum(p.numel() for p in model.parameters()) > 100_000
    x = torch.randn(2, 40, 32)
    with torch.no_grad():
        pres, sal = model(x)
        cpres, csal = model.forward_causal(torch.nn.functional.pad(x, (0, 0, 0, la)))
        # causal: the output for frame t does not change when frames after t + lookahead do
        x2 = x.clone()
        x2[:, 30:] = 0.0
        pres2, _ = model(x2)
    assert pres.shape == (2, 40, 1) and sal.shape == (2, 40, 1, 32)
    assert torch.allclose(pres, cpres[:, la:]) and torch.allclose(sal, csal[:, la:])
    assert torch.allclose(pres[:, : 30 - la], pres2[:, : 30 - la], atol=1e-5)
    assert not torch.allclose(pres[:, 30 - la : 30], pres2[:, 30 - la : 30], atol=1e-5)


@pytest.mark.parametrize("lookahead", [0, 5])
def test_learned_streaming_matches_offline(tmp_path, lookahead):
    _, spec = _checkpoint(tmp_path, channels=4, dilations=(1, 2), lookahead=lookahead)
    audio = synth_whistle(44100, 2.0, np.random.default_rng(0), n_notes=4).audio.astype(np.float32)

    def run(chunk):
        p = Pipeline(spec.model_copy(update={"chunk_size": chunk}), data_dir=tmp_path)
        r = p.analyze(audio)
        offline_pres, _ = p.detector.infer(r.filtered.values)
        return r, offline_pres[:, 0]  # the one class of this checkpoint: whistle

    (a, offline), (b, _) = run(4096), run(700)
    key = [(round(e.start, 4), round(e.end, 4), round(e.median_hz(), 1)) for e in a.events]
    assert key and key == [
        (round(e.start, 4), round(e.end, 4), round(e.median_hz(), 1)) for e in b.events
    ]
    # Streamed scores are the offline (input-aligned) scores, `lookahead` frames late.
    n = len(offline) - lookahead
    np.testing.assert_allclose(a.scores[lookahead:], offline[:n], atol=1e-4)
    np.testing.assert_allclose(b.scores[lookahead:], offline[:n], atol=1e-4)
    # ...and with lookahead the last frames still produce events: flush pads the future.
    assert max(e.end for e in a.events) > a.filtered.times[-1 - lookahead] - 0.3
