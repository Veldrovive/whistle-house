# whistle-house

Whistle a tune → your lights change. Speech shouldn't.

v2 of whistled-tune detection. The pipeline is modular: transform → filters → whistle detector →
tune recognizer → outputs, and every stage is swappable. A web manager lets you watch the live
spectrogram, collect and label training data from the room mic, define tunes and hook them to
outputs, compare pipelines on recordings, and train the learned whistle detector.

## Quick start

```bash
uv sync --extra ml                 # python deps incl. torch
mise install                       # node for the manager
(cd manager && npm install && npm run build)
uv run whistle-house serve         # http://<pi>:8000
```

No mic available (or v1 still holds it)? `uv run whistle-house serve --replay some.wav`.

Workflow:

1. **Live**: check the mic and spectrogram. Record clips labelled `whistle`, `speech`, `noise`, …
   Include 20-30 s of `silence` (nothing happening): it calibrates the noise gate.
2. **Tunes**: create a tune, record a few examples, press *Fit*, tick the outputs it should fire.
3. **Outputs**: add a Home Assistant event or webhook.
4. **Models**: train the learned detector on a GPU box, not the Pi: set `WHISTLE_HOUSE_EXTERNAL`
   in `.env`, copy `data/`, run `uv run whistle-house warmstart` (downloads public datasets,
   pretrains, fine-tunes on your mic data), copy the model back, then switch a pipeline's
   detector to `learned`. Details in AGENTS.md.
5. **Pipelines**: tweak stages and params, test on recordings, make one active. The mic is noisy:
   add a `spectral_gate` filter and press *Measure from silence* in its panel.

See [AGENTS.md](AGENTS.md) for architecture, layout, and contribution rules.
