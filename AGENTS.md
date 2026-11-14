# AGENTS.md — whistle-house

Instructions for humans and coding agents working in this repo. Keep this file current: if you
change the layout, commands, or a rule below, update it in the same change.

## What this is

Detect whistled tunes from a room microphone (Raspberry Pi 5 + USB mic) and trigger home
automation (Home Assistant events, webhooks). This is v2, a ground-up rebuild of
`~/projects/whistled-tune-detection` (v1). The only thing carried over from v1 is the
STFT → mel → dB transform. v1's main failure was **talking triggered the lights**. v2's goal is a
lightweight **learned whistle detector** that rejects speech and noise, trained on data from
this mic with noise mixed in synthetically.

## Architecture

```
audio ─▶ Transform ─▶ [Filter, Filter, …] ─▶ WhistleDetector ─▶ TuneRecognizer ─▶ Outputs
        (FrameBlock)       (FrameBlock)       (WhistleEvents)     (TuneMatches)
```

* Every slot is **swappable**. Implementations are registered classes (`@register`) with a
  pydantic `Config`. A pipeline is a JSON spec naming one implementation per slot (plus a list of
  filters) with params: `data/pipelines/<name>.json`, `whistle_house.pipeline.PipelineSpec`.
* Types between stages are defined in `src/whistle_house/types.py`. They are the contract.
* **Streaming only.** Stages receive data incrementally and keep their own state. Offline analysis
  (`Pipeline.analyze`) and training-data generation feed whole signals through the *same*
  streaming code. Invariant: streaming in any chunk size == offline. Tests enforce this, so don't
  break it.
* The `Frontend` (transform + filters) is what a learned detector consumes. Checkpoints record
  the frontend spec they were trained on, and a pipeline using a checkpoint must use that
  frontend.

## Layout

```
AGENTS.md            this file (CLAUDE.md just imports it)
pyproject.toml       python package (uv). extras: ml (torch). dev group: pytest, ruff, librosa
mise.toml            node version for the manager (managed by mise)
src/whistle_house/
  types.py           FrequencyAxis, FrameBlock, WhistleEvent, DetectorOutput, TuneMatch
  pipeline.py        PipelineSpec / Frontend / Pipeline / PipelineStore
  settings.py        data-dir layout (WHISTLE_HOUSE_DATA, default ./data)
  tunes.py           Tune model, TuneLibrary (data/tunes.json), note-template fitting
  cli.py             `whistle-house` CLI: serve, devices, analyze, import, train-detector
  stages/
    base.py          stage ABCs + registry (register, build_stage, list_stages)
    transforms/stft.py   StreamingSTFT core + stft_mel (v1 port) + stft_linear
    filters/basic.py     identity, noise_floor, frame_median, band_reject
    detectors/common.py  ContourTracker (links per-frame pitch candidates into notes)
    detectors/peak_tracker.py  heuristic baseline (also auto-labels clean whistle pitch)
    detectors/learned.py       WhistleNet-backed detector (torch, lazy import)
    recognizers/note_template.py  key/tempo-invariant per-note template matching
  audio/             io.py (load/save/resample), sources.py (MicSource, ReplaySource, ArraySource)
  data/
    store.py         recordings: data/recordings/<id>.wav + <id>.json (label, tune, segments, split)
    augment.py       mix_at_snr, gain/EQ/reverb, synth_whistle, synth_pseudo_speech, NoiseBank
    training.py      ExampleGenerator: augmented (features, per-frame f0) examples via the Frontend
  ml/                model.py (WhistleNet, causal CNN), train.py (training loop, status json)
  outputs/           output types (log, webhook, home_assistant_event) + OutputStore (data/outputs.json)
  server/            app.py (FastAPI /api + /ws/live, serves manager/dist), engine.py (LiveEngine)
manager/             Vite + React + TS manager website
  src/api.ts         typed client — mirror of server/app.py routes
  src/pages/         Live, Data, Tunes, Pipelines, Models, Outputs
  src/components/    Spectrogram (canvas + SVG overlays), SchemaForm, AnalysisView, RecordPanel
tests/               pytest: transform parity (v1 + librosa), pipeline, tunes, server API
data/                runtime data, gitignored (recordings, models, tunes, outputs, pipelines)
```

## Commands

```bash
uv sync                      # python deps (+ dev group)
uv sync --extra ml           # + torch, for the learned detector / training
uv run pytest -q             # tests (must pass before committing)
uv run ruff format src tests && uv run ruff check src tests

uv run whistle-house serve                    # live engine + API + built manager on :8000
uv run whistle-house serve --replay some.wav  # no mic: loop a file in real time as the "mic"
uv run whistle-house serve --no-live          # no audio input at all
uv run whistle-house devices
uv run whistle-house analyze clip.wav --pipeline default
uv run whistle-house import *.wav --label speech       # bulk-add to the dataset
uv run whistle-house train-detector --name whistlenet --device auto

# manager (node via mise: `mise install` once; prefix with `mise exec --` if mise isn't activated)
cd manager && npm install
npm run dev      # vite dev server, proxies /api + /ws to :8000 (WHISTLE_HOUSE_API to override)
npm run build    # -> manager/dist, served by `whistle-house serve`
npm run lint
```

## Rules

1. **Keep the manager up to date.** The manager is the one place to see and control the system.
   Any new backend capability ships with its API route (`server/app.py`), its wrapper in
   `manager/src/api.ts`, and UI in a page. Run `npm run build && npm run lint` in `manager/`.
   New *stages* and *output types* need no frontend work: the manager renders their forms from
   the pydantic `Config` JSON schema. So give every field a default and a `description`.
2. **Adding a stage implementation:** subclass the base in `stages/base.py`, set `name`, define
   `Config(StageConfig)`, implement `process`/`push` (+ `reset`, + `flush` for detectors),
   decorate with `@register`, and import the module in the slot's `__init__.py`. Add a test
   (streaming == offline at minimum).
3. **Don't change the behaviour of an existing registered stage** in ways that change its
   output (trained checkpoints depend on the frontend). Add a new name or a new config field
   whose default keeps the old behaviour. `tests/test_transform.py` pins `stft_mel` to v1.
4. **Training runs on another machine** (clone the repo, copy `data/`). The Pi only runs
   inference. Don't spend effort making training fast on the Pi. Inference cost on the Pi does
   matter: a pipeline using the `learned` detector should use `chunk_size` ≈ 4096.
5. **Data is precious.** `data/` is gitignored and holds hand-collected recordings. Never delete
   or rewrite recordings or `data/` contents without asking. Recording splits are deterministic
   from the id (`split_for`). Don't reshuffle them.
6. **Mic contention on the Pi:** v1 runs as `whistle-hass.service` and holds the USB mic.
   Stop it (with the owner's OK) before running `serve` live, or use `--replay`.
7. Use mise for toolchains (node). Don't hand-install runtimes.
8. Keep code typed, small, and documented at module level. `ruff` must pass.

## Data & training strategy

* **Collect on the deployment mic** via the manager's Live page:
  `whistle` (clean, quiet room: becomes positives, with pitch auto-labelled by `peak_tracker`),
  `speech` (lots; the v1 failure mode), `noise`/`music` (TV, kitchen, doors…), `silence`.
  Tune examples are `whistle` clips with a `tune_id`.
* **Noise robustness is synthetic.** `data/training.py` mixes clean whistles with interferers at
  random SNR (−3…25 dB), plus random gain, EQ and reverb. Interferers come from negative
  recordings on this mic, files in `data/noise/**`, and `synth_pseudo_speech` (harmonic stacks
  with moving formants, the spectral shape that fooled v1).
* **External corpora worth dropping into `data/noise/`** (check licences): MUSAN (speech / music /
  noise), ESC-50 (environmental sounds), DEMAND (domestic noise), LibriSpeech or Common Voice
  (speech). Whistle-specific public data is scarce. The MLEnd "Hums and Whistles" dataset may
  be usable for extra positives. Synthetic whistles (`synth_whistle`) fill the gap meanwhile.
* Model: `WhistleNet`, a small causal dilated CNN (about 7k params) over frontend frames. It
  outputs per-frame presence (whistle or not) and a pitch-salience map. It is causal, so
  streaming == offline. Metrics written per epoch to `data/models/<name>.json`, notably
  `false_positive_rate` (the speech-rejection number to drive down).

## Roadmap (next steps)

* Segment-level labelling UI in the Data page (draw `segments` on the spectrogram).
* Evaluation page: run several pipelines over the test split and compare event precision/recall
  and false triggers per hour of speech.
* Hard-negative mining: save live clips that triggered events but no tune, for relabelling.
* Learned detector: try a lookahead (non-causal, delayed) variant. Quantise or export to ONNX
  for cheaper Pi inference.
* Systemd unit for v2 once it replaces v1.
