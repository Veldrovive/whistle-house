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
  settings.py        .env loading; data dir (WHISTLE_HOUSE_DATA) + external root (WHISTLE_HOUSE_EXTERNAL)
  tunes.py           Tune model, TuneLibrary (data/tunes.json), note-template fitting
  noise_profiles.py  per-bin silence stats for spectral_gate (data/noise_profiles/<name>.json)
  cli.py             `whistle-house` CLI: serve, devices, analyze, import, datasets, train-detector, warmstart
  stages/
    base.py          stage ABCs + registry (register, build_stage, list_stages)
    transforms/stft.py   StreamingSTFT core + stft_mel (v1 port) + stft_linear
    filters/basic.py     identity, noise_floor, frame_median, band_reject
    filters/spectral_gate.py  gate against a measured noise profile (needs a silence recording)
    detectors/common.py  ContourTracker (links per-frame pitch candidates into notes)
    detectors/peak_tracker.py  heuristic baseline (also auto-labels clean whistle pitch)
    detectors/learned.py       WhistleNet-backed detector (torch, lazy import)
    recognizers/note_template.py  key/tempo-invariant per-note template matching
  audio/             io.py (load/save/resample), sources.py (MicSource, ReplaySource, ArraySource)
  data/
    store.py         recordings: data/recordings/<id>.wav + <id>.json (label, tune, segments, split)
    augment.py       mix_at_snr, gain/EQ/reverb, synth_whistle, synth_pseudo_speech
    training.py      collect_sources (+ LabelCache pitch auto-labels), ExampleGenerator, MIX_PRESETS
  external/          external datasets: base.py (download/prepare/manifest framework),
                     datasets.py (mlend_whistles, mlend_hums, musan, esc50, demand)
  ml/                model.py (WhistleNet, causal CNN), train.py (training loop, status json)
  outputs/           output types (log, webhook, home_assistant_event) + OutputStore (data/outputs.json)
  server/            app.py (FastAPI /api + /ws/live, serves manager/dist), engine.py (LiveEngine)
manager/             Vite + React + TS manager website
  src/api.ts         typed client — mirror of server/app.py routes
  src/pages/         Live, Data, Tunes, Pipelines, Models, Outputs
  src/components/    Spectrogram (canvas + SVG overlays), SchemaForm, AnalysisView, RecordPanel,
                     NoiseProfilePanel (extra controls under a spectral_gate filter)
tests/               pytest: transform parity (v1 + librosa), pipeline, tunes, noise gate, external data, server API
.env.example         settings template; copy to .env (gitignored)
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
uv run whistle-house datasets list             # external datasets + status
uv run whistle-house datasets fetch all        # training machine only (refuses on a Pi)
uv run whistle-house warmstart                 # fetch -> pretrain -> fine-tune (see below)
uv run whistle-house noise-profile create room --add   # profile from all 'silence' recordings,
                                               # + add a spectral_gate to the default pipeline

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
   inference and must never store external datasets (`datasets fetch` refuses on a Pi). Don't spend effort making training fast on the Pi. Inference cost on the Pi does
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
* **Mic self-noise: `spectral_gate`.** The USB mic is noisy (steady hiss, ~-35 dBFS). Record
  20-30 s of `silence`, measure a noise profile (manager: the panel under a `spectral_gate`
  filter; CLI: `noise-profile create`), and the gate keeps only bins > `n_std` noise std-devs
  above the noise mean. On the real mic it removed every false event on held-out silence
  (`noise_floor`: 30 in 28 s) and, unlike `noise_floor`, does not swallow long held notes.
  Profiles are measured at the filter's position (upstream chain is fingerprinted; a mismatch
  refuses to build) and are immutable: make a new one when the mic, gain or room changes.
* **Noise robustness is synthetic.** `data/training.py` mixes clean whistles with interferers at
  random SNR (−3…25 dB), plus random gain, EQ and reverb. Interferers come from negative
  recordings on this mic, files in `data/noise/**`, and `synth_pseudo_speech` (harmonic stacks
  with moving formants, the spectral shape that fooled v1).
* **External datasets** (`external/`, see the workflow below) give a warm start: MLEnd whistles
  (positives), MLEnd hums (hard negatives), MUSAN speech/music/noise, ESC-50, DEMAND. Any other
  audio dropped into `data/noise/` is used as extra interferers. Synthetic whistles fill gaps.
* **Auto-labelling** uses stricter `peak_tracker` params than live detection
  (`GeneratorConfig.labeler_params`), and external whistle clips with <20% labelled frames are
  dropped. Labels are cached per front-end in `data/cache/labels/` and `<external>/labels/`.
* **Mix presets** (`MIX_PRESETS`): `mic` (mic + synthetic), `pretrain` (external +
  synthetic), `finetune` (mic-heavy + some external, use with `--init-from`).
* Model: `WhistleNet`, a small causal dilated CNN (about 7k params) over frontend frames. It
  outputs per-frame presence (whistle or not) and a pitch-salience map. It is causal, so
  streaming == offline. Metrics written per epoch to `data/models/<name>.json`, notably
  `false_positive_rate` (the speech-rejection number to drive down).

## Training-machine workflow (warm start)

```bash
git clone <repo> && cd whistle-house && uv sync --extra ml
cp .env.example .env            # set WHISTLE_HOUSE_EXTERNAL=/big/disk/whistle-datasets
rsync -a pi:projects/whistle-house/data/ ./data/   # mic recordings, pipelines, tunes
uv run whistle-house datasets fetch all --clean-raw   # ~17 GB MLEnd + 11 GB MUSAN + ~1.3 GB
uv run whistle-house warmstart --skip-fetch --workers 8
#   = train-detector --mix pretrain --external all --name whistlenet-pretrain
#   + train-detector --mix finetune --external all --init-from whistlenet-pretrain --name whistlenet
#     (fine-tune is skipped if data/ has no mic whistle recordings yet)
rsync -a data/models/whistlenet.* pi:projects/whistle-house/data/models/
```

Then on the Pi make a pipeline with the `learned` detector (`checkpoint: whistlenet.pt`,
`chunk_size` 4096, same transform/filters as the training pipeline). Re-run only the fine-tune
step as more mic data is collected. To add a dataset: subclass `ExternalDataset` in
`external/datasets.py` (`download`, `iter_sources`) and decorate with `@register_dataset`.

## Roadmap (next steps)

* Segment-level labelling UI in the Data page (draw `segments` on the spectrogram).
* Evaluation page: run several pipelines over the test split and compare event precision/recall
  and false triggers per hour of speech.
* Hard-negative mining: save live clips that triggered events but no tune, for relabelling.
* Learned detector: try a lookahead (non-causal, delayed) variant. Quantise or export to ONNX
  for cheaper Pi inference.
* Systemd unit for v2 once it replaces v1.
