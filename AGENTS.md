# AGENTS.md — whistle-house

Instructions for humans and coding agents working in this repo. Keep this file current: if you
change the layout, commands, or a rule below, update it in the same change.

## What this is

Detect whistled (and hummed) tunes from a room microphone (Raspberry Pi 5 + USB mic) and trigger
home automation (Home Assistant events, webhooks). This is v2, a ground-up rebuild of
`~/projects/whistled-tune-detection` (v1). The only thing carried over from v1 is the
STFT → mel → dB transform. v1's main failure was **talking triggered the lights**. v2's goal is a
lightweight **learned detector** for whistles and hums that rejects speech and noise, trained
on data from this mic with noise mixed in synthetically.

## Architecture

```
audio ─▶ Transform ─▶ [Filter, Filter, …] ─▶ WhistleDetector ─▶ TuneRecognizer ─▶ Outputs
        (FrameBlock)       (FrameBlock)       (WhistleEvents)     (TuneMatches)
```

* Every slot is **swappable**. Implementations are registered classes (`@register`) with a
  pydantic `Config`. A pipeline is a JSON spec naming one implementation per slot (plus a list of
  filters) with params: `data/pipelines/<name>.json`, `whistle_house.pipeline.PipelineSpec`.
* Types between stages are defined in `src/whistle_house/types.py`. They are the contract.
  A `WhistleEvent` is one note with a `kind` (`whistle` or `hum`; for a hum the contour is the
  fundamental). The recognizer matches a tune within one kind and reports it (`TuneMatch.kind`,
  also sent to Home Assistant), so a tune fires whether it is whistled or hummed.
* **An event is not a note.** The tracker ends an event when the pitch jumps, not when the
  sound stops: a slide through several notes is one event, a leap splits one. Recognizers
  that need notes must find them in the contours (`score_align` does; `note_template`
  assumes one event per note and fails on legato). A recognizer also gets the notes still
  sounding (`process(events, now, active)`) and a `flush(now)` at the end of a stream.
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
  tunes.py           Tune model (Score = sheet music, or a template fitted from examples),
                     TuneLibrary (data/tunes.json), note-template fitting
  noise_profiles.py  per-bin silence stats for spectral_gate (data/noise_profiles/<name>.json)
  cli.py             `whistle-house` CLI: serve, devices, analyze, import, datasets, train-detector, warmstart
  stages/
    base.py          stage ABCs + registry (register, build_stage, list_stages)
    transforms/stft.py   StreamingSTFT core + stft_mel (v1 port) + stft_linear + stft_log
                         (60 Hz-4 kHz, 24 bins/octave, n_fft 4096: covers hums and whistles)
    filters/basic.py     identity, noise_floor, frame_median, band_reject
    filters/spectral_gate.py  gate against a measured noise profile (needs a silence recording)
    detectors/common.py  ContourTracker (links per-frame pitch candidates into notes),
                         drop_harmonics (which peaks of a frame are overtones)
    detectors/peak_tracker.py  heuristic baseline (also auto-labels clean whistle pitch)
    detectors/learned.py       WhistleNet-backed detector (torch, lazy import): one tracker per
                               kind the checkpoint detects (whistle, hum), true streaming
    recognizers/score_align.py    tunes written as sheet music: aligns the score to the pitch
                                  curve, free in key and tempo (see "Tunes" below)
    recognizers/note_template.py  per-note template fitted from examples (one event per note)
  audio/             io.py (load/save/resample), sources.py (MicSource, ReplaySource, ArraySource),
                     render.py (filtered frames -> audio: the "Processed" playback in the manager)
  data/
    store.py         recordings: data/recordings/<id>.wav + <id>.json (label, tune, segments, split)
    augment.py       mix_at_snr, gain/EQ/reverb/saturate, synth_whistle, synth_hum, synth_pseudo_speech,
                     labelled-clip augmentation (splice_notes, warp_clip, reverse/envelope)
    perform.py       simulated performances of a Score (key, tempo, slides, sloppiness) as a
                     pitch curve -> events through the real ContourTracker; for recognizer tests
    training.py      collect_sources (+ LabelCache pitch auto-labels), ExampleGenerator (one f0
                     label column per class), MIX_PRESETS / HUM_MIX_PRESETS
    hum_labels.py    pYIN hum pitch labels (offline, librosa), harmonic-level profile
  external/          external datasets: base.py (download/prepare/manifest framework),
                     datasets.py (mlend_whistles, mlend_hums, musan, esc50, demand)
  ml/                model.py (WhistleNet, causal CNN; HarmonicStack input; ARCHS small/large/harmonic),
                     train.py (training loop, per-class metrics, status json)
  outputs/           output types (log, webhook, home_assistant_event) + OutputStore (data/outputs.json)
  server/            app.py (FastAPI /api + /ws/live, serves manager/dist), engine.py (LiveEngine)
manager/             Vite + React + TS manager website
  src/api.ts         typed client — mirror of server/app.py routes
  src/pages/         Live, Data, Tunes, Pipelines, Models, Outputs
  src/lib/score.ts   pitch spelling, note values, bar lines, WebAudio playback of a Score
  src/components/    ScoreEditor (sheet-music editor, VexFlow; lazy-loaded, ~900 kB with its
                     music font), Spectrogram (canvas + SVG overlays), SchemaForm, AnalysisView
                     (spectrograms + Original/Processed playback), RecordPanel,
                     NoiseProfilePanel (extra controls under a spectral_gate filter)
tests/               pytest: transform parity (v1 + librosa), pipeline, tunes, noise gate, external data,
                     augmentation labels, server API, harmonic model (test_harmonic.py),
                     overtone rejection (test_overtones.py), written tunes against simulated
                     performances (test_score_align.py)
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
uv run whistle-house train-detector --pipeline hum-log --arch harmonic --name humnet  # + hums
uv run whistle-house datasets list             # external datasets + status
uv run whistle-house datasets fetch all        # training machine only (refuses on a Pi)
uv run whistle-house warmstart                 # fetch -> pretrain -> fine-tune (see below)
uv run whistle-house noise-profile create room --add   # profile from all 'silence' recordings,
                                               # + add a spectral_gate to the default pipeline
uv run whistle-house hum-labels                # pYIN pitch + harmonic profile of every 'hum' clip

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
   (streaming == offline at minimum). Filters whose output is "dB above a floor" (0 = floor)
   set `relative_output = True` so Processed playback (`audio/render.py`) maps them correctly.
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
9. **Hum pipelines:** the `harmonic` model needs a log-frequency front end (`stft_log`; training
   refuses anything else) and must not sit behind `noise_floor`, which adapts to held notes and
   swallows a steady hum within a second. Use `spectral_gate` (or no filter) instead.

## Tunes (sheet music, `score_align`)

* A tune is written in the manager's Tunes page (`ScoreEditor`): notes and rests with lengths
  in beats, a base tempo, playback. Stored as `Tune.score` (`pitch` = MIDI number or null for
  a rest, `beats`, quarter = 1; `flat` and `beats_per_bar` are display only). A pipeline
  matches written tunes only if its recognizer is `score_align` (the Tunes page warns and
  offers to switch the active pipeline); `note_template` only matches fitted templates.
* **Matching is deliberately loose.** Only intervals and relative note lengths count: any key
  (continuous, not snapped to semitones), 0.5x-2x the written tempo, whistled or hummed.
  How it works (module docstring of `score_align.py` has the detail): the recent events of a
  kind are one pitch curve; it is cut into steps of roughly constant pitch, long steps into
  short pieces; dynamic programming assigns each written note a run of pieces; cost per note
  = time-weighted pitch error (flat beyond 2 tolerances, and at least half of a note must be
  on pitch) + error of its onset-to-onset time against the tempo + a loose check of the
  silent share against the written rest. Key candidates come from votes of every
  (piece, note) pair, tempo candidates from the phrase length; both are refitted from the
  best alignment. Score = `exp(-mean cost per note)`.
* **A tune must stand alone** (default): quiet before the first note (`quiet_before_s`) and
  after the last (`quiet_after_s`, then `settle_s` for the detector's latency), because short
  tunes are often inside longer ones. So a match fires ~0.75 s after the last note. Set
  either to 0 to drop that side.
* Measured on the real mic clips: event boundaries are almost all pitch jumps (median gap
  between events 0.01 s), 14 of 75 whistle and 9 of 15 hum events hold more than one note, and
  held pitches are not on a semitone grid. That is why nothing is quantised to semitones and
  why a semitone step between two notes cannot be told from a slightly mistuned repeat.
* Simulation (`data/perform.py`, `tests/test_score_align.py`), 5 tunes of 4-8 notes: of 300
  moderately sloppy performances (0.3 st pitch error, 12% timing error, legato, slides,
  scoops) none was missed outright and the 5th percentile of scores was 0.56-0.64, against a
  threshold of 0.5; 1 of 400 random 3-9 note melodies scored above it. A tune with
  few distinct pitches (repeats of one note) is the weakest kind. ~2-9 ms per finished note
  for 3 tunes on a Mac; not yet timed on the Pi or tested on real performances of a written
  tune.
* **Missing notes are not allowed yet.** The place for it is marked in `align`: one more
  move that stays at the same boundary and pays a penalty, with a count of misses in the
  state.
* Repeated notes: a gap (even two frames the tracker bridged) marks a new note; without one
  the repeat costs `slur_penalty` but still matches by timing.

## Data & training strategy

* **Collect on the deployment mic** via the manager's Live page:
  `whistle` (clean, quiet room: becomes positives, with pitch auto-labelled by `peak_tracker`),
  `speech` (lots; the v1 failure mode), `noise`/`music` (TV, kitchen, doors…), `silence`,
  `hum` (clean humming: positives for the `harmonic` model, pitch labelled by pYIN; a
  whistle-only model uses them as hard negatives alongside the external hums).
* **Hums (first 6 clips, MacBook mic, one low voice):** f0 82-190 Hz. The fundamental is the
  strongest harmonic in most frames, h2 is 1-9 dB below, h3 -10…-16 dB, h4+ ≤ -25 dB; 75-90%
  of the energy is below 300 Hz, so `stft_mel`'s default range drops what carries the pitch
  (use `stft_log`). Pitch labels come from pYIN (`data/hum_labels.py`: 96-100% of frames
  voiced, cached in `data/cache/labels/hum-*.npz`), not `peak_tracker`. The manager overlays
  them (dashed green) on `hum` recordings. Re-check on the Pi's USB mic, which may roll off
  the fundamental. The speech clips (same voice) have nearly the same per-frame harmonic
  profile as the hums (h2 -11...-16, h3 -12...-16, h4 -19...-26 dB): what separates them is how the comb moves
  over time, which is why the hum model has a longer time receptive field.
* **Hum training data** (only when `hum` is one of the model's classes, i.e. `--arch harmonic`
  or `--classes whistle,hum`): mic `hum` clips and external `hum` clips (MLEnd hums, 4804
  clips / 21.6 h; pYIN takes ~37 core-minutes once, cached in `<external>/labels/*-hum-*.npz`)
  are positives; `synth_hum` fills gaps (source-filter harmonic stack, sometimes high-passed
  like a small mic; the label stays the fundamental). `HUM_MIX_PRESETS` splits each mix
  between whistle, hum and nothing. Hum clips are then never used as interferers. Mic hum
  clips get the same training-only augmentation as mic whistles.
  Clips recorded on a tune's page are `whistle` clips with a `tune_id` (try-outs of a written
  tune, or examples for a fitted template).
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
  dropped. Labels are cached per front-end in `data/cache/labels/` and `<external>/labels/`
  (the cache key includes the labeller params, so changing them relabels).
* **Overtones are not notes.** `peak_tracker` turns every prominent peak into a contour, so a
  whistle with visible harmonics also gave a contour at 2x/3x its pitch: 44 of 151 events on
  the mic whistle clips with live params. Its `reject_harmonics` option drops a peak when a
  peak at least as loud sits at 1/2...1/5 of its frequency (0 of 103 with it on). It is off
  by default (existing pipelines keep their behaviour; tick it in the Pipelines page) and on
  for labelling. Whistle labels were only lightly affected (2-3% of labelled time on real
  clips, 22-31% on synthetic whistles with strong overtones, none with the check), and where
  label events overlap the more confident one now wins. pYIN was tried as the whistle
  labeller instead and is not better: it agrees with the checked labels except in glides and
  in ~1% of MLEnd frames where it drops an octave below a tone that is 20 dB louder.
  Training also makes harmonic-rich positives whose label stays the fundamental:
  `synth_whistle(overtones=True)` (`p_whistle_overtones`) and `saturate` (soft clipping, as an
  overdriven mic does; `p_saturate`). `overtone_rate` in the training metrics is the share of
  detected frames whose pitch is the 2nd-4th harmonic of the label (drive it to 0).
* **Mix presets** (`MIX_PRESETS`): `mic` (mic + synthetic), `pretrain` (external +
  synthetic), `finetune` (mostly external + 15% mic whistles, use with `--init-from`).
* **There will never be many mic clips, so fine-tuning leans on generalisation** and is
  watched for overfitting rather than trusted. Validation runs once per epoch, so fine-tunes
  use short epochs (warmstart: 50 steps) with `--patience`. Per validation, `ml/train.py`
  records: `f1` etc. on the val splits (speaker-disjoint external + held-out mic + synthetic;
  this picks the checkpoint), `mic_train_*` (fresh augmentations of the mic clips being
  trained on), `mic_val_*` (held-out mic clips, if any) and `overfit_gap` = `mic_train_f1` −
  `f1`. Overfitting = `mic_train_f1` still rising while `f1` stalls or falls. A fine-tune
  first validates its init checkpoint as epoch 0 and keeps it unless something beats it on
  val, so the result is never worse on val than the pretrained model. The manager's Models
  page plots these curves. Until some mic whistle clips land in the `val` split (~1 in 10 do,
  by id), val cannot show whether fine-tuning helps *on this mic*, only that it did no harm.
* **Mic-clip augmentation (training only).** In the `finetune` mix, mic whistle clips are
  varied before the usual mixing (`GeneratorConfig.p_mic_*`, primitives in `data/augment.py`):
  `splice_notes` (re-order the take's notes into a new melody; needs gaps between notes),
  `warp_clip` (speed change = pitch + tempo together, ±4 semitones, kept on the frequency
  axis, with slow drift), `reverse_clip`, `envelope_clip` (loudness drift). Each returns
  matching f0 labels. `GeneratorConfig.eval_config()` switches them off and every validation
  set uses it: validation must look like real whistling. Add new ones the same way (a
  function returning `(audio, f0)`, a `p_mic_*` field defaulting to 0, zeroed in
  `eval_config`, a label-follows-audio test in `tests/test_augment.py`).
* The `mic` interferer weight is scaled down when there are fewer than
  `mic_full_weight_clips` (10) mic negative clips, so one silence take doesn't crowd out speech.
* Model: `WhistleNet`, a small causal dilated CNN (about 7k params) over frontend frames. For
  each class it detects (`whistle`; `hum` too for the `harmonic` arch) it outputs per-frame
  presence and a pitch-salience map. It is causal, so streaming == offline, and the `learned`
  detector streams it properly: `forward_stream` keeps each layer's last few input frames, so
  a frame is computed once however long the receptive field is. Metrics written per epoch to
  `data/models/<name>.json`, notably `false_positive_rate` (the speech-rejection number to
  drive down). Multi-class models also write `<class>_<metric>`; the plain names are then the
  mean over classes (that mean F1 picks the checkpoint).
* **Harmonic input (`harmonic` arch, whistles + hums).** A hum's harmonics are equally spaced
  in Hz, but the spacing is the pitch, so no fixed kernel fits every note; on a log axis the
  pattern is the same at every pitch but unevenly spaced (+24, +38.04, +48, +55.7 bins), out
  of reach of small kernels. `HarmonicStack` (first layer, no parameters) gives each pitch
  candidate its own linear frequency axis in units of that pitch: input channel `j` at bin `b`
  is the spectrum at `HARMONIC_GRID[j] * f_b` (a constant bin shift on the log axis). The grid
  is 1/3, 1/2, then 1, 1.5, 2, ... 8. At the true pitch the integer channels hold the
  harmonics and the half-integer ones the gaps (on the mic hums 35-50 dB below the
  fundamental, and still ~10 dB below their neighbours at h7-h8 where those rise above the noise);
  at 2 f0 / 3 f0 the sub-harmonic channels are filled; at f0/2 only even channels are. A
  whistle is channel 1 alone. So every bin sees the whole comb (f/3 to 8f) through a plain
  3x3 convolution, and `freq_context` (each block pools its features over all bins and feeds
  that back) covers the rest of the axis. ~31k params, 0.73 s time receptive field, 12 frames
  (140 ms) of lookahead. `tests/test_harmonic.py` pins the alignment on the real front end.
* **Model sizes** (`ml/model.ARCHS`, `--arch`, the "model" select on the Models page): `small`
  is the whistle-only Pi model above; `harmonic` is the whistle + hum model (cheap enough for
  the Pi: about the cost per chunk that `small` had before streaming was made incremental). `large` (about 190k params, frequency dilation, 16 frames of
  lookahead) is a reference model that shows how well detection can be done with this data;
  it is not meant to be cheap on the Pi. A lookahead model is delayed, not non-causal: the
  `learned` detector reports its events and scores `lookahead` frames late and pads the future
  on flush, so streaming == offline still holds. Train `large` with `--batch-size 16` on a
  16 GB Mac (32 runs out of GPU memory and swaps). Fine-tuning keeps the checkpoint's
  architecture.

## Training-machine workflow (warm start)

```bash
git clone <repo> && cd whistle-house && uv sync --extra ml
cp .env.example .env            # set WHISTLE_HOUSE_EXTERNAL=/big/disk/whistle-datasets
rsync -a pi:projects/whistle-house/data/ ./data/   # mic recordings, pipelines, tunes
uv run whistle-house datasets fetch all --clean-raw   # ~17 GB MLEnd + 11 GB MUSAN + ~1.3 GB
uv run whistle-house warmstart --skip-fetch --workers 8
#   = train-detector --mix pretrain --external all --name whistlenet-pretrain
#   + train-detector --mix finetune --external all --init-from whistlenet-pretrain --name whistlenet
#       --steps-per-epoch 50 --epochs 60 --patience 15   (short epochs = frequent validation)
#     (fine-tune is skipped if data/ has no mic whistle recordings yet)
rsync -a data/models/whistlenet.* pi:projects/whistle-house/data/models/
```

For the whistle + hum model run the same recipe on a log-frequency pipeline:
`uv run whistle-house warmstart --skip-fetch --workers 8 --pipeline hum-log --arch harmonic
--name humnet` (the first run also relabels the external whistles for that front end and
pYIN-labels the external hums).

Then on the Pi make a pipeline with the `learned` detector (`checkpoint: whistlenet.pt`,
`chunk_size` 4096, same transform/filters as the training pipeline). Its `kinds` param limits
what a pipeline reports (e.g. `["whistle"]` to ignore hums). Re-run only the fine-tune
step as more mic data is collected. To add a dataset: subclass `ExternalDataset` in
`external/datasets.py` (`download`, `iter_sources`) and decorate with `@register_dataset`.

## Roadmap (next steps)

* Hums: the `harmonic` model is built but has only been smoke-tested (up to 500 steps on mic +
  synthetic data: hums, whistles, noise and silence came out right, `overtone_rate` reached 0,
  speech still raised some false hum notes). Next: the real warm start (external data), then judge it on
  `hum_false_positive_rate` against speech. Open questions: does singing count as a hum
  (today it is neither class; MUSAN music is an interferer); sustained instrument notes are
  harmonic combs too; re-record hums on the Pi's USB mic.

* Written tunes: evaluate `score_align` on real performances (MLEnd has 8 known songs with
  the song in the manifest: write one's opening as a tune, measure recall on it and false
  triggers on the other seven); show why a try-out did not match (best alignment and score
  below threshold) in the manager; allow a missing note; time it on the Pi.
* Segment-level labelling UI in the Data page (draw `segments` on the spectrogram).
* Evaluation page: run several pipelines over the test split and compare event precision/recall
  and false triggers per hour of speech.
* Hard-negative mining: save live clips that triggered events but no tune, for relabelling.
* Learned detector: quantise or export to ONNX for cheaper Pi inference.
* Systemd unit for v2 once it replaces v1.
