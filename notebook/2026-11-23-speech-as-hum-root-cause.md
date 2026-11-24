# 2026-11-23 — why speech fires as hum on the Fifine, and the `room` training recipe

The live system on dirk (pipeline `humnet-long-e240-gate`) fired tunes during an ordinary
conversation: 29 Home Assistant events between 17:10 and 17:51, nearly all `hum` matches. This
is the investigation and the retraining that followed. All computation on lgn9.

## Setup

- dirk's 35 Fifine recordings copied next to the 41 Mac-mic ones in lgn9's `data/recordings`
  (additive). Speech on the Fifine: 5 clips, 110 s (3 of them recorded right after the
  conversation).
- Harness: `scratch/svh/` on lgn9 (not in git). Every clip goes through the real streaming
  code in 4096-sample chunks (`Frontend.push` -> `LearnedDetector.process` -> `score_align`
  with dirk's four tunes). Reported: share of frames with hum presence >= 0.5, the same among
  pYIN-voiced frames, hum events per minute, tune matches per minute.
- External speech: MUSAN speech, val split (22 files, 3.35 h), random 30 s segments.

## 1. Is the confusion visible in the training data? No.

Share of frames called hum (events per minute in brackets):

| speech                                  | minutes | `humnet-long-pretrain-e240` (live) | `humnet-long-pretrain` (60 ep) | `humnet-deep` | `humnet` |
|-----------------------------------------|---------|------------------------------------|--------------------------------|---------------|----------|
| MUSAN val (never trained on)            | 75 / 30 | 0.40% (0.9)                        | 0.33%                          | 0.9%          | 1.9%     |
| MUSAN train                             | 75      | 0.15% (0.4)                        |                                |               |          |
| owner, Mac mic, train split (17 clips)  | 7.5     | 0.07% (0.1)                        | 0.10%                          | 0.5%          | 1.7%     |
| owner, Mac mic, held out (2 clips, 16 s)| 0.3     | 0.08%                              | 4.9%                           | 8.5%          | 12.5%    |
| **owner, Fifine (5 clips)**             | 1.85    | **10.1% (23.9)**                   | **21.6%**                      | **19.0%**     | **22.3%**|
| owner, Fifine, behind `adaptive_gate`   | 1.85    | 19.9% (47.7)                       |                                |               |          |

25-65 times the rate on the pretraining speech. The validation `false_positive_rate` (2.1%)
could not show it: it is over every non-target frame of mixed examples, and speech there is
MUSAN speech. Note the Mac speech clips are training interferers for every model (the
`pretrain` mix uses the `mic` interferer group), so only the two held-out ones say anything.

Per clip (live model): 1.1%, 4.0%, 9.6%, 12.1%, 16.3%. The 1.1% clip is the one from
2026-11-22 that AGENTS.md's 1.4% came from; the four later ones are 4-16%.

Hums and whistles are fine on every model (Fifine hums 91% of voiced frames, whistles 58% of
frames; nothing on silence or noise clips).

## 2. What makes it fire

Waveform manipulations of the same speech, live model, share of frames called hum:

| manipulation                                              | MUSAN val | Mac speech | Fifine speech |
|-----------------------------------------------------------|-----------|------------|---------------|
| none                                                      | 0.5-0.7%  | 0.07%      | 10.1%         |
| level to -45 / -55 dBFS                                   | 0.2 / 0.1 | 0.08 / 0.04|               |
| + Fifine room noise (quiet / fridge / fan take)           | 0.1-0.3   | 0.08-0.13  |               |
| gain -12 / +12 / +24 dB                                   |           | 0.01 / 0.3 / 1.1 | 2.9 / 17.4 / 23.4 |
| white noise at 20 / 10 dB SNR                             | 0.8 / 2.0 | 0.5 / 1.9  |               |
| synthetic reverb, RT60 0.3 s DRR +6 / 0.5 s 0 / 0.7 s -6  | 1.6 / 5.6 / 9.9 | 0.4 / 2.6 / 6.8 | - / 35 / - |
| tilt -3 / -6 dB per octave                                | 1.5 / 3.0 | 0.8 / 2.4  | 29.8 / -      |
| tilt +3 / +6 dB per octave                                | 0.1 / -   |            | 0.6 / 0.0     |
| **Fifine response** (below)                               | **8.4-13.0** | **36.0** | -             |
| **inverse Fifine response**                               |           |            | **0.0**       |
| Fifine response + reverb 0.5 s / DRR 0                    | 32.4      | 49.9       |               |

"Fifine response" = long-term spectrum of the owner's speech on the Fifine minus the same on
the Mac mic (or minus MUSAN's), smoothed to 1/3 octave, relative to 90-130 Hz, applied as a
linear-phase filter with the loudness kept. Fifine minus Mac:

| Hz | 50 | 71 | 100 | 141 | 200 | 283 | 400 | 566 | 800 | 1131 | 1600 | 2263 | 3200 |
|----|----|----|-----|-----|-----|-----|-----|-----|-----|------|------|------|------|
| dB | -13 | -14 | -2 | +3 | -7 | -13 | -18 | -18 | -17 | -16 | -14 | -14 | -14 |

So, relative to the fundamental of a low voice, everything from ~280 Hz to 3 kHz arrives
13-18 dB weaker on the Fifine in that room than on the laptop mic (20-30 dB weaker than
audiobook speech). Whether that is the mic, the room (a mode near 100-150 Hz) or the distance
is not separated; a clip spoken 10 cm from the Fifine would tell.

What the model sees at the true pitch (front-end dB at h x f0, median over voiced frames,
relative to h1):

| frames                          | h2    | h3    | h4    | h5    | h8    |
|---------------------------------|-------|-------|-------|-------|-------|
| MUSAN speech                    | +2.3  | -2.1  | -8.0  | -13.8 | -20.4 |
| owner speech, Mac mic           | -1.4  | -6.5  | -14.0 | -19.9 | -27.1 |
| **owner speech, Fifine**        | -8.8  | -14.1 | -24.9 | -27.3 | -38.0 |
| ... of those, called hum        | -15.0 | -20.6 | -30.5 | -36.5 | -46.7 |
| **MLEnd hums** (training hums)  | -8.3  | -14.9 | -22.6 | -26.7 | -30.5 |
| owner hums, Mac mic             | -7.2  | -13.4 | -21.8 | -27.7 | -35.7 |
| owner hums, Fifine              | -17.2 | -35.3 | -44.5 | -47.8 | -50.1 |

Speech on the Fifine has the harmonic roll-off of a training-set hum. **The models tell a hum
from speech mostly by how much energy sits above the fundamental** (a closed mouth has no
formants), and this channel moves that by more than the gap between the classes. Same voice,
same words: 0.07% -> 36% with the response applied, 10% -> 0% with it removed.

Also:

- False hums rise with the voice's pitch: 2% of voiced frames at 60-100 Hz, 14% at 100-130,
  36% at 130-170, 50% at 170-230 Hz (MLEnd hums are mostly above 130 Hz).
- Louder = more hum (Fifine speech +12 dB: 17%, -12 dB: 2.9%). In the v1 recipe a lone
  interferer was drawn 0-30 dB quieter than an example with a hum in it.
- Reverb alone does it too (MUSAN 0.7% -> 5.6-9.9%). In v1, reverb was only ever applied to
  the whistle / hum, never to the interferer, and `random_reverb` has almost no direct sound.
  Whether the Fifine clips are more reverberant than the Mac ones could not be shown (the
  envelope decay after speech offsets is the same, ~150-160 dB/s, on both).
- Not the cause: absolute level, room noise, the room's steady tones.

## 3. Model, data, or something else?

- **Data (training recipe).** Every architecture takes the same shortcut: `harmonic` (0.73 s
  of context), `harmonic_deep` (1.45 s) and `harmonic_long` (2.94 s) are all at 10-22% on the
  Fifine speech and 13-32% on MUSAN with the Fifine response. More context and longer training
  help inside the training distribution (MUSAN raw 1.9% -> 0.9% -> 0.3-0.5%) and somewhat on
  the Fifine (60 vs 240 epochs: 21.6% vs 10.1%), but do not remove it.
- **Something else, 1: the gate.** The live pipeline put an `adaptive_gate` in front of a
  model trained ungated: 10.1% -> 19.9% of frames, 24 -> 48 events per minute.
- **Something else, 2: the recognizer.** Live settings were `threshold` 0.1 with
  `off_pitch_cost` 0.75 and `median_weight` 1. On the 1.85 min of Fifine speech: 2.2 matches
  per minute gated (0.5 ungated) at 0.1, none at 0.5, none with default settings. But of the
  28 live hum matches 40% scored above 0.5, so the threshold alone is not a fix.

## 4. The `room` recipe (`--augment room`)

`data/augment.py` + `GeneratorConfig` (all off by default: `--augment v1` generates exactly
the old examples, checked bit for bit against the previous commit):

- `random_channel`: a random smooth frequency response (tilt up to 4 dB/octave, one or two
  shelves up to 20 dB with the corner at 80 Hz-1.5 kHz, broad bumps), drawn independently for
  the clean signal and each interferer (p 0.8 each), and for the whole mix (p 0.4).
- `room_reverb` (direct-to-reverb ratio -6...18 dB, RT60 0.15-0.9 s) on the clean signal and
  on interferers alike (p 0.5 each).
- a second, quieter interferer under the first (p 0.4): speech in a noisy room.
- one level distribution for every example (loud parts at -65...-10 dBFS) whatever it holds.
- speech weight among interferers 3 -> 6.

`--augment channel` is the ablation with only the per-source response.

Runs (lgn9, `harmonic_long`, same settings as `humnet-long-pretrain`: 60 x 500 steps, batch
32, on the Mac-only recordings via `--data data-mac`, so **no Fifine clip is seen**):

```bash
CUDA_VISIBLE_DEVICES=<n> scripts/train-forever.sh .venv/bin/whistle-house --data data-mac \
  train-detector --pipeline hum-log --arch harmonic_long --name humnet-long-<aug>-pretrain \
  --mix pretrain --augment <aug> --external all --epochs 60 --steps-per-epoch 500 \
  --batch-size 32 --val-examples 512 --workers 24 --wandb-project whistle-house
```

Epochs take 125-160 s instead of 78 s: lgn9 was saturated by another user's CPU jobs, and the
new augmentation costs more per example.

Trap: a bare `uv run` on lgn9 re-syncs the venv without the `ml` extra and removes torch
(5.8 GB to copy back). Use `.venv/bin/whistle-house` / `.venv/bin/python` there.

## 5. Results

Pretrains finished 20:12 (87 and 85 min of epoch time, one attempt each). Best val epoch 57
(`room`: F1 0.906, hum F1 0.911) and 54 (`channel`: 0.916 / 0.919); the old recipe's 0.921 is
on easier validation examples, so these are not comparable (each run validates on its own
augment).

Fine-tune (`data/`, now with the Fifine clips; 11 mic whistle + 13 mic hum clips, 34 mic
interferers in train):

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train-forever.sh .venv/bin/whistle-house train-detector \
  --pipeline hum-log --name humnet-long-room --mix finetune --augment room --external all \
  --init-from humnet-long-room-pretrain --steps-per-epoch 50 --epochs 60 --patience 15 \
  --lr 5e-4 --batch-size 32 --val-examples 512 --workers 24 --wandb-project whistle-house
```

Best at epoch 5 (val F1 0.9151 -> 0.9166), stopped at 20. `mic_train_f1` 0.896 -> 0.937 at
the kept epoch and still rising to 0.949 after it while val fell: the usual overfitting sign,
and the kept checkpoint is from before it.

Share of frames called hum (`scratch/svh/e6_compare.py`; the pretrains have never seen a
Fifine clip, the fine-tune has seen the train-split ones):

| clips                                         | v1, 60 ep | v1, 240 ep (live) | `channel` | `room` | `room` + fine-tune |
|-----------------------------------------------|-----------|-------------------|-----------|--------|--------------------|
| **Fifine speech, all 5 (110 s)**              | 21.6%     | 10.1%             | 2.5%      | 0.89%  | 0.39%              |
| Fifine speech, val + test clips (38 s)        |           | 13.6%             |           | 1.28%  | **0.34%**          |
| ... hum events per minute of speech           | 40.1      | 23.9              | 7.1       | 2.7    | 1.1                |
| ... seconds of hum per minute of speech       | 12.2      | 5.7               | 1.3       | 0.36   | 0.13               |
| Fifine speech +12 / +24 dB                    | 30.6 / 33.8 | 17.4 / 23.4     | 7.9 / 13.1| 0.86 / 0.93 | 0.40 / 0.53   |
| Fifine speech + reverb 0.5 s / DRR 0          | 42.6      | 35.3              | 16.7      | 1.46   | 1.09               |
| Fifine speech, tilt -3 dB/octave              | 32.9      | 29.8              | 4.3       | 2.18   | 1.15               |
| MUSAN speech val                              | 0.33      | 0.50              | 0.78      | 0.16   | 0.07               |
| MUSAN speech val + Fifine response            | 20.3      | 13.0              | 4.3       | 3.1    | 1.6                |
| MUSAN speech val + response + reverb          | 34.9      | 32.4              | 16.7      | 4.3    | 3.2                |
| Mac speech, held out (16 s)                   | 4.9       | 0.08              | 4.6       | 1.4    | 0.7                |
| MUSAN music val                               | 3.6       | 4.6               | 4.2       | 3.5    | 3.9                |
| MUSAN noise val                               | 2.7       | 1.9               | 1.0       | 1.3    | 1.3                |
| Fifine silence (8 min) / noise (1 min)        | 0 / 0     | 0 / 0             | 0 / 0     | 0 / 0.3 | 0 / 0.5           |

Detection (higher is better):

| clips                                         | v1, 60 ep | v1, 240 ep (live) | `channel` | `room` | `room` + fine-tune |
|-----------------------------------------------|-----------|-------------------|-----------|--------|--------------------|
| Fifine hums, voiced frames found              | 90%       | 91%               | 79%       | 94%    | 94%                |
| Fifine hums -12 dB, frames called hum         | 62.2%     | 66.9%             | 54.6%     | 75.3%  | 75.2%              |
| Fifine whistles, frames called whistle        | 53.1%     | 58.3%             | 63.3%     | 72.8%  | 69.5%              |
| MLEnd hums val, frames called hum             | 71.5%     | 71.7%             | 75.1%     | 76.1%  | 75.4%              |
| hummed tune clips matched (defaults, >= 0.5)  | 9/9       | 8/9               | 7/9       | 9/9    | 9/9                |
| whistled tune clips matched                   | 2/3       | 2/3               | 2/3       | 2/3    | 2/3 (3/3 with the lenient settings) |

Operating point on the Fifine clips (speech frames called hum / voiced hum frames found):

| presence threshold | 0.3        | 0.5        | 0.7        | 0.9        |
|--------------------|------------|------------|------------|------------|
| v1, 240 ep (live)  | 14.9 / 94  | 10.1 / 91  | 5.9 / 88   | 2.1 / 80   |
| `room`             | 2.0 / 95   | 0.9 / 94   | 0.3 / 93   | 0.1 / 88   |
| `room` + fine-tune | 1.2 / 95   | 0.4 / 94   | 0.1 / 93   | 0.0 / 89   |

Tune matches on the Fifine speech with the live recognizer settings: live model 0.54 per
minute at threshold 0.1 (2.2 behind the gate), `room` models none at 0.1 or 0.5. On the worst
synthetic case (MUSAN + response + reverb, 30 min): 0.93 -> 0.10 per minute at 0.1, 0.13 ->
0.03 at 0.5.

Reading:

- The diagnosis holds as a prediction: a model that never saw the Fifine goes from 21.6% to
  0.9% of the owner's speech frames called hum by changing only the augmentation, and finds
  more of his hums and whistles than before, not fewer.
- `room` is flat in level (0.86-0.93% from -12 to +24 dB) where v1 went 6% -> 34%.
- The response alone (`channel`) does most of the speech part but is unstable (best-val
  checkpoints at epochs 12 / 30 / 54: 0.0% / 11.7% / 2.5%), costs hum recall, and keeps the
  level and reverb shortcuts. Use `room`.
- Small new costs of `room`: 0.3-0.5% of the Fifine `noise` clips called hum (v1: 0), and
  0.1-0.2% of speech frames called whistle (v1: 0).
- **Not shown:** anything about a real conversation. The evidence is 110 s of speech (38 s
  unseen by the fine-tune), one voice. The live failure was ~0.7 triggers per minute over 40
  minutes; minutes of recorded conversation are needed to state a rate.

Deployment state (2026-11-23 20:30):

- `humnet-long-room.{pt,json,onnx}` on lgn9, the Mac and dirk (`.pt` sha256 `34d2c20f...`,
  ONNX export matches PyTorch). dirk has a pipeline `humnet-long-room` (ungated, onnx,
  `settle_s` 0.5, the owner's leniency options, threshold 0.5). It is **not active**. On dirk
  `analyze` gives no events on the two held-out speech clips and `Rising (hum)` at 0.90-0.91
  on the held-out hum clips.
- The owner had switched hums off at 18:06 by setting `kinds: ["whistle"]` on the active
  `humnet-long-e240-gate`. Left as it is.
- Running: `humnet-long-room-pretrain-e240` (same command with `--epochs 240`, GPU 1, tmux
  window `train-channel`, started 20:20, ~78 s per epoch -> about 01:35). With v1, 240 epochs
  halved the Fifine speech rate against 60. When done: `e6_compare.py`, then the fine-tune
  above with `--init-from humnet-long-room-pretrain-e240 --name humnet-long-room-e240`.

## 6. Later the same evening: 21 new Fifine clips, and how to fine-tune

The owner recorded 11 speech clips in different manners (281 s; notes on each), 8 hum clips
(139 s) and 2 whistle clips (44 s). Every model above had never seen them
(`scratch/svh/e7_new_clips.py`):

| new speech, all 11 clips                | live (v1, 240 ep) | v1, 60 ep | `room` pretrain | `humnet-long-room` |
|-----------------------------------------|-------------------|-----------|-----------------|--------------------|
| frames called hum                       | 13.8%             | 19.3%     | 5.1%            | 3.0%               |
| hum events per minute                   | 26.3              | 33.7      | 11.1            | 7.0                |
| tune matches / min, live settings >= 0.1| 1.92              | 2.13      | 0.43            | 0.21               |
| tune matches / min, live settings >= 0.5| 0.64              | 0.64      | 0.21            | 0.21               |
| tune matches / min, default settings    | 0                 | 0         | 0               | 0                  |
| new hums, voiced frames found           | 96.4%             | 96.6%     | 95.9%           | 95.1%              |

So 4.6x fewer frames than the live model, not the 25x of the first five clips: those were
easy. Per clip, `humnet-long-room`: Instruction 0.0, Angry 0.0, Phone call 0.1, Explaining
0.1, Questions 0.2, Exited announcer 1.4, Weather man 2.3, Calling to other room 2.8, Good
boy 7.1, Thinking 9.0, Bedtime story 9.5% (one match there, 0.74). Sing-song and drawn-out
speech is what is left, and "Thinking" holds actual hums ("hmm").

### The fine-tune recipe was not doing much

Same fine-tune as before with the new clips in `data/` (`humnet-long-room-ft2`): best epoch 6,
stopped at 21. Variants showed why that is a poor instrument:

- `--patience 15` with `--epochs 60 --steps-per-epoch 50`: OneCycle warms up for 18 epochs,
  and val F1 (mostly external data, 512 examples) peaks at epoch 4-6, so the kept checkpoint
  has hardly moved from the pretrain. With the mic negatives weighted 3x, "best" was epoch 0:
  the init checkpoint itself.
- The state the run stops in (epoch 15-21) is at the LR peak. Leave-out runs judged there
  disagreed with each other (held-out hard clips: 3.0% -> 0.3% in one fold, 2.4% -> 7.4% in the
  other, for the same change).

A schedule that runs to its end (`--epochs 20 --steps-per-epoch 100 --patience 0`, last epoch
kept) is consistent. Leave-out test: two views of `data/` each without three of the hard
train-split clips (`data-foldA`: Bedtime story, Good boy, Weather man; `data-foldB`:
Thinking, Calling to other room, Exited announcer), frames called hum on the three left out:

| model (last epoch)                        | fold A clips (74 s) | fold B clips (75 s) |
|-------------------------------------------|---------------------|---------------------|
| `room` pretrain (no fine-tune)            | 9.1%                | 9.4%                |
| `humnet-long-room` (saw no new clip)      | 6.3%                | 4.9%                |
| **left out**, mic weight 3                | 2.8%                | 2.4%                |
| **left out**, mic weight 9                | 1.8%                | 1.9%                |
| trained on them, mic weight 3 / 9         | 0.2-0.3% / 0.1%     | 1.9-2.6% / 0.7-1.9% |

More speech from the room helps on speech the model has not heard (9% -> 2-3%), a little
more with the mic's clips drawn 3x as often, and the fold B clips stay near 2% even when
trained on (the hums in "Thinking"). Hum recall did not move (93-96% of voiced frames in every
run).

Two options came out of this (code, CLI, API, Models page, tests): `--augment room-mic` (room
with the `mic` interferer weight 3 -> 9) and `--keep last` (`<name>.pt` = the last epoch).

### `humnet-long-room2`

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train-forever.sh .venv/bin/whistle-house train-detector \
  --pipeline hum-log --name humnet-long-room2 --mix finetune --augment room-mic --keep last \
  --external all --init-from humnet-long-room-pretrain --steps-per-epoch 100 --epochs 20 \
  --patience 0 --lr 5e-4 --batch-size 32 --val-examples 512 --workers 24
```

5.8 min. Train split: 13 mic whistle, 19 mic hum clips, 42 mic interferers (19 of them
Fifine speech or noise). Val F1 0.9219 -> 0.9211 (it would not have been kept as "best").

| clips                                              | live  | `humnet-long-room` | `humnet-long-room2` |
|----------------------------------------------------|-------|--------------------|---------------------|
| Fifine speech it never saw (5 clips, 118 s)        | 9.7%  | 0.17%              | **0.0%**            |
| all Fifine speech (16 clips, 6.5 min; 11 in train) | 12.8% | 2.25%              | 0.27%               |
| ... hum events per minute                          | 25.6  | 5.4                | 1.2                 |
| ... tune matches / min, live settings >= 0.1       | 1.53  | 0.15               | 0.15 (one: "Thinking", 0.55) |
| ... +24 dB / + reverb / tilt -3 dB per octave      | 23.5 / 33.4 / 31.2 | 2.2 / 5.1 / 3.0 | 0.25 / 0.64 / 0.44 |
| Fifine hums, voiced frames found (all / unseen)    | 94% / 93% | 95% / 94%      | 95% / 94%           |
| hummed tune clips matched / whistled               | 8/9, 2/3 | 9/9, 2/3        | 9/9, 2/3            |
| Fifine noise clips called hum                      | 0     | 0.47%              | 0.08%               |
| MUSAN speech val raw / + response / + both         | 0.5 / 9.0 / 22.6 | 0.07 / 0.77 / 1.28 | 0.29 / 1.04 / 0.81 |
| Mac speech held out (16 s)                         | 0.08  | 0.68               | 0.30                |
| MUSAN music / noise val                            | 4.6 / 1.9 | 3.9 / 1.3      | 3.3 / 1.3           |

("Fifine response" here was re-measured with the new speech clips, so the MUSAN probe rows
differ from section 5.)

The honest number for unheard speech of the hard kind is the leave-out row (about 2% of
frames), not 0.27%: most of the speech in that row is in its training set. The five clips it
never saw are the easy manners.

State (21:50): `humnet-long-room2.{pt,json,onnx}` on lgn9, the Mac and dirk (`.pt` sha256
`228ab3a4...`); dirk's inactive pipeline is now `humnet-long-room2` (the `humnet-long-room`
pipeline from two hours earlier was removed, its model files are still there). On dirk
`analyze`: no events on Questions, Angry, Phone call; "Thinking" gives `Falling (hum)` 0.55.
21:55: on the owner's word, `humnet-long-room2` was made the active pipeline (manager API)
and the service restarted; engine running on the Fifine with it. `humnet-long-e240-gate`
keeps its `kinds: ["whistle"]`.

When `humnet-long-room-pretrain-e240` finishes: the command above with `--init-from
humnet-long-room-pretrain-e240 --name humnet-long-room2-e240`, then `e6_compare.py` /
`e7_new_clips.py` against `humnet-long-room2`.

## Open

- Fine-tune checkpoint selection: val F1 cannot see what a mic fine-tune is for. A probe of
  held-out mic negatives (frames called hum / whistle) in `validate()` would let "best" mean
  something there; until then `--keep last` and a look at the recordings.
- "Thinking"-type sounds ("hmm", "mmm") are hums. Whether they should trigger is a question
  for the recognizer (a tune must stand alone, note lengths), not the detector.

- A live conversation with the new model is still the only real measure of triggers per
  hour: clips recorded on purpose are not a conversation.
- One clip of speech 10 cm from the Fifine would separate the mic's response from the room's.
- Hum pitch below 100 Hz is found less often (65% of frames on the Fifine hums at 60-100 Hz
  with the live model; MLEnd has almost no hums that low). Not rechecked on the new model.
- A real room-response set (measured impulse responses) for validation: the reverb in
  training and in the probes comes from the same synthetic family.
