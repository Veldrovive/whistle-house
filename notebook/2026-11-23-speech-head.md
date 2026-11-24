# 2026-11-23 (late) — a speech output, and holding hums back after speech

Follows `2026-11-23-speech-as-hum-root-cause.md`. The owner's idea: also detect *speech* (presence
only), and suppress hum detections once speaking has been heard, so a word that ends like a
hum is not a note. Whistles are left alone (never confused with speech). All training and
measurement on lgn9; scripts in `notebook/scripts/2026-11-23-speech-vs-hum/` (`e8_*`,
`clip_look.py`, `label_stats.py`; on lgn9 under `scratch/svh/`).

## What was built

- **Labels** (`data/training.py`, `GeneratorConfig.aux = ("speech",)`): one more label column
  after the f0 columns. An interferer drawn from a speech file (mic `speech` clips, external
  `speech` entries: `TrainingSources.speech_files`) is tracked through reverb, channel and
  mixing; `speech_activity` marks the 20 ms frames within 30 dB of its loud parts (gaps under
  0.3 s filled). Label 1 there if the speech is no more than 10 dB under the rest of the mix,
  -1 (left out of the loss) if 10-20 dB under, else 0. Whole example -1 when the interferer
  is synthetic pseudo-speech or a stretch of a speech file quieter than -60 dBFS. Recipes
  without `aux` are bit-identical to before (hash of 12 examples each of `mic`/`v1`,
  `mic`/`room`, `finetune`/`room-mic`).
  In the mixes (96 examples each): `pretrain`/`room` 43% of frames speech, 20% ignored, 37%
  none; `finetune`/`room-mic` 29 / 26 / 45%. Inside a speech example 96% of frames are active
  on average: the label says "speech is going on", it does not follow word gaps.
- **Head** (`ml/model.py`, `WhistleNetConfig.aux`): a second linear layer on the same pooled
  features as the presence head; its outputs are extra columns of the presence output
  (K + A). `add_aux` puts one on a checkpoint that has none without changing its other
  outputs. Loss: masked BCE (`aux_weight` 1). Val metrics `speech_f1` etc., outside the mean
  that picks a checkpoint. `--aux speech` on the CLI, a "speech head" tick on the Models page.
  ONNX export carries it (`OnnxStream.aux`).
- **Hold** (`stages/detectors/learned.py`): `speech_hold_s` (0 = off), `speech_threshold`
  (0.5), `speech_min_s` (0.1), `speech_holds` (`["hum"]`). Once the speech output has been
  above threshold for `speech_min_s`, held kinds get no pitch candidates until
  `speech_hold_s` after the last speech frame, and a held-kind note still sounding at that
  moment is dropped. The speech score is reported as `kind_scores["speech"]` (dotted trace in
  the manager). Streaming == offline still holds (test).

## Runs

| name | what | time |
|------|------|------|
| `humnet-long-room2-sp` (+ `-foldA`, `-foldB`) | head added to `humnet-long-room-pretrain` in the mic fine-tune: `--mix finetune --augment room-mic --keep last --patience 0 --aux speech --init-from humnet-long-room-pretrain --epochs 40 --steps-per-epoch 100 --lr 5e-4 --batch-size 32 --val-examples 512` | 25 min each (box shared with three other runs) |
| `humnet-long-speech-pretrain` | from scratch with the head: the `humnet-long-room-pretrain` command + `--aux speech`, `--data data-mac` (no Fifine clip) | 89 min |
| `humnet-long-speech` (+ folds) | the `humnet-long-room2` fine-tune command from it (20 x 100) | 9 min each |

The folds leave three hard speech clips out each (A: Bedtime story, Good boy, Weather man;
B: Exited announcer, Calling to other room, Thinking). Val/test-split clips are never trained
on by any fine-tune.

## Results, head added in the fine-tune (`e8_speech_head.py ft1`)

Speech output on (>= 0.5), % of frames:

| clips | `sp` | `spA` | `spB` |
|-------|------|-------|-------|
| Fifine speech, all 16 (6.5 min) | 89.5 (95.7 of voiced) | 87.9 | 88.3 |
| Fifine speech left out of fold A | 93.5 | **92.0** | 93.0 |
| Fifine speech left out of fold B | 84.6 | 81.1 | **78.9** |
| Fifine speech, val+test clips (5) | 87.7 | 86.1 | 87.4 |
| Mac speech (19 clips) | 96.6 | 96.5 | 96.5 |
| MUSAN speech val (30 min) | 92.0 | 92.2 | 92.3 |
| Fifine hums, all 21 | 4.5 | 3.5 | 4.0 |
| Fifine hums, 20 of 21 clips | 0.0-0.7 each (two train clips 1-3.5) | | |
| Fifine hum `fb3c1d2fe46f` (test split) | **48** | 39 | 45 |
| Mac hums | 0.0 | 0.0 | 0.0 |
| MLEnd hums val (17 min) | 1.2 | 0.9 | 0.8 |
| whistles (Fifine, Mac, MLEnd) | 0.0 / 0.1 / 0.4 | | |
| Fifine silence, noise | 0.0, 0.06 | | |
| MUSAN music / noise val | 5.6 / 4.1 | | |

Bold = clips that model never saw. So the head generalises on speech (79-92% of frames of
unseen hard clips, 91-99% of their voiced frames) and stays off on hums, with one exception:
`fb3c1d2fe46f` is a fast hummed melody (about 50 notes in 20 s, 0.2-0.5 s each, steps of a
third to a fifth), which has the pitch movement of talking. It is in the test split, so nothing trained
on it. The joint fine-tune also lost hum presence on that clip (89% of voiced frames with
`humnet-long-room2`, 56-69% here); all other hum clips are unchanged. Before any hold the
hum output of these models is slightly worse on negatives than `room2`'s (Mac speech 0.03%
-> 0.3-0.4% of frames, MUSAN noise 2.9% -> 4.1%; twice the fine-tune steps and a second
task), which the hold more than takes back on speech.

Hum events with the hold (events per minute; `sp` saw every train-split clip, the fold
columns are the clips that model did not see):

| speech | `room2` | `sp` hold 0 | 0.5 s | 1 s | 2 s |
|--------|---------|-------------|-------|-----|-----|
| Fifine, all 16 clips | 1.22 | 0.92 | 0 | 0 | 0 |
| MUSAN speech val | 0.57 | 0.73 | 0.07 | 0.07 | 0.03 |
| Mac speech | 0.13 | 1.29 | 0 | 0 | 0 |
| left out of fold A, `spA` | | 2.44 | 0 | 0 | 0 |
| left out of fold B, `spB` | | 11.2 | 4.0 | 3.2 | 1.6 |

Tune matches at dirk's lenient recognizer settings: `room2` one (Thinking -> Falling 0.55);
with any hold none, on any speech group, for all three models. Hummed tune clips matched
9/9 and whistled 2/3 at every hold (same as `room2`). Seconds of hum events per minute of
humming: Fifine all 47.6 -> 45.6 at 1 s (all of the loss is `fb3c`: 10.2 s of events -> 3.3 s),
MLEnd hums 45.2 -> 44.3.

What is left in fold B's unseen clips at 1 s hold: "Thinking" (2 events: an "hmm" with nobody
talking in the second before it is a hum by any definition) and "Calling to other room"
(distant speech, the speech output is on for only 63% of it).

**Threshold / minimum length** (`e8_sweep.py`: frame-level replay of the hold from the saved
scores). Raising `speech_threshold` to 0.7 or `speech_min_s` to 0.3-0.6 gives back most of
`fb3c` but also most of the false hum frames: the hum-like stretches of speech are where the
speech output is least sure, and what removes them is being within a second of confident
speech. At 0.9 the hold does nothing at all: this head (a linear layer trained for 4,000
steps at a fine-tune learning rate) rarely gets that sure.
Defaults stay 0.5 / 0.1.

## Results, trained from scratch with the head (`e8_speech_head.py final`)

Pretrain val (epoch 60): F1 0.889 (whistle 0.891, hum 0.886), the same as
`humnet-long-room-pretrain` without the head, and speech F1 0.895 (recall 0.92, false
positive rate 0.07). After the fine-tune: F1 0.911, speech F1 0.881.

| | `room2` | `sp` (head added in the fine-tune) | `pre` (no Fifine clip seen) | `speech` | 
|---|---|---|---|---|
| speech output on, Fifine speech, % of frames | | 89.5 | 77.2 | 85.0 |
| the same, MUSAN speech val | | 92.0 | 88.7 | 88.9 |
| the same, Fifine hums / `fb3c` alone | | 4.5 / 48 | 3.2 / 34 | 3.8 / 42 |
| hum presence on Fifine speech, % of frames | 0.27 | 0.33 | 3.15 | 0.42 |
| hum events / min, Fifine speech, no hold -> 1 s | 1.22 | 0.92 -> 0 | 6.12 -> 0.77 | 0.61 -> 0.15 |
| hum events / min, MUSAN speech, no hold -> 1 s | 0.57 | 0.73 -> 0.07 | 0.33 -> 0.07 | 0.40 -> 0.07 |
| hum events / min, MUSAN music, no hold -> 1 s | 3.5 | 5.0 -> 2.5 | 6.4 -> 5.7 | 8.4 -> 7.3 |
| `fb3c`: hum presence, % of voiced frames | 89 | 56 | 89 | 63 |
| hummed / whistled tune clips matched | 9/9, 2/3 | 9/9, 2/3 | 8/9, 2/3 | 9/9, 2/3 |

Leave-out folds, hum events per minute on the three clips the model did not see, no hold
-> 1 s: fold A 8.1 -> 0.8 (head added in the fine-tune: 2.4 -> 0), fold B 6.4 -> 1.6 (11.2
-> 3.2). Six clips, 2.4 minutes: the two ways of training are not told apart by this.

So training the head from the start buys nothing measurable over adding it in the mic
fine-tune (25 minutes instead of 100), and its hum output is a little worse on music.
`humnet-long-room2-sp` stays the model to try. Two things the from-scratch run does show:

- The pretrain model has never heard the Fifine, and its speech output is still on for 77%
  of the owner's speech there; with a 1 s hold its hum events on that speech go from 6.1 to
  0.8 per minute. The hold does not depend on the mic fine-tune.
- `fb3c` (the fast hummed melody) reads as speech for every model with a speech output,
  34-48% of its frames, with or without having seen the Fifine. It is a property of the
  labels (fast pitch movement = talking), not of one training run.

Threshold sweep (`e8_final_sweep.txt`): as for the first models. On `speech`, the hold at
0.5 / 0.1 s takes hum frames on Fifine speech from 0.42% to 0.14%, and at 0.9 it does
nothing.

## Open

- A conversation on the live system is still the measure.
- Fast hummed melodies read as speech. The owner's tunes are slow three-note ones, so the
  hold costs them nothing in these clips; a fast tune would need the speech output trained
  on more fast humming (MLEnd has it: 1% of its frames fire) or a longer `speech_min_s`.
- The hold looks only backwards. "Hmm, well..." gives a hum note before the speech starts;
  a single note matches no tune, but a recognizer-side rule (a tune followed by speech
  within a second is not a tune) would need the speech score there.
- Speech labels do not follow pauses (96% of a speech example is "speech"), which is right for
  a hold but would be wrong for anything that needs word timing.
