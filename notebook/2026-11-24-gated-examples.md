# 2026-11-24 — a steady hum in the room, and training for a gate in the pipeline

Around midnight a steady electrical hum appeared on dirk's mic and `humnet-long-room2`
(ungated) drew a hum note along it. The owner's fix is the `adaptive_gate` (he recorded a
`silence` take with the hum on, "Weird hum", and made a profile `weird-hum` from it), and
asked for gated examples in training so a model is at home behind the gate. All training
and measurement on lgn9; script `notebook/scripts/2026-11-23-speech-vs-hum/e9_gate.py` (on
lgn9 `scratch/svh/`).

## The hum

Three new Fifine clips with it on and nobody in the room (`a73ef3fe863c` silence 30 s,
`34809240aec2` noise 22 s, `84d4029aa168` other 4 s). A tone at 120 Hz (mains x 2), 16 dB
over its neighbouring bins, per-bin std 1.0-1.3 dB over 30 s (the room's noise without it:
4 dB). Clip RMS -66 dBFS against -69 for the quiet room, so the tone carries about as much
power as all the rest of the room noise. Something weaker also sits at 130-300 Hz (profile
max -8.5 dB there against -24.5 quiet).

## What the models that exist do with it (`e9_gate.py base`)

`+g6` = behind `adaptive_gate` over the six Fifine profiles including `weird-hum` (the
filter of dirk's `humnet-long-e240-gate` pipeline, `dwell_s` 60); `+g5` = the same without
`weird-hum`. Neither model was trained on gated frames.

| | `room2` | `room2` +g6 | `sp` | `sp` +g6 |
|---|---|---|---|---|
| hum clips with the 120 Hz tone: % of frames called hum | 22.4 | 0 | 91.9 | 0 |
| the same: hum events / min (seconds of events / min) | 16.0 (15) | 0 | 10.7 (58) | 0 |
| Fifine hums: hum presence, % of voiced frames | 95.0 | 94.3 | 91.5 | 92.2 |
| Fifine whistles: whistle presence, % of frames | 58.8 | 46.6 | 59.9 | 43.0 |
| Fifine speech: % of frames called hum | 0.27 | 0.79 | 0.33 | 0.71 |
| Fifine speech: hum events / min | 1.22 | 1.84 | 0.92 | 1.53 |
| the same with a 1 s speech hold | | | 0 | 0 |
| MUSAN noise val: hum events / min | 0.56 | 0.28 | 4.2 | 0.28 |

`sp` = `humnet-long-room2-sp`. `+g5` gives the same numbers as `+g6` on every Fifine group,
the hum clips included: a 16 dB line that the gate lets through is not enough for either
model to call a hum once the rest of the frame is gated to 0.

So the gate does remove the tone, and a model trained without gated frames keeps its hums
behind it. What it loses there: a fifth to a quarter of the whistle frames, and it calls
2-3 times as many speech frames a hum.

## What was built

- `augment.synth_background`: a room's steady background at unit RMS. Coloured noise (random
  response, up to 6 dB per octave more at the bottom) plus 0-3 steady tonal sources: a tone
  or a harmonic series at 45-1500 Hz, 40% of them on the mains (50 / 60 / 100 / 120 Hz, all
  locked to one grid frequency), strongest partial -20...+15 dB re the noise's RMS, pitch
  within 0.1%, sometimes a slow beat against a neighbour 10-25 dB weaker.
- `GeneratorConfig.p_background`: an example gets a steady background under it, a mic
  `silence` take (`TrainingSources.backgrounds`) half the time when there are any, else the
  synthetic one, at -85...-45 dBFS RMS and at least 10 dB under the example's loud parts.
- `GeneratorConfig.p_gate`: the example's frames go through `SpectralGate` with a profile
  (per-bin mean and std) measured on the 3 s of floor noise + background just before the
  example, as a profile of the room is. `gate_offset_db` 1.5 (std) shifts the profile as a
  gain drift would; with `p_gate_mismatch` 0.15 the profile was measured with the
  background 6-20 dB quieter than it is, so part of it leaks; the gate's `n_std` is drawn
  from 1.5-3. `SpectralGate.set_profile` takes the statistics directly. `adaptive_gate`
  with its belief on one profile is this gate (tested when it was built).
- `--augment room-gate` / `room-mic-gate`: `room` / `room-mic` + `p_background` 0.6,
  `p_gate` 0.5. Train on the pipeline without filters (the generator refuses otherwise);
  the gate goes in the pipeline that uses the model. The old recipes generate bit-identical
  examples (hash of 10 examples each of `mic`/`v1`, `mic`/`room`, `finetune`/`room-mic` with
  and without `--aux speech`, before and after).
- Cost: a gated fine-tune epoch took 51 s against 38.5 s ungated (16 workers).

## Runs

| name | what | time |
|------|------|------|
| `humnet-long-gate-ft` | the `humnet-long-room2-sp` command with `--augment room-mic-gate`: `--mix finetune --keep last --patience 0 --aux speech --init-from humnet-long-room-pretrain --epochs 40 --steps-per-epoch 100 --lr 5e-4 --batch-size 32 --val-examples 512`, all Fifine clips | 31 min |
| `humnet-long-gate-ft-nohum` | the same on `data-nohum` (without the three clips that have the hum) | 30 min |
| `humnet-long-room2-sp2` | control: the `humnet-long-room2-sp` command again (`room-mic`, no gated examples), all Fifine clips including the three with the hum | 25 min |
| `humnet-long-gate-pretrain` | from scratch: the `humnet-long-speech-pretrain` command with `--augment room-gate` (`--data data-mac`) | 118 min |
| `humnet-long-gate` (+ `-nohum`) | fine-tune from it, 20 x 100 (`scratch/svh/chain_gate.sh`) | 13 min each |

The pretrain these fine-tunes start from never saw a gated frame, and on the half-gated
validation set of `room-mic-gate` it already scores F1 0.910 (0.922 on the ungated set of
`room-mic`): most of what a gate does to the picture is harmless to the model. After the
fine-tune: 0.905 on the half-gated set (`sp2`: 0.913 on the ungated one).

## Results, gated examples in the fine-tune only (`e9_gate.py ft`)

`gft` = `humnet-long-gate-ft`, `gftnh` = `-nohum`. Plain column = no filter, `+g6` = behind
the six-profile `adaptive_gate`. `+g5` (no `weird-hum` profile) gave the same numbers as
`+g6` on every Fifine group for every model.

| | `sp` | `sp` +g6 | `sp2` | `sp2` +g6 | `gft` | `gft` +g6 | `gftnh` | `gftnh` +g6 |
|---|---|---|---|---|---|---|---|---|
| 120 Hz clips: % of frames called hum | 91.9 | 0 | 0.08 | 0 | 0.37 | 0 | 2.3 | 0 |
| 120 Hz clips: seconds of hum events / min | 58 | 0 | 0.15 | 0 | 0.26 | 0 | 1.6 | 0 |
| Fifine whistles: whistle presence, % of frames | 59.9 | 43.0 | 54.9 | 38.3 | 57.9 | 57.4 | 57.6 | 57.2 |
| Fifine whistles: whistle events / min | 60.2 | 54.3 | 55.3 | 47.6 | 56.4 | 58.1 | 56.0 | 57.4 |
| whistled tune clips matched | 2/3 | 2/3 | 2/3 | 3/3 | 2/3 | 3/3 | 2/3 | 3/3 |
| Fifine hums: hum presence, % of voiced frames | 91.5 | 92.2 | 91.1 | 91.9 | 93.8 | 91.6 | 92.7 | 90.9 |
| hummed tune clips matched | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| MLEnd hums val: hum presence, % of voiced frames | 93.8 | 90.6 | 93.9 | 91.0 | 93.4 | 93.0 | 93.5 | 93.4 |
| Fifine speech: % of frames called hum | 0.33 | 0.71 | 0.26 | 0.67 | 0.74 | 0.53 | 0.62 | 0.62 |
| Fifine speech: hum events / min | 0.92 | 1.53 | 0.61 | 1.38 | 1.53 | 0.77 | 1.22 | 1.38 |
| the same with a 1 s speech hold | 0 | 0 | 0 | 0 | 0.31 | 0 | 0.61 | 0 |
| Fifine speech: speech output on, % of frames | 89.5 | 88.8 | 90.8 | 88.9 | 87.9 | 88.2 | 85.3 | 86.2 |
| MUSAN speech val: hum events / min | 0.75 | 0.6 | 0.4 | 0.5 | 0.75 | 1.45 | 0.85 | 0.9 |
| MUSAN music val: hum events / min | 4.1 | 3.4 | 4.6 | 1.5 | 4.5 | 4.4 | 5.7 | 5.3 |
| MUSAN noise val: hum events / min | 4.2 | 0.28 | 4.2 | 0.14 | 4.4 | 4.1 | 5.4 | 4.1 |
| Fifine silence (8 clips, 7.9 min), noise (3): hum or whistle events | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

What this says:

- **Behind the gate, a model trained with gated examples keeps its whistles.** 57% of the
  whistle frames with or without the gate, where the models trained without them drop to
  38-47% behind it, and MLEnd hums stay at 93% where the others lose 3 points. The owner's
  hums are found as often either way (91-94% of voiced frames, all nine tune clips).
- **The gate takes the 120 Hz hum out for every model**, also without a profile of it.
- **Without a gate, steady backgrounds in training do most of it too.** `gftnh` never saw
  the hum clips and calls 2.3% of their frames a hum (1.6 s of events a minute) against 92%
  (58 s) for `sp`; what taught it is the synthetic steady tones and the other `silence`
  takes. With the clips among its negatives a model ignores this hum entirely, gated
  examples or not (`sp2`, `gft`: under 0.4%). So one `silence` take of a new hum does two
  jobs: a profile for the gate (works at once) and a negative for the next fine-tune.
- **Speech**: behind the gate the gated-trained models call 0.5-0.6% of the owner's speech
  frames a hum (0.8-1.4 events a minute), the others 0.7% (1.4-1.5); with the 1 s speech
  hold all of them give none. Ungated, the gated-trained models are a little worse than
  `sp2` (1.2-1.5 events a minute against 0.6; 6.5 minutes of speech, so 8-10 events against
  4).
- **MUSAN through a Fifine gate** is not a case that occurs (no profile fits those
  recordings: the gate there mostly thresholds at the Fifine's floor). The models trained
  without gated frames fire less on music and noise behind it; the gated-trained ones fire
  as they do ungated.

## Results, gated examples from the start (`e9_gate.py final`)

Pretrain val (epoch 60, half-gated set): F1 0.884 (whistle 0.887, hum 0.881), speech F1
0.870. After the fine-tune: 0.896, speech F1 0.864 (its init scores 0.900 on that set).
`gpre` = `humnet-long-gate-pretrain` (never heard the Fifine), `gate` = `humnet-long-gate`,
`gatenh` = `-nohum`; `gft` repeated from above. `+g5` again equals `+g6` on every Fifine
group.

| | `gft` | `gft` +g6 | `gpre` | `gpre` +g6 | `gate` | `gate` +g6 | `gatenh` | `gatenh` +g6 |
|---|---|---|---|---|---|---|---|---|
| 120 Hz clips: % of frames called hum | 0.37 | 0 | 0 | 0 | 0.02 | 0 | 0.48 | 0 |
| 120 Hz clips: seconds of hum events / min | 0.26 | 0 | 0 | 0 | 0 | 0 | 0.45 | 0 |
| Fifine whistles: whistle presence, % of frames | 57.9 | 57.4 | 63.4 | 67.9 | 57.6 | 59.8 | 57.6 | 58.8 |
| Fifine whistles: whistle events / min | 56.4 | 58.1 | 64.8 | 64.4 | 60.6 | 62.0 | 62.3 | 60.9 |
| whistled tune clips matched | 2/3 | 3/3 | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 |
| Fifine hums: hum presence, % of voiced frames | 93.8 | 91.6 | 91.7 | 88.9 | 92.6 | 92.5 | 91.2 | 91.2 |
| the same, val+test clips (6, never trained on) | 90.3 | 83.2 | 81.5 | 70.9 | 83.7 | 83.4 | 78.6 | 78.2 |
| hummed tune clips matched | 9/9 | 9/9 | 9/9 | 8/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| MLEnd hums val: hum presence, % of voiced frames | 93.4 | 93.0 | 93.4 | 93.4 | 93.5 | 93.6 | 92.9 | 93.1 |
| Fifine speech: % of frames called hum | 0.74 | 0.53 | 2.13 | 1.34 | 0.72 | 0.40 | 0.48 | 0.26 |
| Fifine speech: hum events / min | 1.53 | 0.77 | 4.13 | 2.76 | 1.38 | 0.77 | 0.92 | 0.46 |
| the same with a 1 s speech hold | 0.31 | 0 | 0.77 | 0.15 | 0.46 | 0.15 | 0.61 | 0.15 |
| Fifine speech: speech output on, % of frames | 87.9 | 88.2 | 73.9 | 76.6 | 79.2 | 78.1 | 81.4 | 79.6 |
| the same, val+test clips (5, never trained on) | 85.8 | 85.6 | 66.6 | 68.4 | 70.5 | 69.8 | 73.8 | 72.2 |
| MUSAN speech val: hum events / min (1 s hold) | 0.75 (0.3) | 1.45 (0.1) | 0.45 (0.05) | 0.35 (0) | 0.45 (0) | 0.3 (0) | 0.3 (0) | 0.35 (0) |
| MUSAN music val: hum events / min | 4.5 | 4.4 | 3.1 | 9.4 | 4.1 | 6.4 | 3.9 | 7.0 |
| MUSAN noise val: hum events / min | 4.4 | 4.1 | 3.5 | 2.5 | 4.2 | 1.3 | 3.8 | 1.1 |
| MUSAN noise val: whistle events / min | 10.0 | 7.2 | 13.3 | 16.0 | 14.8 | 14.4 | 14.8 | 14.1 |
| Fifine silence (8 clips, 7.9 min), noise (3): hum or whistle events | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

What this says:

- **Gated examples from the start buy nothing clear over gated examples in the fine-tune
  only** (131 minutes against 31). `gate` behind the gate has a little more of the whistles
  (59.8% of frames against 57.4, 62 events a minute against 58) and is cleaner on MUSAN
  speech. `gft` has the surer speech output on the owner's speech (88% of frames against
  78; on clips neither trained on, 86 against 70), so the 1 s hold leaves it no hum event
  on the 6.5 minutes of Fifine speech where `gate` keeps one; it draws half as many
  whistle events on MUSAN noise (7 a minute against 14) and matched all three whistled
  tune clips behind the gate (`gate`: two). Hums are the same. `humnet-long-gate-ft` stays
  the model to try behind the gate. Third time in two days that a from-scratch run did not
  beat changing only the mic fine-tune (`room2-sp`, `gate-ft`): try the fine-tune first.
- **A model that sees steady backgrounds from the start ignores the 120 Hz hum without a
  gate and without ever having heard it.** `gpre` was trained on no Fifine clip: 0% of the
  hum clips' frames, no event (`sp`, fine-tuned from the ungated pretrain without the
  clips: 92%). `gatenh`: 0.5% of frames, one event in 56 s. The gate is still the
  surer fix (0 for every model), but the model no longer depends on it for this.
- `gpre` also finds more of the Fifine whistles than any fine-tuned model (63% of frames
  ungated, 68% gated): the mic fine-tune costs whistle frames here as it did for the
  ungated models (`room2`-family 55-60%). Not followed up.
- Behind the gate the from-scratch models fire more on MUSAN music (6-9 hum events a
  minute against 3-4 ungated) and less on MUSAN noise. As above, a Fifine gate on MUSAN is
  not a case that occurs.

## On dirk (2026-11-24 08:07)

The owner said to deploy. dirk's checkout was at the Mac's commit and clean; the uncommitted
working-tree changes were copied over it (rsync of the changed and new files, nothing
committed), `humnet-long-gate-ft.{onnx,json,pt}` copied, manager rebuilt, service restarted.
New pipeline `humnet-long-gate-ft`: `stft_log`, `adaptive_gate` over the seven profiles of
the owner's `humnet-long-e240-gate` pipeline (the six above + `hum-2`, which he measured at
01:17 from a second hum take, `12633bbfa2be`; the measurements above used six), `dwell_s`
60, `learned` with `speech_hold_s` 1.0, whistles and hums, recognizer as in
`humnet-long-room2` (threshold 0.5). Made active through the API.

Before the restart, both pipelines over 12 recordings on dirk itself (`dirk_check.py` in the scripts folder,
4096-sample chunks, the owner's tunes):

| | `humnet-long-room2` | `humnet-long-gate-ft` |
|---|---|---|
| three takes with the hum (111 s): hum events | 9, 4, 2 | 0, 0, 0 |
| three speech clips (82 s): hum events / tune matches | 7 / 1 (Falling 0.55) | 0 / 0 |
| three hummed tune clips matched | 3 | 3 |
| three whistled tune clips matched | 2 | 3 |
| ms per chunk (93 ms of audio): median / p95 / worst | 14.6 / 17.9 / 24.8 | 15.8 / 18.5 / 32.5 |

The third whistled clip (`0c1b6a9bd088`) matched through one event reported as a hum, not
as whistle notes; not looked into. The service with the new pipeline and one Live page
open: 26% of a core, 214 MB. On the Live page the gate sat on `fifine` 100% and no note was
drawn in the half minute watched; whether the hum was on then is not known.

## Open

- Not yet seen on dirk: a conversation, and the hum coming on.

- The gate in training is `spectral_gate` with one profile. `adaptive_gate` choosing the
  wrong profile for a while, or blending two, is only covered by the mismatch case (15% of
  gated examples).
- A hum that starts while someone is humming a tune at its pitch: the gate takes the
  fundamental of those notes out. Not measured.
