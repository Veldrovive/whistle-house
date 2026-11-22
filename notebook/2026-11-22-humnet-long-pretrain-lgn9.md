# 2026-11-22 — `humnet-long-pretrain` on lgn9

Pretrain of the `harmonic_long` arch (whistle + hum, 2.94 s receptive field, 0.37 s lookahead)
on the external datasets, on lgn9 (8x H100 80 GB, shared). Fine-tune happens later on the Mac.

## Command

Same settings as `humnet-pretrain` / `humnet-deep-pretrain` so the runs compare:

```bash
CUDA_VISIBLE_DEVICES=<gpu> scripts/train-forever.sh uv run whistle-house train-detector \
  --pipeline hum-log --arch harmonic_long --name humnet-long-pretrain \
  --mix pretrain --external all \
  --epochs 60 --steps-per-epoch 500 --batch-size 32 --val-examples 512 \
  --workers 8 --wandb-project whistle-house
```

## Setup notes

- Install: torch 2.15.0+cu132, CUDA available, 136 tests pass. The first `uv sync` only looked
  stuck: uv was copying ~5.8 GB of CUDA wheels from the uv cache on `/z` (NFS) into the venv on
  `/scratch4` (no hardlinks across filesystems). It finished on its own; no lock was involved.
- Datasets fetched into `/z/home/adempst/data/whistle-house` (`fetch all --clean-raw`).
  MLEnd whistles: 4 files missing upstream (0727, 1411, 4541, 5677), skipped.

## Trade-offs vs. the earlier runs

- **GPU.** The first attempt shared GPU 5 with another user's job (stopped after epoch 1, see
  the log). The run that counts had GPU 5 to itself.
- **Thread limit + 24 workers** (`OMP/OPENBLAS/MKL_NUM_THREADS=1`, `--workers 24` instead of 8).
  Changes nothing about what is learned (`workers` is resume-free in `ml/train.py`); see the
  log for why.
- **Batch size unchanged (32)**, same steps, epochs, val examples and LR as the earlier runs.
- **Windows are 4.5 s** for this arch (3 s for the others; see AGENTS.md), so val numbers are
  not strictly comparable with `humnet-pretrain` / `humnet-deep-pretrain`. Compare on the
  recordings with `inspect-model` after fine-tuning.
- No wandb group (`train-detector` has no group flag; earlier runs went through `warmstart`).

## Log

- 11:30 datasets prepared: MLEnd whistles 1797 clips / 8.1 h, MLEnd hums 4804 / 21.6 h,
  MUSAN speech 60.5 h + music 42.6 h + noise 6.2 h, ESC-50 2000 clips, DEMAND 7 clips. ~10 GB.
- 11:35 launched on GPU 5 (`whistle-house:train` tmux window). Labelling: kept 1111/1140
  external whistle clips after QA; pYIN labelled 2931 hum clips in ~4 min.
- 11:41 training. Status file: device cuda, window_s 4.5. Train split: 1111 external + 6 mic
  whistle clips, 2931 external + 4 mic hum clips, interferers 23 mic / 386 speech / 593 music /
  2049 noise. Val: 122 + 1 whistle, 399 + 2 hum.
  wandb run `cnwtyubo` (project whistle-house).
- GPU memory: this run 12.2 GB; GPU 5 total 73.4 / 80 GB with the other user's job (61 GB).
  ~6.6 GB headroom: if their job grows (it reaches ~70 GB on other cards), one of the two can
  OOM. `train-forever.sh` resumes this run from the last validation if it dies.
- Epoch 1 (shared GPU): 444 s, f1 0.760 (whistle 0.815, hum 0.704). wandb `cnwtyubo`.
- 11:58 owner freed GPU 5; first attempt stopped and **restarted from scratch** (wandb
  `sp13n99j`). Diagnosis before the restart: GPU 5 at ~0% util for our process; the 8 data
  workers each ran ~12 threads (numpy/BLAS pools sized to the 112-core machine), together ~97
  cores, machine load 194. The bottleneck was example generation thrashing, not the GPU.
  Fix: BLAS pools at 1 thread per process, 24 workers.
- After the fix: epoch 1 in **78 s** (5.7x faster), GPU util 45-96% (mean ~65%), our CPU ~14
  cores (from ~97), machine load 194 -> 20. GPU memory 12.2 GB.
  Workers are now ~58% busy, i.e. waiting on the training loop: the main process (~90% of a
  core: targets in numpy, host->device copy, kernel launches) plus the GPU on a small model at
  batch 32 is the limit. Further speed would need a larger batch, AMP, `torch.compile` or
  targets on the GPU; all change the recipe vs `humnet-pretrain` / `humnet-deep-pretrain`, and
  at ~78 min for 60 epochs not worth it here. Batch size stays 32.
- Epoch 1 (restart): f1 0.780 (whistle 0.800, hum 0.760), FPR 4.5%.
- **Lesson for any training machine with many cores:** BLAS pools in the data workers must be
  one thread each. Now the default in code: `threads.limit_worker_threads` runs in every
  DataLoader worker and labelling worker (checked: pools 64 -> 1 threads per worker). The env
  vars are no longer needed. (The Mac has few cores, so it likely never showed there.)
- **Batch size, measured** (model step alone on GPU 5, synthetic 4.5 s windows = 140 frames x
  146 bins, while the run was training on the same GPU):

  | batch | step    | per example | numpy targets | peak mem |
  |-------|---------|-------------|---------------|----------|
  | 32    | 132 ms  | 4.14 ms     | 12 ms         | 4.2 GB   |
  | 64    | 235 ms  | 3.67 ms     | 18 ms         | 8.5 GB   |
  | 128   | 477 ms  | 3.73 ms     | 41 ms         | 16.9 GB  |
  | 256   | 978 ms  | 3.82 ms     | 91 ms         | 33.8 GB  |

  Step time is linear in batch: the GPU is already compute-bound at 32 (the model is small in
  parameters, 76k, but its activations cover every time-frequency cell through 12 blocks). A
  larger batch would gain ~10% examples/s at most and change the recipe, so it stays 32. If
  speed ever matters, the levers are bf16 autocast / `channels_last`, not batch size.

## Result: `humnet-long-pretrain` (batch 32) — done 13:20

60 epochs in 79.6 min of epoch time, one attempt, no crashes. Best checkpoint epoch 57 (val,
4.5 s windows):

| metric                | mean  | whistle | hum   |
|-----------------------|-------|---------|-------|
| F1                    | 0.921 | 0.916   | 0.926 |
| precision             | 0.928 | 0.918   | 0.937 |
| recall                | 0.914 | 0.914   | 0.915 |
| false positive rate   | 0.022 | 0.022   | 0.022 |
| pitch acc (1st)       | 0.958 | 0.984   | 0.932 |
| overtone rate         | 0.008 | 0.000   | 0.016 |

Files on lgn9: `~/projects/whistle-house/data/models/humnet-long-pretrain.pt` (345 KB,
sha256 `4285d439…0c15`) and `.json`. Next (on the Mac): fine-tune with `--init-from
humnet-long-pretrain` (command in the handoff / AGENTS.md), then `inspect-model` to compare
speech-as-hum with `humnet` / `humnet-deep` on held-out speech.

## Side experiment: `humnet-long-pretrain-b128` (running)

Question from the owner: does a larger batch give a better final model? Same command with
`--batch-size 128 --workers 32`, same 60 x 500 steps and LR 2e-3, so 4x the examples per
update and per epoch; only the batch differs. GPU 7 (free per gpu-tracker), started 12:24,
333 s/epoch -> ETA ~17:55. wandb `1m4m6038`. Uses the in-code worker thread limit (no env
vars): each worker <= 1 core (the ~127 OpenBLAS threads per worker exist but idle).
Epoch 1: f1 0.798 vs 0.780 for batch 32.

## Longer run: `humnet-long-pretrain-e240` — done 18:42

The batch-32 run was still improving at the end (val F1 0.912 @ 40, 0.919 @ 50, 0.921 @ 60,
train loss still falling; part of that is OneCycle's LR annealing). Fresh run, 240 epochs
instead of 60 (the OneCycle schedule spans the whole run, so a finished run cannot be
extended with `--resume`), everything else as `humnet-long-pretrain`. 240 x 500 x 32 = the
same number of examples as the b128 run (60 x 500 x 128) with 4x the updates, so the three
runs also answer the batch question:

| run                       | batch | epochs | examples | updates |
|---------------------------|-------|--------|----------|---------|
| humnet-long-pretrain      | 32    | 60     | 0.96 M   | 30 k    |
| humnet-long-pretrain-b128 | 128   | 60     | 3.84 M   | 30 k    |
| humnet-long-pretrain-e240 | 32    | 240    | 3.84 M   | 120 k   |

GPU 5 (free per gpu-tracker), started 13:27, 82 s/epoch -> ETA ~18:55. wandb `y4qe2uoa`.

### Result (e240)

Done 18:42 (315 min of epoch time, one attempt). Best epoch 181:

| run (best epoch)               | F1    | whistle F1 | hum F1 | FPR   | pitch acc | hum pitch acc |
|--------------------------------|-------|------------|--------|-------|-----------|---------------|
| humnet-long-pretrain (57/60)   | 0.921 | 0.916      | 0.926  | 0.022 | 0.958     | 0.932         |
| humnet-long-pretrain-e240 (181/240) | 0.928 | 0.923 | 0.933  | 0.021 | 0.964     | 0.940         |
| humnet-long-pretrain-b128 (31/60, unfinished) | 0.924 | 0.921 | 0.927 | 0.023 | 0.961 | 0.936 |

e240 val F1 along the way: 0.894 @ 30, 0.907 @ 60, 0.918 @ 90, 0.919 @ 120, 0.921 @ 150,
0.925 @ 180, 0.927 @ 210, 0.927 @ 240 (mean of the last 5 epochs 0.927). So 4x the training
buys +0.007 F1 on val, flat over the last ~40 epochs: it is near what this arch + data give
with this recipe. Files: `data/models/humnet-long-pretrain-e240.pt` (sha256 `7b5c96e4…b0c8`)
+ `.json`. wandb `y4qe2uoa`.

### b128 hang

`humnet-long-pretrain-b128` stopped making progress at 15:17 (wandb's last log: global step
15890, 390 steps into epoch 32) but did not exit: main process at 100% CPU in state R, GPU 7 at
100%, all 32 workers alive and idle (queues full), SIGINT ignored. Looks like a hung CUDA op
the main process spins on; no Xid or ECC errors visible (dmesg is not readable here; a stack
dump with py-spy was not run). Nothing caught it for 3.5 h because `train-forever.sh` only
retries on exit and my watcher only looked for an exit. 18:44: SIGTERM -> `train-forever.sh`
resumed from the epoch-31 state; ETA ~21:25. The watcher now also flags a stall (no status
update for 15 min).
Possible improvement: a step-progress watchdog in `ml/train.py` (or `train-forever.sh`) that
kills a run whose status/step counter has not moved for N minutes, so the retry covers hangs.
