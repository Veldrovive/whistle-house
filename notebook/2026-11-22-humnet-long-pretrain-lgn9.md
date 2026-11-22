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
- **Lesson for any training machine with many cores:** run training with
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Worth making the default in
  the DataLoader worker init (the Mac has few cores, so it likely never showed there).
