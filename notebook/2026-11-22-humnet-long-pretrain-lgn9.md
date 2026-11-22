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

- **Shared GPU.** No GPU was free (another user's 8-GPU job, 100% util, 52-70 GB of 80 GB per
  card). With the owner's OK this run shares a busy GPU. Effects: wall-clock time is longer
  than an idle H100 would give and not comparable as a speed number; there is some OOM risk
  for either job if the other one's memory peaks. Nothing that changes what is learned.
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
