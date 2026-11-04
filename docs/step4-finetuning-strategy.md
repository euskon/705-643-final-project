# Step 4: fine-tuning strategy

## Setup

* **Model:** the pretrained backbone (`encoder` = convolutional set-abstraction
  stages, `decoder` = feature propagation) from `runs/pretrain/<model>/backbone.pt`,
  plus a **fresh 2-class SegHead** (not climbable / climbable). The pretraining head
  (4 pseudolabel classes) is discarded.
* **Labels:** per point, on each sample's own cylinder only (`label_mode:
  center_only`). **Score per sample** = mean P(climbable) over its cylinder.
* **Data:** `data/finetune/<name>` from Step 3. Train/val = spatially
  separated crag groups (mostly RRG). Test = Horseshoe Canyon Ranch (hand
  `Climb` positives and `Nc` negatives, plus generated negatives) and the held-out
  test groups.

## Strategies (`cfgs/finetune/*.yaml`)

| strategy | frozen | LR (encoder / decoder / head) | when it should win |
|---|---|---|---|
| `linear_probe` | encoder + decoder (incl. BatchNorm stats) | – / – / 1e-3 | tiny datasets; also measures how linearly separable "climbable" is in the pretrained features |
| `decoder_head` | encoder | – / 1e-3 / 1e-3 | generic geometry is right, the per-point mapping isn't |
| `full_discriminative` | none | 5e-5 / 2.5e-4 / 5e-4 | enough data; standard choice |
| `gradual_unfreeze` | head only, then +decoder at epoch 5, then +encoder at epoch 10 | discriminative, as above | small data with a large domain gap; protects pretrained features early on |
| `scratch` | none, random init | 1e-3 | baseline: what pretraining buys |
| `s3dis_init` | none, S3DIS init (no Step 2) | discriminative | baseline: what *our* LiDAR pretraining buys over S3DIS |

The epoch counts are upper bounds. Early stopping on val loss (patience 10,
`min_delta` 0.001) decides the effective number of epochs, and `best.pt` is
the checkpoint at the best val loss.

## Protocol for choosing

1. `bash scripts/s4_sweep.sh` (PointNeXt-S backbone), then
   `BACKBONE=runs/pretrain/pointnet2 bash scripts/s4_sweep.sh`.
2. **Select on val only** (`s4_compare.py` sorts by val sample AUROC; look
   at `val_sample_auroc_hard` too, which uses positives vs `band` + hand `Nc` only,
   i.e. the cases that matter).
3. LR: for the winning strategy, re-run with `LRS="1e-3 3e-4 1e-4"` (or
   `optimizer.lr=` overrides). Keep the best by val.
4. Report **test** once for the chosen configuration: Horseshoe Canyon AUROC
   (real Climb vs real Nc) is the headline number. Per-source accuracies show
   whether the model only rejects easy talus or also hard rock.
5. Ablations from the same dataset, by config only:
   `dataset.exclude_sources=[far_cliff]`, `[aug_squash,aug_layback]`,
   `dataset.min_confidence=0.7`, `dataset.label_mode=with_background`.

Expectations: with a few hundred to ~2k positives, `gradual_unfreeze` or
`full_discriminative` should beat `linear_probe`, and both pretrained inits
should beat `scratch`. If `linear_probe` ≈ `full_discriminative`, the
pretraining features carry the signal. If `s3dis_init` ≈ ours, the
pseudolabel pretraining isn't adding much, so revisit the Step 1 classes.

## Metrics logged (W&B group `finetune`)

Per epoch on val: point CE loss (the checkpoint monitor), IoU per class,
sample AUROC / AP / balanced accuracy / F1, hard-negative AUROC, and accuracy
per sample source. `s4_eval_finetune.py` writes the same for val and test,
per area, plus `predictions_best.csv` (one score per sample, with lon/lat for maps).
