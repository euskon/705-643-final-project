# Plan: fix neighbourhood scale (V4 training run)

Resolves **problem 2** of `docs/fable-investigation-project-issues.md`: the
model's neighbourhoods are nearly empty, so it barely sees local shape. The
full explanation and measurements are in
`docs/fable-investigation-neighbourhood-scale.md`.

## 0. Before you start

* **Order.** Do this after the problem 1 fix
  (`research-prompts/fable-investigation-project-issues-resolution-plan.md`,
  fine-tuning set V4). Both plans change `BlockDataset.__getitem__` in
  `crag/datasets.py`: plan 1 keeps every labelled point exactly once when
  resampling, and this plan changes the point count, point order and
  thinning. Build on plan 1's version of that function; don't write a second
  copy. If plan 1 is not done yet, do steps 1-5 here (code, configs, check
  script, tests) and stop before step 6 (training).
* **Environment.** `conda activate crag-pointnext`. The check script in step
  5 needs the GPU (it uses the model's own CUDA operations).
* **Run `pytest` before changing anything**, so any test that already fails
  is known.

## 1. The fix in detail

### Step 1: make the first radius about 2 m

Change `coord_scale` from **0.16 to 0.05** in both `cfgs/pretrain/default.yaml`
and `cfgs/finetune/default.yaml`. (They must match: training and
evaluation build blocks from the same setting.) The radius stays 0.1 model
units, so the model files and the downloaded weights still fit. On the ground
the stages become:

| stage | today | new |
|---|---|---|
| 1 | 0.625 m | 2 m |
| 2 | 1.25 m | 4 m |
| 3 | 2.5 m | 8 m |
| 4 | 5 m | 16 m (most of the block) |

Why 2 m and not 1.5 m: at 1.5 m the first stage still gets only about 9 real
points; at 2 m it gets about 18, and every later stage is full.

**What we give up:** no stage looks at anything smaller than 2 m. There is
little to lose: with points 0.3-0.5 m apart, the data holds almost no detail
under 1 m anyway. Fine detail can come in as **per-point input features**
instead (the surface direction we already compute from each point's close
neighbours: problem 3). The two fixes work together: the features describe
the surface right at each point, and the stages describe the shape around
it.

**Every place the scale is read** (all must pick up the new value from the
config; none may keep its own copy):

| place | how it gets the scale |
|---|---|
| `crag/datasets.py` `BlockSpec` (default `0.16`) | change the default to `0.05` too, so code that builds a `BlockSpec` without a config doesn't silently use the old value |
| `crag/pretrain.py` `block_spec()` (fallback `0.16`) | change the fallback to `0.05` |
| `crag/gt_eval.py` (Horseshoe waypoint scoring) | uses `spec.coord_scale`; no change, but check it is called with the run's own spec |
| `scripts/s2_eval_pretrain.py`, `scripts/s4_eval_finetune.py` | rebuild the config from the checkpoint, so they use the scale the run was trained with; no change needed, but confirm in the tests |
| `scripts/s2_pretrain.py` docstring | the example uses `dataset.num_points=16384`: update it |

**Why the model needs no other changes.** The radius, the jitter limits and
the stage offsets are all in model units. PointNeXt divides each
neighbour's offset by the radius, and the decoder joins stages through
nearest neighbours, which don't depend on scale. Only the number of metres
per model unit changes. Height values get smaller (a 40 m wall is 2.0
model units instead of 6.4); normalisation in the first layer absorbs that.

### Step 2: keep the augmentation sizes in metres

Some augmentation settings are in model units, so a new scale changes their
size on the ground:

| setting | today (0.16) | unchanged at 0.05 | set to |
|---|---|---|---|
| `jitter_sigma` | 0.005 = 3 cm | 10 cm | **0.0015** (3 cm) |
| `jitter_clip` | 0.02 = 12.5 cm | 40 cm | **0.006** (12 cm) |

`scale: [0.9, 1.1]` is a ratio and does not change. Rotation around the
vertical axis and the centring step don't depend on scale either.

### Step 3: 4,096 points per block

Set `num_points: 4096` in both configs. The simulation at 4,096 with a 2 m
radius gives 15 / 24 / 27 / 24 real points per stage (Horseshoe 15 / 32 / 32
/ 29), so still well filled.

* Blocks with more than 4,096 points lose some at random; that acts as extra
  augmentation. About a quarter of fine-tuning blocks and about 45% of
  pretraining blocks are affected (pretraining blocks: median 3,900 points,
  5% over 10,600).
* Smaller blocks get far fewer copies.
* Memory halves, so `batch_size` can double with `grad_accum_steps` halved,
  keeping the same effective batch:

| config | today | new |
|---|---|---|
| pretrain | batch 8 × accumulate 4 = 32 | batch 16 × accumulate 2 = 32 |
| fine-tune | batch 8 × accumulate 2 = 16 | batch 16 × accumulate 1 = 16 |

Also set `val_batch_size: 16` in the pretrain config. If the GPU runs out of
memory (10 GB RTX 3080), go back to batch 8 and the old accumulation; the
results should be the same.

### Step 3b: shuffle the point order (new)

The neighbour search does **not** pick the nearest 32 points. It walks
through the points in stored order and takes the **first 32** it finds
inside the radius (`ball_query_kernel_fast` in
`third_party/PointNeXt/openpoints/cpp/`). Today this barely matters, because
almost no ball holds 32 points. With the 2 m radius, stages 2-4 are full, so
the order decides *which* neighbours are kept.

Today `BlockDataset.__getitem__` sorts the chosen points
(`np.sort(rng.choice(...))`), so they stay in the stored order, which follows
the tile files and roughly the aircraft's scan lines. A full ball would keep
mostly points from one scan line or one corner of the ball, not a fair
spread.

**Change:** replace the sort with a random order (`rng.permutation`). In
training the order is new each time; in evaluation it is fixed per block
(the generator is already seeded per item), so validation stays repeatable.
Apply the same change to `crag/gt_eval.py`, which picks points with
`rng.choice` and doesn't sort them, but keep it explicit with
`rng.permutation`. Labelled-point masks (`center`, the per-point weight from
plan 1) must be reordered together with the points.

### Step 4 (optional): make training blocks as sparse as Horseshoe sometimes

Horseshoe is about a third sparser than Red River Gorge. During training,
drop a random 0-40% of a block's points (never the labelled ones) half the
time. The model then learns to work with sparse blocks and doesn't treat
density as a clue about the area.

**Implementation:**

* New config key, default off:
  `dataset.thin: {prob: 0.5, max_frac: 0.4}`.
* In `BlockDataset.__getitem__`, **training only**, after the block or window
  is read and **before** resampling to `num_points`: with probability `prob`,
  draw `f` uniformly from 0 to `max_frac` and drop a random `f` share of the
  points.
* Fine-tuning: never drop points coded 0 or 1 (the labelled ones).
* Pretraining: every point is labelled, so drop at random; keep at least 64
  points (the same floor `_window` already uses).
* Evaluation never thins.
* Turn it on only after the A/B test in step 6 has run without it, so its
  effect can be measured separately.

### Step 5: a permanent check

New script `scripts/check_neighbourhoods.py`. For a dataset and a config, it
reports the median **real** points per stage, overall and per area. Target:
at least 12 in stage 1 and 24 in later stages. Run it whenever the scale,
the point count or the thinning changes.

**How it works** (use the model's own steps, not a copy, so the answer is
exact):

1. Build `BlockDataset` from the config (validation mode, so it is
   repeatable), for `--dataset pretrain` or `--dataset finetune:v4`.
2. Take `--n-blocks` blocks (default 200), stratified by area.
3. For each block, use the OpenPoints CUDA operations the model uses:
   `furthest_point_sample` for the centres and `ball_query` (via the
   `QueryAndGroup` grouper) for the neighbours. Use the radius, `nsample` and
   strides from the model config (`model.encoder_args`), doubling the radius
   each stage exactly as the encoder does.
4. Count **distinct** point indices in each ball (copies and padding
   removed), centre included.
5. Print a table (stage × median, 10th percentile) overall and per area, and
   write `<out>/neighbourhoods.json`.
6. Exit with an error if any stage's overall median is under its target.
   `--min-stage1 12 --min-later 24` sets the targets.

```bash
python scripts/check_neighbourhoods.py --cfg cfgs/pretrain/pointnext-s.yaml --dataset pretrain
python scripts/check_neighbourhoods.py --cfg cfgs/finetune/default.yaml --arch cfgs/pretrain/pointnext-s.yaml --dataset finetune:v4
python scripts/check_neighbourhoods.py ... dataset.coord_scale=0.16 dataset.num_points=8192   # old settings, as a reference
```

Expected results (from the simulation): old settings about 2 / 3 / 3 / 4;
new settings about 15 / 24 / 27 / 24. Run it for both PointNeXt-S and
PointNet++. PointNet++ has 4 grouping stages starting at radius 0.1; its MSG
variant has two radii per stage, so report both.

### Step 5b: guard against mixing scales (new)

A backbone trained at one scale gives wrong results when fine-tuned or
evaluated at another, and nothing would warn us. Add a check:

* Every checkpoint already stores its config (`cfg` in `best.pt`,
  `last.pt` and `backbone.pt`).
* In `crag/finetune.py` `build_model`, read `dataset.coord_scale` from the
  backbone checkpoint's config. If it differs from the fine-tuning config's
  value by more than 1%, stop with a clear error naming both values.
* Exception: the S3DIS checkpoints (`s3dis_init`) have no `dataset.coord_scale`
  of ours. Skip the check for them and log a warning.
* Same check in `scripts/s2_eval_pretrain.py` and
  `scripts/s4_eval_finetune.py` if they're given a config override.

### Step 6: retrain

**6a. Archive the old runs first.** Both training scripts use `resume: auto`:
if `runs/pretrain/pointnext-s/last.pt` exists, a "new" run **continues the
old 0.16-scale model** instead of starting fresh. Before any training:

```bash
mkdir -p runs/archive-v3-scale016
mv runs/pretrain/pointnext-s runs/pretrain/pointnet2 runs/archive-v3-scale016/
mv runs/finetune runs/archive-v3-scale016/finetune
```

(Ask before moving if plan 1 already archived them elsewhere; don't move
anything twice.) The guard from step 5b catches a mix-up if one slips
through.

**6b. Test cheaply first: the `scratch` A/B test.** The `scratch` strategy
doesn't need pretraining. On fine-tuning set V4, train four PointNeXt-S
scratch runs:

| run name | `coord_scale` | `num_points` | what it isolates |
|---|---|---|---|
| `scale-ab-cs016-n8192` | 0.16 | 8192 | today's settings |
| `scale-ab-cs005-n8192` | 0.05 | 8192 | the radius alone |
| `scale-ab-cs005-n4096` | 0.05 | 4096 | radius + point count (the proposed setting) |
| `scale-ab-cs0067-n4096` | 0.067 | 4096 | 1.5 m first radius, to check 2 m is the better choice |

Run each with **two seeds** (`seed=42` and `seed=7`). Horseshoe has only
about 180 hand points, so a single run's difference under about 0.05 is
noise. Keep the jitter settings in metres for each scale (step 2:
`jitter_sigma = 0.03 × coord_scale`, `jitter_clip = 0.12 × coord_scale`).
Thinning stays off.

```bash
python scripts/s4_finetune.py --strategy scratch --run-name scale-ab-cs005-n4096-s42 \
  init.arch=cfgs/pretrain/pointnext-s.yaml dataset.coord_scale=0.05 dataset.num_points=4096 \
  datatransforms.kwargs.jitter_sigma=0.0015 datatransforms.kwargs.jitter_clip=0.006 seed=42
python scripts/s4_eval_finetune.py --run runs/finetune/scale-ab-cs005-n4096-s42
```

**Pass condition**, for the proposed setting against today's, averaged over
the two seeds:

* validation sample AUROC higher by 0.02 or more, **and**
* Horseshoe Climb vs Nc AUROC not lower, **and**
* the gap above plan 1's group B baseline (plain wall geometry) larger.

If it passes, continue with 6c. If it doesn't, stop and find out why before
spending hours on pretraining. Things to check: the step 5 check numbers,
whether training loss falls more slowly (the learning rate may need
retuning, e.g. try `optimizer.lr=5e-4`), and how predictions differ on a few
blocks.

**6c. Optional thinning test.** Repeat the winning setting with
`dataset.thin` on. Keep it only if Horseshoe Climb vs Nc improves without
validation falling by more than 0.02.

**6d. Retrain pretraining** for both models at the winning setting (about
2-3 h each; faster with 4,096 points). Check pretraining's own results
(`scripts/s2_eval_pretrain.py`): cliff IoU on validation and test areas
should be at least V3's (0.47-0.48 and 0.56-0.57). Wider neighbourhoods
should help it see walls.

**6e. Fine-tuning sweep** on V4 with the new backbones, the same strategies
as plan 1 section 9.2.

**6f. Write up** in `docs/results/v4/README.md`, in its own section: the
neighbourhood check table (old and new), the A/B table with both seeds, the
thinning result, pretraining IoU against V3, and the sweep.

## 2. Tests to add (`tests/test_training.py`)

| test | checks |
|---|---|
| `test_configs_share_scale` | `cfgs/pretrain/default.yaml` and `cfgs/finetune/default.yaml` have the same `coord_scale`, `num_points` and jitter settings |
| `test_jitter_in_metres` | `jitter_sigma / coord_scale` is about 0.03 m in both configs (catches a scale change that forgets the jitter) |
| `test_point_order_shuffled` | a training item's points are not in sorted index order; two validation reads of the same item are identical |
| `test_order_keeps_masks_aligned` | after shuffling, the `center` mask and labels still match their points |
| `test_thinning` | with `thin` on: about the right share is dropped, labelled points are never dropped, evaluation is never thinned, at least 64 points stay |
| `test_scale_guard` | fine-tuning with a backbone saved at 0.16 and a config at 0.05 raises an error; an S3DIS checkpoint only warns |
| `test_check_neighbourhoods_synthetic` | on a synthetic flat grid of points 0.3 m apart, the script's stage 1 count matches what geometry predicts (π × 2² ÷ 0.3² ≈ 140, capped at 32) (GPU only; skipped without CUDA) |

Run the whole suite after each step.

## 3. Files that change

| file | change |
|---|---|
| `cfgs/pretrain/default.yaml`, `cfgs/finetune/default.yaml` | `coord_scale: 0.05`, `num_points: 4096`, `jitter_sigma: 0.0015`, `jitter_clip: 0.006`, batch size / accumulation, `val_batch_size`, `thin` (off); update the comments explaining the scale |
| `cfgs/pretrain/pointnext-s.yaml`, `pointnet2.yaml`, `pointnet2-msg.yaml` | update the comments (0.1 / 0.05 = 2 m on the ground) |
| `crag/datasets.py` | update the docstring on why the scale is what it is (measured spacing, not voxel size); `BlockSpec` default 0.05; random point order; optional thinning |
| `crag/pretrain.py` | `block_spec` fallback 0.05; pass `thin` to the dataset |
| `crag/finetune.py` | scale guard; pass `thin` to the dataset |
| `crag/gt_eval.py` | explicit random point order |
| `scripts/check_neighbourhoods.py` | new |
| `scripts/s2_pretrain.py` | docstring example |
| `tests/test_training.py` | new tests |
| `docs/pipeline.md` | the new settings and the check script in the runbook |
| `docs/results/v4/README.md` | results section |

## 4. Risks

| risk | what we do |
|---|---|
| The A/B test shows no gain | stop before pretraining; see 6b. A likely reason is that problem 3 (inputs with no surface information) still limits it |
| Learning rate tuned for the old scale | if training loss falls slowly, retry the A/B winner at half the learning rate |
| Out of GPU memory at batch 16 | back to batch 8 and the old accumulation |
| A 16 m stage-4 radius covers nearly the whole block, so it sees the block's edges | expected and fine: it gives the whole-block view; the labelled points are near the centre, away from the edges |
| An old 0.16 backbone used by mistake | archive step 6a plus the guard in step 5b |
| Thinning hides a real density difference between areas | measured separately in 6c, and kept only if it helps Horseshoe |
