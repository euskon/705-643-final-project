# Plan: give the model useful point information (input channels)

Resolves **problem 3** of `docs/fable-investigation-project-issues.md`: the
model's feature channels hold only the point's position inside the block (x,
y, z and a copy of z) and nothing about the surface. The full explanation
and measurements are in `docs/fable-investigation-input-channels.md`.

## 0. Before you start

* **Order.** Do this after plan 1 (labelling, V4:
  `research-prompts/fable-investigation-project-issues-resolution-plan.md`)
  and plan 2 (neighbourhood scale:
  `research-prompts/fable-investigation-project-issues-resolution-plan2.md`).
  All three change `BlockDataset.__getitem__` and `make_item` in
  `crag/datasets.py`; build on their versions. The channel tests in step 10
  only mean something once the plan 1 shortcut is gone and plan 2's
  neighbourhoods are filled.
* If plans 1-2 are not done, steps 1-9 (code, data, tests) can still be done;
  stop before step 10 (training).
* **Environment:** `conda activate crag-pointnext`.
* **Run `pytest` first**, so any test that already fails is known.
* **Disk.** 38 GB was free on 2026-11-15; `data/pretrain` is 59 GB. Step 5
  rebuilds the pretraining blocks in place and adds about 15 GB. Run `df -h .`
  and stop if less than 25 GB is free; ask the user what to clear (for
  example archived runs, or old fine-tuning sets `data/finetune/v1`, `v2`).

## 1. The goal

Every point's feature channels describe **the point and its surface**, never
its position inside the block. They are computed **one way, in one place**,
from data that is measured the same way in every area.

**Channels to build** (all five stored; which ones are used is a config
choice):

| name | meaning | formula | range given to the model |
|---|---|---|---|
| `hag` | height above the local ground | z minus the lowest bare-earth ground height within 10 m (1 m grid from `crag/walls.py`) | metres ÷ 20, clipped to -0.1..3 |
| `vert` | verticality | the vertical part of the surface normal, as a positive number (1 = flat ground, 0 = vertical wall) | 0..1 |
| `planarity` | flat face vs line vs scatter | already in the stage files | 0..1 |
| `ret_pos` | return position | return number ÷ number of returns (1 = last return) | 0..1 |
| `single` | single return | 1 if the pulse came back once, else 0 | 0 or 1 |

`intensity` stays stored (as today) for the optional test in step 10.

**Default feature sets:**

| stage | `feature_keys` | why |
|---|---|---|
| pretraining | `hag,planarity` | the other channels define the pretraining "cliff" label, so they'd turn it into a lookup (doc section 7, option 1) |
| fine-tuning | `hag,vert,planarity,ret_pos,single` | the recommended set (doc section 8) |

The old `pos,heights` keeps working, so the reference run can be repeated.

## 2. Step 1: one definition of the channels

New module **`crag/channels.py`**. This is the only place channel values are
computed; everything else calls it.

```python
CHANNELS = ("hag", "vert", "planarity", "ret_pos", "single", "intensity")
STORED_DTYPE = np.float16

def point_channels(c: dict, idx: np.ndarray | None = None) -> np.ndarray:
    """(N, len(CHANNELS)) float16 for crop/tile dict ``c`` (optionally rows ``idx``)."""
```

* Input: a dict with the arrays `AreaCloud.crop` returns (step 3), or a tile
  with its stage arrays (pretraining).
* `hag`: `(c["hag"]) / 20`, clipped. NaN (no ground grid) -> 0 and logged.
* `vert`: `abs(normals[:, 2])`. Normals are already unit length; points with
  no normal (zeros) -> 1 (treated as flat) and counted.
* `planarity`: from the stage files; missing -> 0 and counted.
* `ret_pos`: `return_number / max(number_of_returns, 1)`, clipped to 0..1.
* `single`: `number_of_returns == 1`.
* `intensity`: same as today: divided by the tile's 99th percentile, clipped
  to 0..1. Keep the per-tile percentile, so pretraining and fine-tuning
  values match.
* Constant `CHANNEL_SCALE_VERSION = 1`, written next to the blocks (step 5),
  so a later change to a formula can't silently mix with old blocks.

## 3. Step 2: add height above ground to the stage files

`hag` needs the bare-earth grid and its 10 m neighbourhood. That is already
built per tile **with a 10 m border** in `stage_tile`
(`crag/pseudolabels.py`) for `wall_dist`.

* In `crag/walls.py`, add `point_height_above_ground(xyz, surface, cfg,
  ground, window_m=10.0)`: build the grid as `wall_grid` does, take
  `ndimage.minimum_filter(dem, size=2*window_m/res + 1)`, and return
  `z - local_min` at each point's cell. Reuse the grid rather than building
  it twice (make `wall_grid` return it, or compute both in one call).
* **Do not change the pseudolabel digest.** Changing `PseudoLabelConfig`
  would force recomputing every stage file and change the labels. Instead:
  * add a separate pass, `scripts/s1_add_channels.py --areas ...` (default:
    all processed areas, plus every fine-tuning area, including Horseshoe
    Canyon Ranch), which for each tile loads the tile with its border,
    computes `hag`, and writes `stage/<tile>.hag.npy` (float16, length `n`);
  * skip a tile whose file exists with the right length unless `--force`;
  * run it with `--workers 3` (19 GB of RAM; the pseudolabel run used 3
    workers).
* Check the result before going on, on one Red River Gorge tile and one
  Horseshoe tile:
  * ground points: `hag` median about 0 (within ±0.5 m);
  * cliff points: `hag` spread from about 0 to the wall's height;
  * vegetation: `hag` up to typical tree heights (10-35 m).

## 4. Step 3: expose the new arrays when cutting blocks

`crag/crop.py` `AreaCloud`:

* `_tile`: also load `planarity` from the stage file (if present) and
  `stage/<tile>.hag.npy` (if present, with the right length).
* `crop`: also return `planarity`, `hag`, `return_number` and
  `number_of_returns` (it already reads the last two, to compute `multi`).
  Missing arrays are filled with NaN or 0, like `wall_dist` today.

## 5. Step 4: store the channels in the blocks

### Fine-tuning blocks (`crag/finetune_data.build_blocks`)

* Replace the `inten` array with `point_channels(crop, idx)`: shape
  `(N, 6)`, float16.
* For augmented samples (`aug_squash`, `aug_layback`; off by default after
  plan 1), the channels come from the original points. `hag` and `vert` are
  then wrong for the reshaped wall, so **refuse** to build augmentations
  together with `CHANNEL_SCALE_VERSION >= 1` unless `--allow-stale-aug` is
  given.
* Write `<dataset>/channels.json`:
  `{"channels": [...CHANNELS], "version": CHANNEL_SCALE_VERSION}`.

### Pretraining blocks (`crag/pseudolabels.make_blocks`)

* Same change: `feat` becomes `(N, 6)` from `point_channels` (tile plus stage
  arrays, without the border points).
* Add `--rebuild-blocks` to `scripts/s1_pseudolabel.py`: reuse the existing
  stage files (labels unchanged) and rewrite only the blocks and the index,
  **one area at a time, in place**. Don't build a full second copy; the disk
  can't hold it.
* Write `data/pretrain/channels.json` the same way.
* Afterwards, the index files and label counts must match the old ones (same
  blocks, same points). Compare `index.csv` before and after for two areas.

### Reading old blocks

V3 and older blocks have a 1-D `feat` (intensity only). `BlockDataset`
detects that and treats it as `("intensity",)`. Asking for any other channel
on such a dataset raises a clear error ("rebuild with channels").

## 6. Step 5: the dataset and `make_item`

`crag/datasets.py`:

* `FEATURE_DIMS`: add `hag`, `vert`, `planarity`, `ret_pos`, `single` (1
  each). Keep `pos` (3), `heights` (1) and `intensity` (1).
* `BlockDataset.__init__`: read `channels.json` from `root`, and map each
  requested key to its column in `feat`.
* `__getitem__`: carry the whole `(N, C)` array through windowing (pretraining
  `_window` already concatenates whatever `feat` holds), thinning (plan 2) and
  resampling, so channels stay with their points. Then pass a dict of named
  channel columns to `make_item`.
* `make_item`: build `x` in the order of `feature_keys`. `pos` and `heights`
  still come from the augmented positions; every other key comes from the
  stored columns.
* **Augmentation:** all five channels are unaffected by spinning around the
  vertical axis, centring and jitter. Scaling (×0.9-1.1) changes `pos` but
  not `hag`. That is a small mismatch (up to 10%) and acceptable; note it in
  the docstring rather than adding code.
* Update the module docstring: what each channel means, and that the feature
  channels must never encode position inside the block.

## 7. Step 6: the model's input size and loading weights

* **Set `in_channels` from `feature_keys` automatically**, instead of
  hard-coding `in_channels: 4` in the model files:
  * `crag/pretrain.build_model_from`: set `cfg.model.encoder_args.in_channels
    = feature_dim(keys)` before building, and log it;
  * `crag/finetune.build_model`: the same, overriding the value copied from
    the backbone's `config.json`. Today it raises an error on a mismatch;
    change that to a log line: "first layer rebuilt: backbone had N
    channels, fine-tuning uses M".
  * Keep the model YAMLs' `in_channels`, with a comment that the code
    overrides it.
* **Weight loading:** `load_pretrained` already skips tensors whose shape
  differs and lists them. After a channel change, check the list holds
  **only the first layer**:
  * PointNeXt-S: `encoder.encoder.0.*` (the per-point first step);
  * PointNet++: the first grouping layer's first convolution (input =
    channels + 3 offsets).

  Add an assert in `crag/finetune.build_model`: if any mismatched tensor is
  outside those, stop with an error, since that would mean an architecture
  mismatch, not a channel change.
* PointNet++ MSG (`pointnet2-msg.yaml`) has two branches at the first stage;
  both first convolutions are expected in the list.

## 8. Step 7: Horseshoe waypoint scoring uses the same channels

`crag/gt_eval.score_waypoints` builds blocks itself. Today it passes
intensity only. Change it to `point_channels(crop, idx)` plus the run's
`feature_keys`, so it gives the model exactly what training did. Add an
assert that the channel list matches the checkpoint's config.

## 9. Step 8: two new checks in the leak test

Extend `scripts/s3_leak_test.py` (from plan 1) with a per-channel section.
For every stored channel, take the median over each sample's labelled
points, then report:

1. **Label check**: AUROC for climbable vs not climbable, on validation and
   on Horseshoe Climb vs Nc.
2. **Area check**: AUROC for Horseshoe vs Red River Gorge samples (all
   labels together).

Flag a channel when the area check is above 0.8 while its label check is
within 0.45-0.55. It identifies the survey without saying anything about
the rock. Run it on V3 first as a reference. `intensity` has label AUROC 0.49
there, and its medians differ by area (0.14 Horseshoe, 0.20 Red River Gorge,
up to 0.82 elsewhere). Whether it crosses 0.8 on the Horseshoe vs Red River
Gorge check hasn't been measured yet. Write `leak_test.json` with these
numbers.

A flag is a warning, not a stop: it tells us to watch that channel in step
10.

## 10. Step 9: configs

* `cfgs/finetune/default.yaml`:
  `dataset.feature_keys: hag,vert,planarity,ret_pos,single`.
* `cfgs/pretrain/default.yaml`: `dataset.feature_keys: hag,planarity`.
* Update the comments that say "4 channels -> the S3DIS checkpoints' first
  layer loads": it no longer does, by design.

## 11. Step 10: experiments

All on V4 with plan 2's scale settings, `scratch` strategy (no pretraining
needed), PointNeXt-S, **two seeds** (`seed=42`, `seed=7`). Archive nothing
new; these are new run names.

### 10a. Add one group at a time

| run name | `feature_keys` | question it answers |
|---|---|---|
| `chan-a-posheights` | `pos,heights` | reference (today's channels) |
| `chan-b-hag` | `hag` | does removing position alone change anything? |
| `chan-c-shape` | `hag,vert,planarity` | does surface shape help? |
| `chan-d-returns` | `hag,vert,planarity,ret_pos,single` | do the vegetation flags help? (proposed default) |
| `chan-e-intensity` | d + `intensity_rank` | does intensity add anything once it can't identify the area? |
| `chan-f-walldist` | d + `wall_dist` | fine-tuning only: does the wall-distance channel help? |

```bash
python scripts/s4_finetune.py --strategy scratch --run-name chan-d-returns-s42 \
  init.arch=cfgs/pretrain/pointnext-s.yaml dataset.feature_keys=hag,vert,planarity,ret_pos,single seed=42
python scripts/s4_eval_finetune.py --run runs/finetune/chan-d-returns-s42
```

* `intensity_rank` (run e only): within each block, replace intensity by its
  rank ÷ number of points (0..1), computed in `make_item` before resampling.
  That removes the per-survey brightness. Add it as a derived key in
  `FEATURE_DIMS`; it doesn't need to be stored.
* `wall_dist` (run f only): add it to `CHANNELS` as metres ÷ 10, clipped to
  0..2. It's already in the stage files, so only the fine-tuning blocks
  change. It is never used in pretraining.
* Runs e and f only start after a-d are done.

### 10b. How to decide

Compare means over the two seeds. Horseshoe has about 180 hand points, so
differences under about 0.05 are noise.

* **Keep a group** if Horseshoe Climb vs Nc AUROC rises by 0.05 or more, or
  rises at all while the gap above plan 1's plain-geometry baseline grows,
  and validation does not fall by more than 0.02.
* **Drop a group** if it helps validation but not Horseshoe, especially if
  the step 8 area check flagged it. That is the sign of an area shortcut.
* **b vs a:** if `hag` alone does as well as `pos,heights`, position wasn't
  needed. That's the expected result: it confirms the model can't use the
  block position any more, at no cost.
* If **nothing beats `chan-b-hag`** on Horseshoe, the channels aren't the
  limit. Write that down and go to problem 4 (Horseshoe points) and more
  real negatives, rather than adding more channels.

### 10c. Pretraining with the new channels

Only after 10b has chosen the fine-tuning set:

1. Rebuild the pretraining blocks (step 4) if not done yet.
2. Pretrain both models with `feature_keys: hag,planarity` and plan 2's
   settings. New run names, `runs/pretrain/pointnext-s-ch1` and
   `runs/pretrain/pointnet2-ch1`, so nothing resumes an old run.
3. Check pretraining's own results (`scripts/s2_eval_pretrain.py`): cliff
   IoU against V3 (0.47-0.48 validation, 0.56-0.57 test). It may fall a
   little, since the model no longer gets absolute height; a large fall
   (more than 0.05) means `hag` isn't working as intended, so check step 2.
4. Fine-tune from these backbones with the chosen channels (`decoder_head`,
   `full_discriminative`, `gradual_unfreeze`), and compare against
   `scratch` with the same channels.
5. **Decision:** if no pretrained run beats `scratch` on Horseshoe Climb vs
   Nc by 0.05 or more, record that pretraining doesn't help with these
   channels. Keep fine-tuning from scratch for now, and note "change the
   pretraining task" (doc section 7, option 3) as future work.

### 10d. Write up

Add a "channels" section to `docs/results/v4/README.md`:

* the step 8 table (label and area check per channel);
* the 10a table, both seeds and the mean;
* the pretraining IoU against V3;
* pretrained vs scratch.

## 12. Tests to add

| file | test | checks |
|---|---|---|
| `tests/test_labels_negatives_crossref.py` | `test_hag_on_synthetic_wall` | a synthetic 15 m wall on flat ground: ground points `hag` ≈ 0, the wall top ≈ 15 m (÷ 20 in the stored value), trees above the wall top higher still |
| same | `test_point_channels_ranges` | every channel within its range; `vert` ≈ 1 on flat ground and ≈ 0 on the wall; `single` and `ret_pos` match hand-made return numbers |
| same | `test_crop_returns_channel_arrays` | `AreaCloud.crop` returns `hag`, `planarity`, return arrays, with NaN when the stage files lack them |
| `tests/test_training.py` | `test_feature_keys_select_columns` | `make_item` places each requested channel in the right column, in `feature_keys` order |
| same | `test_channels_follow_points` | after windowing, thinning and resampling, each point keeps its own channel values (build a block whose channels equal the point index) |
| same | `test_no_position_in_features` | with the default fine-tuning keys, shifting the whole block in x, y or z leaves `x` unchanged |
| same | `test_old_blocks_read_as_intensity` | a 1-D `feat` reads as intensity; asking for `hag` raises the "rebuild" error |
| same | `test_in_channels_auto` | the model is built with `in_channels = feature_dim(feature_keys)`; loading a 4-channel backbone into a 5-channel model reports only the first-layer tensors |
| same | `test_gt_eval_uses_run_channels` | `score_waypoints` gives the model the same channel count as the checkpoint |

Run the whole suite after each step.

## 13. Files that change

| file | change |
|---|---|
| `crag/channels.py` | new: `CHANNELS`, `point_channels`, version constant |
| `crag/walls.py` | `point_height_above_ground` (sharing the grid with `wall_grid`) |
| `scripts/s1_add_channels.py` | new: writes `stage/<tile>.hag.npy` |
| `crag/crop.py` | load `planarity` and `hag`; return them and the return arrays from `crop` |
| `crag/finetune_data.py` | store `point_channels` in fine-tuning blocks; write `channels.json`; refuse stale augmentations |
| `crag/pseudolabels.py` | store `point_channels` in pretraining blocks; `channels.json` |
| `scripts/s1_pseudolabel.py` | `--rebuild-blocks` (reuse the stage files, rewrite blocks in place) |
| `crag/datasets.py` | new keys; read `channels.json`; carry `(N, C)` through windowing, thinning and resampling; `make_item` by name; `intensity_rank`; docstring |
| `crag/pretrain.py`, `crag/finetune.py` | `in_channels` from `feature_keys`; first-layer-only mismatch assert |
| `crag/gt_eval.py` | use `point_channels` and the run's keys |
| `scripts/s3_leak_test.py` | per-channel label and area checks |
| `cfgs/pretrain/default.yaml`, `cfgs/finetune/default.yaml` | new `feature_keys`; updated comments |
| `cfgs/pretrain/pointnext-s.yaml`, `pointnet2.yaml`, `pointnet2-msg.yaml` | comment: `in_channels` is set by the code |
| `docs/pipeline.md` | the channel pass (`s1_add_channels.py`) and `--rebuild-blocks` in the runbook |
| `docs/results/v4/README.md` | channels section |
| `tests/...` | the tests above |

## 14. Risks

| risk | what we do |
|---|---|
| Not enough disk for the pretraining rebuild | check first (section 0); rebuild one area at a time in place; ask before deleting anything |
| `hag` wrong where the provider's ground points are sparse (under dense trees, or tiles that fall back to surface returns) | the step 2 checks; count, per area, the share of points whose 10 m window had no ground cell |
| The model leans on `ret_pos` / `single` as a survey identifier (sensors record different numbers of returns) | the step 8 area check; drop the group if it flags and doesn't help Horseshoe |
| Pretraining gets weaker without absolute height | expected; measured in 10c. Fine-tuning is what counts |
| A typo in `feature_keys` silently uses the wrong channels | `make_item` raises on an unknown key (as today); the run's `config.json` records the keys; `gt_eval` asserts they match |
| Old V3 blocks used by accident with the new keys | the "rebuild with channels" error from step 4 |

## 15. Decisions for the user (defaults used unless you say otherwise)

| decision | default | other options |
|---|---|---|
| `hag` window | lowest ground within 10 m | 15 m (taller, more leaning walls) |
| pretraining channels | `hag,planarity` | the full fine-tuning set (pretraining becomes easy; doc section 7, option 2) |
| intensity | stored, used only in run e | drop it entirely |
| full surface direction (two horizontal parts) | not in this plan | a later plan: needs the rotation augmentation to spin normals, and outward-oriented normals so overhangs differ from slabs |
| aerial photo colour | not in this plan | a later plan, after the geometry channels prove useful |
