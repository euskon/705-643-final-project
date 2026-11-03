# Multi-area pipeline: runbook

Four steps, each a handful of plain Python scripts in `scripts/` (or one-word
`make` targets). Every script is incremental (re-running skips finished
work), deterministic (fixed seeds, stable hashes, cached downloads), and
prints what it did. W&B tracking reads the API key from `.keys`
(gitignored), never prints it, and every script takes
`--wandb online|offline|disabled`.

```
Step 1  pretraining data     rank areas -> registry -> LiDAR tiles -> pseudolabelled blocks
Step 2  pretraining          OpenPoints PointNeXt-S / PointNet++ (S3DIS init) on pseudolabels
Step 3  fine-tuning data     route-level GPS (OSM/openclimbing + hand GT) x Mountain Project -> verified positives + hard negatives
Step 4  fine-tuning          frozen/unfrozen strategies, fresh 2-class head, compared on held-out crags
```

> Step numbering: your notes had two "Step 3"s. The fine-tuning data has
> to exist before fine-tuning can run, so route GPS + hard negatives are
> Step 3 here and the fine-tuning strategy is Step 4.

## 0. Environment (once)

```bash
bash scripts/setup_env.sh        # conda env `crag-pointnext`, compiles OpenPoints' CUDA ops,
conda activate crag-pointnext    # downloads the two S3DIS checkpoints
make test                        # 18 tests incl. GPU model checks (~10 s)
```

PointNeXt is a git submodule (`third_party/PointNeXt`, with its nested
`openpoints`). Torch 2.6 + cu124 matches the local CUDA 12.4 toolkit used to
compile `pointnet2_batch`.

## Step 1: pseudolabelled pretraining dataset

| # | command | what it does | output |
|---|---|---|---|
| 1a | `python scripts/s1_rank_areas.py` | MP page views for each candidate in `registry/mp_candidates.csv` (1 cached request/area, 60 s apart) | `registry/mp_ranked.csv` |
| 1b | `python scripts/s1_build_registry.py` | top 50 + Horseshoe Canyon (holdout) -> crags (MP sub-area pins with routes), bbox, UTM zone, USGS EPT dataset(s), split | `registry/areas.json`, `data/mountain_project/route_finder_<id>.csv` |
| 1c | `python scripts/s1_download_lidar.py --dry-run` then without `--dry-run` | 200 m UTM tiles within 150 m of crags, from USGS 3DEP EPT (resumable, worker processes) | `data/lidar/<slug>/E…_N….npz` |
| 1d | `python scripts/s1_pseudolabel.py` | normals -> 5 per-point classes (existing heuristics) -> 20 m blocks voxelised at 0.25 m | `data/pretrain/<slug>/`, `data/pretrain/summary.csv` |

* **Adding an area:** append `{"slug": "...", "name": "...", "mp_id": 123}`
  to `registry/areas.json`, then re-run 1b-1d. They only process what is missing.
* **Registry schema:** `docs/registry-schema.md`.
* **Classes:** ground, vegetation, steep_terrain, cliff, climbable (cliff
  inside a Mountain Project route cylinder), plus ignore (-1). See the docstring of
  `crag/pseudolabels.py`. MP pins are used here only, never as route locations.
* **Splits:** whole areas (registry `split`: train/val/test, deterministic
  hash; `holdout` = never used, e.g. Horseshoe Canyon). Blocks inside
  `registry/heldout_regions.json` (fine-tuning test crags, written by 3c)
  are dropped too.
* **Scale knobs:** `--max-tiles-per-area` (default 150, keeps the most
  route-dense tiles), `CRAG_DATA_DIR=/big/disk` to move data. Horseshoe
  Canyon: 49 tiles, 146 MB raw, ~1 min download, ~1 min pseudolabels.

## Step 2: pretraining

> **Order:** run the Step 3 *data* scripts (3a-3c, minutes to an hour) before
> launching pretraining. 3c writes `registry/heldout_regions.json`, and the
> pretraining loader drops every block inside those fine-tuning test crags, so
> the multi-day pretraining never sees them, not even with pseudolabels.

```bash
python scripts/s2_pretrain.py --cfg cfgs/pretrain/pointnext-s.yaml     # backbone 1 (S3DIS PointNeXt-S)
python scripts/s2_pretrain.py --cfg cfgs/pretrain/pointnet2.yaml       # backbone 2 (S3DIS PointNet++)
python scripts/s2_pretrain.py --cfg cfgs/pretrain/pointnext-s.yaml --smoke   # 6-step pipeline check
python scripts/s2_eval_pretrain.py --run runs/pretrain/pointnext-s     # best checkpoint: val/test areas + hand GT
```

* Config = `cfgs/pretrain/default.yaml` + the model file + `key=value` overrides
  (e.g. `batch_size=4 grad_accum_steps=8 dataset.num_points=16384`).
* RTX 3080 settings: 8 blocks x 8192 points, grad accumulation 4 (effective 32), AMP.
  Measured peak memory is only 0.45 GB (PointNeXt-S) / 0.82 GB (PointNet++)
  per step, so `batch_size=32 grad_accum_steps=1` or 16384 points also fit.
* **Checkpoints:** `best.pt` + `backbone.pt` (encoder+decoder only) whenever val loss
  improves by more than `checkpoint.min_delta` (0.001); `last.pt` every epoch
  and on Ctrl+C. Re-running the same command resumes (`resume=none` to restart).
* **Coordinates:** fixed scale `0.04 / 0.25 = 0.16` instead of per-block unit
  sphere, so the S3DIS filters see the same neighbourhood density and metric
  size stays meaningful (see the docstring of `crag/datasets.py`).
* The eval script also scores the 186 hand-gathered Horseshoe waypoints
  (Climb vs Nc AUROC of P(climbable)) next to the steepness heuristic.

## Step 3: route-level GPS and hard negatives

Proposal, findings and verification: `docs/step3-routes-and-hard-negatives.md`.

```bash
python scripts/s3_survey_openclimbing.py --resolve-mp   # 3a: best-documented US areas (route-level GPS)
python scripts/s3_collect_routes.py                      # 3b: OSM routes per area x MP routes + hand GT
python scripts/s3_build_finetune_set.py --name v1        # 3c: snap, negatives, verification, blocks, splits
```

Outputs: `data/routes/verified_routes.csv`, `data/finetune/v1/` (with `index.csv`,
`report.json`, `samples.geojson` to open in QGIS/Google Earth, and
`review_negatives.csv`, a sheet for manual checking).

## Step 4: fine-tuning strategy

Rationale and protocol: `docs/step4-finetuning-strategy.md`.

```bash
python scripts/s4_finetune.py --strategy linear_probe          # or decoder_head, full_discriminative, gradual_unfreeze, scratch, s3dis_init
python scripts/s4_eval_finetune.py --run runs/finetune/linear_probe
bash scripts/s4_sweep.sh                                       # all strategies, then s4_compare.py
BACKBONE=runs/pretrain/pointnet2 bash scripts/s4_sweep.sh      # same for the second backbone
```

## W&B

Project `crag-detection` (`--wandb-project` to change). Job types: `dataset`
/ `data` (steps 1 and 3 summaries and tables), `pretrain`, `finetune`, `eval`,
`analysis`. Runs resume under the same id when training resumes. The best
checkpoint is uploaded as a model artifact at the end of a run.

## Where the old proof of concept fits

The single-area proof of concept lives, self-contained, in
`proof-of-concept/` (notebook, PointNet++ model/trainer, inference,
`eval_pseudolabels.py`, its docs and outputs). The parts this pipeline reuses
were copied into `crag/`, with the same behaviour and defaults:

| proof of concept | pipeline | change |
|---|---|---|
| `labeling.py` (normals, `PointCloud`, route cylinders, negatives) + `config.LabelConfig` | `crag/labeling.py` | `build_positives` takes the area's CRS converters; patch/sample code dropped |
| `groundtruth.py` + `config.GroundTruthConfig` | `crag/groundtruth.py` | `snap_one` (was `_snap_one`); waypoints projected to the area's UTM zone |
| `usgs_ept.read_ept` | `crag.tiles.read_ept` | output CRS required |
