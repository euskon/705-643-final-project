# Crag detection from USGS LiDAR

**Multi-area pipeline (current):** pseudolabelled pretraining on the 50 most
popular Mountain Project areas, then fine-tuning on route-level GPS
(OpenStreetMap / openclimbing.org + hand-gathered ground truth) with verified
hard negatives, using OpenPoints (PointNeXt-S / PointNet++). Runbook:
**[docs/pipeline.md](docs/pipeline.md)**.

```bash
bash scripts/setup_env.sh && conda activate crag-pointnext && make test
python scripts/s1_rank_areas.py && python scripts/s1_build_registry.py      # step 1: registry
python scripts/s1_download_lidar.py && python scripts/s1_pseudolabel.py     # step 1: data
python scripts/s2_pretrain.py --cfg cfgs/pretrain/pointnext-s.yaml          # step 2
python scripts/s2_eval_pretrain.py --run runs/pretrain/pointnext-s
python scripts/s3_collect_routes.py && python scripts/s3_build_finetune_set.py   # step 3
bash scripts/s4_sweep.sh                                                    # step 4
```

| path | role |
|---|---|
| `registry/` | `areas.json` (one entry per area), MP candidates + ranking, held-out regions |
| `crag/` | the pipeline package (registry, MP, EPT index, tiles, cliff labelling, pseudolabels, datasets, models, trainer, OSM routes, cross-reference, ground truth, negatives, fine-tuning) |
| `scripts/` | `s1_*` ... `s4_*` entry points, `setup_env.sh` |
| `cfgs/` | pretraining (`pointnext-s`, `pointnet2`, `pointnet2-msg`) and fine-tuning strategy configs |
| `third_party/PointNeXt` | OpenPoints / PointNeXt submodule |
| `docs/` | runbook, registry schema, Step 3 / Step 4 design notes |
| `tests/` | `python -m pytest tests` |
| `data/handgathered/` | Horseshoe Canyon Ranch GPS waypoints + photos (fine-tuning test set) |
| `data/mountain_project/`, `data/raw/`, `data/routes/` | committed scrapes and caches (MP route exports and pages, USGS EPT index, Overpass) and the Step 3 route tables, so nothing is re-fetched |
| `data/lidar/`, `data/pretrain/`, `data/finetune/`, `runs/`, `pretrained/` | generated locally (gitignored) |
| `proof-of-concept/` | the original single-area Horseshoe Canyon Ranch notebook and code ([README](proof-of-concept/README.md)) |
| `proposal/`, `research-prompts/` | project proposal and background research notes |
