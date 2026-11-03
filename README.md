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
| `crag/` | the pipeline package (registry, MP, EPT index, tiles, pseudolabels, datasets, models, trainer, OSM routes, cross-reference, negatives, fine-tuning) |
| `scripts/` | `s1_*` ... `s4_*` entry points, `setup_env.sh` |
| `cfgs/` | pretraining (`pointnext-s`, `pointnet2`, `pointnet2-msg`) and fine-tuning strategy configs |
| `third_party/PointNeXt` | OpenPoints / PointNeXt submodule |
| `docs/` | runbook, registry schema, Step 3 / Step 4 design notes |
| `tests/` | `python -m pytest tests` |

---

## Proof of concept: Horseshoe Canyon Ranch, Jasper AR

Fine-tunes a ModelNet40-pretrained PointNet++ to classify 5 m columns of USGS
3DEP LiDAR as **climbable / not climbable**, with Mountain Project route
locations as weak labels. Background reasoning is in the three research notes
in this folder; the walkthrough is `crag_poc.ipynb`.

| file | role |
|---|---|
| `config.py` | every knob: bbox, EPT dataset, label/patch/split settings |
| `usgs_ept.py` | stream 3DEP Entwine Point Tiles with PDAL → UTM 15N numpy, cached |
| `mountain_project.py` | one cached route-finder CSV export → routes + wall pins |
| `labeling.py` | normals, pin → cliff snapping, 2.5 m route cylinders, negatives, LiDAR-coverage/overhang diagnostics |
| `dataset.py` | whole-wall train/val/test split, normalisation, `PatchDataset` |
| `model.py` | PointNet++ SSG (yanx27 layout, MIT) + pretrained loading + freeze modes |
| `trainer.py` | training loop, metrics, `compare_modes` (scratch / head / finetune) |
| `inference.py` | sliding-window P(climbable) map |

## Run

```bash
conda env create -f ../environment-final-proj.yaml   # env: final-proj-torch
conda activate final-proj-torch
cd final-project
jupyter execute --inplace crag_poc.ipynb   # or open it in VS Code with the final-proj-torch kernel
```

The notebook sets `FAST_DEV_RUN = not torch.cuda.is_available()`: on a Mac it
trains 2 epochs to prove the pipeline end to end; on a CUDA machine it trains
60 epochs per mode and draws a 4 m probability grid. On the first run it
downloads ~19 M LiDAR points (~4 min, ~300 MB in `data/`) and the 18 MB
pretrained checkpoint (`pretrained/`); Mountain Project is hit once and cached.

## Sources

* LiDAR: `USGS_LPC_AR_North_Corridor_B1_2017_LAS_2018` (fallback
  `AR_BuffaloRiver_2010`) from `s3://usgs-lidar-public` — see
  [hobuinc/usgs-lidar](https://github.com/hobuinc/usgs-lidar).
* Routes: Mountain Project area
  [105903004](https://www.mountainproject.com/area/105903004/horseshoe-canyon-ranch)
  via its route-finder CSV export (one request, cached; robots.txt asks for a
  60 s crawl delay, which `mountain_project.py` enforces).
* Pretrained weights:
  [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
  `log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth`.
