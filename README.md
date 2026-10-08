# Crag detection PoC — Horseshoe Canyon Ranch, Jasper AR

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
