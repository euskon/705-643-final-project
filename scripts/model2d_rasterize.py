#!/usr/bin/env python
"""2D model, step 1 - turn every fine-tuning sample into a top-down image for the 2D models.

Reads ``data/finetune/<name>/index.csv`` (from ``s3_build_finetune_set.py``:
one row per sample the 3D model trains or is scored on) and, for each row,
grids the LiDAR returns of the same square around the same point
(``crag/model2d_raster.py``). Labels, sources and train/val/test splits are copied
unchanged, so the 2D and 3D models see the same samples.

    python scripts/model2d_rasterize.py                      # data/finetune/v3 -> data/raster/v3
    python scripts/model2d_rasterize.py --name v3 --res-m 0.25 --out v3-res025
    python scripts/model2d_rasterize.py --preview 0          # skip the preview picture

Output: ``data/raster/<out>/``

* ``images.npy``  float32 ``(N, C, H, W)``, north up; channels in ``meta.json``
* ``index.csv``   one row per image (``row`` = position in ``images.npy``): sample id, label,
  source, split, coordinates, plus how well the scanner covered the square
* ``meta.json``   channels, cell size, counts per split
* ``preview.png`` a few samples per source, to check by eye

Needs the raw tiles (``data/lidar``) of the areas in the index.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from crag.cli import setup_logging
from crag.crop import AreaCloud
from crag.paths import FINETUNE_DIR, LIDAR_DIR
from crag.model2d_raster import CHANNELS, RASTER_DIR, RasterConfig, rasterize_sample, slope_deg

log = logging.getLogger("rasterize_finetune")

# copied from the fine-tuning index when present (the rest of its columns describe the 3D block arrays)
KEEP = ["sample_id", "slug", "label", "source", "confidence", "split", "group", "x", "y", "lon", "lat", "base_z", "top_z",
        "name", "mp_route_id", "route_uid", "n_center", "pair_id", "has_opposite_neighbour", "judge_result", "judge_confidence"]


def preview(images: np.ndarray, index: pd.DataFrame, cfg: RasterConfig, path: Path, per_source: int, seed: int = 0) -> None:
    """Slope picture (0-90 deg) of a few random samples of every source; the dot is the sample point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = sorted(index["source"].unique())
    fig, axes = plt.subplots(len(sources), per_source, figsize=(1.9 * per_source, 2.05 * len(sources)), squeeze=False)
    half = cfg.size_m / 2
    for r, src in enumerate(sources):
        rows = index[index["source"] == src]
        rows = rows.sample(min(len(rows), per_source), random_state=seed)
        for c in range(per_source):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(rows):
                ax.axis("off")
                continue
            s = rows.iloc[c]
            ax.imshow(slope_deg(images[s["row"], 0], cfg.res_m), cmap="magma", vmin=0, vmax=90, extent=(-half, half, -half, half))
            ax.plot(0, 0, "o", ms=3, color="cyan")
            ax.set_title(f"{src} | {s['slug'][:14]}", fontsize=6)
    fig.suptitle(f"slope of the bare-earth height (black = flat, yellow = vertical), {cfg.size_m:g} m squares, north up", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="v3", help="fine-tuning dataset to convert: data/finetune/<name>")
    ap.add_argument("--out", help="output name -> data/raster/<out> (default: same as --name)")
    ap.add_argument("--size-m", type=float, default=20.0, help="side of the square; 20 = the 3D block's footprint")
    ap.add_argument("--res-m", type=float, default=0.5, help="cell size in metres")
    ap.add_argument("--preview", type=int, default=8, help="samples per source in preview.png (0 = no preview)")
    args = ap.parse_args()
    setup_logging()

    cfg = RasterConfig(size_m=args.size_m, res_m=args.res_m)
    src = FINETUNE_DIR / args.name / "index.csv"
    if not src.exists():
        sys.exit(f"{src} not found - build it with scripts/s3_build_finetune_set.py --name {args.name}")
    index = pd.read_csv(src)
    synthetic = index["source"].astype(str).str.startswith("aug_")
    if synthetic.any():  # copies of positive blocks reshaped in 3D; there is no LiDAR at their location to grid
        log.info("skipping %d synthetic 3D augmentation samples (aug_*)", int(synthetic.sum()))
        index = index[~synthetic]
    missing = [s for s in index["slug"].unique() if not (LIDAR_DIR / s).exists()]
    if missing:
        sys.exit(f"no LiDAR tiles on disk for: {', '.join(missing)}")
    index = index[[c for c in KEEP if c in index.columns]].reset_index(drop=True)

    out = RASTER_DIR / (args.out or args.name)
    out.mkdir(parents=True, exist_ok=True)
    ignore = RASTER_DIR / ".gitignore"  # generated data: keep it out of git without touching the shared .gitignore
    if not ignore.exists():
        ignore.write_text("*\n")
    images = np.zeros((len(index), len(CHANNELS), cfg.n, cfg.n), np.float32)
    stats = []
    t0 = time.time()
    for slug, sub in index.groupby("slug", sort=False):
        cloud = AreaCloud(slug)
        # neighbouring samples share tiles: visit them in tile order so the tile cache is hit
        order = sub.assign(_tx=sub["x"] // cloud.size, _ty=sub["y"] // cloud.size).sort_values(["_tx", "_ty"]).index
        for i in order:
            images[i], st = rasterize_sample(cloud, float(index.at[i, "x"]), float(index.at[i, "y"]), cfg)
            stats.append({"row": i, **st})
        log.info("%s: %d images (%.0f s so far)", slug, len(sub), time.time() - t0)
    st = pd.DataFrame(stats).set_index("row").sort_index()
    index.insert(0, "row", np.arange(len(index)))
    for c in st.columns:
        index[c] = st[c].to_numpy()

    np.save(out / "images.npy", images)
    index.to_csv(out / "index.csv", index=False)
    by_split = index.groupby(["split", "label"]).size().rename("n").reset_index().to_dict("records")
    meta = {
        "source_dataset": str(src.parent.relative_to(FINETUNE_DIR.parent)).replace("\\", "/"),
        "channels": list(CHANNELS),
        "shape": list(images.shape),
        "orientation": "row 0 = north edge, column 0 = west edge",
        "heights": "metres above the image's lowest dem cell (z0 in index.csv)",
        "config": asdict(cfg),
        "samples_by_split_label": by_split,
        "samples_by_source": index["source"].value_counts().to_dict(),
        "no_points": int((index["n_points"] == 0).sum()),
        "dem_from_surface_fallback": int((~index["used_ground"] & (index["n_points"] > 0)).sum()),
        "median_frac_empty": float(index["frac_empty"].median()),
        "median_frac_dem_filled": float(index["frac_dem_filled"].median()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    if args.preview > 0:
        preview(images, index, cfg, out / "preview.png", args.preview)

    print(index.groupby(["split", "source"]).size().unstack(0).fillna(0).astype(int).to_string())
    print(index.groupby("source")[["n_points", "frac_empty", "frac_dem_filled"]].median().round(2).to_string())
    log.info("%d images %s -> %s (%.0f MB, %.0f s)", len(index), tuple(images.shape[1:]), out, images.nbytes / 1e6, time.time() - t0)


if __name__ == "__main__":
    main()
