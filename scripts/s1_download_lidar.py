#!/usr/bin/env python
"""Step 1c - download USGS LiDAR tiles around every registry area's crags (incremental).

Each area's tiles (``--tile-size-m`` squares within ``--crag-radius-m`` of a
crag pin) are fetched from the USGS 3DEP Entwine Point Tiles listed in the
registry and cached as ``data/lidar/<slug>/<tile>.npz``. Already-downloaded
tiles are skipped, so re-running after an interruption (or after adding an
area) only fetches what is missing.

    python scripts/s1_download_lidar.py --dry-run          # plan + size estimate only
    python scripts/s1_download_lidar.py                     # all enabled areas
    python scripts/s1_download_lidar.py --areas smith-rock,red-rocks --workers 8
    python scripts/s1_download_lidar.py --max-tiles-per-area 40   # cap huge regions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from crag import registry, tiles
from crag.cli import parse_list, setup_logging
from crag.paths import AREAS_JSON, LIDAR_DIR

log = logging.getLogger("download_lidar")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=AREAS_JSON)
    ap.add_argument("--areas", help="comma-separated slugs (default: all enabled)")
    ap.add_argument("--splits", default="train,val,test,holdout", help="only areas in these pretraining splits")
    ap.add_argument("--tile-size-m", type=float, default=200.0)
    ap.add_argument("--crag-radius-m", type=float, default=150.0)
    ap.add_argument("--max-tiles-per-area", type=int, default=150, help="keep the most route-dense tiles (0 = no cap)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    setup_logging()

    reg = registry.load(args.registry)
    splits = set(parse_list(args.splits))
    areas = [a for a in registry.areas(reg, parse_list(args.areas)) if a.get("split", "train") in splits]
    summary = []
    for area in areas:
        usgs = area.get("usgs")
        if not (usgs or {}).get("datasets"):
            if usgs is None:
                log.warning("%s: USGS datasets not resolved yet - run s1_build_registry.py; skipping", area["slug"])
            else:
                log.warning("%s: no USGS 3DEP EPT coverage over its crags; skipping", area["slug"])
            continue
        plan = tiles.plan_tiles(area, args.tile_size_m, args.crag_radius_m, args.max_tiles_per_area or None)
        have = sum(tiles.tile_path(area["slug"], t).exists() for t in plan)
        row = {"slug": area["slug"], "tiles": len(plan), "cached": have}
        if args.dry_run:
            ppm2 = area["usgs"]["datasets"][0]["points_per_m2"]
            row["est_points_M"] = round(len(plan) * args.tile_size_m**2 * ppm2 / 1e6, 1)
            row["est_GB"] = round(row["est_points_M"] * 1e6 * 11 / 1e9, 2)  # ~11 B/pt compressed
            summary.append(row)
            continue
        t0 = time.time()
        results = tiles.download_area(area, plan, workers=args.workers)
        status = pd.Series([r["status"] for r in results]).value_counts().to_dict()
        manifest = {
            "slug": area["slug"],
            "tile_size_m": args.tile_size_m,
            "crag_radius_m": args.crag_radius_m,
            "max_tiles": args.max_tiles_per_area,
            "tiles": [t.id for t in plan],
            "status": status,
        }
        out = LIDAR_DIR / area["slug"] / "manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=1))
        row.update(status)
        row["minutes"] = round((time.time() - t0) / 60, 1)
        summary.append(row)
        log.info("%s: %s", area["slug"], status)
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
    if args.dry_run and len(df):
        print(f"\nTOTAL tiles={df['tiles'].sum()}  ~{df['est_points_M'].sum():.0f}M points  ~{df['est_GB'].sum():.1f} GB")


if __name__ == "__main__":
    main()
