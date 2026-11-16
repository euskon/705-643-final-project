#!/usr/bin/env python
"""One-time automatic snap of the Horseshoe Canyon Ranch waypoints (plan 4, step 5).

Always reads the **originals** in ``data/handgathered/original/`` (never the working
files), runs the pipeline's wall check on every Climb / Nc row and writes the working
files at the original paths: same rows, same order, same original columns, plus the snap
columns. ``lat/lon`` become the snapped position; ``raw_lat/raw_lon`` keep the original.

    python scripts/snap_hcr_waypoints.py            # refuses if judge decisions would be lost
    python scripts/snap_hcr_waypoints.py --force

Next: ``scripts/hcr_judge_cases.py``, the judges, ``scripts/hcr_judge_apply.py``, then
``scripts/s3_collect_routes.py`` to refresh ``data/routes/verified_routes.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from crag import hcr_points, registry
from crag.cli import setup_logging
from crag.crop import AreaCloud
from crag.groundtruth import GroundTruthConfig, snap_one
from crag.labeling import LabelConfig
from crag.negatives import local_pc
from crag.walls import grid_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default=hcr_points.HCR_SLUG)
    ap.add_argument("--force", action="store_true", help="overwrite working files that already carry judge decisions")
    args = ap.parse_args()
    setup_logging()
    hcr_points.require_hcr(args.area)
    area = next(a for a in registry.load()["areas"] if a["slug"] == args.area)
    crs, cloud, cfg, lab = registry.area_crs(area), AreaCloud(args.area), GroundTruthConfig(), LabelConfig()

    def snap(xy):
        pc, _ = local_pc(cloud, xy[0], xy[1], 40.0)
        if pc is None:
            return {"status": "no_lidar"}
        return snap_one(pc, np.asarray(xy, float), lab, cfg, grid_for(cloud, xy[0], xy[1]))

    p = hcr_points.snap_waypoint_files(snap, crs.to_work, crs.to_lonlat, cfg=cfg, force=args.force)
    lab_rows = p[p["comment"] != "?"]
    print("status by label:\n", lab_rows.groupby(["comment", "snap_status"]).size().unstack(fill_value=0).to_string())
    flags = lab_rows["snap_flags"].str.split(";").explode()
    print("\nflags:\n", flags[flags != ""].value_counts().to_string())
    print(f"\nmedian move {lab_rows['snap_dist_m'].median():.1f} m; wrote {', '.join(hcr_points.FILES)} in {hcr_points.HAND_DIR}")


if __name__ == "__main__":
    main()
