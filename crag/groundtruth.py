"""Hand-gathered ground truth: GPS waypoints taken on site at the wall base.

Each waypoint in ``data/handgathered/Waypoints_*.csv`` is labelled ``Climb``
(climbable rock, often with the route name), ``Nc`` (rock that is not
climbable) or ``?`` (unlabelled). The points are where the person *stood*, so
they are a few metres out from the face plus 3-10 m of GPS error;
:func:`snap_one` moves a point onto the nearest LiDAR cliff column whose base
is at the person's standing height. Step 3 uses the same rule to snap OSM
route starts.

Carried over from ``proof-of-concept/groundtruth.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crag.geo import AreaCRS, utm_epsg
from crag.labeling import LabelConfig, PointCloud, wall_direction
from crag.paths import PROJECT_DIR

LABELS = {"Climb": 1, "Nc": 0, "?": -1}


@dataclass
class GroundTruthConfig:
    """Hand-gathered GPS waypoints (``data/handgathered``) used as ground truth."""

    paths: tuple[Path, ...] = (
        PROJECT_DIR / "data" / "handgathered" / "Waypoints_2026-10-24.csv",
        PROJECT_DIR / "data" / "handgathered" / "Waypoints_2026-10-25.csv",
    )
    # handheld GPS under a cliff is ~3-10 m off and the waypoint is where the
    # person stood, a few metres out from the face; MP pins use 30 m
    snap_radius_m: float = 12.0
    # a snapped cliff's base must be within this of the ground under the
    # waypoint, so a tiered cliff can't snap onto the band above / below
    max_dz_m: float = 6.0
    # column around the waypoint whose low points give the standing height
    ground_radius_m: float = 3.0


def load_waypoints(cfg: GroundTruthConfig | None = None, to_work=None) -> pd.DataFrame:
    """All waypoint CSVs as one frame with ``gt_id``, ``label`` (1/0/-1) and metric ``x, y``.

    ``to_work`` converts lon/lat to the point cloud's CRS; by default the UTM
    zone of the waypoints themselves (EPSG:32615 for Horseshoe Canyon Ranch).
    """
    cfg = cfg or GroundTruthConfig()
    frames = []
    for path in cfg.paths:
        df = pd.read_csv(path, dtype={"route_index": str, "name": str})
        df["source_file"] = path.name
        df["gt_id"] = [f"{path.stem}#{i}" for i in range(len(df))]
        frames.append(df)
    gt = pd.concat(frames, ignore_index=True)
    unknown = set(gt["comment"]) - set(LABELS)
    if unknown:
        raise ValueError(f"unexpected labels in waypoint comment column: {unknown}")
    gt["label"] = gt["comment"].map(LABELS).astype(int)
    if to_work is None:
        to_work = AreaCRS(utm_epsg(gt["lon"].mean(), gt["lat"].mean())).to_work
    xy = to_work(gt["lon"].to_numpy(), gt["lat"].to_numpy())
    gt["x"], gt["y"] = xy[:, 0], xy[:, 1]
    gt["route_id"] = gt["mp_url"].str.extract(r"/route/(\d+)/", expand=False)
    return gt


def snap_one(pc: PointCloud, xy: np.ndarray, lab: LabelConfig | None = None, g: GroundTruthConfig | None = None) -> dict:
    """Nearest qualifying cliff column to a point where someone stood (or a route start).

    A candidate qualifies when its ``route_radius_m`` cylinder holds at least
    ``min_steep_points`` steep points (the same rule as the MP positives) and
    its base (5th-percentile Z) is within ``max_dz_m`` of the ground under the
    point (10th-percentile Z of a ``ground_radius_m`` column). Returns
    ``ground_z, status`` (``ok`` / ``no_cliff``) and, when ``ok``,
    ``snap_x, snap_y, snap_dist_m, base_z, out_dx, out_dy, n_steep_snap``.
    """
    lab, g = lab or LabelConfig(), g or GroundTruthConfig()
    ground_idx = pc.column(xy, g.ground_radius_m)
    ground_z = float(np.quantile(pc.xyz[ground_idx, 2], 0.1)) if len(ground_idx) else np.nan
    near = pc.steep_tree_xy.query_ball_point(xy, g.snap_radius_m)
    out = {"ground_z": ground_z, "status": "no_cliff"}
    if not near:
        return out
    cand = pc.xyz[pc.steep_idx[near], :2]
    cand = cand[np.argsort(np.linalg.norm(cand - xy, axis=1))]
    _, first = np.unique(np.round(cand / 0.5).astype(np.int64), axis=0, return_index=True)
    for c in cand[np.sort(first)]:
        n_steep = len(pc.steep_tree_xy.query_ball_point(c, lab.route_radius_m))
        if n_steep < lab.min_steep_points:
            continue
        idx = pc.column(c, lab.route_radius_m)
        base_z = float(np.quantile(pc.xyz[idx, 2], 0.05))
        if np.isfinite(ground_z) and abs(base_z - ground_z) > g.max_dz_m:
            continue
        out_dir = wall_direction(pc, idx)
        return {
            **out,
            "snap_x": c[0],
            "snap_y": c[1],
            "snap_dist_m": float(np.linalg.norm(c - xy)),
            "base_z": base_z,
            "out_dx": out_dir[0],
            "out_dy": out_dir[1],
            "n_steep_snap": n_steep,
            "status": "ok",
        }
    return out
