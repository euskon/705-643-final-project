"""Hand-gathered ground truth: GPS waypoints taken on site at the wall base.

Each waypoint in ``data/handgathered/Waypoints_*.csv`` is labelled ``Climb``
(climbable rock, often with the route name), ``Nc`` (rock that is not
climbable) or ``?`` (unlabelled). The points are where the person *stood*, so
they are a few metres out from the face plus 3-10 m of GPS error;
:func:`snap_to_cliff` moves each one onto the nearest LiDAR cliff column whose
base is at the person's standing height.

Example::

    gt = load_waypoints(cfg.gt)
    gt = snap_to_cliff(pc, gt, cfg)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ExperimentConfig, GroundTruthConfig
from labeling import PointCloud, wall_direction
from usgs_ept import lonlat_to_work

LABELS = {"Climb": 1, "Nc": 0, "?": -1}
AREAS = {0: "North Forty", 1: "West Side", 2: "East Side"}


def load_waypoints(cfg: GroundTruthConfig) -> pd.DataFrame:
    """Both waypoint CSVs as one frame with ``gt_id``, ``label`` (1/0/-1) and UTM ``x, y``."""
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
    xy = lonlat_to_work(gt["lon"].to_numpy(), gt["lat"].to_numpy())
    gt["x"], gt["y"] = xy[:, 0], xy[:, 1]
    gt["route_id"] = gt["mp_url"].str.extract(r"/route/(\d+)/", expand=False)
    return gt


def _snap_one(pc: PointCloud, xy: np.ndarray, cfg: ExperimentConfig) -> dict:
    lab, g = cfg.label, cfg.gt
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


def snap_to_cliff(pc: PointCloud, gt: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Add the nearest qualifying cliff column to every waypoint.

    A candidate qualifies when its ``route_radius_m`` cylinder holds at least
    ``min_steep_points`` steep points (the same rule as the MP positives) and
    its base (5th-percentile Z) is within ``max_dz_m`` of the ground under the
    waypoint (10th-percentile Z of a ``ground_radius_m`` column). New columns:
    ``snap_x, snap_y, snap_dist_m, ground_z, base_z, out_dx, out_dy,
    n_steep_snap, status`` (``ok`` / ``no_cliff``), plus ``eval_x, eval_y``:
    the snapped point when there is one, else the raw waypoint.
    """
    rows = [_snap_one(pc, np.array([x, y]), cfg) for x, y in gt[["x", "y"]].to_numpy()]
    cols = ["snap_x", "snap_y", "snap_dist_m", "ground_z", "base_z",
            "out_dx", "out_dy", "n_steep_snap", "status"]
    snapped = pd.DataFrame(rows, index=gt.index).reindex(columns=cols)
    gt = pd.concat([gt, snapped], axis=1)
    ok = gt["status"] == "ok"
    gt["eval_x"] = np.where(ok, gt["snap_x"], gt["x"])
    gt["eval_y"] = np.where(ok, gt["snap_y"], gt["y"])
    return gt
