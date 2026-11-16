"""Hand-gathered ground truth: GPS waypoints taken on site at the wall base.

Each waypoint in ``data/handgathered/Waypoints_*.csv`` is labelled ``Climb``
(climbable rock, often with the route name), ``Nc`` (rock that is not
climbable) or ``?`` (unlabelled). The points are where the person *stood*, so
they are a few metres out from the face plus 3-10 m of GPS error.

:func:`snap_one` is the **wall check** every route GPS point goes through
before it is used for training or testing (hand-gathered waypoints and
OpenStreetMap route starts alike; ``crag.negatives.snap_routes``):

1. *which wall* - the **anchor**: the nearest spot within ``snap_radius_m``
   with enough cliff points, a foot at the person's standing height, and a
   wall at least ``min_wall_height_m`` (6 m) tall on the bare-earth ground
   grid (``crag.walls.wall_height_at``);
2. *where on it* - the spot **straight in front of the person** (the GPS
   point projected onto the wall line through the anchor). If that spot
   fails the checks, it slides along the wall toward the anchor. So points
   keep their order and spacing along the wall.

Points with walls in reach that are all too short get ``too_short``; points
with no wall at all ``off_wall``. :func:`wall_foot` then gives the foot of
the wall at any spot on or near it - the one centring rule for every
fine-tuning sample, whatever its label.

The Horseshoe waypoint files are snapped once and judged
(``scripts/snap_hcr_waypoints.py``, ``scripts/hcr_judge_apply.py``); their
``lat/lon`` are then final (``presnapped``) and are only confirmed, never
moved again. The untouched originals are in ``data/handgathered/original``.

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
from crag.walls import WallGrid, wall_frame_at, wall_grid, wall_height_at

LABELS = {"Climb": 1, "Nc": 0, "?": -1}


@dataclass
class GroundTruthConfig:
    """Hand-gathered GPS waypoints (``data/handgathered``) used as ground truth."""

    paths: tuple[Path, ...] = (
        PROJECT_DIR / "data" / "handgathered" / "Waypoints_2026-10-24.csv",
        PROJECT_DIR / "data" / "handgathered" / "Waypoints_2026-10-25.csv",
    )
    # handheld GPS under a cliff is ~3-10 m off and the waypoint is where the
    # person stood, a few metres out from the face; Horseshoe Canyon's East
    # Side waypoints sit 7-12 m out (docs/hcr-waypoint-audit.md). Points with
    # no wall this close are dropped as ``off_wall``
    snap_radius_m: float = 20.0
    # the wall at the snapped spot must be at least this tall (user's choice,
    # 2026-11-15: below 6 m is bouldering height)
    min_wall_height_m: float = 6.0
    # "grid": height on the bare-earth ground grid (new) | "column": spread of
    # cliff points in the upright column (old, comparison runs only)
    height_measure: str = "grid"
    # "toward_wall": straight in front of the person (new) | "nearest": the
    # nearest passing spot (old, comparison runs only)
    snap_method: str = "toward_wall"
    # lighter cliff-point check for the spot straight in front (the anchor
    # already proved the wall is real)
    target_min_steep_points: int = 3
    # how far toward / away from the wall line the straight-in-front spot may
    # be looked for (walls curve away from the straight line through the anchor)
    across_search_m: float = 8.0
    # flag a point slid further than this along the wall
    max_along_shift_m: float = 3.0
    # confirmation radius for points already snapped and saved
    presnapped_radius_m: float = 3.0
    # a snapped cliff's base must be within this of the ground under the
    # waypoint, so a tiered cliff can't snap onto the band above / below
    max_dz_m: float = 6.0
    # column around the waypoint whose low points give the standing height
    ground_radius_m: float = 3.0


SNAP_FILE_COLS = ("raw_lat", "raw_lon", "auto_lat", "auto_lon", "snap_status", "snap_dist_m", "along_shift_m", "wall_height_m",
                  "wall_height_column_m", "nearest_wall_height_m", "skipped_shorter_wall", "presnapped", "snap_flags",
                  "snap_min_wall_height_m", "snap_method")
JUDGE_COLS = ("judge_a_choice", "judge_b_choice", "judge_c_choice", "judge_confidence", "judge_reason_a", "judge_reason_b",
              "judge_reason_c", "judge_result")
UNUSABLE_STATUS = ("too_short", "off_wall", "no_lidar", "judge_none")


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
    # files written by scripts/snap_hcr_waypoints.py carry the snap / judge columns; old files don't
    if "presnapped" not in gt:
        gt["presnapped"] = False
    gt["presnapped"] = gt["presnapped"].fillna(False).astype(bool)
    for c, default in (("snap_status", ""), ("judge_result", ""), ("judge_confidence", ""), ("snap_flags", "")):
        gt[c] = gt[c].fillna(default).astype(str) if c in gt else default
    return gt


def _spot(pc: PointCloud, c: np.ndarray, lab: LabelConfig, g: GroundTruthConfig, grid, ground_z: float, min_pts: int) -> dict | None:
    """The wall at spot ``c`` if it has enough cliff points and its foot is at the standing height."""
    n_steep = len(pc.steep_tree_xy.query_ball_point(c, lab.route_radius_m))
    if n_steep < min_pts:
        return None
    idx = pc.column(c, lab.route_radius_m)
    base_z = float(np.quantile(pc.xyz[idx, 2], 0.05))
    if np.isfinite(ground_z) and abs(base_z - ground_z) > g.max_dz_m:
        return None
    cz = pc.xyz[idx[pc.steep[idx]], 2]
    col_h = float(np.quantile(cz, 0.95) - np.quantile(cz, 0.05)) if len(cz) >= 2 else 0.0
    grid_h = wall_height_at(grid, c, base_z)[0] if grid is not None else float("nan")
    return {"xy": np.asarray(c, float), "n_steep": n_steep, "idx": idx, "base_z": base_z, "col_h": col_h, "grid_h": grid_h,
            "h": grid_h if g.height_measure == "grid" else col_h}


def snap_one(pc: PointCloud, xy: np.ndarray, lab: LabelConfig | None = None, g: GroundTruthConfig | None = None, grid: WallGrid | None = None) -> dict:
    """The wall check for a point where someone stood (or a route start). See the module docstring.

    ``grid`` is the bare-earth wall grid around the point (``crag.walls.grid_for``); without
    one it is built from ``pc`` treating every point as surface (synthetic scenes only).
    Returns ``ground_z, wall_dist_m, status`` (``ok`` / ``too_short`` / ``off_wall``) and, when
    ``ok``: ``snap_x, snap_y, snap_dist_m, base_z, wall_height_m`` (ground grid),
    ``wall_height_column_m``, ``anchor_x, anchor_y, along_shift_m, out_dx, out_dy, n_steep_snap``.
    ``nearest_wall_height_m, nearest_wall_dist_m`` describe the nearest spot that passed every
    check except height; ``skipped_shorter_wall`` is true when the anchor is more than 3 m
    further than it."""
    lab, g = lab or LabelConfig(), g or GroundTruthConfig()
    xy = np.asarray(xy, float)
    if grid is None and g.height_measure == "grid" and len(pc.xyz) >= 16:
        grid = wall_grid(pc.xyz, np.ones(len(pc.xyz), bool))
    ground_idx = pc.column(xy, g.ground_radius_m)
    ground_z = float(np.quantile(pc.xyz[ground_idx, 2], 0.1)) if len(ground_idx) else np.nan
    wall_dist = np.nan
    if pc.wall_dist is not None and len(pc.xyz):
        _, i = pc.tree_xy.query(xy)
        wall_dist = float(pc.wall_dist[i])
    near = pc.steep_tree_xy.query_ball_point(xy, g.snap_radius_m) if len(pc.steep_idx) else []
    out = {"ground_z": ground_z, "wall_dist_m": wall_dist, "status": "off_wall", "nearest_wall_height_m": np.nan, "nearest_wall_dist_m": np.nan,
           "skipped_shorter_wall": False}
    if not near:
        return out
    cand = pc.xyz[pc.steep_idx[near], :2]
    cand = cand[np.argsort(np.linalg.norm(cand - xy, axis=1))]
    _, first = np.unique(np.round(cand / 0.5).astype(np.int64), axis=0, return_index=True)
    anchor = None
    for c in cand[np.sort(first)]:
        sp = _spot(pc, c, lab, g, grid, ground_z, lab.min_steep_points)
        if sp is None:
            continue
        if not np.isfinite(out["nearest_wall_dist_m"]):
            out["nearest_wall_height_m"], out["nearest_wall_dist_m"] = sp["h"], float(np.linalg.norm(c - xy))
        if sp["h"] >= g.min_wall_height_m:
            anchor = sp
            break
    if anchor is None:
        if np.isfinite(out["nearest_wall_dist_m"]):
            out["status"] = "too_short"
        return out
    a_xy = anchor["xy"]
    out["skipped_shorter_wall"] = bool(out["nearest_wall_height_m"] < g.min_wall_height_m and np.linalg.norm(a_xy - xy) - out["nearest_wall_dist_m"] > 3.0)
    sp, shift = anchor, 0.0
    if g.snap_method == "toward_wall" and grid is not None:
        out_a, along = wall_frame_at(grid, a_xy)
        s = float((xy - a_xy) @ along)
        # walls curve, so at each position along the wall look toward / away from it
        # (nearest first) for the wall itself; this keeps the position along the wall
        us = np.arange(0.0, g.across_search_m + 0.25, 0.5)
        us = np.ravel(np.column_stack([us, -us]))[1:]
        found = None
        for t in np.r_[np.arange(abs(s), 0.0, -0.5), 0.0] * (1.0 if s >= 0 else -1.0):
            if t == 0.0:
                break  # the anchor itself
            for u in us:
                spot = a_xy + t * along + u * out_a
                if float(grid.distance(spot)[0]) > 2.0:
                    continue
                c = _spot(pc, spot, lab, g, grid, ground_z, g.target_min_steep_points)
                if c is None or c["h"] < g.min_wall_height_m:
                    continue
                # sit on real points: the nearest cliff point within 1 m of the accepted spot
                d, j = pc.steep_tree_xy.query(spot)
                if d <= 1.0:
                    c2 = _spot(pc, pc.xyz[pc.steep_idx[j], :2], lab, g, grid, ground_z, g.target_min_steep_points)
                    if c2 is not None and c2["h"] >= g.min_wall_height_m:
                        c = c2
                found = (c, abs(s - t))
                break
            if found:
                break
        sp, shift = found if found else (anchor, abs(s))
    elif g.snap_method not in ("toward_wall", "nearest"):
        raise ValueError(f"snap_method={g.snap_method!r}")
    c = sp["xy"]
    out_dir = wall_frame_at(grid, c)[0] if grid is not None else wall_direction(pc, sp["idx"])
    return {
        **out,
        "snap_x": float(c[0]),
        "snap_y": float(c[1]),
        "snap_dist_m": float(np.linalg.norm(c - xy)),
        "base_z": sp["base_z"],
        "wall_height_m": sp["h"],
        "wall_height_column_m": sp["col_h"],
        "anchor_x": float(a_xy[0]),
        "anchor_y": float(a_xy[1]),
        "along_shift_m": float(shift),
        "out_dx": float(out_dir[0]),
        "out_dy": float(out_dir[1]),
        "n_steep_snap": sp["n_steep"],
        "status": "ok",
    }


def wall_foot(pc: PointCloud, xy, grid: WallGrid | None = None, min_points: int = 3, radius_m: float = 2.5, search_m: float = 5.0,
              max_walk_m: float = 15.0, step_m: float = 0.5, gap_m: float = 1.5) -> dict:
    """The foot of the wall at ``xy`` (any spot on or near a wall) - the one centring rule for
    every fine-tuning sample.

    1. the cliff column at ``xy`` (cliff points within ``radius_m``); with fewer than
       ``min_points``, move straight toward the nearest cliff point within ``search_m``
       (only toward / away from the wall, so the position along the wall is kept);
    2. the outward direction ``out`` (ground grid, else the points' normals);
    3. the foot is the outer edge of the bare-earth wall-mask strip on the line through the
       spot along ``out`` (the strip nearest the spot, looking both ways) - so it doesn't
       depend on whether a sample starts on the face, at its base or in front of it. Without
       a wall mask: walk outward over 1 m slices of cliff points until a gap > ``gap_m``;
    4. ``base_z`` = 5th-percentile height of all points within ``radius_m`` of the foot.

    Returns ``foot_x, foot_y, base_z, out_dx, out_dy, foot_move_m, status`` (``ok`` / ``no_wall``)."""
    xy = np.asarray(xy, float)
    if grid is None and len(pc.xyz) >= 16:
        grid = wall_grid(pc.xyz, np.ones(len(pc.xyz), bool))
    bad = {"status": "no_wall", "foot_x": np.nan, "foot_y": np.nan, "base_z": np.nan, "out_dx": np.nan, "out_dy": np.nan, "foot_move_m": np.nan}
    if not len(pc.steep_idx):
        return bad
    d, j = pc.steep_tree_xy.query(xy)
    if d > search_m:
        return bad
    def frame(p):
        if grid is not None:
            return wall_frame_at(grid, p)
        o = wall_direction(pc, pc.column(p, radius_m))
        return o, np.array([-o[1], o[0]])
    p0 = xy
    if len(pc.steep_tree_xy.query_ball_point(xy, radius_m)) < min_points:
        o, _ = frame(pc.xyz[pc.steep_idx[j], :2])
        p0 = xy + ((pc.xyz[pc.steep_idx[j], :2] - xy) @ o) * o
        if len(pc.steep_tree_xy.query_ball_point(p0, radius_m)) < min_points:
            p0 = pc.xyz[pc.steep_idx[j], :2].astype(float)
            if len(pc.steep_tree_xy.query_ball_point(p0, radius_m)) < min_points:
                return bad
    out, along = frame(p0)
    foot_o = None
    if grid is not None:
        # start-independent foot: the outer edge of the wall-mask strip on the line through p0
        # along ``out`` (looking both into and out from the wall), the strip nearest p0
        t = np.arange(-max_walk_m, max_walk_m + step_m, step_m)
        w = grid.is_wall(p0[None, :] + t[:, None] * out[None, :])
        segs, cur = [], None
        for k in np.flatnonzero(w):
            if cur is not None and t[k] - cur[1] <= gap_m + 1e-9:
                cur[1] = t[k]
            else:
                cur = [t[k], t[k]]
                segs.append(cur)
        if segs:
            near_seg = min(segs, key=lambda sg: 0.0 if sg[0] <= 0 <= sg[1] else min(abs(sg[0]), abs(sg[1])))
            if min(abs(near_seg[0]), abs(near_seg[1])) <= search_m or near_seg[0] <= 0 <= near_seg[1]:
                foot_o = float(near_seg[1])
    if foot_o is None:  # no wall mask here: walk outward over the cliff points
        near = pc.steep_idx[pc.steep_tree_xy.query_ball_point(p0, max_walk_m + radius_m + 1.0)]
        rel = pc.xyz[near, :2] - p0
        a, o = rel @ along, rel @ out
        o = o[np.abs(a) <= radius_m]
        last_t = 0.0
        for tt in np.arange(0.0, max_walk_m + step_m, step_m):
            m = np.abs(o - tt) <= 0.5
            if m.any():
                foot_o, last_t = float(o[m].mean()), tt
            elif tt - last_t > gap_m:
                break
        if foot_o is None:
            return bad
    foot = p0 + foot_o * out
    idx = pc.column(foot, radius_m)
    if not len(idx):
        return bad
    return {"status": "ok", "foot_x": float(foot[0]), "foot_y": float(foot[1]), "base_z": float(np.quantile(pc.xyz[idx, 2], 0.05)),
            "out_dx": float(out[0]), "out_dy": float(out[1]), "foot_move_m": float(np.linalg.norm(foot - xy))}
