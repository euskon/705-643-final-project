#!/usr/bin/env python
"""Compare cliff ("rock") rules on known rock locations before regenerating pseudolabels.

Known rock = the verified route starts in ``data/routes/verified_routes.csv``
(OSM routes, plus the hand-gathered Horseshoe Canyon ``Climb`` and ``Nc``
waypoints - ``Nc`` is rock too, just not climbable). For each area the
script recomputes normals on the tiles holding those points, applies every
rule variant, and reports:

* ``cliff_share``  - fraction of all points labelled cliff;
* ``recall``       - fraction of known-rock points with >= ``min_steep_points``
  cliff points within ``radius_m`` (the same test route snapping uses);
* ``cliff_near``   - median cliff points within 5 m of a known-rock point;
* ``isolated``     - fraction of cliff points with < ``min_steep_points``
  cliff neighbours in their 2.5 m column (speckle: canopy / noise, not walls).

Variants: ``old`` (normals from all points, single return, slope >= 60),
``surface`` (normals from last non-vegetation returns only), ``surface+p<t>``
(plus planarity >= t), and ``... +multi`` (also counting steep, planar last
returns of multi-return pulses), ``... +wall`` (only within ``wall.near_m`` of
the bare-earth wall mask, ``crag.walls`` - the current default is
``surface +multi +wall``). ``--plots`` writes top-down and side views.

    python scripts/s1_check_rock_rule.py
    python scripts/s1_check_rock_rule.py --areas horseshoe-canyon-ranch,red-river-gorge --plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crag import registry  # noqa: E402
from crag.cli import parse_list, setup_logging  # noqa: E402
from crag.labeling import compute_normals, slope_deg, surface_geometry  # noqa: E402
from crag.paths import LIDAR_DIR, ROUTES_DIR  # noqa: E402
from crag.pseudolabels import VEG_LAS_CLASSES, PseudoLabelConfig, _TileCache, surface_mask, tile_with_halo  # noqa: E402
from crag.tiles import Tile  # noqa: E402
from crag.walls import point_wall_distance  # noqa: E402

DEFAULT_AREAS = "horseshoe-canyon-ranch,red-river-gorge,leavenworth,the-gunks,lake-tahoe,joshua-tree-national-park,rumney"
PLANARITY = (0.3, 0.5, 0.7)


def variants(slope_old, slope_new, plan, rn, nr, las, steep_deg, near_wall=None):
    last = rn >= nr
    ok = last & ~np.isin(las, VEG_LAS_CLASSES)
    single = nr == 1
    out = {
        "old": ok & single & (slope_old >= steep_deg),
        "surface": ok & single & (slope_new >= steep_deg),
        "surface +multi": ok & (slope_new >= steep_deg),
    }
    if near_wall is not None:
        out["surface +multi +wall"] = out["surface +multi"] & near_wall
        out["surface +wall"] = out["surface"] & near_wall
    for t in PLANARITY:
        rock = ok & (slope_new >= steep_deg) & (plan >= t)
        out[f"surface+p{t}"] = rock & single
        out[f"surface+p{t} +multi"] = rock
    return out


def column_counts(xy_query, xy_pts, r):
    if len(xy_pts) == 0 or len(xy_query) == 0:
        return np.zeros(len(xy_query), int)
    return np.asarray(cKDTree(xy_pts).query_ball_point(xy_query, r, return_length=True), int)


def check_area(area, pts_xy, cfg, args, plot_dir=None):
    slug = area["slug"]
    cache = _TileCache(slug, cfg.tile_size_m, maxlen=32)
    size = cfg.tile_size_m
    tids = sorted({Tile(np.floor(x / size) * size, np.floor(y / size) * size, size).id for x, y in pts_xy})
    tids = [t for t in tids if (LIDAR_DIR / slug / f"{t}.npz").exists()]
    rng = np.random.default_rng(0)
    if len(tids) > args.max_tiles:
        tids = sorted(rng.choice(tids, args.max_tiles, replace=False).tolist())
    acc = {}
    n_pts_total = 0
    for tid in tids:
        h = tile_with_halo(tid, cache, cfg.halo_m)
        if h is None:
            continue
        n = h["n"]
        xyz = h["xyz_abs"]
        rn, nr, las = h["return_number"], h["number_of_returns"], h["classification"]
        n_old = compute_normals(xyz, cfg.label.normal_k, cfg.label.normal_voxel_m)[:n]
        n_new, plan = surface_geometry(xyz, surface_mask(rn, nr, las), cfg.label.normal_k, cfg.label.normal_voxel_m)
        n_new, plan = n_new[:n], plan[:n]
        core = xyz[:n]
        tile = Tile.from_id(tid, size)
        inside = (
            (pts_xy[:, 0] >= tile.x0 + args.radius_m) & (pts_xy[:, 0] < tile.x0 + size - args.radius_m)
            & (pts_xy[:, 1] >= tile.y0 + args.radius_m) & (pts_xy[:, 1] < tile.y0 + size - args.radius_m)
        )
        q = pts_xy[inside]
        wd = point_wall_distance(xyz, surface_mask(rn, nr, las), cfg.wall, ground=las == 2)[:n]
        V = variants(slope_deg(n_old), slope_deg(n_new), plan, rn[:n], nr[:n], las[:n], cfg.label.steep_slope_deg, wd <= cfg.wall.near_m)
        n_pts_total += n
        for name, m in V.items():
            a = acc.setdefault(name, {"cliff": 0, "isolated": 0, "hits": [], "near5": []})
            cxy = core[m, :2]
            a["cliff"] += int(m.sum())
            if m.any():
                col = column_counts(cxy, cxy, cfg.label.route_radius_m)
                a["isolated"] += int((col < cfg.label.min_steep_points).sum())
            a["hits"] += (column_counts(q, cxy, args.radius_m) >= cfg.label.min_steep_points).tolist()
            a["near5"] += column_counts(q, cxy, 5.0).tolist()
        if plot_dir is not None and len(q):
            plot_tile(plot_dir / f"{slug}_{tid}.png", core, V, q, rn[:n], nr[:n], las[:n], slope_deg(n_new))
    rows = []
    for name, a in acc.items():
        rows.append({
            "slug": slug, "variant": name, "tiles": len(tids), "n_rock_pts": len(a["hits"]),
            "cliff_share": a["cliff"] / max(n_pts_total, 1),
            "recall": float(np.mean(a["hits"])) if a["hits"] else np.nan,
            "cliff_near": float(np.median(a["near5"])) if a["near5"] else np.nan,
            "isolated": a["isolated"] / max(a["cliff"], 1),
        })
    return rows


def plot_tile(path, xyz, V, q, rn, nr, las, slope_new, half=20.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["old", "surface +multi", "surface +multi +wall"]
    c = q[len(q) // 2]
    m = (np.abs(xyz[:, 0] - c[0]) <= half) & (np.abs(xyz[:, 1] - c[1]) <= half)
    if m.sum() < 100:
        return
    p = xyz[m]
    last = (rn >= nr)[m]
    veg = (~last) | np.isin(las[m], VEG_LAS_CLASSES)
    # side view along the principal horizontal axis of the cliff points
    cm = V["surface +multi"][m]
    axis = np.array([1.0, 0.0])
    if cm.sum() > 10:
        d = p[cm, :2] - p[cm, :2].mean(0)
        axis = np.linalg.svd(d, full_matrices=False)[2][0]
    u = (p[:, :2] - c) @ axis
    w = (p[:, :2] - c) @ np.array([-axis[1], axis[0]])
    slab = np.abs(w - (np.median(w[cm]) if cm.sum() > 10 else 0.0)) <= 4.0
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for k, name in enumerate(names):
        cl = V[name][m]
        col = np.where(veg, 1, np.where(cl, 3, np.where(slope_new[m] < 30, 0, 2)))
        colors = np.array(["#c8b48c", "#4f9a4a", "#e0a040", "#d62728"])[col]
        order = np.argsort(col)  # cliff drawn last
        axes[0, k].scatter(p[order, 0] - c[0], p[order, 1] - c[1], c=colors[order], s=3, linewidths=0)
        axes[0, k].scatter(q[:, 0] - c[0], q[:, 1] - c[1], marker="x", c="k", s=30)
        axes[0, k].set_xlim(-half, half), axes[0, k].set_ylim(-half, half), axes[0, k].set_aspect("equal")
        axes[0, k].set_title(f"{name}: cliff {cl.mean():.1%}")
        o = order[slab[order]]
        axes[1, k].scatter(u[o], p[o, 2], c=colors[o], s=4, linewidths=0)
        axes[1, k].set_aspect("equal")
        axes[1, k].set_title("side view: 8 m slab along the wall")
    fig.suptitle(f"{path.stem}  tan=ground green=vegetation orange=steep_terrain red=cliff  x=known rock")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--areas", default=DEFAULT_AREAS)
    ap.add_argument("--routes", type=Path, default=ROUTES_DIR / "verified_routes.csv")
    ap.add_argument("--radius-m", type=float, default=12.0, help="known rock counts as found with enough cliff points this close")
    ap.add_argument("--max-tiles", type=int, default=12, help="tiles sampled per area")
    ap.add_argument("--plots", type=Path, default=None, help="directory for top-down / side-view PNGs")
    ap.add_argument("--out", type=Path, default=None, help="write the table as CSV")
    args = ap.parse_args()
    setup_logging()

    cfg = PseudoLabelConfig()
    reg = registry.load()
    routes = pd.read_csv(args.routes)
    if args.plots:
        args.plots.mkdir(parents=True, exist_ok=True)
    rows = []
    for area in registry.areas(reg, parse_list(args.areas), enabled_only=False):
        r = routes[routes["slug"] == area["slug"]]
        if r.empty:
            continue
        xy = registry.area_crs(area).to_work(r["lon"].to_numpy(), r["lat"].to_numpy())
        rows += check_area(area, np.asarray(xy), cfg, args, args.plots)
    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing to check")
        return
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    tot = df.groupby("variant").apply(lambda g: pd.Series({
        "recall": np.average(g["recall"].fillna(0), weights=g["n_rock_pts"]),
        "cliff_share": g["cliff_share"].mean(),
        "isolated": g["isolated"].mean(),
    }), include_groups=False)
    print("\nall areas (recall weighted by known-rock points):")
    print(tot.round(3).to_string())
    if args.out:
        df.to_csv(args.out, index=False)
        print(json.dumps({"wrote": str(args.out)}))


if __name__ == "__main__":
    main()
