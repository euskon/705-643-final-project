#!/usr/bin/env python
"""Hand-gathered Horseshoe Canyon waypoints on a 1 m elevation / hillshade map with cliff points.

Checks whether the GPS waypoints (where people stood) sit on the rock, and
where fine-tuning's snapping moves them. See docs/hcr-waypoint-audit.md.

    python scripts/plot_hcr_waypoints.py                 # -> docs/figures/hcr-waypoints/

Outputs (in --out):
  wall_mask_steps.png       how the bare-earth wall mask is built, step by step
  overview.png              whole area, waypoints coloured by label, sub-area boxes
  zoom_<area>_<k>.png       ~150 m windows along each sub-area, raw GPS -> snapped position arrows
  distances.png             histograms: distance to nearest cliff column, snap distance, local relief
  profiles.png              elevation cross-sections through a sample of waypoints (perpendicular to wall)
  waypoints_metrics.csv     per-waypoint numbers
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crag import registry  # noqa: E402
from crag.groundtruth import load_waypoints  # noqa: E402
from crag.walls import WallConfig, wall_grid  # noqa: E402

SLUG = "horseshoe-canyon-ranch"
COL = {1: "#1f77ff", 0: "#ff7f0e", -1: "#888888"}
NAME = {1: "Climb", 0: "Nc", -1: "?"}


def load_cloud():
    xs, cl, last, single, gnd = [], [], [], [], []
    for f in sorted(glob.glob(str(ROOT / "data/lidar" / SLUG / "*.npz"))):
        st = ROOT / "data/pretrain" / SLUG / "stage" / Path(f).name
        d = np.load(f)
        if not st.exists() or len(d["xyz"]) == 0:
            continue
        s = np.load(st)
        xyz = d["xyz"].astype(np.float64)
        xyz[:, :2] += d["origin"]
        n = len(xyz)
        xs.append(xyz)
        cl.append(s["cls"][:n])
        last.append(d["return_number"] >= d["number_of_returns"])
        single.append(d["number_of_returns"] == 1)
        gnd.append(d["classification"] == 2)
    return np.vstack(xs), np.concatenate(cl), np.concatenate(last), np.concatenate(single), np.concatenate(gnd)


def rasterize(xyz, cls, last, single, ground, wcfg=None):
    """Area-wide rasters: the shared wall mask (crag.walls) plus cliff-label layers on its grid."""
    wcfg = wcfg or WallConfig()
    g = wall_grid(xyz, last & (cls != 1), wcfg, ground)
    ny, nx = g.wall.shape
    iy, ix = g.cell(xyz[:, :2])
    flat = iy * nx + ix
    cliff_n = np.bincount(flat[cls == 3], minlength=nx * ny).reshape(ny, nx)
    cliff1 = np.bincount(flat[(cls == 3) & single], minlength=nx * ny).reshape(ny, nx) > 0
    cliffm = np.bincount(flat[(cls == 3) & ~single], minlength=nx * ny).reshape(ny, nx) > 0
    steep = g.slope >= wcfg.min_slope_deg
    tall = g.relief >= wcfg.min_relief_m
    # "cliff column" = >= 8 cliff points within 2.5 m, on the grid
    yy, xx = np.mgrid[-3:4, -3:4]
    disc = (xx**2 + yy**2 <= 2.5**2).astype(float)
    col = ndimage.convolve(cliff_n.astype(float), disc, mode="constant") >= 8
    return dict(x0=g.x0, y0=g.y0, res=g.res, dem=g.dem, cliff_n=cliff_n, col=col & (cliff_n > 0), cliff1=cliff1, cliffm=cliffm,
                wall=g.wall, slope=g.slope, rel5=g.relief, steep=steep, tall=tall, wall_raw=steep & tall)


def extent(r):
    ny, nx = r["dem"].shape
    return [r["x0"], r["x0"] + nx * r["res"], r["y0"], r["y0"] + ny * r["res"]]


def draw_base(ax, r, win=None, cliff=True):
    # layers: hillshade, pink = multi-return cliff label, red = single-return cliff label,
    # black = bare-earth wall (DEM slope >= 50 deg), 5 m contours
    ls = LightSource(azdeg=315, altdeg=45)
    dem = r["dem"]
    rgb = ls.shade(dem, cmap=plt.cm.terrain, vert_exag=2, blend_mode="soft", dx=r["res"], dy=r["res"],
                   vmin=np.percentile(dem, 1), vmax=np.percentile(dem, 99.5))
    ax.imshow(rgb, origin="lower", extent=extent(r), interpolation="nearest")
    if cliff:
        for key, colr, a in (("cliffm", "#ff9ec8", 0.55), ("cliff1", "#d62728", 0.8), ("wall", "#111111", 0.85)):
            c = np.ma.masked_where(~r[key], np.ones_like(r["dem"]))
            ax.imshow(c, origin="lower", extent=extent(r), cmap=matplotlib.colors.ListedColormap([colr]),
                      alpha=a, interpolation="nearest")
        ny, nx = r["dem"].shape
        X = r["x0"] + (np.arange(nx) + 0.5) * r["res"]
        Y = r["y0"] + (np.arange(ny) + 0.5) * r["res"]
        if win:
            sx = (X >= win[0]) & (X <= win[1]); sy = (Y >= win[2]) & (Y <= win[3])
            cs = ax.contour(X[sx], Y[sy], r["dem"][np.ix_(sy, sx)], levels=np.arange(400, 800, 5), colors="k", linewidths=0.3, alpha=0.6)
    if win:
        ax.set_xlim(win[0], win[1]), ax.set_ylim(win[2], win[3])
    ax.set_aspect("equal")


def wp_metrics(gt, r, xyz, cls):
    ys, xs = np.nonzero(r["col"])
    col_xy = np.c_[r["x0"] + (xs + 0.5) * r["res"], r["y0"] + (ys + 0.5) * r["res"]]
    d_col, i_col = cKDTree(col_xy).query(gt[["x", "y"]].to_numpy())
    cliff_tree = cKDTree(xyz[cls == 3, :2])
    d_pt, _ = cliff_tree.query(gt[["x", "y"]].to_numpy())
    n5 = cliff_tree.query_ball_point(gt[["x", "y"]].to_numpy(), 5.0, return_length=True)
    n12 = cliff_tree.query_ball_point(gt[["x", "y"]].to_numpy(), 12.0, return_length=True)
    # relief within 10 m on the DEM
    rel = []
    for x, y in gt[["x", "y"]].to_numpy():
        ix, iy = int((x - r["x0"]) / r["res"]), int((y - r["y0"]) / r["res"])
        p = r["dem"][max(iy - 10, 0):iy + 11, max(ix - 10, 0):ix + 11]
        rel.append(float(p.max() - p.min()) if p.size else np.nan)
    ys2, xs2 = np.nonzero(r["wall"])
    wall_xy = np.c_[r["x0"] + (xs2 + 0.5) * r["res"], r["y0"] + (ys2 + 0.5) * r["res"]]
    d_wall, i_wall = cKDTree(wall_xy).query(gt[["x", "y"]].to_numpy())
    out = gt.copy()
    out["dist_wall_m"] = d_wall
    out["wall_x"], out["wall_y"] = wall_xy[i_wall, 0], wall_xy[i_wall, 1]
    out["dist_cliff_col_m"] = d_col
    out["near_col_x"], out["near_col_y"] = col_xy[i_col, 0], col_xy[i_col, 1]
    out["dist_cliff_pt_m"] = d_pt
    out["cliff_pts_5m"] = n5
    out["cliff_pts_12m"] = n12
    out["relief_10m"] = rel
    return out


def join_snapped(gt, dataset):
    """Wall-check result of every waypoint, from the fine-tune build's positives_all.csv (matched by id)."""
    p = ROOT / "data/finetune" / dataset / "positives_all.csv"
    if not p.exists():
        return gt.assign(snap_x=np.nan, snap_y=np.nan, status="not built", snap_dist_m=np.nan)
    chk = pd.read_csv(p, low_memory=False)
    chk = chk[chk["source"].str.startswith("handgathered")][["route_uid", "status", "snap_x", "snap_y", "wall_height_m"]]
    chk["gt_id"] = chk.pop("route_uid").str.removeprefix("hand:")
    m = gt.merge(chk, on="gt_id", how="left")
    m.loc[m["status"] != "ok", ["snap_x", "snap_y"]] = np.nan
    m["snap_dist_m"] = np.hypot(m["snap_x"] - m["x"], m["snap_y"] - m["y"])
    return m


def wall_profile(r, x, y, nx_, ny_, half=30.0, step=0.5):
    t = np.arange(-half, half + step, step)
    px, py = x + t * nx_, y + t * ny_
    ix = np.clip(((px - r["x0"]) / r["res"]).astype(int), 0, r["dem"].shape[1] - 1)
    iy = np.clip(((py - r["y0"]) / r["res"]).astype(int), 0, r["dem"].shape[0] - 1)
    return t, r["dem"][iy, ix]


def wall_steps(r, gt, path, half=60.0):
    """Six panels on one East Side window: DEM -> slope -> relief -> masks -> result."""
    e = gt[gt.area_name == "East Side"]
    cx, cy = e.x.median(), e.y.median()
    ix0, ix1 = int((cx - half - r["x0"]) / r["res"]), int((cx + half - r["x0"]) / r["res"])
    iy0, iy1 = int((cy - half - r["y0"]) / r["res"]), int((cy + half - r["y0"]) / r["res"])
    ext = [r["x0"] + ix0 * r["res"], r["x0"] + ix1 * r["res"], r["y0"] + iy0 * r["res"], r["y0"] + iy1 * r["res"]]
    cut = lambda a: a[iy0:iy1, ix0:ix1]
    w = e[(abs(e.x - cx) < half) & (abs(e.y - cy) < half)]
    panels = [
        ("1. bare-earth DEM: lowest last return per 1 m cell (m)", cut(r["dem"]), "terrain", None),
        ("2. slope of the DEM, smoothed sigma=1 m (deg)", cut(r["slope"]), "magma", (0, 80)),
        ("3. relief: max-min elevation in 5x5 m (m)", cut(r["rel5"]), "viridis", (0, 15)),
        ("4. slope >= 50 deg (white)", cut(r["steep"]).astype(float), "gray", (0, 1)),
        ("5. AND relief >= 4 m (white)", cut(r["wall_raw"]).astype(float), "gray", (0, 1)),
        ("6. opened with 2x2 m (drops 1-cell specks) = wall mask", cut(r["wall"]).astype(float), "gray", (0, 1)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, (title, img, cmap, lim) in zip(axes.flat, panels):
        im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None,
                       interpolation="nearest")
        if cmap != "gray":
            fig.colorbar(im, ax=ax, shrink=0.7)
        ax.scatter(w.x, w.y, s=14, c=[COL[l] for l in w.label], edgecolors="w", linewidths=0.4)
        ax.set_title(title, fontsize=10), ax.set_aspect("equal")
    fig.suptitle("Bare-earth wall mask, East Side window (dots = waypoints: blue Climb, orange Nc)")
    fig.tight_layout(), fig.savefig(path, dpi=90), plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs/figures/hcr-waypoints")
    ap.add_argument("--zoom-m", type=float, default=160.0)
    ap.add_argument("--dataset", default="v3", help="fine-tune set whose snapped (wall-checked) positions to draw")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    area = next(a for a in registry.areas(registry.load(), [SLUG], enabled_only=False))
    gt = load_waypoints(to_work=registry.area_crs(area).to_work)
    xyz, cls, last, single, ground = load_cloud()
    print(f"{len(xyz):,} points, cliff {np.mean(cls == 3):.1%}")
    r = rasterize(xyz, cls, last, single, ground)
    print(f"cliff label: single-return {np.mean((cls == 3) & single):.1%}, multi-return {np.mean((cls == 3) & ~single):.1%}")
    gt = join_snapped(wp_metrics(gt, r, xyz, cls), args.dataset)
    gt.to_csv(args.out / "waypoints_metrics.csv", index=False)

    # ---- overview
    pad = 120
    win = [gt.x.min() - pad, gt.x.max() + pad, gt.y.min() - pad, gt.y.max() + pad]
    fig, ax = plt.subplots(figsize=(14, 12))
    draw_base(ax, r, win)
    for lab in (1, 0, -1):
        g = gt[gt.label == lab]
        ax.scatter(g.x, g.y, s=18, c=COL[lab], edgecolors="k", linewidths=0.4, label=f"{NAME[lab]} ({len(g)})", zorder=5)
    for name, g in gt.groupby("area_name"):
        ax.annotate(name, (g.x.mean(), g.y.max() + 25), ha="center", fontsize=12, weight="bold",
                    bbox=dict(fc="white", alpha=0.8))
    ax.legend(loc="lower right", title="black = DEM wall, red/pink = single/multi-return cliff label")
    ax.set_title("Horseshoe Canyon Ranch - hand-gathered waypoints on 1 m hillshade + cliff columns (UTM m)")
    fig.tight_layout(), fig.savefig(args.out / "overview.png", dpi=110), plt.close(fig)

    # ---- zooms: tile each sub-area's waypoints into zoom-m windows
    Z = args.zoom_m
    for name, g in gt.groupby("area_name"):
        cells = {}
        for i, row in g.iterrows():
            cells.setdefault((int((row.x - g.x.min()) // Z), int((row.y - g.y.min()) // Z)), []).append(i)
        for k, ids in enumerate(sorted(cells.values(), key=lambda v: gt.loc[v, "x"].mean())):
            w = gt.loc[ids]
            cx, cy = w.x.mean(), w.y.mean()
            win = [cx - Z / 2 - 15, cx + Z / 2 + 15, cy - Z / 2 - 15, cy + Z / 2 + 15]
            fig, ax = plt.subplots(figsize=(12, 12))
            draw_base(ax, r, win)
            for _, p in w.iterrows():
                if np.isfinite(p.snap_x) and p.snap_dist_m > 0.3:
                    ax.annotate("", (p.snap_x, p.snap_y), (p.x, p.y),
                                arrowprops=dict(arrowstyle="->", color="white", lw=1.2), zorder=4)
                ax.scatter(p.x, p.y, s=60, c=COL[p.label], edgecolors="k", linewidths=0.8, zorder=5)
                if p.status == "off_wall":  # failed the wall check
                    ax.scatter(p.x, p.y, s=140, marker="x", c="k", linewidths=2, zorder=6)
                ax.annotate(f"{p['name'][:18] if isinstance(p['name'], str) else ''} {p.dist_wall_m:.0f}m",
                            (p.x, p.y), xytext=(4, 4), textcoords="offset points", fontsize=7, color="white",
                            path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground="black")], zorder=6)
            ax.scatter([], [], c=COL[1], label="Climb"), ax.scatter([], [], c=COL[0], label="Nc")
            ax.plot([], [], c="white", label="-> wall-checked position")
            ax.scatter([], [], marker="x", c="k", label="rejected: off wall"), ax.legend(loc="lower right")
            ax.set_title(f"{name} #{k + 1}: name + distance to bare-earth wall (black); red/pink = single/multi-return cliff label; 5 m contours")
            fig.tight_layout(), fig.savefig(args.out / f"zoom_{name.replace(' ', '_')}_{k + 1}.png", dpi=100), plt.close(fig)

    # ---- distance histograms
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, col, bins, title in [
        (axes[0], "dist_wall_m", np.arange(0, 41, 1), "distance to bare-earth wall, DEM slope>=50 (m)"),
        (axes[1], "snap_dist_m", np.arange(0, 21, 0.5), "GPS -> wall-checked position (m)"),
        (axes[2], "cliff_pts_5m", np.arange(0, 401, 10), "cliff points within 5 m"),
        (axes[3], "relief_10m", np.arange(0, 41, 1), "DEM relief within 10 m (m)"),
    ]:
        for lab in (1, 0):
            v = gt.loc[gt.label == lab, col].dropna()
            ax.hist(v.clip(upper=bins[-1]), bins=bins, alpha=0.6, color=COL[lab], label=f"{NAME[lab]} median {v.median():.1f}")
        ax.set_title(title), ax.legend()
    fig.tight_layout(), fig.savefig(args.out / "distances.png", dpi=100), plt.close(fig)

    # ---- profiles through the wall for a sample
    s = pd.concat([gt[gt.label == 1].sample(min(9, (gt.label == 1).sum()), random_state=0),
                   gt[gt.label == 0].sample(min(6, (gt.label == 0).sum()), random_state=0)])
    fig, axes = plt.subplots(3, 5, figsize=(22, 11))
    for ax, (_, p) in zip(axes.flat, s.iterrows()):
        d = np.array([p.wall_x - p.x, p.wall_y - p.y])
        d = d / (np.linalg.norm(d) + 1e-9) if np.linalg.norm(d) > 0.1 else np.array([1.0, 0.0])
        t, z = wall_profile(r, p.x, p.y, d[0], d[1])
        ax.plot(t, z, c="k")
        ax.axvline(0, c=COL[p.label], lw=2, label="GPS waypoint")
        ax.axvline(p.dist_wall_m, c="#111111", ls="--", label="nearest DEM wall")
        if np.isfinite(p.snap_x):
            ax.axvline(np.dot([p.snap_x - p.x, p.snap_y - p.y], d), c="green", ls=":", label="snapped")
        ax.set_title(f"{NAME[p.label]}: {str(p['name'])[:22]}", fontsize=9)
        ax.set_xlabel("m toward nearest wall"), ax.set_aspect("equal", adjustable="datalim")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout(), fig.savefig(args.out / "profiles.png", dpi=90), plt.close(fig)

    wall_steps(r, gt, args.out / "wall_mask_steps.png")

    cols = ["dist_wall_m", "dist_cliff_col_m", "dist_cliff_pt_m", "snap_dist_m", "cliff_pts_5m", "relief_10m"]
    print(gt.groupby(["area_name", "comment"])[cols].median().round(1).to_string())
    print(gt.groupby("comment")[cols].describe(percentiles=[.5, .9]).T.round(1).to_string())
    print("wall check:", gt.groupby(["comment", "status"]).size().to_dict())


if __name__ == "__main__":
    import matplotlib.patheffects  # noqa: F401
    main()
