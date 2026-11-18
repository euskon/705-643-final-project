#!/usr/bin/env python
"""Maps of route GPS points before and after the wall check (snapping), for chosen crags.

For each crag (a Mountain Project wall name in ``verified_routes.csv``) this draws, over the
same background (hillshade of the bare-earth ground grid, 2 m contours, the wall mask and the
cliff LiDAR points coloured by height above the ground):

* ``<crag>_1_before.png``: the raw route GPS points (OpenStreetMap route starts);
* ``<crag>_2_after.png``: what the wall check did to each: raw point -> anchor (the nearest
  spot on a real wall >= 6 m) -> snapped spot (straight in front of the raw point) -> wall
  foot (where the sample is centred), coloured by the result;
* ``<crag>_3_steps.png``: one route, step by step: every cliff spot searched within 20 m and
  why it passed or failed, the anchor and the wall line, the spot straight in front, the
  final position and its foot.

    python scripts/plot_snapping.py
    python scripts/plot_snapping.py --crags "Drive-By Crag,Bob Marley" --out docs/figures/v4/snapping
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from crag import registry
from crag.crop import AreaCloud
from crag.groundtruth import GroundTruthConfig, _spot
from crag.hcr_points import hillshade
from crag.labeling import LabelConfig
from crag.negatives import centre_on_foot, local_pc, snap_routes
from crag.paths import PROJECT_DIR, ROUTES_DIR
from crag.walls import grid_for, grid_from_crop, wall_frame_at

STATUS_COL = {"ok": "#1f9e3a", "too_short": "#e08a00", "off_wall": "#d62728", "no_lidar": "#7f7f7f"}
STATUS_TXT = {"ok": "passed: snapped to a wall", "too_short": "rejected: walls in reach are all under 6 m",
              "off_wall": "rejected: no wall within 20 m", "no_lidar": "no LiDAR"}


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def background(ax, crop: dict, grid, cx: float, cy: float, half: float, cbar: bool = True, fig=None):
    """Hillshade + contours + wall mask + cliff points, in metres from the window centre."""
    ny, nx = grid.dem.shape
    ext = [grid.x0 - cx, grid.x0 + nx * grid.res - cx, grid.y0 - cy, grid.y0 + ny * grid.res - cy]
    from scipy import ndimage

    # light smoothing: cells without LiDAR ground points are filled from their nearest neighbour,
    # which looks blocky in a hillshade (the wall check itself uses the same 1 m smoothing)
    ax.imshow(hillshade(ndimage.gaussian_filter(grid.dem, 1.0), grid.res), extent=ext, origin="lower", cmap="gray", vmin=0, vmax=1)
    xs = np.linspace(ext[0], ext[1], nx, endpoint=False) + grid.res / 2
    ys = np.linspace(ext[2], ext[3], ny, endpoint=False) + grid.res / 2
    ax.contour(xs, ys, grid.dem, levels=np.arange(np.floor(grid.dem.min() / 2) * 2, grid.dem.max() + 2, 2.0), colors="#8a6d3b", linewidths=0.3, alpha=0.6)
    ax.imshow(np.ma.masked_where(~grid.wall, grid.wall), extent=ext, origin="lower", cmap="Reds", alpha=0.35, vmin=0, vmax=1.3)
    st = crop["steep"]
    rel = crop["xyz"][st, :2] - [cx, cy]
    h = crop["hag"][st] if np.isfinite(crop["hag"]).any() else crop["xyz"][st, 2] - crop["xyz"][:, 2].min()
    sc = ax.scatter(rel[:, 0], rel[:, 1], c=np.clip(h, 0, 40), s=2, cmap="viridis", vmin=0, vmax=40, linewidths=0, alpha=0.8)
    if cbar and fig is not None:
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="cliff LiDAR points: height above the local ground (m)")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xlabel("metres east of the crag centre")
    ax.set_ylabel("metres north of the crag centre")
    bar = 20 if half > 40 else 5
    ax.plot([-half * 0.92, -half * 0.92 + bar], [-half * 0.92] * 2, "-", c="white", lw=3)
    ax.text(-half * 0.92 + bar / 2, -half * 0.88, f"{bar} m", color="white", ha="center", fontsize=8)
    ax.annotate("N", xy=(half * 0.9, half * 0.9), xytext=(half * 0.9, half * 0.75), color="white", ha="center", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="white"))


def bg_legend():
    return [Line2D([], [], marker="s", ls="", mfc="#f4a6a6", mec="none", ms=9, label="wall mask (steep bare-earth ground, >= 4 m relief)"),
            Line2D([], [], marker="o", ls="", mfc="#3b8e8c", mec="none", ms=5, label="cliff LiDAR points"),
            Line2D([], [], c="#8a6d3b", lw=0.8, label="2 m contours of the bare-earth ground")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", default="red-river-gorge")
    ap.add_argument("--crags", default="Drive-By Crag,Bob Marley,The Gallery,Gold Coast")
    ap.add_argument("--out", type=Path, default=PROJECT_DIR / "docs/figures/v4/snapping")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    area = next(a for a in registry.load()["areas"] if a["slug"] == args.slug)
    crs, cloud = registry.area_crs(area), AreaCloud(args.slug)
    routes = pd.read_csv(ROUTES_DIR / "verified_routes.csv")
    routes = routes[(routes["slug"] == args.slug) & (routes["label"] == 1)].reset_index(drop=True)
    xy = crs.to_work(routes["lon"].to_numpy(), routes["lat"].to_numpy())
    routes["x0"], routes["y0"] = xy[:, 0], xy[:, 1]
    lab, g = LabelConfig(), GroundTruthConfig()
    summary = []
    for crag in [c.strip() for c in args.crags.split(",")]:
        own = routes[routes["mp_wall"] == crag]
        if own.empty:
            print(f"{crag}: no routes")
            continue
        cx, cy = float(own["x0"].median()), float(own["y0"].median())
        d = np.hypot(own["x0"] - cx, own["y0"] - cy)
        half = float(np.clip(np.quantile(d, 0.9) + 20, 50, 130))
        inwin = routes[(np.abs(routes["x0"] - cx) <= half) & (np.abs(routes["y0"] - cy) <= half)].reset_index(drop=True)
        snapped = snap_routes(cloud, inwin, crs)
        feet = [centre_on_foot(cloud, None, np.array([r.snap_x, r.snap_y])) if r.status == "ok" else {"status": "x"} for r in snapped.itertuples()]
        snapped["foot_x"] = [f.get("foot_x", np.nan) for f in feet]
        snapped["foot_y"] = [f.get("foot_y", np.nan) for f in feet]
        crop = cloud.crop(cx, cy, half + 10)
        grid = grid_from_crop(crop)
        s = snapped.assign(rx=snapped["x"] - cx, ry=snapped["y"] - cy, ax_=snapped["anchor_x"] - cx, ay_=snapped["anchor_y"] - cy,
                           sx=snapped["snap_x"] - cx, sy=snapped["snap_y"] - cy, fx=snapped["foot_x"] - cx, fy=snapped["foot_y"] - cy)
        n_ok = int((s["status"] == "ok").sum())
        stats = s["status"].value_counts().to_dict()
        tag = slugify(crag)

        # 1. before
        fig, ax = plt.subplots(figsize=(11, 11.5))
        background(ax, crop, grid, cx, cy, half, fig=fig)
        ax.scatter(s["rx"], s["ry"], marker="x", c="yellow", s=40, linewidths=1.6, zorder=5)
        ax.legend(handles=bg_legend() + [Line2D([], [], marker="x", ls="", c="yellow", mew=1.6, label=f"raw route GPS point ({len(s)})")], loc="upper center",
                  bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8)
        ax.set_title(f"Red River Gorge, {crag} area: raw OpenStreetMap route start points, before the wall check", fontsize=11)
        fig.tight_layout()
        fig.savefig(args.out / f"{tag}_1_before.png", dpi=110)
        plt.close(fig)

        # 2. after
        fig, ax = plt.subplots(figsize=(11, 11.5))
        background(ax, crop, grid, cx, cy, half, fig=fig)
        for r in s.itertuples():
            col = STATUS_COL.get(r.status, "#7f7f7f")
            if r.status == "ok":
                if np.hypot(r.ax_ - r.sx, r.ay_ - r.sy) > 0.3:
                    ax.plot([r.rx, r.ax_], [r.ry, r.ay_], ":", c="white", lw=0.7, zorder=4)
                    ax.plot(r.ax_, r.ay_, "^", ms=5, mfc="none", mec="white", mew=0.8, zorder=5)
                ax.annotate("", (r.sx, r.sy), (r.rx, r.ry), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=9), zorder=6)
                if np.isfinite(r.fx):
                    ax.plot([r.sx, r.fx], [r.sy, r.fy], "-", c="#00e5ff", lw=1.0, zorder=6)
                    ax.plot(r.fx, r.fy, "s", ms=4, mfc="#00e5ff", mec="k", mew=0.4, zorder=7)
                ax.plot(r.sx, r.sy, "o", ms=5, mfc=col, mec="k", mew=0.5, zorder=7)
            ax.plot(r.rx, r.ry, "x", c="yellow" if r.status == "ok" else col, ms=6, mew=1.4, zorder=8)
        hs = bg_legend() + [Line2D([], [], marker="x", ls="", c="yellow", mew=1.4, label="raw GPS point"),
                            Line2D([], [], marker="^", ls=":", c="0.4", mfc="none", mec="k", label="anchor: nearest spot on a real wall (white; drawn only if different)"),
                            Line2D([], [], marker="o", ls="-", c=STATUS_COL["ok"], mec="k", label="snapped: straight in front, on that wall"),
                            Line2D([], [], marker="s", ls="-", c="#00e5ff", mec="k", label="wall foot: where the training sample is centred")]
        hs += [Line2D([], [], marker="x", ls="", c=STATUS_COL[k], mew=1.4, label=f"{STATUS_TXT[k]} ({v})") for k, v in stats.items() if k != "ok"]
        ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8)
        moved = s.loc[s["status"] == "ok", "snap_dist_m"]
        ax.set_title(f"{crag} area: after the wall check - {n_ok} of {len(s)} passed, median move {moved.median():.1f} m (max {moved.max():.1f} m)", fontsize=11)
        fig.tight_layout()
        fig.savefig(args.out / f"{tag}_2_after.png", dpi=110)
        plt.close(fig)

        # 3. one route step by step: one that slid along the wall if any (shows step 2), else the
        # one that moved furthest
        ok = s[(s["status"] == "ok")]
        if len(ok):
            slid = ok[(ok["along_shift_m"] > 0.4) & (ok["snap_dist_m"] > 2)]
            ex = slid.loc[slid["snap_dist_m"].idxmax()] if len(slid) else ok.loc[ok["snap_dist_m"].idxmax()]
            raw = np.array([ex["x"], ex["y"]])
            pc, _ = local_pc(cloud, raw[0], raw[1], 40.0)
            gfull = grid_for(cloud, raw[0], raw[1])
            gi = pc.column(raw, g.ground_radius_m)
            ground_z = float(np.quantile(pc.xyz[gi, 2], 0.1))
            near = pc.steep_tree_xy.query_ball_point(raw, g.snap_radius_m)
            cand = pc.xyz[pc.steep_idx[near], :2]
            _, first = np.unique(np.round(cand / 0.5).astype(np.int64), axis=0, return_index=True)
            cand = cand[first]
            why = []
            for c in cand:
                n = len(pc.steep_tree_xy.query_ball_point(c, lab.route_radius_m))
                if n < lab.min_steep_points:
                    why.append("few")
                    continue
                sp = _spot(pc, c, lab, g, gfull, ground_z, lab.min_steep_points)
                why.append("dz" if sp is None else ("short" if sp["h"] < g.min_wall_height_m else "pass"))
            why = np.array(why)
            anchor = np.array([ex["anchor_x"], ex["anchor_y"]])
            snap = np.array([ex["snap_x"], ex["snap_y"]])
            foot = np.array([ex["foot_x"], ex["foot_y"]])
            _, along = wall_frame_at(gfull, anchor)
            target = anchor + float((raw - anchor) @ along) * along
            zc = raw
            zh = 26.0
            zcrop = cloud.crop(zc[0], zc[1], zh + 8)
            zgrid = grid_from_crop(zcrop)
            fig, axs = plt.subplots(1, 3, figsize=(21, 7.6))
            cols = {"few": ("#bbbbbb", "under 8 cliff points within 2.5 m"), "dz": ("#9467bd", "foot not at the standing height (> 6 m off)"),
                    "short": ("#e08a00", "wall under 6 m (ground grid)"), "pass": ("#1f9e3a", "passes every check")}
            rel = lambda p: np.asarray(p) - zc
            for k, ax in enumerate(axs):
                background(ax, zcrop, zgrid, zc[0], zc[1], zh, cbar=False)
                ax.plot(0, 0, "x", c="yellow", ms=14, mew=3, zorder=9)
            ax = axs[0]
            ax.add_patch(plt.Circle((0, 0), g.snap_radius_m, fill=False, ec="yellow", ls="--", lw=1.2))
            for key, (col, txt) in cols.items():
                m = why == key
                if m.any():
                    rc = cand[m] - zc
                    ax.scatter(rc[:, 0], rc[:, 1], s=14, c=col, edgecolors="k", linewidths=0.2, label=f"{txt} ({m.sum()})", zorder=6)
            ra = rel(anchor)
            ax.plot(*ra, "^", ms=13, mfc="none", mec="white", mew=2.2, zorder=8)
            ax.legend(loc="lower right", fontsize=7.5, title="cliff spots within 20 m (every 0.5 m)", title_fontsize=8)
            ax.set_title("1. search: every cliff spot within 20 m is checked, nearest first;\nthe first that passes every check is the anchor (white triangle)", fontsize=10)
            ax = axs[1]
            ls = np.array([ra - 30 * along, ra + 30 * along])
            ax.plot(ls[:, 0], ls[:, 1], "--", c="white", lw=1.2, label="wall line through the anchor (along the wall)")
            rt = rel(target)
            ax.plot([0, rt[0]], [0, rt[1]], ":", c="yellow", lw=1.5, label="straight from the raw point to the wall line")
            ax.plot(*ra, "^", ms=13, mfc="none", mec="white", mew=2.2, label="anchor")
            ax.plot(*rt, "D", ms=9, mfc="none", mec="yellow", mew=2, label="spot straight in front of the person")
            rs = rel(snap)
            ax.plot(*rs, "o", ms=11, mfc=STATUS_COL["ok"], mec="k", mew=1, zorder=9, label="snapped: the nearest passing spot to it\n(a lighter check: 3 cliff points, 6 m wall)")
            ax.legend(loc="lower right", fontsize=7.5)
            ax.set_title(f"2. where on that wall: the spot straight in front of the raw point,\nso neighbouring routes keep their spacing (slid {ex['along_shift_m']:.1f} m along the wall)", fontsize=10)
            ax = axs[2]
            rf = rel(foot)
            ax.annotate("", rs, (0, 0), arrowprops=dict(arrowstyle="-|>", color=STATUS_COL["ok"], lw=2))
            ax.plot(*rs, "o", ms=11, mfc=STATUS_COL["ok"], mec="k", mew=1, zorder=9, label=f"snapped (moved {ex['snap_dist_m']:.1f} m)")
            ax.plot([rs[0], rf[0]], [rs[1], rf[1]], "-", c="#00e5ff", lw=2)
            ax.plot(*rf, "s", ms=10, mfc="#00e5ff", mec="k", mew=1, zorder=9, label="wall foot: outer edge of the wall mask")
            ax.add_patch(plt.Circle(rf, 2.5, fill=False, ec="#00e5ff", lw=1))
            ax.legend(loc="lower right", fontsize=7.5)
            ax.set_title(f"3. result: wall {ex['wall_height_m']:.0f} m tall on the ground grid; the training sample\nis centred on the wall foot and labels a 5 m-wide strip of the face", fontsize=10)
            name = ex["name"] if isinstance(ex["name"], str) else "unnamed route"
            fig.suptitle(f"{crag}: how the wall check snaps one route ('{name}'); yellow x = raw GPS point; metres from it", fontsize=12)
            for a in axs:
                a.set_xlabel("metres east of the raw GPS point")
                a.set_ylabel("metres north of the raw GPS point")
            fig.tight_layout()
            fig.savefig(args.out / f"{tag}_3_steps.png", dpi=100)
            plt.close(fig)
        summary.append({"crag": crag, "routes_in_window": len(s), **{f"n_{k}": v for k, v in stats.items()}, "median_move_m": round(float(moved.median()), 2),
                        "max_move_m": round(float(moved.max()), 2), "window_m": round(2 * half)})
        print(summary[-1])
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
