#!/usr/bin/env python
"""2D model - the two figures of a results write-up (docs/results/model2d/<data>/).

    python scripts/model2d_figures.py --data v4 --run gradient-full_discriminative-nopos-s0

* ``examples_correct_vs_wrong.png``: bare-earth elevation of 2 correctly and 2 wrongly
  scored test samples per sample type, for one trained run at its validation cutoff
  (random picks, fixed seed).
* ``model_inputs_gradient.png``: what the gradient model sees for one test sample per type:
  elevation, the east and north gradient channels, and slope with uphill arrows; plus one
  sample after a training rotation.
* ``hcr_climb_vs_nc_by_wall.png``: the Horseshoe Canyon hand points on a hillshade map of
  each wall section, with the LiDAR wall mask and Mountain Project wall pins (route count),
  and below each map the same points laid out along the wall. Filled = kept in the
  fine-tuning set; hollow = judged but dropped by the set's sample rules.

Inputs: ``data/raster/<data>`` (images), ``runs/model2d/<data>/<group>/<run>``
(predictions + cutoff), ``data/finetune/<data>`` (hand points kept / dropped), the
Horseshoe LiDAR tiles and ``data/mountain_project/route_finder_105903004.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from crag import model2d_train as m2  # noqa: E402
from crag.model2d_raster import CHANNELS, RASTER_DIR  # noqa: E402
from crag.paths import FINETUNE_DIR  # noqa: E402

HCR = "horseshoe-canyon-ranch"
HCR_MP_CSV = ROOT / "data" / "mountain_project" / "route_finder_105903004.csv"
SECTIONS = ["North Forty", "West Side", "East Side"]
TYPE_LABELS = {  # row order and row titles of the examples figure
    "osm": "climbable: OSM route start",
    "handgathered": "climbable: Horseshoe hand point",
    "handgathered_nc": "not climbable: Horseshoe hand-checked",
    "band": "not climbable: band (between routes)",
    "far_cliff": "not climbable: far cliff",
    "stepback": "not climbable: stepback (slope below)",
}
BLUE, ORANGE = "#2a6fdb", "#f07d1a"


def find_run(data: str, run: str) -> Path:
    hits = [p for p in m2.runs_root(data).glob(f"*/{run}") if (p / "eval_best.json").exists()]
    if not hits:
        sys.exit(f"no scored run {run!r} under {m2.runs_root(data)}")
    return hits[0]


def examples(data: str, run_dir: Path, out: Path, per_side: int = 2, seed: int = 0) -> None:
    cut = json.loads((run_dir / "eval_best.json").read_text())["val_cutoff"]
    pred = pd.read_csv(run_dir / "predictions_best.csv")
    index = pd.read_csv(RASTER_DIR / data / "index.csv")
    images = np.load(RASTER_DIR / data / "images.npy", mmap_mode="r")
    meta = json.loads((RASTER_DIR / data / "meta.json").read_text())
    half = meta["config"]["size_m"] / 2
    p = pred[pred["split"] == "test"].merge(index[["sample_id", "row"]], on="sample_id")
    p["correct"] = np.where(p["label"] == 1, p["score"] >= cut, p["score"] < cut)
    types = [t for t in TYPE_LABELS if t in set(p["source"])]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(types), 2 * per_side, figsize=(3.6 * per_side * 2, 3.5 * len(types)), squeeze=False)
    for r, t in enumerate(types):
        g = p[p["source"] == t]
        picks = []
        for ok in (True, False):
            pool = g[g["correct"] == ok]
            take = pool.iloc[rng.choice(len(pool), min(per_side, len(pool)), replace=False)] if len(pool) else pool
            picks += [(ok, s) for _, s in take.iterrows()] + [(ok, None)] * (per_side - len(take))
        for c, (ok, s) in enumerate(picks):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(TYPE_LABELS[t], fontsize=9)
            if s is None:
                for side in ax.spines.values():
                    side.set_visible(False)
                ax.text(0.5, 0.5, "none", ha="center", va="center", transform=ax.transAxes, color="0.5")
                continue
            dem = np.asarray(images[int(s["row"]), CHANNELS.index("dem")])
            ax.imshow(dem, cmap="terrain", extent=(-half, half, -half, half), origin="upper")
            if np.ptp(dem) > 2:
                ys = np.linspace(half, -half, dem.shape[0])
                xs = np.linspace(-half, half, dem.shape[1])
                ax.contour(xs, ys, dem, levels=np.arange(np.floor(dem.min()), dem.max(), 2.0), colors="k", linewidths=0.4, alpha=0.6)
            ax.plot(0, 0, "+", color="red", ms=14, mew=2)
            colour = "#1a7a35" if ok else "#c62828"
            for side in ax.spines.values():
                side.set_edgecolor(colour)
                side.set_linewidth(2.5)
            ax.set_title(f"{'CORRECT' if ok else 'WRONG'}  score {s['score']:.2f}\nrelief {np.ptp(dem):.0f} m  ({s['slug']})", fontsize=8, color=colour)
    fig.suptitle(f"Bare-earth elevation, {2 * half:g} m x {2 * half:g} m around each test sample (north up, red + = sample point, contours every 2 m)\n"
                 f"{data} 2D run {run_dir.name}; called climbable when score >= {cut:.2f} (cutoff tuned on validation). "
                 f"Left two: correct, right two: wrong. Random picks, seed {seed}.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)


INPUT_TYPES = ["osm", "handgathered", "handgathered_nc", "band"]  # one example each in the inputs figure


def inputs(data: str, run_dir: Path, out: Path, seed: int = 1, grad_lim: float = 4.0) -> None:
    """What the gradient model sees: elevation -> east / north gradient channels, for one test
    sample per type, plus one sample after a training rotation (channels recomputed after it)."""
    import torch

    from crag.model2d_raster import gradient, slope_deg

    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    std, res_m, size = float(ck["norm"]["std"][0]), float(ck["res_m"]), int(ck["cfg"]["model"]["input_size"]) if "cfg" in ck else 224
    pred = pd.read_csv(run_dir / "predictions_best.csv")
    index = pd.read_csv(RASTER_DIR / data / "index.csv")
    images = np.load(RASTER_DIR / data / "images.npy", mmap_mode="r")
    half = json.loads((RASTER_DIR / data / "meta.json").read_text())["config"]["size_m"] / 2
    p = pred[pred["split"] == "test"].merge(index[["sample_id", "row"]], on="sample_id")
    rng = np.random.default_rng(seed)
    rows = []
    for t in INPUT_TYPES:
        g = p[p["source"] == t]
        if len(g):
            s = g.iloc[int(rng.integers(len(g)))]
            dem = np.asarray(images[int(s["row"]), CHANNELS.index("dem")], dtype=np.float32)
            rows.append((TYPE_LABELS[t], s, dem, ""))
    if len(rows) > 1:  # the Horseshoe climbable example again, turned a quarter turn as in training
        lab, s, dem, _ = rows[1]
        rows.append((lab, s, np.rot90(dem, 1).copy(), "same sample turned 90 deg anticlockwise\n(training augmentation; score is the original's)"))

    ext = (-half, half, -half, half)
    n = rows[0][2].shape[-1]
    xs = np.linspace(-half, half, n)
    ys = np.linspace(half, -half, n)
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 3.9 * len(rows)), squeeze=False)
    heads = ["1. Elevation (bare earth, m above lowest cell)", "2. Input channel 1: east gradient\n(m of rise per m going east)",
             "3. Input channel 2: north gradient\n(m of rise per m going north)", "Slope (deg) and uphill direction\n(for reading only, not an input)"]
    for r, (lab, s, dem, note) in enumerate(rows):
        g = gradient(dem, res_m)
        east, north = g[0], g[1]
        ax = axes[r]
        im0 = ax[0].imshow(dem, cmap="terrain", extent=ext)
        if np.ptp(dem) > 2:
            ax[0].contour(xs, ys, dem, levels=np.arange(np.floor(dem.min()), dem.max(), 2.0), colors="k", linewidths=0.4, alpha=0.6)
        im1 = ax[1].imshow(east, cmap="RdBu_r", vmin=-grad_lim, vmax=grad_lim, extent=ext)
        im2 = ax[2].imshow(north, cmap="RdBu_r", vmin=-grad_lim, vmax=grad_lim, extent=ext)
        slope = slope_deg(dem, res_m)
        im3 = ax[3].imshow(slope, cmap="magma", vmin=0, vmax=90, extent=ext)
        k = 3  # one arrow per 1.5 m, unit length, only on ground steeper than 30 deg
        mag = np.hypot(east, north) + 1e-9
        steep = (slope >= 30)[::k, ::k]
        ax[3].quiver(*np.meshgrid(xs[::k], ys[::k]), np.where(steep, (east / mag)[::k, ::k], np.nan),
                     np.where(steep, (north / mag)[::k, ::k], np.nan), color="cyan", scale=22, width=0.005, alpha=0.9)
        for a in ax:
            a.plot(0, 0, "+", color="lime" if a is ax[3] else "red", ms=12, mew=2)
            a.set_xticks([-half, 0, half])
            a.set_yticks([-half, 0, half])
            a.tick_params(labelsize=7)
        title = f"{lab}\n{s['slug']}, model score {s['score']:.2f}"
        ax[0].set_ylabel(title + (f"\n{note}" if note else ""), fontsize=8.5)
        if r == 0:
            for a, h in zip(ax, heads):
                a.set_title(h, fontsize=9.5)
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
        for a, im in ((ax[1], im1), (ax[2], im2)):
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, extend="both").ax.tick_params(labelsize=7)
        fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.suptitle(f"What the 2D gradient model sees: {2 * half:g} m x {2 * half:g} m around each test sample, {res_m:g} m pixels ({n} x {n}), north up, + = sample point\n"
                 f"Red = ground rises in that direction, blue = it falls; colours stop at +-{grad_lim:g} m/m (a 76 deg slope), walls go far beyond. "
                 f"Before the network, both channels are divided by {std:.2f} (training-set spread) and enlarged to {size} x {size}.\n"
                 f"Model scores (P(climbable)) from {data} run {run_dir.name}. Triangle-shaped patches are ground filled in by interpolation where LiDAR saw no ground. The last row is the second sample turned 90 deg: the gradients are recomputed after the turn, so they still mean east and north.",
                 fontsize=10)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def hand_points(data: str) -> pd.DataFrame:
    """Every judged Horseshoe hand point: wall-foot position, label, section, kept in ``data`` or why dropped."""
    d = FINETUNE_DIR / data
    pa = pd.read_csv(d / "positives_all.csv")
    h = pa[pa["source"].astype(str).str.startswith("handgathered") & (pa["status"] == "ok")].copy()
    h["x"], h["y"] = h["foot_x"].fillna(h["snap_x"]).fillna(h["x"]), h["foot_y"].fillna(h["snap_y"]).fillna(h["y"])
    kept = set(pd.read_csv(d / "index.csv")["sample_id"])
    h["kept"] = h["route_uid"].isin(kept)
    drops = pd.read_csv(d / "dropped_samples.csv").drop_duplicates("sample").set_index("sample")["reason"]
    h["dropped_by"] = h["route_uid"].map(drops)
    return h[h["area_name"].isin(SECTIONS)].reset_index(drop=True)


def along_wall(xy: np.ndarray) -> np.ndarray:
    """Distance along the wall for points strung out along it: a nearest-neighbour path from
    the point furthest from the centre, cumulative step length (metres)."""
    n = len(xy)
    if n == 0:
        return np.zeros(0)
    start = int(np.argmax(np.hypot(*(xy - xy.mean(0)).T)))
    order, left, pos = [start], set(range(n)) - {start}, np.zeros(n)
    while left:
        cur = order[-1]
        nxt = min(left, key=lambda j: np.hypot(*(xy[j] - xy[cur])))
        pos[nxt] = pos[cur] + np.hypot(*(xy[nxt] - xy[cur]))
        order.append(nxt)
        left.remove(nxt)
    return pos


def mp_pins(to_work) -> pd.DataFrame:
    mp = pd.read_csv(HCR_MP_CSV)
    mp["wall"] = mp["Location"].str.split(" > ").str[0]
    w = mp.groupby("wall").agg(n=("Route", "size"), lat=("Area Latitude", "first"), lon=("Area Longitude", "first")).reset_index()
    xy = to_work(w["lon"].to_numpy(), w["lat"].to_numpy())
    return w.assign(x=xy[:, 0], y=xy[:, 1])


def hcr_by_wall(data: str, out: Path, margin_m: float = 25.0) -> None:
    from crag import registry
    from crag.crop import AreaCloud
    from crag.walls import local_grid

    pts = hand_points(data)
    pins = mp_pins(registry.area_crs(registry.areas(registry.load(), [HCR], enabled_only=False)[0]).to_work)
    cloud = AreaCloud(HCR)
    fig = plt.figure(figsize=(6.2 * len(SECTIONS), 9.5))
    gs = fig.add_gridspec(2, len(SECTIONS), height_ratios=[3.2, 1])
    for c, sec in enumerate(SECTIONS):
        g = pts[pts["area_name"] == sec].reset_index(drop=True)
        x0, x1 = g["x"].min() - margin_m, g["x"].max() + margin_m
        y0, y1 = g["y"].min() - margin_m, g["y"].max() + margin_m
        half = max(x1 - x0, y1 - y0) / 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        grid = local_grid(cloud, cx, cy, half)
        ax = fig.add_subplot(gs[0, c])
        if grid is not None:
            ny, nx = grid.dem.shape
            ext = (grid.x0, grid.x0 + nx * grid.res, grid.y0, grid.y0 + ny * grid.res)
            dem = grid.dem
            rgb = LightSource(azdeg=315, altdeg=45).shade(dem, cmap=plt.cm.terrain, vert_exag=2, blend_mode="soft", dx=grid.res, dy=grid.res,
                                                          vmin=np.nanpercentile(dem, 1), vmax=np.nanpercentile(dem, 99.5))
            ax.imshow(rgb, extent=ext, origin="lower")
            ax.imshow(np.ma.masked_where(~grid.wall, np.ones_like(dem)), extent=ext, origin="lower", cmap="gray_r", vmin=0, vmax=1, alpha=0.85)
            xs, ys = np.linspace(ext[0], ext[1], nx), np.linspace(ext[2], ext[3], ny)
            ax.contour(xs, ys, dem, levels=np.arange(np.floor(np.nanmin(dem) / 5) * 5, np.nanmax(dem), 5), colors="k", linewidths=0.3, alpha=0.5)
        for lab, col in ((1, BLUE), (0, ORANGE)):
            k = g[(g["label"] == lab) & g["kept"]]
            dr = g[(g["label"] == lab) & ~g["kept"]]
            ax.scatter(dr["x"], dr["y"], s=26, facecolors="none", edgecolors=col, linewidths=1.2, zorder=3)
            ax.scatter(k["x"], k["y"], s=34, color=col, edgecolors="k", linewidths=0.5, zorder=4)
        pin = pins[(pins["x"].between(cx - half, cx + half)) & (pins["y"].between(cy - half, cy + half))]
        ax.scatter(pin["x"], pin["y"], marker="*", s=260, color="gold", edgecolors="k", linewidths=0.8, zorder=5)
        for _, q in pin.iterrows():
            ax.annotate(f"{q['wall']} ({q['n']})", (q["x"], q["y"]), xytext=(6, 6), textcoords="offset points", fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8), zorder=6)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        n = lambda lab, kept: int(((g["label"] == lab) & (g["kept"] == kept)).sum())  # noqa: E731
        ax.set_title(sec, fontsize=12, weight="bold")
        ax.legend(handles=[
            Line2D([], [], ls="", marker="o", color=BLUE, mec="k", label=f"climbable, kept ({n(1, True)})"),
            Line2D([], [], ls="", marker="o", mfc="none", mec=BLUE, label=f"climbable, dropped ({n(1, False)})"),
            Line2D([], [], ls="", marker="o", color=ORANGE, mec="k", label=f"not climbable (Nc), kept ({n(0, True)})"),
            Line2D([], [], ls="", marker="o", mfc="none", mec=ORANGE, label=f"not climbable (Nc), dropped ({n(0, False)})"),
            Line2D([], [], ls="", marker="*", color="gold", mec="k", ms=12, label="Mountain Project wall pin"),
        ], loc="lower left", fontsize=7, framealpha=0.85)
        # along-wall strip
        g["along"] = along_wall(g[["x", "y"]].to_numpy(float))
        bx = fig.add_subplot(gs[1, c])
        jit = np.random.default_rng(0).uniform(-0.12, 0.12, len(g))
        for lab, col, yv in ((1, BLUE, 1), (0, ORANGE, 0)):
            m = g["label"] == lab
            bx.scatter(g.loc[m & ~g["kept"], "along"], yv + jit[(m & ~g["kept"]).to_numpy()], s=26, facecolors="none", edgecolors=col, linewidths=1.2)
            bx.scatter(g.loc[m & g["kept"], "along"], yv + jit[(m & g["kept"]).to_numpy()], s=34, color=col, edgecolors="k", linewidths=0.5)
        bx.set_yticks([0, 1], ["Nc", "climbable"])
        bx.set_ylim(-0.6, 1.6)
        bx.set_xlabel("distance along the wall (m)")
        bx.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Horseshoe Canyon Ranch: hand-checked climbable (blue) vs not-climbable (orange) spots at the wall foot, and Mountain Project wall pins (stars, route count)\n"
                 f"Filled = kept in fine-tuning set {data}; hollow = judged but dropped by its sample rules (duplicate, strip too narrow, too few points). "
                 f"Black = steep wall from LiDAR. Bottom: the same points laid out along each wall.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="v4", help="dataset version")
    ap.add_argument("--run", default="gradient-full_discriminative-nopos-s0", help="run folder name for the examples figure")
    ap.add_argument("--out", type=Path, help="default: docs/results/model2d/<data>/")
    ap.add_argument("--only", choices=["examples", "hcr", "inputs"], help="make one figure only")
    args = ap.parse_args()
    out = args.out or ROOT / "docs" / "results" / "model2d" / args.data
    out.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "examples"):
        examples(args.data, find_run(args.data, args.run), out / "examples_correct_vs_wrong.png")
        print("->", out / "examples_correct_vs_wrong.png")
    if args.only in (None, "inputs"):
        inputs(args.data, find_run(args.data, args.run), out / "model_inputs_gradient.png")
        print("->", out / "model_inputs_gradient.png")
    if args.only in (None, "hcr"):
        hcr_by_wall(args.data, out / "hcr_climb_vs_nc_by_wall.png")
        print("->", out / "hcr_climb_vs_nc_by_wall.png")


if __name__ == "__main__":
    main()
