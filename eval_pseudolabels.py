"""Score the surface-normal pseudolabels (and the model) against hand-gathered ground truth.

Plan and metric definitions: ``docs/ground-truth-evaluation-plan.md``. In short:

* **Q1 rock recall** - is there a cliff column (>= ``min_steep_points`` steep
  points in a ``route_radius_m`` cylinder) within r of each Climb / Nc
  waypoint? The false-cliff rate is the same test at random far-field points.
* **Q2 Climb vs Nc** - do steepness scores separate climbable from
  non-climbable rock (AUROC, PR)?
* **Q3 label agreement** - which weak samples in ``samples.csv`` sit within r
  of each waypoint, and how far the MP wall pins are from named routes.
* **Q4 model** - P(climbable) of a trained checkpoint at each waypoint.
* **Sweep** - Q1/Q2 over the heuristic's knobs, on the LiDAR near the waypoints.

Run everything and write ``runs/gt_eval/``::

    python eval_pseudolabels.py                 # all of it
    python eval_pseudolabels.py --no-sweep --mode finetune
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

from config import PROJECT_DIR, ExperimentConfig
from dataset import normalize_patch, resample
from groundtruth import AREAS, load_waypoints, snap_to_cliff
from labeling import (
    PointCloud,
    compute_normals,
    extract_patch,
    route_profile,
    slope_deg,
    steep_fraction,
    wall_direction,
)
from model import PointNet2Cls
from mountain_project import filter_bbox, load_routes
from usgs_ept import load_region, lonlat_to_work

OUT_DIR = PROJECT_DIR / "runs" / "gt_eval"
PROFILE_HEIGHT_M = 15.0  # route_profile height when no route length is known


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def max_cliff_count(steep_tree: cKDTree, xy: np.ndarray, r: float, cyl_r: float) -> np.ndarray:
    """Per query point: the most steep points any ``cyl_r`` cylinder within ``r`` holds.

    Cylinder centres are the steep points themselves (thinned to a 0.5 m grid),
    the same candidates :func:`labeling.place_route_cylinders` uses.
    """
    out = np.zeros(len(xy), dtype=int)
    if steep_tree.n == 0:
        return out
    for i, near in enumerate(steep_tree.query_ball_point(xy, r)):
        if not near:
            continue
        cand = steep_tree.data[near]
        _, first = np.unique(np.round(cand / 0.5).astype(np.int64), axis=0, return_index=True)
        out[i] = steep_tree.query_ball_point(cand[first], cyl_r, return_length=True).max()
    return out


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def best_f1(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """(best F1 over thresholds, that threshold)."""
    ok = np.isfinite(s)
    p, r, t = precision_recall_curve(y[ok], s[ok])
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.argmax(f1[:-1])) if len(t) else 0
    return (float(f1[i]), float(t[i])) if len(t) else (np.nan, np.nan)


def bootstrap_ci(fn, *arrays, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI of ``fn(*arrays)`` (resampling rows)."""
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        try:
            v = fn(*(a[idx] for a in arrays))
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def by_scope(gt: pd.DataFrame):
    """Yield ("all", gt) then one (area name, rows) per area."""
    yield "all", gt
    for a, name in AREAS.items():
        sub = gt[gt["area"] == a]
        if len(sub):
            yield name, sub


def random_far_points(
    pc: PointCloud, avoid_xy: np.ndarray, n: int, min_dist: float, min_points: int, seed: int = 0
) -> np.ndarray:
    """Random region points at least ``min_dist`` from every ``avoid_xy`` that have LiDAR."""
    rng = np.random.default_rng(seed)
    tree = cKDTree(avoid_xy)
    minx, miny, maxx, maxy = pc.bounds_xy
    out = []
    for _ in range(n * 50):
        if len(out) >= n:
            break
        c = rng.uniform([minx + 20, miny + 20], [maxx - 20, maxy - 20])
        if tree.query(c)[0] >= min_dist and len(pc.column(c, 5.0)) >= min_points:
            out.append(c)
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Q1 + Q2: the heuristic at each waypoint
# ---------------------------------------------------------------------------


def point_scores(pc: PointCloud, gt: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Heuristic scores per waypoint.

    ``max_steep_r{r}`` (from the raw waypoint) drives Q1; the column scores
    at the evaluation point (snapped, else raw) drive Q2.
    """
    lab = cfg.label
    gt = gt.copy()
    raw = gt[["x", "y"]].to_numpy()
    for r in cfg.gt.tolerances_m:
        gt[f"max_steep_r{r:g}"] = max_cliff_count(pc.steep_tree_xy, raw, r, lab.route_radius_m)
    rows = []
    for c in gt[["eval_x", "eval_y"]].to_numpy():
        frac, _ = steep_fraction(pc, c, cfg.patch.radius_m)
        idx = pc.column(c, lab.route_radius_m)
        steep = idx[pc.steep[idx]]
        row = {"steep_frac_5m": frac, "n_steep_2p5m": len(steep)}
        if len(steep) >= 3:
            base_z = float(np.quantile(pc.xyz[idx, 2], 0.05))
            prof = route_profile(pc, idx, base_z, base_z + PROFILE_HEIGHT_M, wall_direction(pc, idx))
            row |= prof | {"wall_height_m": float(np.ptp(pc.xyz[steep, 2]))}
        rows.append(row)
    return pd.concat([gt, pd.DataFrame(rows, index=gt.index)], axis=1)


def rock_recall_table(
    scores: pd.DataFrame, far_counts: dict[float, np.ndarray], cfg: ExperimentConfig
) -> pd.DataFrame:
    rows = []
    k = cfg.label.min_steep_points
    for r in cfg.gt.tolerances_m:
        far = far_counts[r] >= k
        rows.append({"scope": "random far-field", "r_m": r, "n": len(far),
                     "rate": far.mean(), "ci": bootstrap_ci(np.mean, far)})
        for scope, sub in by_scope(scores):
            for name, lab in (("Climb", 1), ("Nc", 0), ("Climb+Nc", None)):
                s = sub if lab is None else sub[sub["label"] == lab]
                hit = (s[f"max_steep_r{r:g}"] >= k).to_numpy()
                if len(hit):
                    rows.append({"scope": f"{scope} {name}", "r_m": r, "n": len(hit),
                                 "rate": hit.mean(), "ci": bootstrap_ci(np.mean, hit)})
    df = pd.DataFrame(rows)
    df["ci_lo"], df["ci_hi"] = zip(*df.pop("ci"))
    return df


Q2_SCORES = ["max_steep_r10", "n_steep_2p5m", "steep_frac_5m", "coverage",
             "max_gap_m", "lean_deg", "wall_height_m"]


def separation_table(scores: pd.DataFrame) -> pd.DataFrame:
    """AUROC (Climb = positive) of each score, overall and per area."""
    rows = []
    for scope, sub in by_scope(scores):
        y = sub["label"].to_numpy()
        for col in Q2_SCORES:
            s = sub[col].to_numpy(dtype=float)
            auc = safe_auroc(y, s)
            lo, hi = bootstrap_ci(safe_auroc, y, s)
            rows.append({"scope": scope, "score": col, "n_climb": int((y == 1).sum()),
                         "n_nc": int((y == 0).sum()), "auroc": auc, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Q3: agreement with the weak labels
# ---------------------------------------------------------------------------


def label_agreement(samples: pd.DataFrame, gt: pd.DataFrame, r: float) -> pd.DataFrame:
    """Crosstab of waypoint label x which weak labels lie within ``r`` of its eval point."""
    tree = cKDTree(samples[["x", "y"]].to_numpy())
    lab = samples["label"].to_numpy()
    near = tree.query_ball_point(gt[["eval_x", "eval_y"]].to_numpy(), r)
    cat = []
    for idx in near:
        has_pos, has_neg = (lab[idx] == 1).any(), (lab[idx] == 0).any()
        cat.append("both" if has_pos and has_neg else "weak positive" if has_pos
                   else "weak negative" if has_neg else "no weak label")
    gt = gt.assign(weak=cat, gt_label=gt["label"].map({1: "Climb", 0: "Nc"}))
    tab = pd.crosstab(gt["gt_label"], gt["weak"])
    return tab.reindex(columns=["weak positive", "both", "weak negative", "no weak label"], fill_value=0)


def mp_pin_error(gt: pd.DataFrame, routes: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """For named Climb waypoints: distance to the route's MP wall pin and to its assigned cylinder."""
    named = gt[gt["route_id"].notna()].copy()
    named["route_id"] = named["route_id"].astype("int64")
    named = named.merge(routes[["route_id", "route", "wall", "lat", "lon"]].rename(
        columns={"lat": "pin_lat", "lon": "pin_lon"}), on="route_id", how="left")
    pin = lonlat_to_work(named["pin_lon"].to_numpy(), named["pin_lat"].to_numpy())
    named["pin_dist_m"] = np.hypot(named["x"] - pin[:, 0], named["y"] - pin[:, 1])
    cyl = samples[samples["source"] == "mp_route"].drop_duplicates("route").set_index("route")
    has = named["route"].isin(cyl.index)
    named["cylinder_dist_m"] = np.nan
    named.loc[has, "cylinder_dist_m"] = np.hypot(
        named.loc[has, "eval_x"].to_numpy() - cyl.loc[named.loc[has, "route"], "x"].to_numpy(),
        named.loc[has, "eval_y"].to_numpy() - cyl.loc[named.loc[has, "route"], "y"].to_numpy(),
    )
    return named[["name", "area_name", "wall", "pin_dist_m", "cylinder_dist_m", "status", "snap_dist_m"]]


# ---------------------------------------------------------------------------
# Q4: the model
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path, device: torch.device) -> PointNet2Cls:
    model = PointNet2Cls(num_classes=2)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


@torch.no_grad()
def model_scores(
    model: PointNet2Cls, pc: PointCloud, gt: pd.DataFrame, cfg: ExperimentConfig,
    ring_m: tuple[float, ...] = (2.5, 5.0), batch_size: int = 64,
) -> pd.DataFrame:
    """``prob_center`` at each eval point and ``prob_max`` over 8-direction rings around it."""
    device = next(model.parameters()).device
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    offsets = np.vstack([[0.0, 0.0]] + [np.c_[r * np.cos(angles), r * np.sin(angles)] for r in ring_m])
    centers, owner = [], []
    for i, c in enumerate(gt[["eval_x", "eval_y"]].to_numpy()):
        for o in offsets:
            centers.append(c + o)
            owner.append(i)
    probs = np.full(len(centers), np.nan)
    batch, where = [], []

    def flush():
        x = torch.from_numpy(np.stack(batch)).to(device)
        probs[where] = torch.softmax(model(x), 1)[:, 1].cpu().numpy()
        batch.clear()
        where.clear()

    for j, c in enumerate(centers):
        pts = extract_patch(pc, c, cfg.patch)
        if len(pts) < cfg.patch.min_points:
            continue
        pts = normalize_patch(resample(pts, cfg.patch.num_points, np.random.default_rng(j)), cfg.patch)
        batch.append(np.ascontiguousarray(pts.T, dtype=np.float32))
        where.append(j)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    owner = np.asarray(owner)
    out = gt.copy()
    out["prob_center"] = probs[:: len(offsets)]
    out["prob_max"] = [np.nanmax(probs[owner == i]) if np.isfinite(probs[owner == i]).any() else np.nan
                       for i in range(len(gt))]
    return out


def model_table(scored: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    train = samples[samples["split"] == "train"][["x", "y"]].to_numpy()
    d = cKDTree(train).query(scored[["eval_x", "eval_y"]].to_numpy())[0]
    scored = scored.assign(near_train=np.where(d < 7.5, "<7.5 m from train", ">=7.5 m from train"))
    rows = []
    groups = list(by_scope(scored)) + [(k, g) for k, g in scored.groupby("near_train")]
    for scope, sub in groups:
        y = sub["label"].to_numpy()
        for col in ("prob_center", "prob_max"):
            s = sub[col].to_numpy(dtype=float)
            ok = np.isfinite(s)
            f1_05 = f1_score(y[ok], s[ok] >= 0.5, zero_division=0) if ok.any() else np.nan
            lo, hi = bootstrap_ci(safe_auroc, y, s)
            rows.append({"scope": scope, "score": col, "n": int(ok.sum()), "auroc": safe_auroc(y, s),
                         "ci_lo": lo, "ci_hi": hi, "f1@0.5": f1_05, "best_f1": best_f1(y, s)[0]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sweep over the heuristic's knobs
# ---------------------------------------------------------------------------


def threshold_sweep(
    cloud: dict, gt: pd.DataFrame, far_xy: np.ndarray, cfg: ExperimentConfig,
    r: float = 10.0, buffer_m: float = 30.0,
) -> pd.DataFrame:
    """Rock recall, false-cliff rate and Climb-vs-Nc AUROC for each knob setting.

    Normals are recomputed only on the LiDAR within ``buffer_m`` of a waypoint
    or far-field point (the buffer keeps the PCA neighbourhoods at the query
    points intact). Normal settings are varied one at a time around the
    defaults; slope / single-return / min-points are crossed with each.
    """
    xyz, nret = cloud["xyz"], cloud["number_of_returns"]
    queries = np.vstack([gt[["x", "y"]].to_numpy(), far_xy])
    d = cKDTree(queries).query(xyz[:, :2], distance_upper_bound=buffer_m)[0]
    keep = np.isfinite(d)
    sub_xyz, sub_ret = xyz[keep], nret[keep]
    print(f"sweep subset: {keep.sum():,} of {len(xyz):,} points")
    lab = cfg.label
    normal_settings = [(lab.normal_k, lab.normal_voxel_m)]
    normal_settings += [(k, lab.normal_voxel_m) for k in (8, 24)]
    normal_settings += [(lab.normal_k, v) for v in (0.25, 1.0)]
    gt_xy, y = gt[["x", "y"]].to_numpy(), gt["label"].to_numpy()
    rows = []
    for k, vox in normal_settings:
        slope = slope_deg(compute_normals(sub_xyz, k, vox))
        for single in (True, False):
            for deg in (45, 50, 55, 60, 65, 70, 75):
                steep = slope >= deg
                if single:
                    steep &= sub_ret == 1
                tree = cKDTree(sub_xyz[steep, :2])
                g = max_cliff_count(tree, gt_xy, r, lab.route_radius_m)
                f = max_cliff_count(tree, far_xy, r, lab.route_radius_m)
                for m in (4, 8, 16):
                    rows.append({
                        "normal_k": k, "normal_voxel_m": vox, "single_return": single,
                        "steep_slope_deg": deg, "min_steep_points": m,
                        "rock_recall": float((g >= m).mean()),
                        "climb_recall": float((g[y == 1] >= m).mean()),
                        "nc_recall": float((g[y == 0] >= m).mean()),
                        "false_cliff_rate": float((f >= m).mean()),
                        "climb_vs_nc_auroc": safe_auroc(y, g.astype(float)),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def plot_area_maps(pc: PointCloud, gt: pd.DataFrame, samples: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (a, name) in zip(axes, AREAS.items()):
        g = gt[gt["area"] == a]
        x0, x1 = g["x"].min() - 40, g["x"].max() + 40
        y0, y1 = g["y"].min() - 40, g["y"].max() + 40
        st = pc.xyz[pc.steep_idx, :2]
        m = (st[:, 0] > x0) & (st[:, 0] < x1) & (st[:, 1] > y0) & (st[:, 1] < y1)
        ax.scatter(*st[m].T, s=0.3, c="0.6", rasterized=True, label="steep LiDAR point")
        s = samples[samples["x"].between(x0, x1) & samples["y"].between(y0, y1)]
        ax.scatter(*s[s["label"] == 1][["x", "y"]].to_numpy().T, s=40, marker="s",
                   facecolors="none", edgecolors="tab:blue", label="weak positive (mp_route)")
        ax.scatter(*s[s["label"] == 0][["x", "y"]].to_numpy().T, s=40, marker="x",
                   c="tab:orange", label="weak negative")
        for lab, color, txt in ((1, "tab:green", "GT Climb"), (0, "tab:red", "GT Nc")):
            gg = g[g["label"] == lab]
            ax.scatter(gg["x"], gg["y"], s=18, c=color, label=txt, zorder=3)
            ok = gg[gg["status"] == "ok"]
            ax.quiver(ok["x"], ok["y"], ok["snap_x"] - ok["x"], ok["snap_y"] - ok["y"],
                      angles="xy", scale_units="xy", scale=1, color=color, width=0.002, zorder=3)
        ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal", title=f"{name}: raw → snapped",
               xlabel="UTM 15N easting (m)")
    axes[0].set_ylabel("northing (m)")
    axes[0].legend(loc="lower left", fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_score_hists(scores: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for ax, col in zip(axes.flat, Q2_SCORES):
        for lab, color, txt in ((1, "tab:green", "Climb"), (0, "tab:red", "Nc")):
            v = scores.loc[scores["label"] == lab, col].dropna()
            ax.hist(v, bins=20, alpha=0.55, color=color, label=f"{txt} (n={len(v)})", density=True)
        ax.set_title(col)
    axes.flat[0].legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sweep(sweep: pd.DataFrame, cfg: ExperimentConfig, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    base = sweep[(sweep["normal_k"] == cfg.label.normal_k)
                 & (sweep["normal_voxel_m"] == cfg.label.normal_voxel_m)]
    for (single, m), g in base.groupby(["single_return", "min_steep_points"]):
        g = g.sort_values("steep_slope_deg")
        ax = axes[0]
        ax.plot(g["false_cliff_rate"], g["rock_recall"], "o-",
                label=f"single_return={single}, min_pts={m}")
        for _, row in g.iterrows():
            ax.annotate(f"{row['steep_slope_deg']:.0f}°", (row["false_cliff_rate"], row["rock_recall"]),
                        fontsize=6, xytext=(2, 2), textcoords="offset points")
        axes[1].plot(g["steep_slope_deg"], g["climb_vs_nc_auroc"], "o-",
                     label=f"single_return={single}")
    axes[0].set(xlabel="false-cliff rate (random far-field)", ylabel="rock recall (Climb+Nc), r=10 m",
                title=f"trade-off over steep_slope_deg (k={cfg.label.normal_k}, voxel={cfg.label.normal_voxel_m})")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0.5, c="0.5", ls="--")
    axes[1].set(xlabel="steep_slope_deg", ylabel="AUROC Climb vs Nc (max steep count)",
                title="does steepness separate Climb from Nc?")
    handles, labels = axes[1].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[1].legend(uniq.values(), uniq.keys(), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _md(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without the optional ``tabulate`` dependency."""
    if index:
        df = df.reset_index()
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", default="finetune", help="runs/hcr-<mode>/best.pt for Q4")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--n-far", type=int, default=150, help="random far-field points")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    cfg = ExperimentConfig()
    args.out.mkdir(parents=True, exist_ok=True)

    cloud = load_region(cfg.region)
    normals_path = cfg.region.cache_dir / f"normals_k{cfg.label.normal_k}_vox{cfg.label.normal_voxel_m}.npy"
    if normals_path.exists():
        normals = np.load(normals_path)
    else:
        normals = compute_normals(cloud["xyz"], cfg.label.normal_k, cfg.label.normal_voxel_m)
        np.save(normals_path, normals)
    pc = PointCloud.build(cloud["xyz"], cfg.label, normals, cloud["number_of_returns"])

    gt_all = snap_to_cliff(pc, load_waypoints(cfg.gt), cfg)
    gt_all.to_csv(args.out / "gt_snapped.csv", index=False)
    gt = gt_all[gt_all["label"] >= 0].reset_index(drop=True)
    samples = pd.read_csv(cfg.region.cache_dir / "samples.csv")
    routes = filter_bbox(load_routes(), cfg.region.bbox_lonlat)
    pins = lonlat_to_work(routes["lon"].to_numpy(), routes["lat"].to_numpy())

    far_xy = random_far_points(pc, np.vstack([pins, gt[["x", "y"]].to_numpy()]), args.n_far,
                               cfg.label.ignore_buffer_m, cfg.patch.min_points, seed=cfg.split.seed)
    far_counts = {r: max_cliff_count(pc.steep_tree_xy, far_xy, r, cfg.label.route_radius_m)
                  for r in cfg.gt.tolerances_m}

    scores = point_scores(pc, gt, cfg)
    scores.to_csv(args.out / "gt_scores.csv", index=False)
    q1 = rock_recall_table(scores, far_counts, cfg)
    q2 = separation_table(scores)
    q3 = {r: label_agreement(samples, gt, r) for r in cfg.gt.tolerances_m}
    pin_err = mp_pin_error(gt, routes, samples)
    snap_summary = gt_all.groupby(["area_name", "comment", "status"]).size().unstack(fill_value=0)
    snap_d = gt.loc[gt["status"] == "ok", "snap_dist_m"].describe()[["count", "50%", "75%", "max"]]

    report = ["# Ground-truth evaluation of the pseudolabels\n",
              "Generated by `eval_pseudolabels.py`; plan in `docs/ground-truth-evaluation-plan.md`.\n",
              f"Waypoints used: {len(gt)} ({(gt['label'] == 1).sum()} Climb, {(gt['label'] == 0).sum()} Nc); "
              f"random far-field points: {len(far_xy)}.\n",
              "## Snapping\n", _md(snap_summary),
              f"\nSnap distance (ok only): median {snap_d['50%']:.1f} m, 75th pct {snap_d['75%']:.1f} m, "
              f"max {snap_d['max']:.1f} m.\n",
              f"## Q1. Rock recall (cliff column with >= {cfg.label.min_steep_points} steep points within r)\n",
              _md(q1, False), "\n## Q2. Climb vs Nc separation (AUROC, Climb = positive)\n", _md(q2, False),
              "\n## Q3. Agreement with weak labels in samples.csv\n"]
    for r, tab in q3.items():
        report += [f"\n**r = {r:g} m**\n", _md(tab)]
    report += ["\n### MP pin error for named Climb waypoints\n",
               _md(pin_err.describe().loc[["count", "50%", "mean", "max"]]), "\n"]

    if not args.no_model:
        ckpt = PROJECT_DIR / "runs" / f"hcr-{args.mode}" / "best.pt"
        if ckpt.exists():
            from trainer import pick_device
            model = load_checkpoint(ckpt, pick_device())
            scored = model_scores(model, pc, gt, cfg)
            scored.to_csv(args.out / f"gt_model_{args.mode}.csv", index=False)
            summary = json.loads((ckpt.parent / "summary.json").read_text())
            report += [f"## Q4. Model `{args.mode}` on the waypoints\n",
                       f"Checkpoint trained {summary.get('epochs_run')} epochs "
                       "(2 = FAST_DEV_RUN, so treat these as a pipeline check).\n",
                       _md(model_table(scored, samples), False), "\n"]
        else:
            print(f"no checkpoint at {ckpt}, skipping Q4")

    if not args.no_sweep:
        sweep = threshold_sweep(cloud, gt, far_xy, cfg)
        sweep.to_csv(args.out / "sweep.csv", index=False)
        plot_sweep(sweep, cfg, args.out / "sweep.png")
        top = sweep.assign(gap=sweep["rock_recall"] - sweep["false_cliff_rate"]).nlargest(10, "gap")
        report += ["## Sweep (r = 10 m): top 10 by rock recall − false-cliff rate\n",
                    _md(top, False), "\n",
                    "Default setting:\n",
                    _md(sweep[(sweep["normal_k"] == cfg.label.normal_k)
                          & (sweep["normal_voxel_m"] == cfg.label.normal_voxel_m)
                          & (sweep["single_return"] == cfg.label.steep_single_return_only)
                          & (sweep["steep_slope_deg"] == cfg.label.steep_slope_deg)
                          & (sweep["min_steep_points"] == cfg.label.min_steep_points)
                          ], False), "\n"]

    plot_area_maps(pc, gt, samples, args.out / "area_maps.png")
    plot_score_hists(scores, args.out / "score_hists.png")
    (args.out / "report.md").write_text("\n".join(report))
    print(f"wrote {args.out}/report.md")


if __name__ == "__main__":
    main()
