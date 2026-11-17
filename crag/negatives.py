"""Verified positives and hard negatives for fine-tuning (proposal: docs/step3-routes-and-hard-negatives.md).

Positives: verified route starts (OpenStreetMap routes matched to Mountain
Project, or hand-gathered Climb waypoints) that pass the wall check
(``crag.groundtruth.snap_one``): moved to the foot of the nearest real wall
within 20 m whose base is at the ground height of the start point (6 m
tolerance). Points with no such wall (``off_wall``) are dropped, Nc
waypoints included.

Negatives, hardest first, each with a ``source`` and a ``confidence``:

``handgathered_nc``  rock judged *not climbable* on site (Horseshoe Canyon).
                     Confidence 1.0 - the only fully verified hard negatives.
``band``             cliff columns on the *same cliff line* between
                     ``band_min_m`` and ``band_max_m`` from every known route,
                     only where the neighbourhood is well documented
                     (>= ``doc_min_routes`` verified routes within
                     ``doc_radius_m`` and MP-wall coverage >= ``min_coverage``).
                     At a popular, completely mapped crag, a stretch of cliff
                     between routes that nobody climbs is most likely bad rock.
                     ``bracketed`` (routes on both sides) raises confidence.
``stepback``         talus / plateau 12-20 m out from / behind a route (the
                     proof of concept's contextual negatives; easy).
``far_cliff``        cliff > ``far_min_m`` from any route or MP crag pin:
                     medium hardness, noisiest (could be unclimbed good rock),
                     lowest confidence.

Verification applied to every generated negative:

* exclusion: no negative cylinder within ``exclusion_m`` of ANY known route
  location (all OSM routes incl. unmatched/boulders, all hand Climb points);
* geometry recorded for audit: cliff height, steep-point count, fraction of
  multi-return (vegetation) points on the face;
* ``validate_on_handgathered``: run the band generator at Horseshoe Canyon
  using only the *named* Climb waypoints as "documented routes", then check
  every negative against all 186 hand labels -> a measured precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dataclasses import replace

from crag.crop import AreaCloud
from crag.groundtruth import UNUSABLE_STATUS, GroundTruthConfig, snap_one, wall_foot
from crag.labeling import LabelConfig, PointCloud
from crag.walls import grid_for, wall_frame_at, wall_height_at

log = logging.getLogger(__name__)
DEFAULT_ROUTE_LENGTH_M = 15.0
SNAP_COLS = ("snap_x", "snap_y", "snap_dist_m", "ground_z", "wall_dist_m", "base_z", "wall_height_m", "wall_height_column_m", "top_z", "out_dx", "out_dy",
             "n_steep_snap", "anchor_x", "anchor_y", "along_shift_m", "nearest_wall_height_m", "nearest_wall_dist_m", "skipped_shorter_wall")
FOOT_COLS = ("foot_x", "foot_y", "base_z", "out_dx", "out_dy", "wall_height_m", "foot_move_m")


@dataclass
class NegativeConfig:
    exclusion_m: float = 7.5  # = 2.5 m route cylinder + 5 m input column
    band_min_m: float = 10.0
    band_max_m: float = 40.0
    doc_radius_m: float = 75.0
    doc_min_routes: int = 3
    min_coverage: float = 0.5
    grid_m: float = 2.0
    neg_spacing_m: float = 6.0
    band_per_positive: float = 1.0
    stepback_m: tuple[float, ...] = (12.0, 20.0)
    max_stepback_steep_frac: float = 0.25
    far_min_m: float = 300.0
    far_per_area: int = 40
    cylinder_m: float = 2.5
    min_steep_points: int = 8
    # every sample's wall must be this tall at its foot (ground grid) - the same rule as the route wall check
    min_wall_height_m: float = 6.0
    dup_m: float = 2.0       # two samples with the same label closer than this: keep one
    pair_min_m: float = 1.0  # a climbable and a not-climbable foot closer than this: dropped unless both confirmed


def local_pc(cloud: AreaCloud, cx: float, cy: float, half: float) -> tuple[PointCloud | None, dict]:
    c = cloud.crop(cx, cy, half)
    if len(c["xyz"]) < 32:
        return None, c
    steep_idx = np.flatnonzero(c["steep"])
    pc = PointCloud(
        xyz=c["xyz"],
        normals=c["normals"],
        steep=c["steep"],
        tree_xy=cKDTree(c["xyz"][:, :2]),
        steep_idx=steep_idx,
        steep_tree_xy=cKDTree(c["xyz"][steep_idx, :2] if len(steep_idx) else np.zeros((1, 2)) + 1e12),
        wall_dist=c["wall_dist"] if "wall_dist" in c and np.isfinite(c["wall_dist"]).any() else None,
    )
    return pc, c


def _ground_z(pc: PointCloud, xy, r: float = 3.0) -> float:
    idx = pc.column(xy, r)
    return float(np.quantile(pc.xyz[idx, 2], 0.1)) if len(idx) else np.inf


def _grid(cloud, x, y):
    """Cached bare-earth wall grid around a point (None for clouds that can't make one)."""
    try:
        return grid_for(cloud, float(x), float(y))
    except Exception:  # pragma: no cover - defensive: a broken tile must not stop a build
        log.exception("wall grid failed at %.0f, %.0f", x, y)
        return None


def centre_on_foot(cloud, pc: PointCloud | None, xy, cfg_min_height: float | None = None) -> dict:
    """``wall_foot`` at ``xy`` plus the ground-grid wall height at the foot: the one centring
    step every wall sample goes through, whatever its label. ``status``: ``ok`` / ``no_wall`` /
    ``too_short`` (when ``cfg_min_height`` is given)."""
    if pc is None:
        pc, _ = local_pc(cloud, xy[0], xy[1], 25.0)
        if pc is None:
            return {"status": "no_wall"}
    grid = _grid(cloud, xy[0], xy[1])
    f = wall_foot(pc, xy, grid)
    if f["status"] != "ok":
        return f
    f["wall_height_m"] = wall_height_at(grid, [f["foot_x"], f["foot_y"]], f["base_z"])[0] if grid is not None else float("nan")
    if cfg_min_height is not None and not f["wall_height_m"] >= cfg_min_height:
        f["status"] = "too_short"
    return f


def snap_routes(cloud: AreaCloud, routes: pd.DataFrame, crs, label: LabelConfig | None = None, gt: GroundTruthConfig | None = None) -> pd.DataFrame:
    """The wall check for route start points (see ``crag.groundtruth.snap_one``).

    Adds ``x, y`` (start), ``snap_*``, ``wall_dist_m``, ``base_z``, ``wall_height_m``,
    ``out_dx/dy`` and ``status``: ``ok``, ``too_short`` (walls in reach, all under the minimum
    height), ``off_wall`` (no wall within the search radius) or ``no_lidar``. Only ``ok``
    points may be used. Route *ways* use their lower end.

    Rows marked ``presnapped`` (the saved Horseshoe points) are never moved: the check only
    confirms a tall-enough wall within ``presnapped_radius_m``; if it fails the point is kept
    and marked ``presnapped_lost``. Rows whose saved ``snap_status`` is unusable keep it."""
    label, gt = label or LabelConfig(), gt or GroundTruthConfig()
    out = []
    for r in routes.itertuples(index=False):
        xy = crs.to_work(r.lon, r.lat)
        row = r._asdict()
        saved = str(getattr(r, "snap_status", "") or "")
        if saved in UNUSABLE_STATUS:
            out.append({**row, "x": float(xy[0]), "y": float(xy[1]), "status": saved})
            continue
        pc, _ = local_pc(cloud, xy[0], xy[1], 40.0)
        if pc is None:
            out.append({**row, "x": xy[0], "y": xy[1], "status": "no_lidar"})
            continue
        if pd.notna(getattr(r, "lat2", np.nan)):  # route line: pick the lower end as the start
            xy2 = crs.to_work(r.lon2, r.lat2)
            if _ground_z(pc, xy2) < _ground_z(pc, xy):
                xy = xy2
        xy = np.asarray(xy, float)
        grid = _grid(cloud, xy[0], xy[1])
        pre = getattr(r, "presnapped", False)
        if isinstance(pre, (bool, np.bool_)) and pre:
            s = snap_one(pc, xy, label, replace(gt, snap_radius_m=gt.presnapped_radius_m, snap_method="nearest"), grid)
            lost = s["status"] != "ok"
            if lost:
                log.warning("presnapped point %s: no wall of %.0f m within %.0f m any more - kept as saved", getattr(r, "route_uid", "?"), gt.min_wall_height_m, gt.presnapped_radius_m)
                idx = pc.column(xy, label.route_radius_m)
                base = float(np.quantile(pc.xyz[idx, 2], 0.05)) if len(idx) else np.nan
                o = wall_frame_at(grid, xy)[0] if grid is not None else np.array([np.nan, np.nan])
                s = {**s, "base_z": base, "wall_height_m": wall_height_at(grid, xy, base)[0] if grid is not None and np.isfinite(base) else np.nan,
                     "out_dx": float(o[0]), "out_dy": float(o[1])}
            s.update(snap_x=float(xy[0]), snap_y=float(xy[1]), snap_dist_m=0.0, along_shift_m=0.0, status="ok", presnapped_lost=bool(lost))
        else:
            s = snap_one(pc, xy, label, gt, grid)
        row.update(x=float(xy[0]), y=float(xy[1]), **s)
        if s["status"] == "ok":
            length = r.length_m if pd.notna(getattr(r, "length_m", np.nan)) and r.length_m > 0 else DEFAULT_ROUTE_LENGTH_M
            row["top_z"] = s["base_z"] + float(length)  # reference only: labels no longer use the route length
        out.append(row)
    df = pd.DataFrame(out)
    # the snap columns only appear when some route snapped; an area with no
    # LiDAR (or no cliff near any route) must still return the full schema
    for c in SNAP_COLS:
        if c not in df:
            df[c] = np.nan
    if "presnapped_lost" not in df:
        df["presnapped_lost"] = False
    df["presnapped_lost"] = df["presnapped_lost"].fillna(False).astype(bool)
    ll = crs.to_lonlat(df["snap_x"].fillna(df["x"]).to_numpy(), df["snap_y"].fillna(df["y"]).to_numpy())
    df["snap_lon"], df["snap_lat"] = ll[:, 0], ll[:, 1]
    return df


def spacing_report(snapped: pd.DataFrame, min_raw_m: float = 2.0, max_raw_m: float = 20.0) -> pd.DataFrame:
    """Pairs of wall-checked points that the snap squashed together.

    For every pair of ``ok`` points (same area) whose raw GPS points are ``min_raw_m`` to
    ``max_raw_m`` apart: raw spacing *along the wall* (using the wall direction at the first
    point) against the distance between the snapped positions. ``squashed_pair`` when the
    snapped spacing is under half the raw along-wall spacing. Nothing is dropped here."""
    cols = ["slug", "uid_a", "uid_b", "label_a", "label_b", "raw_dist_m", "raw_along_m", "snap_dist_m", "squashed_pair"]
    ok = snapped[(snapped["status"] == "ok") & snapped["out_dx"].notna()]
    rows = []
    for slug, g in ok.groupby("slug") if "slug" in ok else [("", ok)]:
        raw, snp = g[["x", "y"]].to_numpy(float), g[["snap_x", "snap_y"]].to_numpy(float)
        along = np.column_stack([-g["out_dy"].to_numpy(float), g["out_dx"].to_numpy(float)])
        uid, lab = g["route_uid"].to_numpy(), g["label"].to_numpy()
        for i, j in cKDTree(raw).query_pairs(max_raw_m):
            d = float(np.linalg.norm(raw[i] - raw[j]))
            if d < min_raw_m:
                continue
            ra, sd = float(abs((raw[i] - raw[j]) @ along[i])), float(np.linalg.norm(snp[i] - snp[j]))
            rows.append([slug, uid[i], uid[j], int(lab[i]), int(lab[j]), d, ra, sd, bool(sd < 0.5 * ra)])
    return pd.DataFrame(rows, columns=cols)


def cylinder_features(pc: PointCloud, xy, cfg: NegativeConfig) -> dict:
    idx = pc.column(xy, cfg.cylinder_m)
    if len(idx) == 0:
        return {"n_points": 0, "n_steep": 0, "cliff_height_m": 0.0, "multi_frac": np.nan, "steep_frac": 0.0}
    st = idx[pc.steep[idx]]
    z = pc.xyz[st, 2]
    return {
        "n_points": int(len(idx)),
        "n_steep": int(len(st)),
        "cliff_height_m": float(np.quantile(z, 0.95) - np.quantile(z, 0.05)) if len(st) >= 3 else 0.0,
        "steep_frac": float(len(st) / len(idx)),
    }


def band_candidates(cloud: AreaCloud, pos: pd.DataFrame, known_xy: np.ndarray, coverage_by_wall: dict, cfg: NegativeConfig, default_coverage: float | None = None) -> pd.DataFrame:
    """Wall feet ``band_min..band_max`` m from every known route, in documented neighbourhoods.

    Each candidate cliff spot (2 m grid) is first centred on the foot of its wall
    (``centre_on_foot``, the same step route points go through) and must stand on a wall of
    at least ``min_wall_height_m``; the distance rules are then checked **at the foot**, since
    moving to the foot can bring a candidate closer to a route. Returns ``x, y`` = the foot,
    plus ``base_z, out_dx, out_dy, wall_height_m``; ``attrs["dropped"]`` counts the drops."""
    ok = pos[pos["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame()
    doc_xy = ok[["snap_x", "snap_y"]].to_numpy()
    doc_tree, known_tree = cKDTree(doc_xy), cKDTree(known_xy)
    walls = ok["mp_wall"].to_numpy() if "mp_wall" in ok else np.array([None] * len(ok))
    seen, seen_foot, rows = set(), set(), []
    dropped = {"no_wall": 0, "too_short": 0, "exclusion_after_centring": 0}
    for p in ok.itertuples(index=False):
        pc, crop = local_pc(cloud, p.snap_x, p.snap_y, cfg.band_max_m + 5)
        if pc is None or len(pc.steep_idx) == 0:
            continue
        cand = pc.xyz[pc.steep_idx, :2]
        keys = np.floor(cand / cfg.grid_m).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        for c, k in zip(cand[first], keys[first]):
            key = (int(k[0]), int(k[1]))
            if key in seen:
                continue
            seen.add(key)
            d_known = known_tree.query(c)[0]
            if not (cfg.band_min_m - 5.0 <= d_known <= cfg.band_max_m + 5.0):  # coarse pre-filter; the real test is at the foot
                continue
            near_doc = doc_tree.query_ball_point(c, cfg.doc_radius_m)
            if len(near_doc) < cfg.doc_min_routes:
                continue
            if cylinder_features(pc, c, cfg)["n_steep"] < cfg.min_steep_points:
                continue
            f = centre_on_foot(cloud, pc, c, cfg.min_wall_height_m)
            if f["status"] != "ok":
                dropped[f["status"]] += 1
                continue
            c = np.array([f["foot_x"], f["foot_y"]])
            fkey = (int(np.floor(c[0])), int(np.floor(c[1])))
            if fkey in seen_foot:
                continue
            seen_foot.add(fkey)
            d_known = known_tree.query(c)[0]
            if d_known < max(cfg.exclusion_m, cfg.band_min_m) or d_known > cfg.band_max_m:
                dropped["exclusion_after_centring"] += int(d_known < cfg.exclusion_m)
                continue
            near_doc = doc_tree.query_ball_point(c, cfg.doc_radius_m)
            if len(near_doc) < cfg.doc_min_routes:
                continue
            feat = cylinder_features(pc, c, cfg)
            vec = doc_xy[near_doc] - c
            u = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
            bracketed = bool((u @ u.T).min() < -0.5)
            w = {x for x in walls[near_doc] if isinstance(x, str)}
            cov = max((coverage_by_wall.get(x, 0.0) for x in w), default=default_coverage if default_coverage is not None else 0.0)
            if cov < cfg.min_coverage:
                continue
            idx = pc.column(c, cfg.cylinder_m)
            multi = float(crop["multi"][idx].mean()) if len(idx) else np.nan
            rows.append({"x": c[0], "y": c[1], "dist_to_route_m": float(d_known), "n_doc": len(near_doc), "bracketed": bracketed, "coverage": float(cov), **feat,
                         "multi_frac": multi, **{k: f[k] for k in ("base_z", "out_dx", "out_dy", "wall_height_m", "foot_move_m")}})
    df = pd.DataFrame(rows)
    df.attrs["dropped"] = dropped
    if df.empty:
        return df
    span = max(1.0 - cfg.min_coverage, 1e-6)
    df["confidence"] = (
        0.5 + 0.2 * df["bracketed"] + 0.2 * ((df["coverage"] - cfg.min_coverage) / span).clip(0, 1) + 0.1 * ((df["n_doc"] - cfg.doc_min_routes) / 5).clip(0, 1)
    ).round(3)
    return df


def greedy_select(df: pd.DataFrame, n_max: int, spacing: float) -> pd.DataFrame:
    """Highest confidence first, at least ``spacing`` apart."""
    if df.empty or n_max <= 0:
        return df.iloc[:0]
    df = df.sort_values(["confidence", "n_steep"], ascending=False)
    chosen, pts = [], []
    for i, r in df.iterrows():
        if pts and np.min(np.hypot(np.array(pts)[:, 0] - r["x"], np.array(pts)[:, 1] - r["y"])) < spacing:
            continue
        chosen.append(i)
        pts.append((r["x"], r["y"]))
        if len(chosen) >= n_max:
            break
    return df.loc[chosen]


def stepback_negatives(cloud: AreaCloud, pos: pd.DataFrame, known_xy: np.ndarray, cfg: NegativeConfig) -> pd.DataFrame:
    known = cKDTree(known_xy)
    rows = []
    for p in pos[pos["status"] == "ok"].itertuples(index=False):
        out = np.array([p.out_dx, p.out_dy])
        for d in cfg.stepback_m:
            for sign, name in ((1, "base"), (-1, "top")):
                c = np.array([p.snap_x, p.snap_y]) + sign * d * out
                if known.query(c)[0] < cfg.exclusion_m:
                    continue
                pc, crop = local_pc(cloud, c[0], c[1], 6.0)
                if pc is None:
                    continue
                col = pc.column(c, 5.0)
                if len(col) < 64 or pc.steep[col].mean() > cfg.max_stepback_steep_frac:
                    continue  # mostly cliff: could be unlisted rock -> not a negative
                rows.append({"x": c[0], "y": c[1], "stepback": name, "dist_to_route_m": float(known.query(c)[0]), **cylinder_features(pc, c, cfg), "confidence": 0.9})
    return pd.DataFrame(rows)


def far_cliff_negatives(cloud: AreaCloud, block_index: pd.DataFrame, anchors_xy: np.ndarray, cfg: NegativeConfig, seed: int = 0) -> pd.DataFrame:
    """Cliff columns far from every route and MP crag pin (from the pretraining block index)."""
    if block_index is None or block_index.empty:
        return pd.DataFrame()
    b = block_index[block_index["n_cliff"] >= cfg.min_steep_points * 3]
    if len(anchors_xy):
        d = cKDTree(anchors_xy).query(b[["cx", "cy"]].to_numpy())[0]
        b = b[d > cfg.far_min_m]
    if b.empty:
        return pd.DataFrame()
    b = b.sample(min(len(b), cfg.far_per_area * 4), random_state=seed)
    rows, dropped = [], {"no_wall": 0, "too_short": 0}
    anchors = cKDTree(anchors_xy) if len(anchors_xy) else None
    for r in b.itertuples(index=False):
        pc, crop = local_pc(cloud, r.cx, r.cy, 10.0)
        if pc is None or len(pc.steep_idx) < cfg.min_steep_points:
            continue
        cand = pc.xyz[pc.steep_idx[:: max(len(pc.steep_idx) // 50, 1)], :2]
        counts = pc.steep_tree_xy.query_ball_point(cand, cfg.cylinder_m, return_length=True)
        c = cand[int(np.argmax(counts))]
        if cylinder_features(pc, c, cfg)["n_steep"] < cfg.min_steep_points:
            continue
        f = centre_on_foot(cloud, None, c, cfg.min_wall_height_m)  # same centring + height rule as every other sample
        if f["status"] != "ok":
            dropped[f["status"]] += 1
            continue
        c = np.array([f["foot_x"], f["foot_y"]])
        if len(anchors_xy) and anchors.query(c)[0] <= cfg.far_min_m:
            continue
        pc, crop = local_pc(cloud, c[0], c[1], 6.0)
        if pc is None:
            continue
        idx = pc.column(c, cfg.cylinder_m)
        rows.append({"x": c[0], "y": c[1], **cylinder_features(pc, c, cfg), "multi_frac": float(crop["multi"][idx].mean()) if len(idx) else np.nan, "confidence": 0.4,
                     **{k: f[k] for k in ("base_z", "out_dx", "out_dy", "wall_height_m", "foot_move_m")}})
        if len(rows) >= cfg.far_per_area:
            break
    df = pd.DataFrame(rows)
    df.attrs["dropped"] = dropped
    return df


def clean_samples(samples: pd.DataFrame, cfg: NegativeConfig) -> tuple[pd.DataFrame, list[dict]]:
    """Checks after centring, for all labels of one area (``x, y`` = wall feet).

    * same-label duplicates: two samples with the same label within ``dup_m`` -> keep the
      higher-confidence one;
    * a climbable and a not-climbable foot under ``pair_min_m`` apart: both dropped, unless
      both positions are ``confirmed`` (kept, with a warning). Close pairs further apart are
      kept on purpose - their labelled strips are split later (``split_strip_widths``).

    Returns (kept samples, list of drops ``{sample, reason, other}``)."""
    drops = []
    if samples.empty:
        return samples, drops
    s = samples.reset_index(drop=True)
    keep = np.ones(len(s), bool)
    xy = s[["x", "y"]].to_numpy(float)
    order = np.lexsort((np.arange(len(s)), -s["confidence"].to_numpy(float)))  # best first, stable
    tree = cKDTree(xy)
    lab = s["label"].to_numpy()
    uid = s["route_uid"].astype(str).to_numpy()
    chosen = np.zeros(len(s), bool)
    for i in order:
        near = [j for j in tree.query_ball_point(xy[i], cfg.dup_m) if chosen[j] and lab[j] == lab[i]]
        if near:
            keep[i] = False
            drops.append({"sample": uid[i], "reason": "duplicate", "other": uid[near[0]], "source": s["source"].iloc[i]})
        else:
            chosen[i] = True
    conf = s["confirmed"].to_numpy(bool) if "confirmed" in s else np.zeros(len(s), bool)
    for i, j in tree.query_pairs(cfg.pair_min_m):
        if not (keep[i] and keep[j]) or lab[i] == lab[j]:
            continue
        if conf[i] and conf[j]:
            log.warning("confirmed climbable / not-climbable pair under %.1f m apart kept: %s, %s", cfg.pair_min_m, uid[i], uid[j])
            continue
        for a, b2 in ((i, j), (j, i)):
            drops.append({"sample": uid[a], "reason": "pair_under_1m", "other": uid[b2], "source": s["source"].iloc[a]})
        keep[i] = keep[j] = False
    return s[keep].reset_index(drop=True), drops


def build_area_samples(cloud: AreaCloud, routes: pd.DataFrame, all_known_lonlat: np.ndarray, crs, coverage_by_wall: dict, cfg: NegativeConfig,
                       block_index: pd.DataFrame | None = None, crag_pins_xy: np.ndarray | None = None, strategies=("handgathered_nc", "band", "far_cliff"),
                       default_coverage: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(wall-checked route points, samples, drop report) for one area.

    ``routes`` has label 1 (routes) and 0 (hand Nc). Every sample - route, hand Nc, band,
    far cliff - ends up centred on the foot of its wall by the same step (``centre_on_foot``)
    and must stand on a wall of at least ``min_wall_height_m``. ``samples`` has ``x, y`` (the
    foot), ``base_z, out_dx, out_dy, wall_height_m, label, source, confidence, route_uid,
    confirmed``. Step-back samples (off the wall) are only made when asked for."""
    snapped = snap_routes(cloud, routes, crs)
    report = {"dropped": {}, "drops": []}

    def count(source, reason, n=1):
        if n:
            report["dropped"][f"{source}:{reason}"] = report["dropped"].get(f"{source}:{reason}", 0) + int(n)

    # 1. route points and hand Nc points: snapped spot -> wall foot
    feet = []
    for r in snapped.itertuples(index=False):
        if r.status != "ok":
            feet.append({"foot_status": r.status})
            continue
        # the height rule was applied by the wall check at the snapped spot (plan 4); at the foot the
        # height is only recorded, so confirmed points are not dropped by grid noise a few metres away
        f = centre_on_foot(cloud, None, np.array([r.snap_x, r.snap_y]))
        feet.append({"foot_status": f["status"], **{k: f.get(k, np.nan) for k in FOOT_COLS}})
    F = pd.DataFrame(feet, index=snapped.index)
    for c in ("foot_status",) + FOOT_COLS:
        snapped[c if c not in ("base_z", "out_dx", "out_dy", "wall_height_m") else f"foot_{c}"] = F[c] if c in F else np.nan
    usable = snapped[(snapped["status"] == "ok") & (snapped["foot_status"] == "ok")]
    for (src, st), n in snapped[snapped["status"] == "ok"].groupby(["source", "foot_status"]).size().items():
        if st != "ok":
            count(src, f"foot_{st}", n)
    confirmed = usable["judge_result"].isin(["agreed", "majority"]).to_numpy() if "judge_result" in usable else np.zeros(len(usable), bool)
    base = pd.DataFrame({
        "x": usable["foot_x"].to_numpy(), "y": usable["foot_y"].to_numpy(), "base_z": usable["foot_base_z"].to_numpy(), "out_dx": usable["foot_out_dx"].to_numpy(),
        "out_dy": usable["foot_out_dy"].to_numpy(), "wall_height_m": usable["foot_wall_height_m"].to_numpy(), "foot_move_m": usable["foot_move_m"].to_numpy(),
        "label": usable["label"].to_numpy(), "source": usable["source"].to_numpy(), "route_uid": usable["route_uid"].to_numpy(), "confirmed": confirmed,
        "confidence": np.where(usable["label"].to_numpy() == 1, np.where(usable.get("match_type", pd.Series("", index=usable.index)).to_numpy() == "fuzzy", 0.9, 1.0), 1.0),
    })
    for c in ("name", "mp_route_id", "match_type", "judge_result", "judge_confidence", "area_name", "length_m"):
        base[c] = usable[c].to_numpy() if c in usable else None
    pos = base[base["label"] == 1]
    nc = base[(base["label"] == 0)] if "handgathered_nc" in strategies else base.iloc[:0]
    known_xy = crs.to_work(all_known_lonlat[:, 0], all_known_lonlat[:, 1]) if len(all_known_lonlat) else np.zeros((0, 2))
    ok_xy = pos[["x", "y"]].to_numpy()
    known_xy = np.vstack([known_xy, ok_xy]) if len(ok_xy) else known_xy
    if len(known_xy) == 0:
        known_xy = np.array([[1e12, 1e12]])
    parts = [pos, nc]
    n_ok = len(pos)
    doc = pd.DataFrame({"status": "ok", "snap_x": pos["x"].to_numpy(), "snap_y": pos["y"].to_numpy(),
                        "mp_wall": usable.loc[usable["label"] == 1, "mp_wall"].to_numpy() if "mp_wall" in usable else None})
    if "band" in strategies:
        band = band_candidates(cloud, doc, known_xy, coverage_by_wall, cfg, default_coverage)
        for k, n in band.attrs.get("dropped", {}).items():
            count("band", k, n)
        band = greedy_select(band, int(round(cfg.band_per_positive * n_ok)), cfg.neg_spacing_m)
        if len(band):
            parts.append(band.assign(source="band", label=0, confirmed=False, route_uid=[f"band:{i}" for i in range(len(band))]))
    if "stepback" in strategies:  # off the wall: not part of the wall-only task (kept for comparison builds)
        sb = stepback_negatives(cloud, snapped[(snapped["label"] == 1)], known_xy, cfg)
        sb = greedy_select(sb, 2 * n_ok, cfg.neg_spacing_m)
        if len(sb):
            parts.append(sb.assign(source="stepback", label=0, confirmed=False, route_uid=[f"stepback:{i}" for i in range(len(sb))]))
    if "far_cliff" in strategies:
        anchors = known_xy if crag_pins_xy is None or not len(crag_pins_xy) else np.vstack([known_xy, crag_pins_xy])
        fc = far_cliff_negatives(cloud, block_index, anchors, cfg)
        for k, n in fc.attrs.get("dropped", {}).items():
            count("far_cliff", k, n)
        if len(fc):
            parts.append(fc.assign(source="far_cliff", label=0, confirmed=False, route_uid=[f"far_cliff:{i}" for i in range(len(fc))]))
    samples = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else base.iloc[:0]
    if len(samples):
        # final verification after centring: no code-made negative within the exclusion radius of a
        # known route (hand Nc points are exempt - they were judged on site next to routes)
        d = cKDTree(known_xy).query(samples[["x", "y"]].to_numpy(float))[0]
        samples["dist_to_route_m"] = np.where(samples["label"] == 1, 0.0, d)
        bad = (d < cfg.exclusion_m) & (samples["label"] == 0) & (samples["source"] != "handgathered_nc")
        for src, n in samples.loc[bad, "source"].value_counts().items():
            count(src, "exclusion_after_centring", n)
        samples = samples[~bad].reset_index(drop=True)
        samples, drops = clean_samples(samples, cfg)
        for dct in drops:
            count(dct["source"], dct["reason"])
        report["drops"] = drops
        ll = crs.to_lonlat(samples["x"].to_numpy(float), samples["y"].to_numpy(float))
        samples["lon"], samples["lat"] = ll[:, 0], ll[:, 1]
    return snapped, samples, report


def waypoint_frame(gt: pd.DataFrame) -> pd.DataFrame:
    """Hand waypoints (``load_waypoints``) as a ``snap_routes`` input, with the saved-snap columns."""
    return pd.DataFrame({"route_uid": gt["gt_id"].to_numpy(), "label": gt["label"].to_numpy(), "lat": gt["lat"].to_numpy(), "lon": gt["lon"].to_numpy(), "lat2": np.nan,
                         "lon2": np.nan, "length_m": np.nan, "presnapped": gt["presnapped"].to_numpy(bool) if "presnapped" in gt else False,
                         "snap_status": gt["snap_status"].to_numpy() if "snap_status" in gt else "",
                         "judge_result": gt["judge_result"].to_numpy() if "judge_result" in gt else "",
                         "judge_confidence": gt["judge_confidence"].to_numpy() if "judge_confidence" in gt else ""})


def validate_on_handgathered(cloud: AreaCloud, gt: pd.DataFrame, crs, cfg: NegativeConfig, doc_fractions=(0.3, 0.5, 0.75, 0.9),
                             repeats: int = 2, match_m: float = 8.0, seed: int = 0) -> dict:
    """Measure how often band negatives are really not climbable, as a function of documentation coverage.

    Horseshoe Canyon Ranch has on-site labels for both climbable (126) and
    non-climbable (60) rock along the walls that were surveyed. For each
    ``doc_fraction`` f we pretend only a random fraction f of the Climb
    waypoints were "documented routes" (simulating an OSM coverage of f),
    generate band negatives from them exactly as in production (band, cliff
    check, greedy spacing, one per documented route), and compare every
    negative with the labels the generator did NOT see: held-out Climb points
    (a negative near one is *contradicted*) and Nc points (*confirmed*).
    ``precision = confirmed / (confirmed + contradicted)``; the same is
    reported for random cliff columns (baseline) - its precision is about the
    share of Nc among labelled rock.
    """
    # every label goes through the wall check first; off-wall points are left out
    gt = gt[gt["label"] >= 0].reset_index(drop=True)
    checked = snap_routes(cloud, waypoint_frame(gt).assign(mp_wall="hand"), crs)
    checked = checked[checked["status"] == "ok"].reset_index(drop=True)
    snapped_all = checked[checked["label"] == 1].reset_index(drop=True)
    climb = snapped_all
    lab_xy = checked[["snap_x", "snap_y"]].to_numpy()
    labels = checked["label"].to_numpy()
    rng = np.random.default_rng(seed)
    lo, hi = lab_xy.min(0) - 30, lab_xy.max(0) + 30

    def score(df, unseen_mask):
        if df is None or df.empty:
            return {"n": 0, "confirmed_nc": 0, "contradicted_climb": 0, "precision": float("nan")}
        tree = cKDTree(lab_xy[unseen_mask])
        lab = labels[unseen_mask]
        d, i = tree.query(df[["x", "y"]].to_numpy())
        near = d <= match_m
        conf, contra = int((lab[i[near]] == 0).sum()), int((lab[i[near]] == 1).sum())
        return {"n": int(len(df)), "confirmed_nc": conf, "contradicted_climb": contra, "precision": conf / (conf + contra) if conf + contra else float("nan")}

    def random_cliff(n, known):
        kt = cKDTree(known)
        rows = []
        for _ in range(50 * n):
            if len(rows) >= n:
                break
            c = rng.uniform(lo, hi)
            pc, _ = local_pc(cloud, c[0], c[1], 4.0)
            if pc is None or len(pc.steep_idx) == 0:
                continue
            c = pc.xyz[pc.steep_idx[rng.integers(len(pc.steep_idx))], :2]
            if kt.query(c)[0] >= cfg.exclusion_m and cylinder_features(pc, c, cfg)["n_steep"] >= cfg.min_steep_points:
                rows.append({"x": c[0], "y": c[1]})
        return pd.DataFrame(rows)

    results = []
    for f in doc_fractions:
        for rep in range(repeats):
            doc = rng.random(len(climb)) < f
            pos = snapped_all[doc].reset_index(drop=True)
            known = pos[["snap_x", "snap_y"]].to_numpy()
            if len(known) < 3:
                continue
            # labels the generator saw (documented Climb points) are excluded from scoring
            seen = np.zeros(len(labels), bool)
            seen[np.flatnonzero(labels == 1)[doc]] = True
            band = band_candidates(cloud, pos, known, {"hand": 1.0}, cfg)
            band = greedy_select(band, len(known), cfg.neg_spacing_m)
            b = score(band, ~seen)
            r = score(random_cliff(max(len(band), 30), known), ~seen)
            results.append({"doc_fraction": f, "repeat": rep, "n_documented": int(len(known)), "band_n": b["n"], "band_confirmed": b["confirmed_nc"],
                            "band_contradicted": b["contradicted_climb"], "band_precision": b["precision"], "random_precision": r["precision"],
                            "random_confirmed": r["confirmed_nc"], "random_contradicted": r["contradicted_climb"]})
    df = pd.DataFrame(results)
    summary = df.groupby("doc_fraction")[["band_precision", "random_precision", "band_n", "band_confirmed", "band_contradicted"]].mean().round(3)
    return {"per_run": df.to_dict("records"), "by_coverage": summary.reset_index().to_dict("records"), "match_m": match_m,
            "n_climb": int(len(climb)), "n_nc": int((labels == 0).sum()), "n_off_wall": int(len(gt) - len(checked))}
