"""Verified positives and hard negatives for fine-tuning (proposal: docs/step3-routes-and-hard-negatives.md).

Positives: verified route starts (OSM matched to MP, or hand-gathered Climb
waypoints) snapped onto the nearest LiDAR cliff column whose base is at the
ground height of the start point (``crag.groundtruth.snap_one``: 12 m search,
6 m height tolerance).

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

from crag.crop import AreaCloud
from crag.groundtruth import GroundTruthConfig, snap_one
from crag.labeling import LabelConfig, PointCloud

log = logging.getLogger(__name__)
DEFAULT_ROUTE_LENGTH_M = 15.0


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
    )
    return pc, c


def _ground_z(pc: PointCloud, xy, r: float = 3.0) -> float:
    idx = pc.column(xy, r)
    return float(np.quantile(pc.xyz[idx, 2], 0.1)) if len(idx) else np.inf


def snap_routes(cloud: AreaCloud, routes: pd.DataFrame, crs, label: LabelConfig | None = None, gt: GroundTruthConfig | None = None) -> pd.DataFrame:
    """Snap route start points onto the cliff. Adds ``x, y`` (start), ``snap_*``, ``base_z``,
    ``top_z``, ``out_dx/dy``, ``status``. Route *ways* use their lower end."""
    label, gt = label or LabelConfig(), gt or GroundTruthConfig()
    out = []
    for r in routes.itertuples(index=False):
        xy = crs.to_work(r.lon, r.lat)
        pc, _ = local_pc(cloud, xy[0], xy[1], 40.0)
        row = r._asdict()
        if pc is None:
            out.append({**row, "x": xy[0], "y": xy[1], "status": "no_lidar"})
            continue
        if pd.notna(getattr(r, "lat2", np.nan)):  # route line: pick the lower end as the start
            xy2 = crs.to_work(r.lon2, r.lat2)
            if _ground_z(pc, xy2) < _ground_z(pc, xy):
                xy = xy2
        s = snap_one(pc, np.asarray(xy, float), label, gt)
        row.update(x=float(xy[0]), y=float(xy[1]), **s)
        if s["status"] == "ok":
            length = r.length_m if pd.notna(getattr(r, "length_m", np.nan)) and r.length_m > 0 else DEFAULT_ROUTE_LENGTH_M
            row["top_z"] = s["base_z"] + float(length)
        out.append(row)
    df = pd.DataFrame(out)
    ll = crs.to_lonlat(df.get("snap_x", df["x"]).fillna(df["x"]).to_numpy(), df.get("snap_y", df["y"]).fillna(df["y"]).to_numpy())
    df["snap_lon"], df["snap_lat"] = ll[:, 0], ll[:, 1]
    return df


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
    """Cliff columns ``band_min..band_max`` m from every known route, in documented neighbourhoods."""
    ok = pos[pos["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame()
    doc_xy = ok[["snap_x", "snap_y"]].to_numpy()
    doc_tree, known_tree = cKDTree(doc_xy), cKDTree(known_xy)
    walls = ok["mp_wall"].to_numpy() if "mp_wall" in ok else np.array([None] * len(ok))
    seen, rows = set(), []
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
            if not (cfg.band_min_m <= d_known <= cfg.band_max_m) or d_known < cfg.exclusion_m:
                continue
            near_doc = doc_tree.query_ball_point(c, cfg.doc_radius_m)
            if len(near_doc) < cfg.doc_min_routes:
                continue
            feat = cylinder_features(pc, c, cfg)
            if feat["n_steep"] < cfg.min_steep_points:
                continue
            vec = doc_xy[near_doc] - c
            u = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
            bracketed = bool((u @ u.T).min() < -0.5)
            w = {x for x in walls[near_doc] if isinstance(x, str)}
            cov = max((coverage_by_wall.get(x, 0.0) for x in w), default=default_coverage if default_coverage is not None else 0.0)
            if cov < cfg.min_coverage:
                continue
            idx = pc.column(c, cfg.cylinder_m)
            multi = float(crop["multi"][idx].mean()) if len(idx) else np.nan
            rows.append({"x": c[0], "y": c[1], "dist_to_route_m": float(d_known), "n_doc": len(near_doc), "bracketed": bracketed, "coverage": float(cov), **feat, "multi_frac": multi})
    df = pd.DataFrame(rows)
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
    b = b.sample(min(len(b), cfg.far_per_area * 3), random_state=seed)
    rows = []
    for r in b.itertuples(index=False):
        pc, crop = local_pc(cloud, r.cx, r.cy, 10.0)
        if pc is None or len(pc.steep_idx) < cfg.min_steep_points:
            continue
        cand = pc.xyz[pc.steep_idx[:: max(len(pc.steep_idx) // 50, 1)], :2]
        counts = pc.steep_tree_xy.query_ball_point(cand, cfg.cylinder_m, return_length=True)
        c = cand[int(np.argmax(counts))]
        feat = cylinder_features(pc, c, cfg)
        if feat["n_steep"] < cfg.min_steep_points:
            continue
        idx = pc.column(c, cfg.cylinder_m)
        rows.append({"x": c[0], "y": c[1], **feat, "multi_frac": float(crop["multi"][idx].mean()) if len(idx) else np.nan, "confidence": 0.4})
        if len(rows) >= cfg.far_per_area:
            break
    return pd.DataFrame(rows)


def build_area_samples(cloud: AreaCloud, routes: pd.DataFrame, all_known_lonlat: np.ndarray, crs, coverage_by_wall: dict, cfg: NegativeConfig,
                       block_index: pd.DataFrame | None = None, crag_pins_xy: np.ndarray | None = None, strategies=("handgathered_nc", "band", "stepback", "far_cliff"),
                       default_coverage: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(positives, negatives) for one area. ``routes`` has label 1 (routes) and 0 (hand Nc)."""
    snapped = snap_routes(cloud, routes, crs)
    pos = snapped[snapped["label"] == 1].reset_index(drop=True)
    nc = snapped[snapped["label"] == 0].reset_index(drop=True)
    known_xy = crs.to_work(all_known_lonlat[:, 0], all_known_lonlat[:, 1]) if len(all_known_lonlat) else np.zeros((0, 2))
    ok_xy = pos.loc[pos["status"] == "ok", ["snap_x", "snap_y"]].to_numpy()
    known_xy = np.vstack([known_xy, ok_xy]) if len(ok_xy) else known_xy
    if len(known_xy) == 0:
        known_xy = np.array([[1e12, 1e12]])
    negs = []
    if "handgathered_nc" in strategies and len(nc):
        x = nc["snap_x"].where(nc["status"] == "ok", nc["x"]).to_numpy()
        y = nc["snap_y"].where(nc["status"] == "ok", nc["y"]).to_numpy()
        d = pd.DataFrame({"x": x, "y": y, "source": "handgathered_nc", "confidence": 1.0, "route_uid": nc["route_uid"], "base_z": nc.get("base_z")})
        negs.append(d)
    n_ok = int((pos["status"] == "ok").sum())
    if "band" in strategies:
        band = band_candidates(cloud, pos, known_xy, coverage_by_wall, cfg, default_coverage)
        band = greedy_select(band, int(round(cfg.band_per_positive * n_ok)), cfg.neg_spacing_m)
        if len(band):
            negs.append(band.assign(source="band"))
    if "stepback" in strategies:
        sb = stepback_negatives(cloud, pos, known_xy, cfg)
        sb = greedy_select(sb, 2 * n_ok, cfg.neg_spacing_m)
        if len(sb):
            negs.append(sb.assign(source="stepback"))
    if "far_cliff" in strategies:
        anchors = known_xy if crag_pins_xy is None or not len(crag_pins_xy) else np.vstack([known_xy, crag_pins_xy])
        fc = far_cliff_negatives(cloud, block_index, anchors, cfg)
        if len(fc):
            negs.append(fc.assign(source="far_cliff"))
    neg = pd.concat(negs, ignore_index=True) if negs else pd.DataFrame(columns=["x", "y", "source", "confidence"])
    if len(neg):
        # audit features for every negative (same definition whatever the strategy)
        feats = []
        for x, y in neg[["x", "y"]].to_numpy():
            pc, crop = local_pc(cloud, x, y, 6.0)
            if pc is None:
                feats.append({})
                continue
            idx = pc.column([x, y], cfg.cylinder_m)
            feats.append({**cylinder_features(pc, [x, y], cfg), "multi_frac": float(crop["multi"][idx].mean()) if len(idx) else np.nan})
        f = pd.DataFrame(feats, index=neg.index)
        for c in f.columns:
            neg[c] = f[c]
        # final verification: nothing within the exclusion radius of a known route
        # (hand Nc points are exempt - they were judged on site next to routes)
        d = cKDTree(known_xy).query(neg[["x", "y"]].to_numpy())[0]
        neg["dist_to_route_m"] = d
        bad = (d < cfg.exclusion_m) & (neg["source"] != "handgathered_nc")
        if bad.any():
            log.info("dropped %d negatives inside the %.1f m exclusion radius", bad.sum(), cfg.exclusion_m)
        neg = neg[~bad].reset_index(drop=True)
        ll = crs.to_lonlat(neg["x"].to_numpy(), neg["y"].to_numpy())
        neg["lon"], neg["lat"] = ll[:, 0], ll[:, 1]
        neg["label"] = 0
    return pos, neg


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
    gt = gt[gt["label"] >= 0].reset_index(drop=True)
    climb = gt[gt["label"] == 1].reset_index(drop=True)
    routes = pd.DataFrame({"route_uid": climb["gt_id"], "label": 1, "lat": climb["lat"], "lon": climb["lon"], "lat2": np.nan, "lon2": np.nan, "length_m": np.nan, "mp_wall": "hand"})
    snapped_all = snap_routes(cloud, routes, crs)
    lab_xy = crs.to_work(gt["lon"].to_numpy(), gt["lat"].to_numpy())
    labels = gt["label"].to_numpy()
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
            known = pos.loc[pos["status"] == "ok", ["snap_x", "snap_y"]].to_numpy()
            if len(known) < 3:
                continue
            # labels the generator saw (documented Climb points) are excluded from scoring
            seen = np.zeros(len(gt), bool)
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
            "n_climb": int(len(climb)), "n_nc": int((labels == 0).sum())}
