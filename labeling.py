"""Weak supervision: turn Mountain Project wall pins + LiDAR into labelled patches.

Pipeline (see the notebook for the pictures):

1. **Normals** - local PCA over ``normal_k`` neighbours (on a voxel-thinned
   copy, so the normal scale doesn't change with point density) gives every point a
   surface normal; slope = angle between the normal and vertical. Points
   steeper than ``steep_slope_deg`` (and, by default, single-return, to drop
   tree canopy) are "cliff" points.
2. **Snap** - an MP wall pin is usually 10-30 m off the rock, so within
   ``snap_search_radius_m`` of the pin we look for columns that actually hold a
   cliff (>= ``min_steep_points`` steep points inside a 2.5 m cylinder) and
   greedily place route cylinders there, nearest-to-pin first, at least
   ``route_spacing_m`` apart. A wall gets at most one cylinder per MP route.
3. **Route cylinder** - each cylinder spans the cliff base (5th percentile Z
   in the cylinder) up to base + the MP route length. It is a *positive*
   (label 1) only if the LiDAR actually saw rock in that height range.
4. **Negatives** (label 0), following the hard-negative notes:
   * contextual: step ``stepback_m`` out from and back into the wall along
     its normal (talus at the base, plateau on top);
   * random: anywhere in the region at least ``ignore_buffer_m`` from every
     wall pin and positive.
   A candidate negative that is itself mostly cliff could be unlisted good
   rock, so it is **ignored** instead of being labelled 0.
5. **Feasibility diagnostics** for every positive: how much of the route's
   height has LiDAR returns and which way the wall leans. Airborne LiDAR
   looks down, so an overhang shows up as a wall whose top sits further out
   than its base - or as a gap with no returns at all under the roof.

The model input for every sample (positive or negative) is the *full vertical
column* of points within ``PatchConfig.radius_m`` of its centre - the label
cylinder is never used to crop the input, so the crop itself cannot leak the
label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from config import ExperimentConfig, LabelConfig, PatchConfig
from usgs_ept import lonlat_to_work, work_to_lonlat

DEFAULT_ROUTE_LENGTH_M = 15.0


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def compute_normals(
    xyz: np.ndarray, k: int = 12, voxel_m: float | None = 0.5, chunk: int = 200_000
) -> np.ndarray:
    """Unit surface normal per point from the PCA of its ``k`` nearest neighbours.

    With ``voxel_m`` the PCA runs on a 3D voxel-thinned copy (one point per
    voxel) and every point takes the normal of its nearest thinned point.
    Without it, a fixed ``k`` spans a smaller patch wherever density is
    higher, so normals in flight-line overlap strips (~2x density) measure
    centimetre roughness and show up as N-S stripes of fake "steep" points.

    The sign is ambiguous (airborne LiDAR has no reliable sensor position per
    point), so normals are flipped to point upward-ish (n_z >= 0); slope and
    wall direction only need the line, not the sign.
    """
    if voxel_m:
        keys = np.floor((xyz - xyz.min(axis=0)) / voxel_m).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        thin = xyz[first]
        _, nearest = cKDTree(thin).query(xyz, workers=-1)
        return compute_normals(thin, k, voxel_m=None, chunk=chunk)[nearest]
    centred = xyz - xyz.mean(axis=0)
    tree = cKDTree(centred)
    normals = np.empty_like(centred, dtype=np.float32)
    for start in tqdm(range(0, len(centred), chunk), desc="normals"):
        stop = min(start + chunk, len(centred))
        _, idx = tree.query(centred[start:stop], k=k, workers=-1)
        nbrs = centred[idx]  # (n, k, 3)
        nbrs = nbrs - nbrs.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nbrs, nbrs) / k
        _, vecs = np.linalg.eigh(cov)
        normals[start:stop] = vecs[:, :, 0]  # eigenvector of the smallest eigenvalue
    normals[normals[:, 2] < 0] *= -1
    return normals


def slope_deg(normals: np.ndarray) -> np.ndarray:
    """0 = flat ground, 90 = vertical wall."""
    return np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]), 0.0, 1.0)))


@dataclass
class PointCloud:
    """The region's points plus the indexes every query below needs."""

    xyz: np.ndarray
    normals: np.ndarray
    steep: np.ndarray  # bool per point
    tree_xy: cKDTree  # 2D tree for vertical-column queries
    steep_idx: np.ndarray  # indices of steep points
    steep_tree_xy: cKDTree

    @classmethod
    def build(
        cls,
        xyz: np.ndarray,
        cfg: LabelConfig,
        normals: np.ndarray | None = None,
        number_of_returns: np.ndarray | None = None,
    ) -> PointCloud:
        if normals is None:
            normals = compute_normals(xyz, cfg.normal_k, cfg.normal_voxel_m)
        steep = slope_deg(normals) >= cfg.steep_slope_deg
        if cfg.steep_single_return_only and number_of_returns is not None:
            steep &= number_of_returns == 1
        steep_idx = np.flatnonzero(steep)
        return cls(
            xyz=xyz,
            normals=normals,
            steep=steep,
            tree_xy=cKDTree(xyz[:, :2]),
            steep_idx=steep_idx,
            steep_tree_xy=cKDTree(xyz[steep_idx, :2]),
        )

    def column(self, center_xy, radius: float) -> np.ndarray:
        """Indices of all points within ``radius`` (horizontal) of ``center_xy``."""
        return np.asarray(
            self.tree_xy.query_ball_point(center_xy, radius), dtype=np.int64
        )

    @property
    def bounds_xy(self) -> tuple[float, float, float, float]:
        lo, hi = self.xyz[:, :2].min(axis=0), self.xyz[:, :2].max(axis=0)
        return lo[0], lo[1], hi[0], hi[1]


def wall_direction(pc: PointCloud, idx: np.ndarray) -> np.ndarray:
    """Horizontal unit vector pointing *out* of the wall (from the cliff into the air).

    The line comes from the mean horizontal normal of the steep points; the
    sign is chosen so it points from the high side (cliff top) to the low
    side (cliff base).
    """
    steep = idx[pc.steep[idx]]
    n = pc.normals[steep, :2].astype(np.float64)
    # normals have an arbitrary sign; align them before averaging
    ref = n[np.argmax(np.linalg.norm(n, axis=1))]
    n = np.where((n @ ref)[:, None] < 0, -n, n)
    d = n.mean(axis=0)
    d /= np.linalg.norm(d) + 1e-12
    z = pc.xyz[idx, 2]
    hi = idx[z >= np.quantile(z, 0.8)]
    lo = idx[z <= np.quantile(z, 0.2)]
    if (pc.xyz[lo, :2].mean(0) - pc.xyz[hi, :2].mean(0)) @ d < 0:
        d = -d
    return d


def route_profile(
    pc: PointCloud, idx: np.ndarray, base_z: float, top_z: float, out_dir: np.ndarray
) -> dict:
    """Feasibility diagnostics for one route cylinder.

    * ``coverage`` - fraction of 1 m height bins between base and top that
      hold at least one steep return (1.0 = LiDAR saw the whole route).
    * ``lean_deg`` - angle of the wall from vertical, from a line fit of the
      steep points' outward offset against height: < 0 the wall lays back
      (slab-ish), ~0 vertical face, > 0 the top sticks out (overhang).
    * ``max_gap_m`` - the tallest run of empty height bins (a hidden roof
      usually shows up as a gap).
    """
    sel = idx[pc.steep[idx]]
    z = pc.xyz[sel, 2]
    keep = (z >= base_z) & (z <= top_z)
    sel, z = sel[keep], z[keep]
    edges = np.arange(base_z, top_z + 1.0, 1.0)
    hist = np.histogram(z, bins=edges)[0] if len(edges) > 1 else np.zeros(1)
    filled = hist > 0
    gaps, run = [], 0
    for f in filled:
        run = 0 if f else run + 1
        gaps.append(run)
    lean = np.nan
    if len(sel) >= 3 and np.ptp(z) > 1.0:
        offset = pc.xyz[sel, :2] @ out_dir
        lean = float(np.degrees(np.arctan(np.polyfit(z, offset, 1)[0])))
    return {
        "n_steep": int(len(sel)),
        "coverage": float(filled.mean()) if len(filled) else 0.0,
        "max_gap_m": float(max(gaps) if gaps else 0.0),
        "lean_deg": lean,
    }


# ---------------------------------------------------------------------------
# positives
# ---------------------------------------------------------------------------


def place_route_cylinders(
    pc: PointCloud,
    pin_xy: np.ndarray,
    n_routes: int,
    cfg: LabelConfig,
    taken: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Greedy cylinder centres on the cliff line nearest to a wall pin, shape (m, 2).

    ``taken`` holds centres already placed for other walls; the spacing rule
    applies to them too (and new centres are appended), so two walls with
    near-identical pins can't claim the same patch - which would otherwise
    put one patch in two splits.
    """
    taken = [] if taken is None else taken
    near = pc.steep_tree_xy.query_ball_point(pin_xy, cfg.snap_search_radius_m)
    if not near:
        return np.empty((0, 2))
    cand = pc.xyz[pc.steep_idx[near], :2]
    cand = cand[np.argsort(np.linalg.norm(cand - pin_xy, axis=1))]
    # thin the candidates first so the loop below stays short
    _, first = np.unique(
        np.round(cand / 0.5).astype(np.int64), axis=0, return_index=True
    )
    cand = cand[np.sort(first)]
    centres: list[np.ndarray] = []
    for c in cand:
        if len(centres) >= n_routes:
            break
        if (
            taken
            and np.min(np.linalg.norm(np.asarray(taken) - c, axis=1))
            < cfg.route_spacing_m
        ):
            continue
        n_steep = len(pc.steep_tree_xy.query_ball_point(c, cfg.route_radius_m))
        if n_steep >= cfg.min_steep_points:
            centres.append(c)
            taken.append(c)
    return np.asarray(centres).reshape(-1, 2)


def build_positives(
    pc: PointCloud, routes: pd.DataFrame, cfg: LabelConfig
) -> pd.DataFrame:
    """One row per placed route cylinder (kept or rejected, with the reason)."""
    rows, taken = [], []
    for (wall, lat, lon), grp in routes.groupby(["wall", "lat", "lon"], sort=False):
        pin = lonlat_to_work(lon, lat)
        if cfg.drop_default_pin_routes and grp["default_pin"].iloc[0]:
            rows.append(
                {
                    "wall": wall,
                    "pin_x": pin[0],
                    "pin_y": pin[1],
                    "status": "default_pin",
                }
            )
            continue
        grp = grp.sort_values("length_m", ascending=False)
        centres = place_route_cylinders(pc, pin, len(grp), cfg, taken)
        if len(centres) == 0:
            rows.append(
                {
                    "wall": wall,
                    "pin_x": pin[0],
                    "pin_y": pin[1],
                    "status": "no_cliff_found",
                }
            )
            continue
        fallback = grp["length_m"].median()
        fallback = DEFAULT_ROUTE_LENGTH_M if np.isnan(fallback) else fallback
        for c, (_, route) in zip(centres, grp.iterrows()):
            idx = pc.column(c, cfg.route_radius_m)
            base_z = float(np.quantile(pc.xyz[idx, 2], 0.05))
            length = route["length_m"] if np.isfinite(route["length_m"]) else fallback
            out_dir = wall_direction(pc, idx)
            prof = route_profile(pc, idx, base_z, base_z + length, out_dir)
            rows.append(
                {
                    "wall": wall,
                    "route": route["route"],
                    "route_id": route["route_id"],
                    "route_type": route["route_type"],
                    "rating": route["rating"],
                    "pin_x": pin[0],
                    "pin_y": pin[1],
                    "x": c[0],
                    "y": c[1],
                    "snap_dist_m": float(np.linalg.norm(c - pin)),
                    "base_z": base_z,
                    "top_z": base_z + length,
                    "length_m": length,
                    "out_dx": out_dir[0],
                    "out_dy": out_dir[1],
                    **prof,
                    "status": "ok"
                    if prof["n_steep"] >= cfg.min_steep_points
                    else "unseen_by_lidar",
                }
            )
    df = pd.DataFrame(rows)
    if "x" in df:
        lonlat = work_to_lonlat(df["x"], df["y"])
        df["lon"], df["lat"] = lonlat[:, 0], lonlat[:, 1]
    return df


# ---------------------------------------------------------------------------
# negatives
# ---------------------------------------------------------------------------


def steep_fraction(pc: PointCloud, center_xy, radius: float) -> tuple[float, int]:
    idx = pc.column(center_xy, radius)
    return (float(pc.steep[idx].mean()) if len(idx) else 0.0), len(idx)


def build_negatives(
    pc: PointCloud,
    positives: pd.DataFrame,
    wall_pins: np.ndarray,
    cfg: LabelConfig,
    patch: PatchConfig,
    seed: int = 0,
) -> pd.DataFrame:
    """Contextual step-back negatives + random far-field negatives."""
    rng = np.random.default_rng(seed)
    pos_xy = positives[["x", "y"]].to_numpy()
    pos_tree = cKDTree(pos_xy)
    anchors = cKDTree(np.vstack([pos_xy, wall_pins]))
    rows, ignored = [], 0

    def try_add(c, source, wall):
        nonlocal ignored
        frac, n = steep_fraction(pc, c, patch.radius_m)
        if n < patch.min_points:
            return False
        if frac > cfg.max_steep_fraction_random_neg:
            ignored += 1  # probably rock nobody has logged: ignore, don't label 0
            return False
        rows.append(
            {
                "x": c[0],
                "y": c[1],
                "source": source,
                "wall": wall,
                "steep_frac": frac,
                "n_points": n,
            }
        )
        return True

    # contextual: out from the face (talus) and back over the top (plateau)
    for _, p in positives.iterrows():
        out = np.array([p["out_dx"], p["out_dy"]])
        for dist in cfg.stepback_m:
            for sign, name in ((1, "stepback_base"), (-1, "stepback_top")):
                c = np.array([p["x"], p["y"]]) + sign * dist * out
                # must not overlap any route's input patch
                if pos_tree.query(c)[0] < patch.radius_m + cfg.route_radius_m:
                    continue
                try_add(c, name, p["wall"])

    # random far-field terrain inside the region
    minx, miny, maxx, maxy = pc.bounds_xy
    pad = patch.radius_m
    n_random, attempts = 0, 0
    while n_random < cfg.n_random_negatives and attempts < cfg.n_random_negatives * 50:
        attempts += 1
        c = rng.uniform([minx + pad, miny + pad], [maxx - pad, maxy - pad])
        if anchors.query(c)[0] < cfg.ignore_buffer_m:
            continue
        n_random += try_add(c, "random", None)

    df = pd.DataFrame(rows)
    df.attrs["ignored_mostly_cliff"] = ignored
    return df


# ---------------------------------------------------------------------------
# patches
# ---------------------------------------------------------------------------


def extract_patch(pc: PointCloud, center_xy, patch: PatchConfig) -> np.ndarray:
    """Raw (n, 3) points of the vertical column around ``center_xy``, centred on it.

    X/Y are relative to the centre, Z relative to the column's lowest point;
    normalisation for the network happens in :mod:`dataset`.
    """
    idx = pc.column(center_xy, patch.radius_m)
    pts = pc.xyz[idx].copy()
    pts[:, 0] -= center_xy[0]
    pts[:, 1] -= center_xy[1]
    pts[:, 2] -= pts[:, 2].min() if len(pts) else 0.0
    return pts.astype(np.float32)


def build_samples(
    pc: PointCloud, routes: pd.DataFrame, cfg: ExperimentConfig
) -> tuple[pd.DataFrame, list[np.ndarray], pd.DataFrame]:
    """Everything above in one call.

    Returns ``(samples, patches, positives_all)``: ``samples`` has one row per
    usable patch (``label`` 1/0, ``wall``, ``source``, centre), ``patches[i]``
    is its raw point column, and ``positives_all`` keeps the rejected route
    cylinders too, for the feasibility report.
    """
    positives_all = build_positives(pc, routes, cfg.label)
    positives = positives_all[positives_all["status"] == "ok"].reset_index(drop=True)
    pins = lonlat_to_work(routes["lon"].to_numpy(), routes["lat"].to_numpy())
    negatives = build_negatives(
        pc, positives, pins, cfg.label, cfg.patch, seed=cfg.split.seed
    )

    pos = positives.assign(label=1, source="mp_route")[
        ["x", "y", "label", "source", "wall", "route"]
    ]
    neg = negatives.assign(label=0, route=None)[
        ["x", "y", "label", "source", "wall", "route"]
    ]
    samples = pd.concat([pos, neg], ignore_index=True)

    patches, keep = [], []
    for i, row in samples.iterrows():
        pts = extract_patch(pc, np.array([row["x"], row["y"]]), cfg.patch)
        if len(pts) >= cfg.patch.min_points:
            patches.append(pts)
            keep.append(i)
    samples = samples.loc[keep].reset_index(drop=True)
    samples["n_points"] = [len(p) for p in patches]
    samples.attrs["ignored_mostly_cliff"] = negatives.attrs.get(
        "ignored_mostly_cliff", 0
    )
    return samples, patches, positives_all
