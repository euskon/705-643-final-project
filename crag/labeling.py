"""Cliff-labelling heuristics shared by pseudolabelling (Step 1) and route snapping (Step 3).

Carried over from the Horseshoe Canyon Ranch proof of concept
(``proof-of-concept/labeling.py``), minus its patch/sample code:

1. **Normals** - local PCA over ``normal_k`` neighbours (on a voxel-thinned
   copy, so the normal scale doesn't change with point density); slope =
   angle between the normal and vertical. Points steeper than
   ``steep_slope_deg`` (and, by default, single-return, to drop tree canopy)
   are "cliff" points.
2. **Snap** - an MP wall pin is usually 10-30 m off the rock, so within
   ``snap_search_radius_m`` of the pin we look for columns that actually hold
   a cliff (>= ``min_steep_points`` steep points inside a 2.5 m cylinder) and
   greedily place route cylinders there, nearest-to-pin first, at least
   ``route_spacing_m`` apart. A wall gets at most one cylinder per MP route.
3. **Route cylinder** - each cylinder spans the cliff base (5th percentile Z)
   up to base + the MP route length; it is ``ok`` only if the LiDAR actually
   saw rock in that height range (``route_profile`` diagnostics).
4. **Negatives** - contextual step-backs out from / over the wall along its
   normal, plus optional random far-field points; a candidate that is itself
   mostly cliff could be unlisted rock, so it is skipped rather than labelled 0.

The changes from the proof of concept: ``build_positives`` takes the
area's CRS converters (no UTM 15N default), and ``PatchConfig`` keeps only the
fields ``build_negatives`` uses. ``LabelConfig`` is unchanged field for field,
so pseudolabel digests (``crag.pseudolabels``) stay the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm.auto import tqdm


@dataclass
class LabelConfig:
    """How Mountain Project routes become climbable / non-climbable patches."""

    # local-PCA neighbourhood used for surface normals
    normal_k: int = 12
    # normals are computed on a copy thinned to one point per voxel of this
    # size, so their scale doesn't depend on point density (flight-line
    # overlap strips have ~2x the points and otherwise show as fake steep
    # stripes); None disables the thinning
    normal_voxel_m: float | None = 0.5
    # a point is "steep" (part of a cliff face) above this slope in degrees
    steep_slope_deg: float = 60.0
    # many 3DEP collections have no vegetation class and most steep-normal
    # multi-return points are tree canopy (~58% at Horseshoe Canyon); a laser
    # pulse that hits bare rock returns once, so only count single returns as
    # cliff
    steep_single_return_only: bool = True
    # MP wall pins are ~10-30 m off; look this far for the cliff line
    snap_search_radius_m: float = 30.0
    # minimum spacing between neighbouring route cylinders on one wall
    route_spacing_m: float = 3.0
    # radius of the cylinder whose points belong to a route (the label region)
    route_radius_m: float = 2.5
    # a route cylinder with fewer steep points than this inside its height
    # range is "unseen by the LiDAR" and is dropped (and reported)
    min_steep_points: int = 8
    # how far to step away from a route, along the wall normal, for the
    # contextual negatives (cliff top plateau and talus at the base)
    stepback_m: tuple[float, ...] = (12.0, 20.0)
    # random negatives must be at least this far from every MP wall pin and
    # every positive (unknown band around crags is ignored, not labelled 0)
    ignore_buffer_m: float = 40.0
    # random negatives that are mostly cliff are probably unlisted rock rather
    # than true negatives, so they are ignored too
    max_steep_fraction_random_neg: float = 0.25
    n_random_negatives: int = 300
    # drop routes whose MP pin is a default (shared / area) pin, see crag.mp.parse_routes
    drop_default_pin_routes: bool = True


@dataclass
class PatchConfig:
    """Context column around a candidate negative (``build_negatives``)."""

    # horizontal radius of the column; must not overlap a route's column
    radius_m: float = 5.0
    # columns with fewer points than this are skipped (no LiDAR there)
    min_points: int = 64


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
    pc: PointCloud,
    routes: pd.DataFrame,
    cfg: LabelConfig,
    to_work,
    to_lonlat,
) -> pd.DataFrame:
    """One row per placed route cylinder (kept or rejected, with the reason).

    ``to_work`` / ``to_lonlat`` convert between lon/lat and the point cloud's
    metric CRS (the area's UTM zone, see :class:`crag.geo.AreaCRS`).
    """
    rows, taken = [], []
    for (wall, lat, lon), grp in routes.groupby(["wall", "lat", "lon"], sort=False):
        pin = to_work(lon, lat)
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
        lonlat = to_lonlat(df["x"], df["y"])
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
