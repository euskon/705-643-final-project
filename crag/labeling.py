"""Normals and cliff points, shared by pseudolabelling (Step 1) and route snapping (Step 3).

Carried over from the Horseshoe Canyon Ranch proof of concept
(``proof-of-concept/labeling.py``); its Mountain Project route-cylinder and
negative-sampling code (``place_route_cylinders``, ``build_positives``,
``build_negatives``) stayed there - pretraining no longer has a climbable
class and fine-tuning snaps verified route starts instead (``crag.negatives``).

* :func:`compute_normals` - local PCA over ``normal_k`` neighbours on a
  voxel-thinned copy (so the normal scale doesn't change with point density),
  every point fitted (the proof of concept's rule).
* :func:`surface_geometry` - the same fit on surface returns only (last
  returns, not LAS vegetation), plus planarity; what pseudolabels use.
* :class:`PointCloud` - points + 2D trees of all and of steep points for
  column queries; :func:`wall_direction` - which way a wall faces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm.auto import tqdm


@dataclass
class LabelConfig:
    """Normal estimation and the cliff-column rule shared by pseudolabels and route snapping."""

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
    # radius of a route's cylinder (the points scored as that route)
    route_radius_m: float = 2.5
    # a cliff column needs at least this many cliff points inside the
    # cylinder before a route / waypoint can snap to it
    min_steep_points: int = 8


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
        thin, nearest = _voxel_thin(xyz, voxel_m)
        return compute_normals(xyz[thin], k, voxel_m=None, chunk=chunk)[nearest]
    return _local_pca(xyz, k, chunk)[0]


def _voxel_thin(xyz: np.ndarray, voxel_m: float) -> tuple[np.ndarray, np.ndarray]:
    """(indices of one point per voxel, index into them of every point's nearest thinned point)."""
    keys = np.floor((xyz - xyz.min(axis=0)) / voxel_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    _, nearest = cKDTree(xyz[first]).query(xyz, workers=-1)
    return first, nearest


def _local_pca(xyz: np.ndarray, k: int, chunk: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """(unit normals with n_z >= 0, eigenvalues sorted ascending) of each point's ``k`` neighbours."""
    centred = xyz - xyz.mean(axis=0)
    tree = cKDTree(centred)
    k = min(k, len(centred))
    normals = np.empty_like(centred, dtype=np.float32)
    evals = np.empty_like(centred, dtype=np.float32)
    for start in tqdm(range(0, len(centred), chunk), desc="normals", disable=len(centred) <= chunk):
        stop = min(start + chunk, len(centred))
        _, idx = tree.query(centred[start:stop], k=k, workers=-1)
        nbrs = centred[idx]  # (n, k, 3)
        nbrs = nbrs - nbrs.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nbrs, nbrs) / k
        vals, vecs = np.linalg.eigh(cov)
        normals[start:stop] = vecs[:, :, 0]  # eigenvector of the smallest eigenvalue
        evals[start:stop] = vals
    normals[normals[:, 2] < 0] *= -1
    return normals, evals


def surface_geometry(
    xyz: np.ndarray, surface: np.ndarray, k: int = 12, voxel_m: float | None = 0.5, chunk: int = 200_000
) -> tuple[np.ndarray, np.ndarray]:
    """(normals, planarity) per point, fitted on the ``surface`` points only.

    ``surface`` marks the returns that can lie on the ground / rock surface
    (last returns that are not LAS vegetation). Fitting only on them keeps
    tree canopy out of the PCA neighbourhood, so rock under foliage gets the
    normal of the rock, not of the leaves in front of it. Same voxel thinning
    and ``k`` as :func:`compute_normals`. Every point (surface or not) takes
    the values of its nearest thinned surface point.

    ``planarity = (l2 - l3) / l1`` from the PCA eigenvalues ``l1 >= l2 >= l3``:
    near 1 for a flat patch (a rock face), near 0 for a line (a trunk or
    branch) or a scattered blob (foliage).
    """
    surface = np.asarray(surface, bool)
    if surface.sum() < 3:
        surface = np.ones(len(xyz), bool)
    s_idx = np.flatnonzero(surface)
    s_xyz = xyz[s_idx]
    if voxel_m:
        thin, _ = _voxel_thin(s_xyz, voxel_m)
        s_xyz = s_xyz[thin]
    normals, evals = _local_pca(s_xyz, k, chunk)
    l1 = np.maximum(evals[:, 2], 1e-12)
    planarity = np.clip((evals[:, 1] - evals[:, 0]) / l1, 0.0, 1.0).astype(np.float32)
    _, nearest = cKDTree(s_xyz).query(xyz, workers=-1)
    return normals[nearest], planarity[nearest]


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
    wall_dist: np.ndarray | None = None  # per point, metres to the bare-earth wall (crag.walls)

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
