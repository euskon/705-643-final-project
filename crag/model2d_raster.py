"""Top-down images of a LiDAR block, for the 2D (ResNet) models.

A block of points is laid on a square grid seen from straight above, and each
grid cell keeps a few numbers about the points that fall in it. One image has
``CHANNELS`` layers:

=====  ==========  ============================================================
 idx   name        per cell
=====  ==========  ============================================================
 0     dem         bare-earth height: the lowest *LAS ground* return (class 2).
                   Cells without one are interpolated from those that have. Blocks
                   with almost no class 2 points use the lowest surface return
                   instead (last returns that are not LAS vegetation) - the
                   same rule as the wall mask (:func:`crag.walls.dem_points`)
 1     zmin        lowest return of any kind (interpolated where empty)
 2     zmax        highest return of any kind (canopy / wall top; interpolated)
 3     count       number of returns (0 = the scanner saw nothing here)
 4     dem_valid   1 where ``dem`` is a real measurement, 0 where it was filled
=====  ==========  ============================================================

Heights are metres above the block's lowest ``dem`` cell (``z0``), so every
image starts at 0 whatever the altitude of the area.

Orientation is a map with north up: row 0 is the northern edge, column 0 the
western edge. :func:`gradient` turns ``dem`` into the two-channel elevation
gradient image of Rafique et al. (2022): d(height)/d(east) and
d(height)/d(north), in metres per metre.

The models built on these images:

* 2D model: :func:`gradient` of ``dem`` (2 channels);
* in-between model: ``zmin``, ``zmax``, ``count`` (3 channels).

Both are derived at load time, so the stored file serves either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError

from crag.paths import DATA_DIR
from crag.walls import WallConfig, dem_points

if TYPE_CHECKING:  # the LiDAR readers need pyproj/PDAL; training only needs the pure functions below
    from crag.crop import AreaCloud

CHANNELS = ("dem", "zmin", "zmax", "count", "dem_valid")
RASTER_DIR = DATA_DIR / "raster"  # output of scripts/model2d_rasterize.py (git-ignored by its own .gitignore)
GROUND_LAS_CLASS = 2
VEG_LAS_CLASSES = (3, 4, 5)


@dataclass
class RasterConfig:
    size_m: float = 20.0  # side of the square, same footprint as the 3D fine-tuning block
    res_m: float = 0.5  # cell size; USGS 3DEP is ~4-40 points per m2, so finer cells are mostly empty
    # use LAS ground for ``dem`` when at least this share of the block's points has it (as crag.walls)
    min_ground_frac: float = 0.02

    @property
    def n(self) -> int:
        return int(round(self.size_m / self.res_m))


def crop_returns(cloud: AreaCloud, cx: float, cy: float, half: float) -> dict:
    """Raw returns with |x-cx|, |y-cy| <= half: ``xyz`` (absolute), ``ground`` and ``surface`` masks.

    The same square as :meth:`crag.crop.AreaCloud.crop` (what the 3D block is
    cut from), but keeping the LAS classification instead of the pseudolabels.
    """
    from crag.tiles import Tile

    s = cloud.size
    xyz, ground, surface = [], [], []
    for i in range(math.floor((cx - half) / s), math.floor((cx + half) / s) + 1):
        for j in range(math.floor((cy - half) / s), math.floor((cy + half) / s) + 1):
            t = cloud._tile(Tile(i * s, j * s, s).id)
            if t is None:
                continue
            p = t["xyz_abs"]
            m = (np.abs(p[:, 0] - cx) <= half) & (np.abs(p[:, 1] - cy) <= half)
            c = t["classification"][m]
            last = t["return_number"][m] >= t["number_of_returns"][m]
            xyz.append(p[m])
            ground.append(c == GROUND_LAS_CLASS)
            surface.append(last & ~np.isin(c, VEG_LAS_CLASSES))
    if not xyz:
        return {"xyz": np.zeros((0, 3)), "ground": np.zeros(0, bool), "surface": np.zeros(0, bool)}
    return {"xyz": np.concatenate(xyz), "ground": np.concatenate(ground), "surface": np.concatenate(surface)}


def _fill(grid: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Fill hole cells from the measured ones.

    Holes inside the measured cells are filled by linear interpolation over a
    triangulation of them (a TIN, the usual way a DEM is made from scattered
    ground points). Copying the nearest value instead leaves a step along every
    boundary between two measured cells' territories, and in a gradient image
    those steps look like a web of thin walls. Holes outside the triangulation
    (image corners) take the nearest filled value.
    """
    if not hole.any():
        return grid
    out = np.where(hole, np.nan, grid)
    known = np.argwhere(~hole)
    if len(known) >= 4:
        try:
            out[hole] = LinearNDInterpolator(known, grid[~hole])(np.argwhere(hole))
        except QhullError:  # measured cells all on one line: nothing to triangulate
            pass
    left = np.isnan(out)
    if left.any():
        _, (fy, fx) = ndimage.distance_transform_edt(left, return_indices=True)
        out = out[fy, fx]
    return out


def rasterize(xyz: np.ndarray, ground: np.ndarray, surface: np.ndarray, x0: float, y0: float, cfg: RasterConfig | None = None) -> tuple[np.ndarray, dict]:
    """Image of the square whose south-west corner is ``(x0, y0)``.

    Returns ``(image, stats)``: ``image`` is float32 ``(len(CHANNELS), n, n)``
    (see the module docstring); ``stats`` has ``z0`` (absolute height of the
    image's zero), ``n_points``, ``n_ground``, ``used_ground`` (False = the
    surface-return fallback), ``frac_empty`` (cells without any return) and
    ``frac_dem_filled`` (cells whose ``dem`` was filled from a neighbour).
    A block without points gives an all-zero image.
    """
    cfg = cfg or RasterConfig()
    n, res = cfg.n, cfg.res_m
    img = np.zeros((len(CHANNELS), n, n), np.float32)
    stats = {"z0": 0.0, "n_points": int(len(xyz)), "n_ground": int(np.sum(ground)), "used_ground": False, "frac_empty": 1.0, "frac_dem_filled": 1.0}
    if len(xyz) == 0:
        return img, stats
    col = np.clip(np.floor((xyz[:, 0] - x0) / res).astype(np.int64), 0, n - 1)
    row = n - 1 - np.clip(np.floor((xyz[:, 1] - y0) / res).astype(np.int64), 0, n - 1)  # north up
    cell = row * n + col
    z = xyz[:, 2].astype(np.float64)

    count = np.bincount(cell, minlength=n * n).reshape(n, n)
    empty = count == 0
    zmin = np.full(n * n, np.inf)
    np.minimum.at(zmin, cell, z)
    zmax = np.full(n * n, -np.inf)
    np.maximum.at(zmax, cell, z)
    zmin = _fill(zmin.reshape(n, n), empty)
    zmax = _fill(zmax.reshape(n, n), empty)

    ground, surface = np.asarray(ground, bool), np.asarray(surface, bool)
    use = dem_points(surface, ground, WallConfig(min_ground_frac=cfg.min_ground_frac))
    dem = np.full(n * n, np.inf)
    np.minimum.at(dem, cell[use], z[use])
    dem = dem.reshape(n, n)
    hole = ~np.isfinite(dem)
    dem = _fill(dem, hole)

    z0 = float(dem.min())
    img[0], img[1], img[2] = dem - z0, zmin - z0, zmax - z0
    img[3] = count
    img[4] = ~hole
    stats.update(z0=z0, used_ground=use is ground,frac_empty=float(empty.mean()), frac_dem_filled=float(hole.mean()))
    return img, stats


def rasterize_sample(cloud: AreaCloud, cx: float, cy: float, cfg: RasterConfig | None = None) -> tuple[np.ndarray, dict]:
    """Image of the ``size_m`` square centred on ``(cx, cy)`` (an area's UTM coordinates)."""
    cfg = cfg or RasterConfig()
    half = cfg.size_m / 2
    c = crop_returns(cloud, cx, cy, half)
    return rasterize(c["xyz"], c["ground"], c["surface"], cx - half, cy - half, cfg)


def gradient(dem: np.ndarray, res_m: float) -> np.ndarray:
    """Two-channel elevation gradient of a north-up height image ``(..., H, W)``.

    Channel 0 is the height change per metre going east, channel 1 going
    north. Compute it *after* any flip / rotation of ``dem``, so the directions
    stay true to the augmented image.
    """
    d_row, d_col = np.gradient(dem.astype(np.float32), res_m, axis=(-2, -1))
    return np.stack([d_col, -d_row], axis=-3)  # rows run north -> south, hence the sign


def slope_deg(dem: np.ndarray, res_m: float) -> np.ndarray:
    """Steepness in degrees (0 = flat, 90 = vertical) of a north-up height image."""
    g = gradient(dem, res_m)
    return np.degrees(np.arctan(np.hypot(g[..., 0, :, :], g[..., 1, :, :])))
