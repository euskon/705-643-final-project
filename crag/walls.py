"""Bare-earth wall mask: one definition of "wall" used by pseudolabels and point checks.

Built on a 1 m grid:

1. bare-earth DEM: the lowest *LAS ground* return (class 2, the data
   provider's own ground classification) in each cell; cells without one take
   the nearest cell that has one. Under a tree there are often no ground
   returns at all, and using any other return there makes a raised "ring wall"
   of canopy around the tree. Tiles with almost no class 2 points (less than
   ``min_ground_frac``) fall back to the lowest surface return (last returns
   that are not LAS vegetation);
2. slope of the DEM after a light Gaussian smooth (``smooth_m``);
3. relief: highest minus lowest DEM cell in a ``relief_window_m`` square;
4. wall = slope >= ``min_slope_deg`` AND relief >= ``min_relief_m``, then a
   ``open_m`` morphological opening removes specks.

A vertical wall gridded on 1 m cells becomes a strip from its foot to its top,
so "distance to wall" for a point beside it is the distance to the foot.
:func:`point_wall_distance` gives that distance for every point; pseudolabels
store it per point (``wall_dist`` in the stage files) and the route-point
check (``crag.groundtruth.snap_one``) uses it. See docs/hcr-waypoint-audit.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

MAX_DIST_M = 100.0  # distances are capped here (and stored as float16)


@dataclass
class WallConfig:
    res_m: float = 1.0
    smooth_m: float = 1.0
    min_slope_deg: float = 50.0
    min_relief_m: float = 4.0
    relief_window_m: float = 5.0
    open_m: float = 2.0
    # a cliff pseudolabel needs its point within this (horizontal) distance
    # of a wall cell; covers overhangs and faces leaning past their foot
    near_m: float = 2.0
    # use LAS ground (class 2) for the DEM when at least this share of points has it
    min_ground_frac: float = 0.02


@dataclass
class WallGrid:
    x0: float
    y0: float
    res: float
    dem: np.ndarray
    slope: np.ndarray
    relief: np.ndarray
    wall: np.ndarray
    dist: np.ndarray  # metres from each cell to the nearest wall cell

    def cell(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.wall.shape
        ix = np.clip(((xy[:, 0] - self.x0) / self.res).astype(int), 0, nx - 1)
        iy = np.clip(((xy[:, 1] - self.y0) / self.res).astype(int), 0, ny - 1)
        return iy, ix

    def distance(self, xy: np.ndarray) -> np.ndarray:
        iy, ix = self.cell(np.atleast_2d(xy))
        return self.dist[iy, ix]


def dem_points(surface: np.ndarray, ground: np.ndarray | None, cfg: WallConfig) -> np.ndarray:
    """Which points the DEM is built from: LAS ground if the tile has enough, else surface returns."""
    if ground is not None and ground.sum() >= max(3, cfg.min_ground_frac * len(ground)):
        return ground
    return surface if surface.sum() >= 3 else np.ones(len(surface), bool)


def wall_grid(xyz: np.ndarray, surface: np.ndarray, cfg: WallConfig | None = None, ground: np.ndarray | None = None) -> WallGrid:
    """The wall mask of the area the points cover.

    ``ground`` = LAS class 2 (preferred for the DEM), ``surface`` = ground/rock-capable returns
    (last returns, not LAS vegetation; the fallback)."""
    cfg = cfg or WallConfig()
    res = cfg.res_m
    p = xyz[dem_points(np.asarray(surface, bool), None if ground is None else np.asarray(ground, bool), cfg)]
    x0, y0 = np.floor(xyz[:, :2].min(0) / res) * res
    nx, ny = (np.floor((xyz[:, :2].max(0) - [x0, y0]) / res)).astype(int) + 1
    ix = np.clip(((p[:, 0] - x0) / res).astype(int), 0, nx - 1)
    iy = np.clip(((p[:, 1] - y0) / res).astype(int), 0, ny - 1)
    dem = np.full(ny * nx, np.inf)
    np.minimum.at(dem, iy * nx + ix, p[:, 2])
    dem = dem.reshape(ny, nx)
    hole = ~np.isfinite(dem)
    if hole.all():
        z = np.zeros_like(dem, dtype=bool)
        return WallGrid(x0, y0, res, np.zeros_like(dem), np.zeros_like(dem), np.zeros_like(dem), z, np.full(dem.shape, MAX_DIST_M))
    if hole.any():
        _, (fy, fx) = ndimage.distance_transform_edt(hole, return_indices=True)
        dem = dem[fy, fx]
    gy, gx = np.gradient(ndimage.gaussian_filter(dem, cfg.smooth_m / res), res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    w = max(int(round(cfg.relief_window_m / res)), 1)
    relief = ndimage.maximum_filter(dem, w) - ndimage.minimum_filter(dem, w)
    wall = (slope >= cfg.min_slope_deg) & (relief >= cfg.min_relief_m)
    k = max(int(round(cfg.open_m / res)), 1)
    if k > 1:
        wall = ndimage.binary_opening(wall, np.ones((k, k), bool))
    if wall.any():
        dist = ndimage.distance_transform_edt(~wall) * res
    else:
        dist = np.full(wall.shape, MAX_DIST_M)
    return WallGrid(x0, y0, res, dem, slope, relief, wall, np.minimum(dist, MAX_DIST_M))


def point_wall_distance(xyz: np.ndarray, surface: np.ndarray, cfg: WallConfig | None = None, ground: np.ndarray | None = None) -> np.ndarray:
    """Horizontal distance (m) from every point to the nearest wall cell."""
    return wall_grid(xyz, surface, cfg, ground).distance(xyz[:, :2]).astype(np.float32)
