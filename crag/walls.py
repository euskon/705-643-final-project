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
    gx: np.ndarray | None = None  # gradient of the smoothed ground grid (m per m), east
    gy: np.ndarray | None = None  # ... north

    def uphill(self, xy, window: int = 5) -> np.ndarray:
        """Horizontal unit vector pointing uphill at ``xy`` (smoothed-grid gradient averaged
        over ``window`` x ``window`` cells). On flat ground beside a wall the gradient says
        nothing, so the direction toward the nearest wall cell is used instead."""
        iy, ix = self.cell(np.atleast_2d(np.asarray(xy, float)))
        iy, ix, h = int(iy[0]), int(ix[0]), window // 2
        sl = (slice(max(iy - h, 0), iy + h + 1), slice(max(ix - h, 0), ix + h + 1))
        g = np.array([self.gx[sl].mean(), self.gy[sl].mean()]) if self.gx is not None else np.zeros(2)
        if np.hypot(*g) < 0.3:  # under ~17 degrees: use "toward the wall"
            dy, dx = np.gradient(self.dist[max(iy - 2, 0):iy + 3, max(ix - 2, 0):ix + 3]) if min(self.dist.shape) > 1 else (np.zeros((1, 1)), np.zeros((1, 1)))
            t = -np.array([dx.mean(), dy.mean()])
            if np.hypot(*t) > 1e-6:
                g = t
        n = np.hypot(*g)
        return g / n if n > 1e-9 else np.array([1.0, 0.0])

    def dem_at(self, xy) -> np.ndarray:
        iy, ix = self.cell(np.atleast_2d(np.asarray(xy, float)))
        return self.dem[iy, ix]

    def is_wall(self, xy) -> np.ndarray:
        iy, ix = self.cell(np.atleast_2d(np.asarray(xy, float)))
        return self.wall[iy, ix]

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
    return WallGrid(x0, y0, res, dem, slope, relief, wall, np.minimum(dist, MAX_DIST_M), gx, gy)


def point_wall_distance(xyz: np.ndarray, surface: np.ndarray, cfg: WallConfig | None = None, ground: np.ndarray | None = None) -> np.ndarray:
    """Horizontal distance (m) from every point to the nearest wall cell."""
    return wall_grid(xyz, surface, cfg, ground).distance(xyz[:, :2]).astype(np.float32)


def wall_height_at(grid: WallGrid, xy, base_z: float, *, max_walk_m: float = 30.0, step_m: float = 0.5, start_m: float = 3.0,
                   gap_m: float = 1.5, past_top_m: float = 1.0) -> tuple[float, float]:
    """(wall height in m, distance walked across the wall strip in m).

    From ``xy`` (a candidate foot), walk uphill across the wall mask. The strip must be
    reached within ``start_m``; breaks up to ``gap_m`` are allowed. Height = highest ground
    value in the strip and ``past_top_m`` beyond it, minus ``base_z``. (0, nan) when no
    strip is reached. ``max_walk_m`` limits how far back the walk goes, not the height."""
    xy = np.asarray(xy, float)
    up = grid.uphill(xy)
    t = np.arange(0.0, max_walk_m + past_top_m + step_m, step_m)
    pts = xy[None, :] + t[:, None] * up[None, :]
    wall = grid.is_wall(pts)
    dem = grid.dem_at(pts)
    first = np.flatnonzero(wall & (t <= start_m))
    if not len(first):
        return 0.0, float("nan")
    last, k = int(first[0]), int(first[0]) + 1
    while k < len(t) and t[k] <= max_walk_m and t[k] - t[last] <= gap_m + 1e-9:
        if wall[k]:
            last = k
        k += 1
    end = t[last] + past_top_m
    top = float(dem[(t >= t[first[0]] - 1e-9) & (t <= end + 1e-9)].max())
    return max(top - float(base_z), 0.0), float(t[last] - t[first[0]])


def wall_frame_at(grid: WallGrid, xy) -> tuple[np.ndarray, np.ndarray]:
    """(out, along) unit vectors: out = downhill (from the wall into the air), along = out turned 90 degrees."""
    out = -grid.uphill(xy)
    return out, np.array([-out[1], out[0]])


VEG_LAS = (3, 4, 5)
GROUND_LAS = 2


def grid_from_crop(c: dict, cfg: WallConfig | None = None) -> WallGrid | None:
    """Wall grid of a raw crop (``AreaCloud.crop``), with the same inputs as the pseudolabels:
    surface = last returns that are not LAS vegetation, ground = LAS class 2."""
    if len(c["xyz"]) < 16:
        return None
    n = len(c["xyz"])
    if "las" in c and "return_number" in c:
        surface = (c["return_number"] >= c["number_of_returns"]) & ~np.isin(c["las"], VEG_LAS)
        ground = c["las"] == GROUND_LAS
    else:  # synthetic clouds in tests
        surface, ground = np.ones(n, bool), None
    return wall_grid(c["xyz"], surface, cfg, ground)


def local_grid(cloud, cx: float, cy: float, half: float = 45.0, cfg: WallConfig | None = None) -> WallGrid | None:
    """Wall grid of the ``2 * half`` square around a point."""
    return grid_from_crop(cloud.crop(cx, cy, half), cfg)


def grid_for(cloud, x: float, y: float, cell_m: float = 100.0, border_m: float = 45.0, maxlen: int = 48) -> WallGrid | None:
    """Cached wall grid covering ``(x, y)`` with at least ``border_m`` around it (one grid per
    ``cell_m`` cell of the area, kept on the cloud object)."""
    cache = cloud.__dict__.setdefault("_grid_cache", {})
    key = (int(np.floor(x / cell_m)), int(np.floor(y / cell_m)))
    if key not in cache:
        if len(cache) >= maxlen:
            cache.pop(next(iter(cache)))
        cache[key] = local_grid(cloud, (key[0] + 0.5) * cell_m, (key[1] + 0.5) * cell_m, cell_m / 2 + border_m)
    return cache[key]


def point_height_above_ground(xyz: np.ndarray, surface: np.ndarray, cfg: WallConfig | None = None, ground: np.ndarray | None = None,
                              window_m: float = 10.0, grid: WallGrid | None = None) -> np.ndarray:
    """Height of every point above the lowest bare-earth ground cell within ``window_m`` (float32)."""
    g = grid or wall_grid(xyz, surface, cfg, ground)
    k = int(round(2 * window_m / g.res)) + 1
    low = ndimage.minimum_filter(g.dem, size=k, mode="nearest")
    iy, ix = g.cell(xyz[:, :2])
    return (xyz[:, 2] - low[iy, ix]).astype(np.float32)
