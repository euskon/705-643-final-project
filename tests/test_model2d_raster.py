"""Top-down images for the 2D models (crag/model2d_raster.py). No LiDAR download or GPU needed."""

import numpy as np

from crag.model2d_raster import CHANNELS, RasterConfig, gradient, rasterize, slope_deg

CFG = RasterConfig(size_m=10.0, res_m=0.5)  # 20 x 20 cells


def _grid_points(z_of_xy, step=0.25, size=10.0):
    """One point at the centre of every ``step`` cell of a ``size`` square with SW corner (100, 200)."""
    c = np.arange(step / 2, size, step)
    x, y = np.meshgrid(c, c)
    x, y = x.ravel() + 100.0, y.ravel() + 200.0
    return np.stack([x, y, z_of_xy(x - 100.0, y - 200.0)], axis=1)


def _raster(xyz, ground=None, surface=None):
    ground = np.ones(len(xyz), bool) if ground is None else ground
    surface = np.ones(len(xyz), bool) if surface is None else surface
    return rasterize(xyz, ground, surface, 100.0, 200.0, CFG)


def test_shape_and_zero_level():
    img, st = _raster(_grid_points(lambda x, y: 500.0 + 0 * x))
    assert img.shape == (len(CHANNELS), 20, 20) and img.dtype == np.float32
    assert st["z0"] == 500.0 and st["frac_empty"] == 0.0 and st["frac_dem_filled"] == 0.0 and st["used_ground"]
    assert np.allclose(img[0], 0) and np.all(img[3] == 4) and np.all(img[4] == 1)  # 4 points per 0.5 m cell


def test_north_is_up_and_gradient_directions():
    # height rises 1 m per metre going north, flat going east
    img, _ = _raster(_grid_points(lambda x, y: 300.0 + y))
    dem = img[0]
    assert dem[0].mean() > dem[-1].mean()  # row 0 = northern edge = highest
    g = gradient(dem, CFG.res_m)
    assert g.shape == (2, 20, 20)
    assert np.allclose(g[0], 0, atol=1e-4) and np.allclose(g[1], 1, atol=1e-4)  # (d/d east, d/d north)
    assert np.allclose(slope_deg(dem, CFG.res_m), 45, atol=1e-2)
    # height rising east instead: column 0 = western edge = lowest
    img, _ = _raster(_grid_points(lambda x, y: 300.0 + 2 * x))
    g = gradient(img[0], CFG.res_m)
    assert img[0][:, 0].mean() < img[0][:, -1].mean()
    assert np.allclose(g[0], 2, atol=1e-4) and np.allclose(g[1], 0, atol=1e-4)


def test_holes_are_interpolated_not_stepped():
    # a tilted plane sampled only on every 4th cell: the filled DEM must still be that plane
    xyz = _grid_points(lambda x, y: 100.0 + 0.5 * x + 0.25 * y, step=2.0)
    img, st = _raster(xyz)
    assert st["frac_dem_filled"] > 0.9 and img[4].sum() == len(xyz)
    inner = slope_deg(img[0], CFG.res_m)[4:-4, 4:-4]  # inside the triangulated region
    assert np.allclose(inner, np.degrees(np.arctan(np.hypot(0.5, 0.25))), atol=0.5)


def test_dem_uses_ground_and_zmax_sees_canopy():
    ground_pts = _grid_points(lambda x, y: 50.0 + 0 * x)
    canopy = ground_pts.copy()
    canopy[:, 2] += 12.0
    xyz = np.concatenate([ground_pts, canopy])
    is_ground = np.r_[np.ones(len(ground_pts), bool), np.zeros(len(canopy), bool)]
    img, st = _raster(xyz, ground=is_ground, surface=is_ground)
    assert st["used_ground"] and np.allclose(img[0], 0) and np.allclose(img[1], 0) and np.allclose(img[2], 12)
    # no LAS ground at all -> surface-return fallback
    img, st = _raster(xyz, ground=np.zeros(len(xyz), bool), surface=is_ground)
    assert not st["used_ground"] and np.allclose(img[0], 0)


def test_empty_block():
    img, st = rasterize(np.zeros((0, 3)), np.zeros(0, bool), np.zeros(0, bool), 0.0, 0.0, CFG)
    assert img.shape == (len(CHANNELS), 20, 20) and not img.any() and st["n_points"] == 0 and st["frac_empty"] == 1.0
