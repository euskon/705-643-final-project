"""Stream USGS 3DEP LiDAR from the public Entwine Point Tile (EPT) buckets.

EPT is an octree of LAZ tiles on S3; PDAL's ``readers.ept`` only downloads the
octree nodes that intersect the requested bounds, so a few km^2 of the ranch
comes down in seconds without touching the 50 billion point dataset as a whole.

The region is fetched as a grid of ``tile_size_m`` squares (one PDAL pipeline
each), reprojected to UTM 15N, noise-filtered and cached as one ``.npz`` so
later runs never hit the network.

Example::

    from config import RegionConfig
    from usgs_ept import load_region

    cloud = load_region(RegionConfig())   # dict of numpy arrays
    cloud["xyz"].shape                    # (N, 3) metres in EPSG:32615
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pdal
from pyproj import Transformer
from tqdm.auto import tqdm

from config import EPT_CRS, NOISE_CLASSES, WGS84, WORK_CRS, RegionConfig

_TO_EPT = Transformer.from_crs(WGS84, EPT_CRS, always_xy=True)
_TO_WORK = Transformer.from_crs(WGS84, WORK_CRS, always_xy=True)
_WORK_TO_WGS = Transformer.from_crs(WORK_CRS, WGS84, always_xy=True)


def lonlat_to_work(lon, lat) -> np.ndarray:
    """WGS84 lon/lat (scalars or arrays) -> (..., 2) UTM 15N metres."""
    x, y = _TO_WORK.transform(
        np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    )
    return np.stack([x, y], axis=-1)


def work_to_lonlat(x, y) -> np.ndarray:
    """UTM 15N metres -> (..., 2) WGS84 lon/lat."""
    lon, lat = _WORK_TO_WGS.transform(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    )
    return np.stack([lon, lat], axis=-1)


def ept_bounds(
    bbox_lonlat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Lat/lon bbox -> (minx, miny, maxx, maxy) in the EPT's Web Mercator."""
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    xs, ys = _TO_EPT.transform([min_lon, max_lon], [min_lat, max_lat])
    return xs[0], ys[0], xs[1], ys[1]


def read_ept(
    ept_url: str, bounds_ept: tuple[float, float, float, float], retries: int = 4
) -> np.ndarray:
    """Run one PDAL pipeline: EPT read inside ``bounds_ept`` -> drop noise -> UTM 15N.

    Returns PDAL's structured array (fields X, Y, Z, Classification, ...).
    """
    minx, miny, maxx, maxy = bounds_ept
    # (filters.range ORs several ranges on one dimension, so it can't express
    # "not 7 and not 18"; filters.expression can)
    not_noise = " && ".join(f"Classification != {c}" for c in NOISE_CLASSES)
    stages = [
        {
            "type": "readers.ept",
            "filename": ept_url,
            "bounds": f"([{minx}, {maxx}], [{miny}, {maxy}])",
        },
        {"type": "filters.expression", "expression": not_noise},
        {"type": "filters.reprojection", "out_srs": WORK_CRS},
    ]
    # S3 occasionally drops a read; retry with backoff before giving up
    for attempt in range(retries):
        try:
            pipeline = pdal.Pipeline(json.dumps(stages))
            pipeline.execute()
            return pipeline.arrays[0]
        except RuntimeError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def _tile_grid(bounds_ept, tile_size_m: float, lat: float):
    """Split Web Mercator bounds into tiles that are ~``tile_size_m`` real metres wide."""
    minx, miny, maxx, maxy = bounds_ept
    # Web Mercator metres are inflated by 1/cos(lat)
    step = tile_size_m / np.cos(np.radians(lat))
    xs = np.append(np.arange(minx, maxx, step), maxx)
    ys = np.append(np.arange(miny, maxy, step), maxy)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            yield x0, y0, x1, y1


def stream_region(region: RegionConfig) -> dict[str, np.ndarray]:
    """Fetch the whole region tile by tile and return it as plain numpy arrays."""
    bounds = ept_bounds(region.bbox_lonlat)
    mid_lat = 0.5 * (region.bbox_lonlat[1] + region.bbox_lonlat[3])
    tiles = list(_tile_grid(bounds, region.tile_size_m, mid_lat))
    parts = []
    for tile in tqdm(tiles, desc=f"EPT {region.ept_name}"):
        arr = read_ept(region.ept_url, tile)
        if len(arr):
            parts.append(arr)
    arr = np.concatenate(parts)
    # neighbouring EPT reads share their edge, so drop exact duplicates
    xyz = np.stack([arr["X"], arr["Y"], arr["Z"]], axis=1)
    xyz, keep = np.unique(xyz, axis=0, return_index=True)
    return {
        "xyz": xyz.astype(np.float64),
        "classification": arr["Classification"][keep].astype(np.uint8),
        "return_number": arr["ReturnNumber"][keep].astype(np.uint8),
        "number_of_returns": arr["NumberOfReturns"][keep].astype(np.uint8),
        "intensity": arr["Intensity"][keep].astype(np.uint16),
    }


def load_region(region: RegionConfig, refresh: bool = False) -> dict[str, np.ndarray]:
    """Cached :func:`stream_region`: reads ``<cache_dir>/<ept_name>.npz`` if present."""
    path = Path(region.cache_dir) / f"{region.ept_name}.npz"
    if path.exists() and not refresh:
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    cloud = stream_region(region)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cloud)
    return cloud
