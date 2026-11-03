"""Per-area metric coordinates: every area works in its own UTM zone (true metres)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyproj import Transformer

WGS84 = "EPSG:4326"
EPT_CRS = "EPSG:3857"  # all USGS 3DEP Entwine Point Tiles are Web Mercator


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=64)
def transformer(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


class AreaCRS:
    """lon/lat <-> UTM metres (and Web Mercator for EPT bounds) for one area."""

    def __init__(self, epsg: int):
        self.epsg = int(epsg)
        self.crs = f"EPSG:{self.epsg}"

    def to_work(self, lon, lat) -> np.ndarray:
        x, y = transformer(WGS84, self.crs).transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.stack([x, y], axis=-1)

    def to_lonlat(self, x, y) -> np.ndarray:
        lon, lat = transformer(self.crs, WGS84).transform(np.asarray(x, float), np.asarray(y, float))
        return np.stack([lon, lat], axis=-1)

    def work_bounds_to_ept(self, x0, y0, x1, y1, pad_m: float = 2.0) -> tuple[float, float, float, float]:
        """UTM rectangle -> enclosing Web Mercator bounds (corners + padding)."""
        xs = np.array([x0 - pad_m, x1 + pad_m, x0 - pad_m, x1 + pad_m])
        ys = np.array([y0 - pad_m, y0 - pad_m, y1 + pad_m, y1 + pad_m])
        mx, my = transformer(self.crs, EPT_CRS).transform(xs, ys)
        return float(mx.min()), float(my.min()), float(mx.max()), float(my.max())


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371008.8 * 2 * np.arcsin(np.sqrt(a))
