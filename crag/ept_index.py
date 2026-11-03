"""Pick the USGS 3DEP Entwine Point Tile dataset(s) covering an area.

Source: hobu's boundary index of the public ``usgs-lidar-public`` bucket
(https://github.com/hobuinc/usgs-lidar, ``boundaries/resources.geojson``):
one MultiPolygon per dataset with its name, EPT URL and point count. It is
downloaded once into ``data/raw/usgs/`` and reused.

For an area we take its *footprint* (the union of circles of
``crag_radius_m`` around every crag pin - the part we will actually
download), intersect it with every dataset boundary and rank the datasets by:

1. coverage of the footprint (>= ``min_coverage`` first),
2. point density (points / boundary area),
3. collection year (newer first).

The ranked list goes into the registry; ``crag.tiles`` downloads each tile
from the first dataset in the list whose boundary contains it, so an area
straddling two collections is still fully covered.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from pyproj import Geod
from shapely.geometry import Point, shape
from shapely.ops import transform as shp_transform, unary_union
from shapely.prepared import prep

from crag.geo import WGS84, transformer
from crag.paths import RAW_DIR

log = logging.getLogger(__name__)

INDEX_URL = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson"
INDEX_PATH = RAW_DIR / "usgs" / "resources.geojson"
_GEOD = Geod(ellps="WGS84")


def download_index(refresh: bool = False) -> Path:
    if INDEX_PATH.exists() and not refresh:
        return INDEX_PATH
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(INDEX_URL, timeout=300)
    resp.raise_for_status()
    INDEX_PATH.write_text(resp.text)
    return INDEX_PATH


def dataset_year(name: str) -> int | None:
    years = [int(y) for y in re.findall(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)", name)]
    if years:
        return max(years)
    m = re.search(r"_D(\d{2})$", name)  # e.g. AR_Eastern_1_D23 -> 2023
    return 2000 + int(m.group(1)) if m else None


@lru_cache(maxsize=1)
def load_index() -> list[dict]:
    feats = json.loads(download_index().read_text())["features"]
    out = []
    for f in feats:
        geom = shape(f["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        area_m2 = abs(_GEOD.geometry_area_perimeter(geom)[0])
        p = f["properties"]
        out.append(
            {
                "name": p["name"],
                "url": p["url"],
                "count": int(p["count"]),
                "geom": geom,
                "prepared": prep(geom),
                "points_per_m2": p["count"] / area_m2 if area_m2 > 0 else 0.0,
                "year": dataset_year(p["name"]),
            }
        )
    return out


@lru_cache(maxsize=4096)
def dataset_by_name(name: str) -> dict:
    for d in load_index():
        if d["name"] == name:
            return d
    raise KeyError(f"USGS EPT dataset {name!r} not in index")


def footprint_lonlat(points_lonlat: np.ndarray, radius_m: float, epsg: int):
    """Union of ``radius_m`` circles around lon/lat points, as a lon/lat polygon."""
    to_work = transformer(WGS84, f"EPSG:{epsg}").transform
    to_ll = transformer(f"EPSG:{epsg}", WGS84).transform
    xy = np.stack(to_work(points_lonlat[:, 0], points_lonlat[:, 1]), axis=1)
    poly = unary_union([Point(x, y).buffer(radius_m, 16) for x, y in xy])
    return shp_transform(to_ll, poly), poly.area


def rank_datasets(
    points_lonlat: np.ndarray,
    epsg: int,
    radius_m: float = 150.0,
    min_coverage: float = 0.9,
    max_keep: int = 4,
) -> list[dict]:
    """Datasets intersecting the crag footprint, best first (see module docstring)."""
    if len(points_lonlat) == 0:
        return []
    fp_ll, fp_area = footprint_lonlat(points_lonlat, radius_m, epsg)
    to_work = transformer(WGS84, f"EPSG:{epsg}").transform
    fp_work = shp_transform(to_work, fp_ll)
    ranked = []
    for d in load_index():
        if not d["geom"].intersects(fp_ll):
            continue
        inter = shp_transform(to_work, d["geom"].intersection(fp_ll))
        cov = inter.area / fp_work.area if fp_work.area else 0.0
        if cov <= 0.001:
            continue
        ranked.append(
            {
                "name": d["name"],
                "url": d["url"],
                "coverage": round(float(cov), 4),
                "points_per_m2": round(float(d["points_per_m2"]), 2),
                "year": d["year"],
            }
        )
    ranked.sort(key=lambda r: (r["coverage"] >= min_coverage, r["points_per_m2"], r["year"] or 0), reverse=True)
    # keep the best plus whatever is needed to fill coverage gaps
    keep, covered = [], None
    for r in ranked:
        g = dataset_by_name(r["name"])["geom"].intersection(fp_ll)
        new = g if covered is None else g.difference(covered)
        if covered is None or shp_transform(to_work, new).area > 0.01 * fp_work.area:
            keep.append(r)
            covered = g if covered is None else covered.union(g)
        if len(keep) >= max_keep:
            break
    return keep


def dataset_for_point(names: list[str], lon: float, lat: float) -> str | None:
    """First dataset (in registry order) whose boundary contains the point."""
    pt = Point(lon, lat)
    for n in names:
        if dataset_by_name(n)["prepared"].contains(pt):
            return n
    return None
