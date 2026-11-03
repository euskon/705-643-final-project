"""Incremental USGS LiDAR download, one square UTM tile per file.

Only land near climbing is fetched: an area's *tile plan* is every
``tile_size_m`` square (aligned to a global UTM grid, so tile ids are stable)
within ``crag_radius_m`` of a crag pin. Each tile is one PDAL ``readers.ept``
request, cropped exactly to the tile after reprojection and saved as
``data/lidar/<slug>/<tile_id>.npz``. A tile file that exists is never fetched
again (empty tiles are saved too, so they aren't retried), which makes the
download resumable and safe to interrupt.

Tile file contents::

    xyz            float32 (N, 3)  metres, x/y relative to the tile's SW corner, z absolute
    origin         float64 (2,)    tile SW corner in the area's UTM zone
    classification, return_number, number_of_returns  uint8 (N,)
    intensity      uint16 (N,)
    dataset        str             USGS dataset name the points came from
    epsg           int
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crag import ept_index
from crag.paths import LIDAR_DIR, add_import_paths
from crag.registry import area_crs

add_import_paths()
from usgs_ept import read_ept  # noqa: E402  (legacy module, reused)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tile:
    x0: float
    y0: float
    size: float

    @property
    def id(self) -> str:
        return f"E{int(self.x0)}_N{int(self.y0)}"

    @property
    def center(self) -> tuple[float, float]:
        return self.x0 + self.size / 2, self.y0 + self.size / 2

    @classmethod
    def from_id(cls, tile_id: str, size: float) -> "Tile":
        e, n = tile_id.split("_")
        return cls(float(e[1:]), float(n[1:]), size)


def tile_path(slug: str, tile: Tile | str) -> Path:
    tid = tile if isinstance(tile, str) else tile.id
    return LIDAR_DIR / slug / f"{tid}.npz"


def crag_points(area: dict, include_default_pins: bool = False) -> np.ndarray:
    """(n, 3) array of crag ``lon, lat, n_routes`` used to plan tiles."""
    crags = [c for c in area.get("crags", []) if include_default_pins or not c.get("default_pin")]
    if not crags:  # no crag list yet: fall back to the area's own pin
        return np.array([[area["lon"], area["lat"], 1.0]])
    return np.array([[c["lon"], c["lat"], c.get("n_routes", 1)] for c in crags], dtype=float)


def plan_tiles(area: dict, tile_size_m: float = 200.0, crag_radius_m: float = 150.0, max_tiles: int | None = None) -> list[Tile]:
    """Tiles within ``crag_radius_m`` of any crag, most route-dense first.

    Priority = number of routes on crags within reach of the tile, so
    ``max_tiles`` keeps the best-documented parts of very large areas.
    """
    crs = area_crs(area)
    pts = crag_points(area)
    xy = crs.to_work(pts[:, 0], pts[:, 1])
    score: dict[tuple[int, int], float] = {}
    s = tile_size_m
    for (x, y), w in zip(xy, pts[:, 2]):
        for i in range(math.floor((x - crag_radius_m) / s), math.floor((x + crag_radius_m) / s) + 1):
            for j in range(math.floor((y - crag_radius_m) / s), math.floor((y + crag_radius_m) / s) + 1):
                # distance from the crag to the tile rectangle
                dx = max(i * s - x, 0.0, x - (i + 1) * s)
                dy = max(j * s - y, 0.0, y - (j + 1) * s)
                if dx * dx + dy * dy <= crag_radius_m**2:
                    score[(i, j)] = score.get((i, j), 0.0) + w
    order = sorted(score, key=lambda k: (-score[k], k))
    if max_tiles:
        order = order[:max_tiles]
    return [Tile(i * s, j * s, s) for i, j in order]


def plan_tiles_for_points(area: dict, lonlat: np.ndarray, radius_m: float = 100.0, tile_size_m: float = 200.0) -> list[Tile]:
    """Tiles within ``radius_m`` of arbitrary points (e.g. verified route starts)."""
    pts = np.column_stack([np.asarray(lonlat, float), np.ones(len(lonlat))])
    return plan_tiles({**area, "crags": [{"lon": p[0], "lat": p[1], "n_routes": 1} for p in pts]}, tile_size_m, radius_m)


def download_tile(area: dict, tile: Tile, retries: int = 4) -> dict:
    """Fetch one tile (if not cached). Returns a small status dict."""
    path = tile_path(area["slug"], tile)
    if path.exists():
        return {"tile": tile.id, "status": "cached"}
    crs = area_crs(area)
    lon, lat = crs.to_lonlat(*tile.center)
    names = [d["name"] for d in area.get("usgs", {}).get("datasets", [])]
    ds_name = ept_index.dataset_for_point(names, float(lon), float(lat)) if names else None
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"origin": np.array([tile.x0, tile.y0]), "epsg": crs.epsg, "tile_size": tile.size}
    if ds_name is None:
        np.savez(path, xyz=np.zeros((0, 3), np.float32), dataset="", **_empty_fields(), **meta)
        return {"tile": tile.id, "status": "no_coverage"}
    url = ept_index.dataset_by_name(ds_name)["url"]
    x1, y1 = tile.x0 + tile.size, tile.y0 + tile.size
    t0 = time.time()
    arr = read_ept(url, crs.work_bounds_to_ept(tile.x0, tile.y0, x1, y1), retries=retries, out_srs=crs.crs)
    x, y = arr["X"], arr["Y"]
    keep = (x >= tile.x0) & (x < x1) & (y >= tile.y0) & (y < y1)
    arr = arr[keep]
    xyz = np.stack([arr["X"] - tile.x0, arr["Y"] - tile.y0, arr["Z"]], axis=1)
    xyz, first = np.unique(xyz, axis=0, return_index=True)  # overlapping EPT nodes
    arr = arr[first]
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        xyz=xyz.astype(np.float32),
        classification=arr["Classification"].astype(np.uint8),
        return_number=arr["ReturnNumber"].astype(np.uint8),
        number_of_returns=arr["NumberOfReturns"].astype(np.uint8),
        intensity=arr["Intensity"].astype(np.uint16),
        dataset=ds_name,
        **meta,
    )
    tmp.rename(path)  # atomic: an interrupted download never leaves a half file
    return {"tile": tile.id, "status": "downloaded", "n_points": int(len(xyz)), "seconds": round(time.time() - t0, 1)}


def _empty_fields() -> dict:
    z8 = np.zeros(0, np.uint8)
    return {"classification": z8, "return_number": z8, "number_of_returns": z8, "intensity": np.zeros(0, np.uint16)}


def download_area(area: dict, tiles: list[Tile], workers: int = 4, max_pool_restarts: int = 5) -> list[dict]:
    """Download every missing tile of an area with a pool of worker *processes*.

    Processes, not threads: an S3 read error inside PDAL's own EPT thread pool
    can call C++ ``std::terminate`` and kill the whole interpreter. A dead
    worker only breaks the pool, which is rebuilt; finished tiles are already
    on disk, so only the unfinished ones are retried.
    """
    slug = area["slug"]
    results = {t.id: {"tile": t.id, "status": "cached"} for t in tiles if tile_path(slug, t).exists()}
    todo = [t for t in tiles if t.id not in results]
    if not todo:
        return list(results.values())
    log.info("%s: %d/%d tiles to download", slug, len(todo), len(tiles))
    ctx = mp.get_context("spawn")  # clean PDAL/GDAL state per worker
    for restart in range(max_pool_restarts + 1):
        todo = [t for t in todo if t.id not in results]
        if not todo:
            break
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                futs = {pool.submit(download_tile, area, t): t for t in todo}
                for fut in as_completed(futs):
                    t = futs[fut]
                    try:
                        r = fut.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as exc:  # keep going; a re-run retries the failed tile
                        log.error("%s %s failed: %s", slug, t.id, exc)
                        r = {"tile": t.id, "status": "failed", "error": str(exc)[:200]}
                    results[t.id] = r
                    done = sum(1 for x in results.values() if x["status"] != "cached")
                    if done % 10 == 0:
                        log.info("%s: %d/%d new tiles done (last: %s)", slug, done, len(tiles), r)
        except BrokenProcessPool:
            log.warning("%s: a download worker crashed (PDAL/S3 error); restarting pool (%d/%d)", slug, restart + 1, max_pool_restarts)
            # tiles that finished before the crash are on disk
            for t in todo:
                if t.id not in results and tile_path(slug, t).exists():
                    results[t.id] = {"tile": t.id, "status": "downloaded"}
    for t in tiles:
        results.setdefault(t.id, {"tile": t.id, "status": "failed", "error": "worker crashed repeatedly"})
    return list(results.values())


def load_tile(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    d["dataset"] = str(d["dataset"])
    return d
