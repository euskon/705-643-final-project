"""Cut a block of LiDAR around any point of an area (waypoints, routes, negatives).

Uses the raw tiles (``data/lidar``) and, when present, the stage-A products
(``data/pretrain/<slug>/stage``: normals, steep mask, base class) so the same
cliff definition is used everywhere.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

from crag.paths import LIDAR_DIR, PRETRAIN_DIR
from crag.tiles import Tile, load_tile


class AreaCloud:
    def __init__(self, slug: str, tile_size_m: float = 200.0, cache_tiles: int = 24):
        self.slug, self.size, self.maxlen = slug, tile_size_m, cache_tiles
        self._cache: OrderedDict[str, dict | None] = OrderedDict()

    def _tile(self, tid: str) -> dict | None:
        if tid in self._cache:
            self._cache.move_to_end(tid)
            return self._cache[tid]
        p = LIDAR_DIR / self.slug / f"{tid}.npz"
        t = None
        if p.exists():
            t = load_tile(p)
            if len(t["xyz"]):
                xyz = t["xyz"].astype(np.float64)
                xyz[:, :2] += t["origin"]
                t["xyz_abs"] = xyz
                sp = PRETRAIN_DIR / self.slug / "stage" / f"{tid}.npz"
                if sp.exists():
                    with np.load(sp) as s:
                        if int(s["n"]) == len(xyz):
                            t["steep"], t["cls"], t["normals"] = s["steep"], s["cls"], s["normals"]
                            if "wall_dist" in s:
                                t["wall_dist"] = s["wall_dist"]
                            if "planarity" in s:
                                t["planarity"] = s["planarity"]
                hp = PRETRAIN_DIR / self.slug / "stage" / f"{tid}.hag.npy"
                if hp.exists():
                    hag = np.load(hp)
                    if len(hag) == len(xyz):
                        t["hag"] = hag
            else:
                t = None
        self._cache[tid] = t
        if len(self._cache) > self.maxlen:
            self._cache.popitem(last=False)
        return t

    def crop(self, cx: float, cy: float, half: float) -> dict:
        """Points with |x-cx|,|y-cy| <= half: ``xyz`` (abs), ``intensity``, ``steep``, ``cls``,
        ``normals``, ``multi`` (non-last return), ``wall_dist`` (m to the bare-earth wall; NaN
        without stage files), ``las`` (the provider's class), ``return_number``,
        ``number_of_returns``, ``planarity`` and ``hag`` (height above the local ground, m;
        both NaN when the stage files lack them)."""
        s = self.size
        parts = {k: [] for k in ("xyz", "intensity", "steep", "cls", "normals", "multi", "wall_dist", "las", "return_number", "number_of_returns", "planarity", "hag", "i99")}
        for i in range(math.floor((cx - half) / s), math.floor((cx + half) / s) + 1):
            for j in range(math.floor((cy - half) / s), math.floor((cy + half) / s) + 1):
                t = self._tile(Tile(i * s, j * s, s).id)
                if t is None:
                    continue
                p = t["xyz_abs"]
                m = (np.abs(p[:, 0] - cx) <= half) & (np.abs(p[:, 1] - cy) <= half)
                parts["xyz"].append(p[m])
                parts["intensity"].append(t["intensity"][m].astype(np.float32))
                parts["steep"].append(t["steep"][m] if "steep" in t else np.zeros(m.sum(), bool))
                parts["cls"].append(t["cls"][m] if "cls" in t else np.full(m.sum(), -1, np.int8))
                parts["normals"].append(t["normals"][m].astype(np.float32) if "normals" in t else np.zeros((m.sum(), 3), np.float32))
                parts["multi"].append(t["return_number"][m] < t["number_of_returns"][m])  # non-last return = canopy
                parts["wall_dist"].append(t["wall_dist"][m].astype(np.float32) if "wall_dist" in t else np.full(m.sum(), np.nan, np.float32))
                parts["las"].append(t["classification"][m])
                parts["return_number"].append(t["return_number"][m])
                parts["number_of_returns"].append(t["number_of_returns"][m])
                parts["planarity"].append(t["planarity"][m].astype(np.float32) if "planarity" in t else np.full(m.sum(), np.nan, np.float32))
                parts["hag"].append(t["hag"][m].astype(np.float32) if "hag" in t else np.full(m.sum(), np.nan, np.float32))
                if "i99" not in t:  # the tile's 99th-percentile intensity (same normalisation as the pretraining blocks)
                    t["i99"] = float(max(np.percentile(t["intensity"], 99), 1.0))
                parts["i99"].append(np.full(int(m.sum()), t["i99"], np.float32))
        if not parts["xyz"]:
            z8, zf = np.zeros(0, np.uint8), np.zeros(0, np.float32)
            return {"xyz": np.zeros((0, 3)), "intensity": zf, "steep": np.zeros(0, bool), "cls": np.zeros(0, np.int8),
                    "normals": np.zeros((0, 3), np.float32), "multi": np.zeros(0, bool), "wall_dist": zf, "las": z8, "return_number": z8,
                    "number_of_returns": z8, "planarity": zf, "hag": zf, "i99": zf}
        return {k: np.concatenate(v) for k, v in parts.items()}

    def has_stage(self) -> bool:
        return (PRETRAIN_DIR / self.slug / "stage").exists()


def voxel_block(xyz_abs: np.ndarray, cx: float, cy: float, voxel_m: float, max_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """(kept indices, local float32 coords (x-cx, y-cy, z-zmin), zmin) after voxel thinning."""
    if len(xyz_abs) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 3), np.float32), 0.0
    idx = rng.permutation(len(xyz_abs))
    _, first = np.unique(np.floor(xyz_abs[idx] / voxel_m).astype(np.int64), axis=0, return_index=True)
    idx = idx[np.sort(first)]
    if len(idx) > max_points:
        idx = idx[rng.choice(len(idx), max_points, replace=False)]
    p = xyz_abs[idx]
    zmin = float(p[:, 2].min())
    local = np.stack([p[:, 0] - cx, p[:, 1] - cy, p[:, 2] - zmin], axis=1).astype(np.float32)
    return idx, local, zmin
