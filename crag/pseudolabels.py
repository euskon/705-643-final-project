"""Per-point pseudolabels for pretraining: generic terrain geometry, no route information.

Classes (``CLASSES``; ``IGNORE = -1`` is masked out of the loss):

====  ==============  ==========================================================
 id   name            rule (evaluated per LiDAR return)
====  ==============  ==========================================================
 0    ground          last return, slope < ``ground_slope_deg`` (30 deg)
 1    vegetation      not the last return of its pulse, or LAS class 3/4/5
 2    steep_terrain   any other last return (30-60 deg)
 3    cliff           last return on a steep (>= 60 deg) surface: a single
                      return (the pulse hit nothing first, bare rock) or the
                      last return of a pulse that clipped foliage first
                      (``multi_return_rock``)
 -1   ignore          LAS buildings (6)
====  ==============  ==========================================================

Slope comes from a PCA of each point's neighbourhood fitted on
the *surface* returns only (last returns that are not LAS vegetation,
:func:`crag.labeling.surface_geometry`), so tree canopy never tilts the
normal of the rock behind it.

Whether rock is *climbable* is left entirely to fine-tuning (verified route
locations, ``crag.finetune_data``); Mountain Project pins aren't used here.

Processing an area (all incremental, keyed by a hash of the config):

A. per tile, with a ``halo_m`` border from the neighbouring tiles: normals +
   planarity -> classes (``stage/<tile>.npz``; also read by ``crag.crop`` for
   the fine-tuning set, so "cliff" means the same thing everywhere);
B. per tile: cut into ``block_size_m`` blocks, voxel-downsample each to
   ``voxel_m`` and write memory-mappable arrays
   (``blocks/<tile>.{points,labels,feat}.npy`` + ``.meta.npz``), plus the area
   ``index.csv`` the training dataset reads.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from crag.labeling import LabelConfig, compute_normals, slope_deg, surface_geometry
from crag.paths import LIDAR_DIR, PRETRAIN_DIR
from crag.registry import area_crs
from crag.tiles import Tile, load_tile

log = logging.getLogger(__name__)

CLASSES = ("ground", "vegetation", "steep_terrain", "cliff")
IGNORE = -1
VEG_LAS_CLASSES = (3, 4, 5)
BUILDING_LAS_CLASS = 6


@dataclass
class PseudoLabelConfig:
    label: LabelConfig = field(default_factory=LabelConfig)
    ground_slope_deg: float = 30.0
    # fit normals on surface points only (last returns, not LAS vegetation),
    # so canopy never enters a rock point's PCA neighbourhood
    surface_normals: bool = True
    # optional gate: a cliff point must also sit on a flat patch, planarity
    # (l2-l3)/l1 of its PCA neighbourhood >= this. Off (None) by default:
    # airborne LiDAR samples vertical faces sparsely, in scan lines, so real
    # walls often look linear and the gate erased most of them
    # (scripts/s1_check_rock_rule.py: cliff points next to known rock fell
    # 3-7x at 0.5 and what was left was mostly speckle)
    min_planarity: float | None = None
    # steep *last return of a multi-return pulse* - the pulse clipped foliage
    # and then hit the surface: "cliff", "ignore" (left out of the loss) or
    # "steep_terrain". "cliff" by default: these points sit in walls as often
    # as single-return cliff points do and fill in tree-lined faces
    # (Red River Gorge: ~3x the cliff points next to known routes)
    multi_return_rock: str = "cliff"
    halo_m: float = 5.0  # neighbour points used for normals at tile edges
    block_size_m: float = 20.0
    voxel_m: float = 0.25
    min_block_points: int = 512
    max_block_points: int = 40000
    tile_size_m: float = 200.0

    def digest(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:10]


def area_dir(slug: str) -> Path:
    return PRETRAIN_DIR / slug


# ---------------------------------------------------------------------------
# A. per-tile base classes
# ---------------------------------------------------------------------------


class _TileCache:
    """Small LRU of loaded raw tiles (absolute coordinates) for halo lookups."""

    def __init__(self, slug: str, size: float, maxlen: int = 16):
        self.slug, self.size, self.maxlen = slug, size, maxlen
        self._d: OrderedDict[str, dict | None] = OrderedDict()

    def get(self, tile_id: str) -> dict | None:
        if tile_id in self._d:
            self._d.move_to_end(tile_id)
            return self._d[tile_id]
        path = LIDAR_DIR / self.slug / f"{tile_id}.npz"
        t = None
        if path.exists():
            t = load_tile(path)
            t["xyz_abs"] = t["xyz"].astype(np.float64)
            t["xyz_abs"][:, :2] += t["origin"]
        self._d[tile_id] = t
        if len(self._d) > self.maxlen:
            self._d.popitem(last=False)
        return t


def surface_mask(rn: np.ndarray, nr: np.ndarray, las: np.ndarray) -> np.ndarray:
    """Returns that can lie on the ground / rock surface: last of their pulse, not LAS vegetation."""
    return (rn >= nr) & ~np.isin(las, VEG_LAS_CLASSES)


def base_classes(
    normals: np.ndarray, rn: np.ndarray, nr: np.ndarray, las: np.ndarray, cfg: PseudoLabelConfig, planarity: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(class per point, cliff mask) from normals, planarity and return info.

    * vegetation: not the last return of its pulse, or LAS class 3/4/5;
    * ground: last return, slope < ``ground_slope_deg``;
    * cliff: single return (the pulse hit nothing before the surface) on a
      steep (>= ``label.steep_slope_deg``) patch (and planar, if ``min_planarity``);
    * steep last return of a multi-return pulse: ``multi_return_rock``;
    * steep_terrain: every other last return (30-60 deg, or failing the planarity gate);
    * ignore: LAS buildings.
    """
    slope = slope_deg(normals)
    last = rn >= nr
    planar = np.ones(len(slope), bool) if (planarity is None or cfg.min_planarity is None) else planarity >= cfg.min_planarity
    rock = last & (slope >= cfg.label.steep_slope_deg) & planar
    single = nr == 1
    cls = np.full(len(slope), 2, np.int8)
    cls[last & (slope < cfg.ground_slope_deg)] = 0
    if not cfg.label.steep_single_return_only:
        single = np.ones(len(slope), bool)
    cls[rock & single] = 3
    multi_rock = rock & ~single
    if cfg.multi_return_rock == "cliff":
        cls[multi_rock] = 3
    elif cfg.multi_return_rock == "ignore":
        cls[multi_rock] = IGNORE
    elif cfg.multi_return_rock != "steep_terrain":
        raise ValueError(f"multi_return_rock={cfg.multi_return_rock!r}")
    cls[(~last) | np.isin(las, VEG_LAS_CLASSES)] = 1
    cls[las == BUILDING_LAS_CLASS] = IGNORE
    return cls, cls == 3


def tile_geometry(xyz: np.ndarray, rn: np.ndarray, nr: np.ndarray, las: np.ndarray, cfg: PseudoLabelConfig) -> tuple[np.ndarray, np.ndarray | None]:
    """(normals, planarity) of a tile + halo, per ``cfg``."""
    lab = cfg.label
    if cfg.surface_normals:
        return surface_geometry(xyz, surface_mask(rn, nr, las), lab.normal_k, lab.normal_voxel_m)
    return compute_normals(xyz, lab.normal_k, lab.normal_voxel_m), None


def tile_with_halo(tile_id: str, cache: _TileCache, halo_m: float) -> dict | None:
    """The tile's points first, then the neighbouring tiles' points within ``halo_m`` of it."""
    t = cache.get(tile_id)
    if t is None or len(t["xyz"]) == 0:
        return None
    tile = Tile.from_id(tile_id, cache.size)
    keys = ("xyz_abs", "return_number", "number_of_returns", "classification")
    parts = {k: [t[k]] for k in keys}
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == dy == 0:
                continue
            nb = cache.get(Tile(tile.x0 + dx * tile.size, tile.y0 + dy * tile.size, tile.size).id)
            if nb is None or len(nb["xyz"]) == 0:
                continue
            p = nb["xyz_abs"]
            keep = (
                (p[:, 0] >= tile.x0 - halo_m) & (p[:, 0] < tile.x0 + tile.size + halo_m)
                & (p[:, 1] >= tile.y0 - halo_m) & (p[:, 1] < tile.y0 + tile.size + halo_m)
            )
            for k in keys:
                parts[k].append(nb[k][keep])
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["n"] = len(t["xyz"])
    return out


def stage_tile(slug: str, tile_id: str, cache: _TileCache, cfg: PseudoLabelConfig, out: Path) -> dict:
    h = tile_with_halo(tile_id, cache, cfg.halo_m)
    if h is None:
        np.savez(out, n=0, digest=cfg.digest())
        return {"tile": tile_id, "n": 0}
    n = h["n"]
    rn, nr, las = h["return_number"], h["number_of_returns"], h["classification"]
    normals, planarity = tile_geometry(h["xyz_abs"], rn, nr, las, cfg)
    normals = normals[:n]
    planarity = None if planarity is None else planarity[:n]
    cls, steep = base_classes(normals, rn[:n], nr[:n], las[:n], cfg, planarity)
    extra = {} if planarity is None else {"planarity": planarity.astype(np.float16)}
    np.savez(out, n=n, digest=cfg.digest(), normals=normals.astype(np.float16), cls=cls, steep=steep, **extra)
    return {"tile": tile_id, "n": n}


# ---------------------------------------------------------------------------
# crag pins (used by the fine-tuning set builder)
# ---------------------------------------------------------------------------


def area_pins(area: dict, crs) -> tuple[np.ndarray, np.ndarray]:
    """UTM xy of (all crag pins, non-default crag pins)."""
    crags = area.get("crags", [])
    if not crags:
        return np.empty((0, 2)), np.empty((0, 2))
    ll = np.array([[c["lon"], c["lat"]] for c in crags])
    xy = crs.to_work(ll[:, 0], ll[:, 1])
    ok = np.array([not c.get("default_pin") for c in crags])
    return xy, xy[ok]


# ---------------------------------------------------------------------------
# B. blocks
# ---------------------------------------------------------------------------


def make_blocks(xyz_abs: np.ndarray, labels: np.ndarray, intensity: np.ndarray, tile: Tile, cfg: PseudoLabelConfig, seed: int) -> dict:
    """Cut a tile into blocks; voxel-downsample; return concatenated arrays + per-block meta."""
    rng = np.random.default_rng(seed)
    bs = cfg.block_size_m
    nb = int(round(tile.size / bs))
    loc = xyz_abs[:, :2] - [tile.x0, tile.y0]
    bi = np.clip((loc // bs).astype(np.int64), 0, nb - 1)
    block_id = bi[:, 0] * nb + bi[:, 1]
    p99 = np.percentile(intensity, 99) if len(intensity) else 1.0
    feat = np.clip(intensity / max(p99, 1.0), 0, 1).astype(np.float16)
    order = np.argsort(block_id, kind="stable")
    starts = np.searchsorted(block_id[order], np.arange(nb * nb + 1))
    pts, labs, feats, meta = [], [], [], []
    offset = 0
    for b in range(nb * nb):
        idx = order[starts[b] : starts[b + 1]]
        if len(idx) < cfg.min_block_points:
            continue
        idx = idx[rng.permutation(len(idx))]  # random but reproducible voxel representative
        keys = np.floor(xyz_abs[idx] / cfg.voxel_m).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        idx = idx[np.sort(first)]
        if len(idx) > cfg.max_block_points:
            idx = idx[rng.choice(len(idx), cfg.max_block_points, replace=False)]
        if len(idx) < cfg.min_block_points:
            continue
        i, j = divmod(b, nb)
        cx, cy = tile.x0 + (i + 0.5) * bs, tile.y0 + (j + 0.5) * bs
        p = xyz_abs[idx]
        zmin = float(p[:, 2].min())
        local = np.stack([p[:, 0] - cx, p[:, 1] - cy, p[:, 2] - zmin], axis=1).astype(np.float32)
        lb = labels[idx]
        counts = np.bincount(lb + 1, minlength=len(CLASSES) + 1)  # [ignore, c0..c4]
        pts.append(local)
        labs.append(lb)
        feats.append(feat[idx])
        meta.append([b, offset, offset + len(idx), cx, cy, zmin, *counts])
        offset += len(idx)
    if not pts:
        return {"points": np.zeros((0, 3), np.float32), "labels": np.zeros(0, np.int8), "feat": np.zeros(0, np.float16), "meta": np.zeros((0, 6 + len(CLASSES) + 1))}
    return {
        "points": np.concatenate(pts),
        "labels": np.concatenate(labs).astype(np.int8),
        "feat": np.concatenate(feats),
        "meta": np.asarray(meta, dtype=np.float64),
    }


META_COLS = ["block", "start", "stop", "cx", "cy", "zmin", "n_ignore"] + [f"n_{c}" for c in CLASSES]


def write_blocks(out_dir: Path, tile_id: str, blocks: dict, digest: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{tile_id}.points.npy", blocks["points"])
    np.save(out_dir / f"{tile_id}.labels.npy", blocks["labels"])
    np.save(out_dir / f"{tile_id}.feat.npy", blocks["feat"])
    np.savez(out_dir / f"{tile_id}.meta.npz", meta=blocks["meta"], digest=digest)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def process_area(area: dict, cfg: PseudoLabelConfig, force: bool = False) -> dict:
    """Run A -> B for one area; skip whatever is already up to date."""
    slug = area["slug"]
    src = LIDAR_DIR / slug
    tile_ids = sorted(
        (p.name[: -len(".npz")] for p in src.glob("E*_N*.npz") if not p.name.endswith(".tmp.npz")),
        key=lambda s: (int(s.split("_")[1][1:]), int(s.split("_")[0][1:])),
    )
    out = area_dir(slug)
    state_path = out / "state.json"
    digest = cfg.digest()
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if not force and state.get("digest") == digest and state.get("tiles") == tile_ids:
        log.info("%s: up to date (%d tiles)", slug, len(tile_ids))
        return state["summary"]
    if not tile_ids:
        log.warning("%s: no LiDAR tiles downloaded", slug)
        return {"slug": slug, "tiles": 0}
    cache = _TileCache(slug, cfg.tile_size_m)
    stage_dir = out / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # A
    for k, tid in enumerate(tile_ids, 1):
        sp = stage_dir / f"{tid}.npz"
        if sp.exists() and not force:
            with np.load(sp) as s:
                if str(s["digest"]) == digest:
                    continue
        r = stage_tile(slug, tid, cache, cfg, sp)
        if k % 10 == 0 or k == len(tile_ids):
            log.info("%s: stage A %d/%d (%s: %d pts)", slug, k, len(tile_ids), tid, r["n"])

    # B
    crs = area_crs(area)
    block_dir = out / "blocks"
    index = []
    for k, tid in enumerate(tile_ids):
        t = cache.get(tid)
        if t is None or len(t["xyz"]) == 0:
            continue
        with np.load(stage_dir / f"{tid}.npz") as s:
            labels = s["cls"]
        seed = int(hashlib.sha1(f"{slug}/{tid}".encode()).hexdigest()[:8], 16)
        blocks = make_blocks(t["xyz_abs"], labels, t["intensity"].astype(np.float32), Tile.from_id(tid, cfg.tile_size_m), cfg, seed)
        write_blocks(block_dir, tid, blocks, digest)
        m = pd.DataFrame(blocks["meta"], columns=META_COLS)
        m.insert(0, "tile", tid)
        index.append(m)
    index = pd.concat(index, ignore_index=True) if index else pd.DataFrame(columns=["tile"] + META_COLS)
    for c in ["block", "start", "stop", "n_ignore"] + [f"n_{c}" for c in CLASSES]:
        index[c] = index[c].astype("int64")
    if len(index):
        ll = crs.to_lonlat(index["cx"].to_numpy(), index["cy"].to_numpy())
        index["lon"], index["lat"] = ll[:, 0], ll[:, 1]
    index.insert(0, "slug", slug)
    index.to_csv(out / "index.csv", index=False)

    counts = {c: int(index[f"n_{c}"].sum()) for c in CLASSES} if len(index) else {}
    counts["ignore"] = int(index["n_ignore"].sum()) if len(index) else 0
    summary = {
        "slug": slug,
        "split": area.get("split"),
        "tiles": len(tile_ids),
        "blocks": int(len(index)),
        "points": int(index["stop"].sub(index["start"]).sum()) if len(index) else 0,
        "blocks_with_cliff": int((index["n_cliff"] > 0).sum()) if len(index) else 0,
        **{f"pts_{k}": v for k, v in counts.items()},
    }
    state_path.write_text(json.dumps({"digest": digest, "tiles": tile_ids, "config": asdict(cfg), "summary": summary}, indent=1, default=str))
    return summary
