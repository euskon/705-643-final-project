"""Per-point pseudolabels for pretraining, built with the existing labeling heuristics.

Classes (``CLASSES``; ``IGNORE = -1`` is masked out of the loss):

====  ==============  ==========================================================
 id   name            rule (evaluated per LiDAR return)
====  ==============  ==========================================================
 0    ground          last return, slope < ``ground_slope_deg`` (30 deg)
 1    vegetation      not the last return of its pulse, or LAS class 3/4/5
 2    steep_terrain   last return, slope >= 30 deg but not a "steep" cliff point
 3    cliff           the PoC's "steep" rule: slope >= 60 deg and single return
 4    climbable       cliff point inside a Mountain Project route cylinder
                      (``labeling.build_positives``: pin snapped to the cliff,
                      2.5 m radius, from the cliff base up the route length)
 -1   ignore          cliff point within ``ignore_buffer_m`` (40 m) of a crag
                      pin / route cylinder but not in a cylinder - possibly
                      climbable rock we can't place a route on; also LAS
                      buildings (6)
====  ==============  ==========================================================

Classes 0-3 teach generic terrain geometry; class 4 vs 3 is the noisy
"rock near documented routes vs other rock" signal. MP pins are only used here
(pretraining), never as route locations for fine-tuning.

Processing an area (all incremental, keyed by a hash of the config):

A. per tile, with a ``halo_m`` border from the neighbouring tiles: normals ->
   slope -> classes 0-3 (``stage/<tile>.npz``);
B. per area: route cylinders from MP routes on the points near crag pins
   (``positives.csv``; stepback negatives as a by-product, ``negatives.csv``);
C. per tile: classes 4 / ignore, then cut the tile into ``block_size_m``
   blocks, voxel-downsample each to ``voxel_m`` and write memory-mappable
   arrays (``blocks/<tile>.{points,labels,feat}.npy`` + ``.meta.npz``), plus
   the area ``index.csv`` the training dataset reads.
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
from scipy.spatial import cKDTree

from crag import mp
from crag.paths import LIDAR_DIR, PRETRAIN_DIR, add_import_paths
from crag.registry import area_crs
from crag.tiles import Tile, load_tile

add_import_paths()
from config import LabelConfig, PatchConfig  # noqa: E402
from labeling import PointCloud, build_negatives, build_positives, compute_normals, slope_deg  # noqa: E402

log = logging.getLogger(__name__)

CLASSES = ("ground", "vegetation", "steep_terrain", "cliff", "climbable")
IGNORE = -1
VEG_LAS_CLASSES = (3, 4, 5)
BUILDING_LAS_CLASS = 6


@dataclass
class PseudoLabelConfig:
    label: LabelConfig = field(default_factory=LabelConfig)
    ground_slope_deg: float = 30.0
    halo_m: float = 5.0  # neighbour points used for normals at tile edges
    crag_context_m: float = 60.0  # points this close to a crag pin feed route placement
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


def base_classes(normals: np.ndarray, rn: np.ndarray, nr: np.ndarray, las: np.ndarray, cfg: PseudoLabelConfig) -> tuple[np.ndarray, np.ndarray]:
    """(classes 0-3 / ignore, steep mask) from normals and return info."""
    slope = slope_deg(normals)
    steep = slope >= cfg.label.steep_slope_deg
    if cfg.label.steep_single_return_only:
        steep &= nr == 1
    last = rn >= nr
    cls = np.full(len(slope), 2, np.int8)
    cls[last & (slope < cfg.ground_slope_deg)] = 0
    cls[steep] = 3
    cls[(~last) | np.isin(las, VEG_LAS_CLASSES)] = 1
    cls[las == BUILDING_LAS_CLASS] = IGNORE
    return cls, steep & (cls == 3)


def stage_tile(slug: str, tile_id: str, cache: _TileCache, cfg: PseudoLabelConfig, out: Path) -> dict:
    t = cache.get(tile_id)
    n = 0 if t is None else len(t["xyz"])
    if n == 0:
        np.savez(out, n=0, digest=cfg.digest())
        return {"tile": tile_id, "n": 0}
    tile = Tile.from_id(tile_id, cfg.tile_size_m)
    parts = [t["xyz_abs"]]
    h = cfg.halo_m
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == dy == 0:
                continue
            nb = cache.get(Tile(tile.x0 + dx * tile.size, tile.y0 + dy * tile.size, tile.size).id)
            if nb is None or len(nb["xyz"]) == 0:
                continue
            p = nb["xyz_abs"]
            keep = (
                (p[:, 0] >= tile.x0 - h) & (p[:, 0] < tile.x0 + tile.size + h)
                & (p[:, 1] >= tile.y0 - h) & (p[:, 1] < tile.y0 + tile.size + h)
            )
            parts.append(p[keep])
    xyz = np.concatenate(parts)
    normals = compute_normals(xyz, cfg.label.normal_k, cfg.label.normal_voxel_m)[:n]
    cls, steep = base_classes(normals, t["return_number"], t["number_of_returns"], t["classification"], cfg)
    np.savez(
        out,
        n=n,
        digest=cfg.digest(),
        normals=normals.astype(np.float16),
        cls=cls,
        steep=steep,
    )
    return {"tile": tile_id, "n": n}


# ---------------------------------------------------------------------------
# B. per-area route cylinders (the proof of concept's positives)
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


def build_area_positives(area: dict, tile_ids: list[str], cache: _TileCache, cfg: PseudoLabelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    crs = area_crs(area)
    _, pins = area_pins(area, crs)
    if len(pins) == 0 or not area.get("mp_id"):
        return pd.DataFrame(), pd.DataFrame()
    pin_tree = cKDTree(pins)
    xyz, nrm, nr = [], [], []
    stage_dir = area_dir(area["slug"]) / "stage"
    for tid in tile_ids:
        t = cache.get(tid)
        if t is None or len(t["xyz"]) == 0:
            continue
        near = pin_tree.query(t["xyz_abs"][:, :2], distance_upper_bound=cfg.crag_context_m)[0] < np.inf
        if not near.any():
            continue
        with np.load(stage_dir / f"{tid}.npz") as s:
            normals = s["normals"]
        xyz.append(t["xyz_abs"][near])
        nrm.append(normals[near].astype(np.float32))
        nr.append(t["number_of_returns"][near])
    if not xyz:
        return pd.DataFrame(), pd.DataFrame()
    xyz, nrm, nr = np.concatenate(xyz), np.concatenate(nrm), np.concatenate(nr)
    pc = PointCloud.build(xyz, cfg.label, normals=nrm, number_of_returns=nr)
    try:
        routes = mp.load_routes(area["mp_id"], area_pin=(area["lat"], area["lon"]))
    except Exception as exc:
        log.warning("%s: no MP routes available (%s) - no climbable class", area["slug"], exc)
        return pd.DataFrame(), pd.DataFrame()
    # only walls whose pin sits on downloaded LiDAR can be snapped
    rxy = crs.to_work(routes["lon"].to_numpy(), routes["lat"].to_numpy())
    on_lidar = cKDTree(xyz[::50, :2]).query(rxy, distance_upper_bound=cfg.label.snap_search_radius_m)[0] < np.inf
    routes = routes[on_lidar].reset_index(drop=True)
    if routes.empty:
        return pd.DataFrame(), pd.DataFrame()
    positives = build_positives(pc, routes, cfg.label, to_work=crs.to_work, to_lonlat=crs.to_lonlat)
    ok = positives[positives.get("status", pd.Series(dtype=str)) == "ok"].reset_index(drop=True)
    negatives = pd.DataFrame()
    if len(ok):
        lab = LabelConfig(**{**asdict(cfg.label), "n_random_negatives": 0})
        negatives = build_negatives(pc, ok, pins, lab, PatchConfig())
    return positives, negatives


# ---------------------------------------------------------------------------
# C. final labels + blocks
# ---------------------------------------------------------------------------


def final_labels(xyz_abs: np.ndarray, cls: np.ndarray, positives_ok: pd.DataFrame, anchors_xy: np.ndarray, cfg: PseudoLabelConfig) -> np.ndarray:
    lab = cls.copy()
    cliff = np.flatnonzero(cls == 3)
    if len(cliff) == 0:
        return lab
    climbable = np.zeros(len(cliff), bool)
    if len(positives_ok):
        cyl = positives_ok[["x", "y"]].to_numpy()
        tree = cKDTree(cyl)
        hits = tree.query_ball_point(xyz_abs[cliff, :2], cfg.label.route_radius_m)
        base, top = positives_ok["base_z"].to_numpy(), positives_ok["top_z"].to_numpy()
        z = xyz_abs[cliff, 2]
        for k, h in enumerate(hits):
            if h:
                h = np.asarray(h)
                climbable[k] = np.any((z[k] >= base[h]) & (z[k] <= top[h]))
    lab[cliff[climbable]] = 4
    if len(anchors_xy):
        d = cKDTree(anchors_xy).query(xyz_abs[cliff, :2], distance_upper_bound=cfg.label.ignore_buffer_m)[0]
        lab[cliff[(~climbable) & (d < np.inf)]] = IGNORE
    return lab


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
    """Run A -> B -> C for one area; skip whatever is already up to date."""
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
    positives, negatives = build_area_positives(area, tile_ids, cache, cfg)
    positives.to_csv(out / "positives.csv", index=False)
    negatives.to_csv(out / "negatives.csv", index=False)
    ok = positives[positives["status"] == "ok"] if "status" in positives else pd.DataFrame()
    crs = area_crs(area)
    all_pins, _ = area_pins(area, crs)
    anchors = np.vstack([all_pins, ok[["x", "y"]].to_numpy()]) if len(ok) else all_pins
    log.info("%s: %d route cylinders placed (%d ok)", slug, len(positives), len(ok))

    # C
    block_dir = out / "blocks"
    index = []
    for k, tid in enumerate(tile_ids):
        t = cache.get(tid)
        if t is None or len(t["xyz"]) == 0:
            continue
        with np.load(stage_dir / f"{tid}.npz") as s:
            cls = s["cls"]
        labels = final_labels(t["xyz_abs"], cls, ok, anchors, cfg)
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
        "route_cylinders_ok": int(len(ok)),
        "stepback_negatives": int(len(negatives)),
        "blocks_with_climbable": int((index["n_climbable"] > 0).sum()) if len(index) else 0,
        **{f"pts_{k}": v for k, v in counts.items()},
    }
    state_path.write_text(json.dumps({"digest": digest, "tiles": tile_ids, "config": asdict(cfg), "summary": summary}, indent=1, default=str))
    return summary
