"""Per-point input channels: the one place their values are computed.

Every channel describes **the point and its surface**, never its position
inside the block (``docs/fable-investigation-input-channels.md``):

==========  ====================================================  ===============
name        meaning                                               stored range
==========  ====================================================  ===============
hag         height above the lowest bare-earth ground within      m / 20, clipped
            10 m (``crag.walls.point_height_above_ground``)       to -0.1..3
vert        vertical part of the surface normal, positive         0..1
            (1 = flat ground, 0 = vertical wall)
planarity   flat face vs line vs scatter (stage files)            0..1
ret_pos     return number / number of returns (1 = last return)   0..1
single      1 if the pulse came back once                         0 or 1
intensity   laser intensity / the tile's 99th percentile          0..1
wall_dist   metres to the bare-earth wall / 10 (fine-tuning       0..2
            blocks only; never used in pretraining)
==========  ====================================================  ===============

Blocks store these as float16 columns; ``channels.json`` next to the blocks
records the column order and ``CHANNEL_SCALE_VERSION``, so a later change to
a formula can't silently mix with old blocks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PRETRAIN_CHANNELS = ("hag", "vert", "planarity", "ret_pos", "single", "intensity")
CHANNELS = PRETRAIN_CHANNELS + ("wall_dist",)
STORED_DTYPE = np.float16
CHANNEL_SCALE_VERSION = 1
HAG_SCALE_M = 20.0
WALL_DIST_SCALE_M = 10.0


def point_channels(c: dict, idx: np.ndarray | None = None, channels: tuple[str, ...] = CHANNELS, stats: dict | None = None) -> np.ndarray:
    """(N, len(channels)) float16 for a crop / tile dict ``c`` (optionally rows ``idx``).

    ``c`` holds ``hag`` (m), ``normals``, ``planarity``, ``return_number``,
    ``number_of_returns``, ``intensity`` with ``i99`` (scalar or per point) and, for
    ``wall_dist``, ``wall_dist`` (m). Missing values: ``hag`` -> 0, no normal -> ``vert`` 1
    (flat), ``planarity`` -> 0, ``wall_dist`` -> 2; counted in ``stats`` when given."""
    n_all = len(c["return_number"])
    idx = np.arange(n_all) if idx is None else np.asarray(idx)
    out = np.zeros((len(idx), len(channels)), np.float32)

    def count(key, bad):
        if stats is not None:
            stats[key] = stats.get(key, 0) + int(bad.sum())

    if stats is not None:
        stats["n"] = stats.get("n", 0) + len(idx)
    for k, name in enumerate(channels):
        if name == "hag":
            h = np.asarray(c["hag"], np.float32)[idx] if "hag" in c else np.full(len(idx), np.nan, np.float32)
            bad = ~np.isfinite(h)
            count("hag_missing", bad)
            out[:, k] = np.clip(np.where(bad, 0.0, h) / HAG_SCALE_M, -0.1, 3.0)
        elif name == "vert":
            nrm = np.asarray(c["normals"], np.float32)[idx]
            bad = ~(np.abs(nrm).sum(1) > 0)
            count("normal_missing", bad)
            out[:, k] = np.where(bad, 1.0, np.clip(np.abs(nrm[:, 2]), 0.0, 1.0))
        elif name == "planarity":
            p = np.asarray(c["planarity"], np.float32)[idx] if "planarity" in c else np.full(len(idx), np.nan, np.float32)
            bad = ~np.isfinite(p)
            count("planarity_missing", bad)
            out[:, k] = np.clip(np.where(bad, 0.0, p), 0.0, 1.0)
        elif name == "ret_pos":
            rn, nr = c["return_number"][idx].astype(np.float32), c["number_of_returns"][idx].astype(np.float32)
            out[:, k] = np.clip(rn / np.maximum(nr, 1.0), 0.0, 1.0)
        elif name == "single":
            out[:, k] = c["number_of_returns"][idx] == 1
        elif name == "intensity":
            i99 = c["i99"][idx] if np.ndim(c["i99"]) else float(c["i99"])
            out[:, k] = np.clip(c["intensity"][idx].astype(np.float32) / np.maximum(i99, 1.0), 0.0, 1.0)
        elif name == "wall_dist":
            w = np.asarray(c["wall_dist"], np.float32)[idx] if "wall_dist" in c else np.full(len(idx), np.nan, np.float32)
            bad = ~np.isfinite(w)
            count("wall_dist_missing", bad)
            out[:, k] = np.clip(np.where(bad, 2.0 * WALL_DIST_SCALE_M, w) / WALL_DIST_SCALE_M, 0.0, 2.0)
        else:
            raise KeyError(f"unknown channel {name!r}")
    return out.astype(STORED_DTYPE)


def write_channels_json(root: Path, channels: tuple[str, ...]) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "channels.json").write_text(json.dumps({"channels": list(channels), "version": CHANNEL_SCALE_VERSION}))


def read_channels_json(root: Path) -> tuple[str, ...] | None:
    """Channel names of the blocks under ``root``; None for old blocks (1-D intensity only)."""
    p = Path(root) / "channels.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if int(d.get("version", 0)) != CHANNEL_SCALE_VERSION:
        raise ValueError(f"{p}: channel version {d.get('version')} != {CHANNEL_SCALE_VERSION} - rebuild the blocks")
    return tuple(d["channels"])
