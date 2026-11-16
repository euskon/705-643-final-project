"""Fine-tuning blocks: one ``block_size_m`` block per sample, centred on the foot of its wall.

**One rule for every sample, whatever its label** (fine-tuning set V4,
``research-prompts/fable-investigation-project-issues-resolution-plan1.md``):
each sample is centred on a wall foot (``crag.groundtruth.wall_foot``), and
the points that carry its label are a **face strip** measured in the wall's
own directions - *along* the wall, *out* from it, *up* from the foot:

* cliff points only, ``|along|`` up to 2.5 m each side (less beside a close
  sample of the other label, :func:`split_strip_widths`), from 15 m behind the
  foot (a face leaning back) to 5 m in front (an overhang), from 1 m below
  the foot up to the wall top (:func:`wall_top`);
* from that strip the same number of points is picked for every sample: 4
  height bands x up to 12 points (:func:`pick_points`); samples with fewer
  than 12 are dropped. Route length is not used for anything.

Stored label codes (int8), remapped at load time by ``label_mode``:

====  ======================================================  =================
code  meaning                                                  center_only map
====  ======================================================  =================
 1    the block's own strip, climbable sample (picked points)       1
 0    the block's own strip, not-climbable sample (picked)          0
 3    another climbable sample's strip inside the block          ignore
 4    another not-climbable sample's strip inside the block      ignore
 2    background (everything else)                               ignore
-1    unknown rock (cliff points near a known route), and the    ignore
      own strip's points that were not picked
====  ======================================================  =================

``label_mode``: ``center_only`` (default; each block supervises only its own
sample, so a sample's score is unambiguous), ``all_cylinders`` (3->1, 4->0)
or ``with_background`` (all_cylinders + background 2->0).

Splits are spatial: samples within ``group_m`` of each other form a group
(≈ one crag); groups are dealt to train/val/test, and whole ``test_areas``
(default: Horseshoe Canyon Ranch) go to test - or, with a Horseshoe fold
(``hcr-a`` / ``hcr-b``), one wall section of it goes to training. Test groups
are written to ``registry/heldout_regions.json`` so pretraining never sees
them.

Augmentation negatives (``aug_squash`` / ``aug_layback``) are off: they
change wall height and lean, and their stored channels would be stale.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from dataclasses import asdict, dataclass

from crag import registry
from crag.channels import CHANNEL_SCALE_VERSION, CHANNELS, point_channels, write_channels_json
from crag.crop import AreaCloud, voxel_block
from crag.paths import FINETUNE_DIR, HELDOUT_REGIONS_JSON

log = logging.getLogger(__name__)

LABEL_MODES = {
    "center_only": {1: 1, 0: 0, 3: -1, 4: -1, 2: -1, -1: -1},
    "all_cylinders": {1: 1, 0: 0, 3: 1, 4: 0, 2: -1, -1: -1},
    "with_background": {1: 1, 0: 0, 3: 1, 4: 0, 2: 0, -1: -1},
}
FT_CLASSES = ("not_climbable", "climbable")
CLIFF_NEGATIVES = ("band", "far_cliff", "handgathered_nc")


def assign_groups(samples: pd.DataFrame, group_m: float) -> np.ndarray:
    keys = np.empty(len(samples), dtype=object)
    for slug, sub in samples.groupby("slug"):
        xy = sub[["x", "y"]].to_numpy()
        pairs = cKDTree(xy).query_pairs(group_m, output_type="ndarray")
        n = len(xy)
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
        comp = connected_components(g, directed=False)[1]
        # stable group key: centroid rounded to 100 m (independent of sample order)
        cent = pd.DataFrame(xy).groupby(comp).transform("mean").to_numpy()
        keys[sub.index] = [f"{slug}:{int(cx // 100)}:{int(cy // 100)}" for cx, cy in cent]
    return keys


HCR_SLUG = "horseshoe-canyon-ranch"
HCR_FOLDS = {"none": (), "hcr-a": ("East Side",), "hcr-b": ("West Side", "North Forty")}  # fold -> wall sections that go to training


def hcr_sections(samples: pd.DataFrame, slug: str = HCR_SLUG) -> pd.Series:
    """Wall section of every Horseshoe sample: its waypoint's ``area_name``; code-made samples
    (band, far cliff) take the section of the nearest waypoint."""
    sec = pd.Series(None, index=samples.index, dtype=object)
    h = samples[samples["slug"] == slug]
    if h.empty or "area_name" not in h:
        return sec
    known = h[h["area_name"].notna() & (h["area_name"].astype(str) != "")]
    sec[known.index] = known["area_name"].astype(str)
    rest = h.index.difference(known.index)
    if len(rest) and len(known):
        _, j = cKDTree(known[["x", "y"]].to_numpy(float)).query(h.loc[rest, ["x", "y"]].to_numpy(float))
        sec[rest] = known["area_name"].astype(str).to_numpy()[j]
    return sec


def apply_hcr_fold(samples: pd.DataFrame, split: np.ndarray, fold: str, slug: str = HCR_SLUG) -> np.ndarray:
    """Move the fold's wall sections of Horseshoe from test to training (never to validation)."""
    if fold not in HCR_FOLDS:
        raise ValueError(f"unknown Horseshoe fold {fold!r}; one of {sorted(HCR_FOLDS)}")
    split = np.array(split, dtype=object)
    sec = hcr_sections(samples, slug)
    m = (samples["slug"] == slug).to_numpy()
    split[m] = np.where(sec[m].isin(HCR_FOLDS[fold]).to_numpy(), "train", "test")
    return split


def assign_splits(samples: pd.DataFrame, val_frac: float, test_frac: float, test_areas=(), val_areas=(), seed: str = "ft") -> np.ndarray:
    """Whole spatial groups -> train / val / test, balanced within each area.

    Hashing each group independently lets one giant group (dense crags chain
    into connected components of ~2k samples) land in val or test and swamp
    it. Instead, each area's groups are placed largest first, each into the
    split furthest below its target share, measured on three counts at once:
    samples, positives and hard (``band``) negatives - so val/test keep enough
    of the cases the hard-negative metrics need. Ties break on a stable hash;
    groups always stay whole (no spatial leakage).
    """
    split = np.empty(len(samples), dtype=object)
    targets = {"train": 1.0 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    for slug, area in samples.groupby("slug"):
        if slug in test_areas or slug in val_areas:
            split[area.index] = "test" if slug in test_areas else "val"
            continue
        counts = pd.DataFrame({"n": 1, "pos": (area["label"] == 1).astype(int), "band": (area["source"] == "band").astype(int)}, index=area.index)
        per_group = counts.groupby(area["group"]).sum()
        totals = per_group.sum().clip(lower=1).astype(float)
        filled = {k: pd.Series(0.0, index=per_group.columns) for k in targets}
        order = sorted(per_group.index, key=lambda g: (-per_group.loc[g, "n"], registry.stable_fraction(seed + g)))
        for g in order:
            # mean relative deficit over the three counts the group actually contains
            dims = per_group.columns[per_group.loc[g] > 0]
            s = max(targets, key=lambda k: (float(((targets[k] * totals[dims] - filled[k][dims]) / totals[dims]).mean()), k == "train"))
            filled[s] += per_group.loc[g]
            split[area.index[area["group"] == g]] = s
    return split


def write_heldout_regions(samples: pd.DataFrame, margin_deg: float = 0.001, path: Path = HELDOUT_REGIONS_JSON) -> None:
    regions = []
    for g, sub in samples[samples["split"] == "test"].groupby("group"):
        regions.append({"name": g, "bbox": [round(sub["lon"].min() - margin_deg, 6), round(sub["lat"].min() - margin_deg, 6),
                                             round(sub["lon"].max() + margin_deg, 6), round(sub["lat"].max() + margin_deg, 6)]})
    path.write_text(json.dumps({"description": "Fine-tuning test groups; pretraining drops blocks inside these boxes (crag/datasets.py).", "regions": regions}, indent=1))
    log.info("wrote %d held-out regions to %s", len(regions), path)


@dataclass
class StripConfig:
    """The labelled face strip and the point pick - identical for every label (stored in report.json)."""

    half_width_m: float = 2.5    # sideways from the foot, each side
    out_back_m: float = 15.0     # behind the foot: a face leaning back
    out_front_m: float = 5.0     # in front of the foot: an overhang
    below_foot_m: float = 1.0
    # wall top: "grid" = the ground-grid wall height at the foot (+ top_margin_m) - matched route lengths
    # best on 2026-11-15 (rank correlation 0.44 vs 0.24-0.30 for the gap rules); "gap" = the gap rule below,
    # also the fallback where the grid gives no height
    top_rule: str = "grid"
    top_margin_m: float = 1.0
    top_gap_m: float = 5.0       # gap rule: the last 1 m height bin before a gap larger than this
    top_cap_m: float = 300.0     # in practice no cap (user's choice, 2026-11-15)
    bands: int = 4               # height bands of the point pick
    per_band: int = 12           # points picked per band (48 at most)
    min_points: int = 12         # a sample with fewer labelled points is dropped, whatever its label
    min_points_split: int = 8    # ... for a strip split beside a close opposite-label sample (plan 1, 6.1)
    pair_reach_m: float = 5.0    # a sample of the other label this close along the wall splits the strips
    pair_gap_m: float = 0.5      # unlabelled gap left between two split strips
    min_side_m: float = 1.0      # a strip side narrower than this -> the sample is dropped


def wall_coords(xyz: np.ndarray, foot_xy, base_z: float, out_dir) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(along, out, up) of every point in the wall's frame at the foot."""
    out_dir = np.asarray(out_dir, float)
    rel = xyz[:, :2] - np.asarray(foot_xy, float)
    return rel @ np.array([-out_dir[1], out_dir[0]]), rel @ out_dir, xyz[:, 2] - float(base_z)


def wall_top(up: np.ndarray, cfg: StripConfig) -> float:
    """Height above the foot where this wall ends: cliff heights ``up`` go into 1 m bins; the
    top is the upper edge of the last bin before the first gap of more than ``top_gap_m``
    with no cliff points (so a second tier above a ledge is not the same wall)."""
    if not len(up):
        return 0.0
    b = np.unique(np.floor(up).astype(np.int64))
    gaps = np.flatnonzero(np.diff(b) - 1 > cfg.top_gap_m)
    last = b[gaps[0]] if len(gaps) else b[-1]
    return float(min(last + 1, cfg.top_cap_m))


def face_strip_mask(xyz: np.ndarray, steep: np.ndarray, foot_xy, base_z: float, out_dir, cfg: StripConfig, left_m: float | None = None,
                    right_m: float | None = None, wall_height_m: float | None = None) -> tuple[np.ndarray, float]:
    """(points in the sample's face strip, wall top in m above the foot). ``left_m`` /
    ``right_m`` are the strip's reach along the wall on each side (default: the full half-width);
    ``wall_height_m`` (ground grid) sets the top when ``cfg.top_rule == "grid"``."""
    left = cfg.half_width_m if left_m is None or not np.isfinite(left_m) else float(left_m)
    right = cfg.half_width_m if right_m is None or not np.isfinite(right_m) else float(right_m)
    along, out, up = wall_coords(xyz, foot_xy, base_z, out_dir)
    m = steep & (along >= -left) & (along <= right) & (out >= -cfg.out_back_m) & (out <= cfg.out_front_m) & (up >= -cfg.below_foot_m)
    if cfg.top_rule == "grid" and wall_height_m is not None and np.isfinite(wall_height_m) and wall_height_m > 0:
        top = min(float(wall_height_m) + cfg.top_margin_m, cfg.top_cap_m)
    else:
        top = wall_top(up[m], cfg)
    return m & (up <= top), top


def pick_points(up: np.ndarray, cfg: StripConfig, rng: np.random.Generator) -> np.ndarray:
    """Indices (into ``up``, the strip points' heights) of the labelled points: the strip's
    height range - foot to its highest point present - in ``bands`` equal bands, up to
    ``per_band`` random points from each. A thin band is not topped up from the others."""
    if not len(up):
        return np.zeros(0, np.int64)
    lo, hi = min(float(up.min()), 0.0), float(up.max())
    band = np.clip(((up - lo) / max(hi - lo, 1e-6) * cfg.bands).astype(np.int64), 0, cfg.bands - 1)
    keep = []
    for b in range(cfg.bands):
        idx = np.flatnonzero(band == b)
        keep.append(idx if len(idx) <= cfg.per_band else rng.choice(idx, cfg.per_band, replace=False))
    return np.sort(np.concatenate(keep))


def split_strip_widths(samples: pd.DataFrame, cfg: StripConfig | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Strip reach along the wall per sample, split between close samples of opposite label.

    Two 5 m strips overlap when their feet are under 5 m apart along the wall; the same rock
    would then carry both labels. For every sample, the nearest opposite-label foot on each
    side (within ``pair_reach_m`` along the wall and across it) at along-wall distance ``d``
    ends the strip at ``d / 2 - pair_gap_m / 2`` on that side - the mirror rule applies to the
    other sample, so a ``pair_gap_m`` gap separates the labels. A side under ``min_side_m``
    drops the sample (``strip_too_narrow``). Same-label neighbours are never split.

    Adds ``strip_left_m, strip_right_m, has_opposite_neighbour, pair_id``; returns
    (kept samples, drops)."""
    cfg = cfg or StripConfig()
    s = samples.reset_index(drop=True).copy()
    drops: list[dict] = []
    while True:
        left = np.full(len(s), cfg.half_width_m)
        right = np.full(len(s), cfg.half_width_m)
        has = np.zeros(len(s), bool)
        pair = np.full(len(s), "", dtype=object)
        other = np.full(len(s), "", dtype=object)
        for slug, g in s.groupby("slug"):
            xy, lab = g[["x", "y"]].to_numpy(float), g["label"].to_numpy()
            out = g[["out_dx", "out_dy"]].to_numpy(float)
            uid = g["sample_id"].astype(str).to_numpy()
            tree = cKDTree(xy)
            for k, i in enumerate(g.index):
                if not np.isfinite(out[k]).all():
                    continue
                along = np.array([-out[k, 1], out[k, 0]])
                best = np.inf
                for j in tree.query_ball_point(xy[k], cfg.pair_reach_m * 1.5):
                    if j == k or lab[j] == lab[k]:
                        continue
                    rel = xy[j] - xy[k]
                    a, o = float(rel @ along), float(rel @ out[k])
                    if abs(a) >= cfg.pair_reach_m or abs(o) > cfg.pair_reach_m:
                        continue
                    has[i] = True
                    w = max(abs(a) / 2 - cfg.pair_gap_m / 2, 0.0)
                    if a >= 0:
                        right[i] = min(right[i], w)
                    else:
                        left[i] = min(left[i], w)
                    if abs(a) < best:
                        best, other[i] = abs(a), uid[j]
                        pair[i] = "|".join(sorted((uid[k], uid[j])))
        narrow = np.minimum(left, right) < cfg.min_side_m
        s["strip_left_m"], s["strip_right_m"], s["has_opposite_neighbour"], s["pair_id"] = left, right, has, pair
        if not narrow.any():
            return s, drops
        for i in np.flatnonzero(narrow):
            drops.append({"sample": str(s["sample_id"].iloc[i]), "reason": "strip_too_narrow", "other": other[i], "source": s["source"].iloc[i]})
        s = s[~narrow].reset_index(drop=True)


def block_codes(crop: dict, idx: np.ndarray, sample: pd.Series, others: pd.DataFrame, known_tree: cKDTree | None, cfg: StripConfig | None = None,
                exclusion_m: float = 7.5, rng: np.random.Generator | None = None) -> tuple[np.ndarray | None, dict]:
    """(label codes of the block's points, info) - or (None, info) when the sample has fewer
    than ``min_points`` labelled points and is dropped. The same code path for every label:
    face strip -> wall top -> balanced pick. ``info``: ``wall_top_m, n_strip, n_center``."""
    cfg = cfg or StripConfig()
    rng = rng or np.random.default_rng(0)
    xyz = crop["xyz"][idx]
    steep = crop["steep"][idx]
    codes = np.full(len(idx), 2, np.int8)
    if known_tree is not None:
        unknown_rock = steep & (known_tree.query(xyz[:, :2], distance_upper_bound=exclusion_m)[0] < np.inf)
        codes[unknown_rock] = -1

    def strip(smp):
        return face_strip_mask(xyz, steep, (smp["x"], smp["y"]), smp["base_z"], (smp["out_dx"], smp["out_dy"]), cfg, smp.get("strip_left_m"), smp.get("strip_right_m"),
                               smp.get("wall_height_m"))

    for _, o in others.iterrows():  # neighbours first (same strip shape), own sample last (wins)
        if o.get("source") == "stepback" or not np.isfinite([o["base_z"], o["out_dx"]]).all():
            continue
        codes[strip(o)[0]] = 3 if o["label"] == 1 else 4
    if sample.get("source") == "stepback":  # off the wall (not part of the wall-only task): the whole column, capped
        own = np.flatnonzero(np.hypot(xyz[:, 0] - sample["x"], xyz[:, 1] - sample["y"]) <= cfg.half_width_m)
        pick = own if len(own) <= cfg.bands * cfg.per_band else rng.choice(own, cfg.bands * cfg.per_band, replace=False)
        top = 0.0
    else:
        m, top = strip(sample)
        own = np.flatnonzero(m)
        pick = own[pick_points(xyz[own, 2] - float(sample["base_z"]), cfg, rng)]
    info = {"wall_top_m": float(top), "n_strip": int(len(own)), "n_center": int(len(pick))}
    split = bool(sample.get("has_opposite_neighbour", False)) if pd.notna(sample.get("has_opposite_neighbour", False)) else False
    if len(pick) < (cfg.min_points_split if split else cfg.min_points):
        return None, info
    codes[own] = -1
    codes[pick] = int(sample["label"])
    return codes, info


def augment_negative(local: np.ndarray, codes: np.ndarray, kind: str, out_dir: np.ndarray, radius: float | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Synthetic negative from a positive block (local coords, centre at 0,0).

    The transform applies to every point above the route base in the block
    (or within ``radius``): transforming only the sample's surroundings would
    leave a notch in a full-height wall - an artefact the model could learn.
    """
    pos = codes == 1
    if pos.sum() < 8:
        return None
    p = local.copy()
    base = float(np.quantile(p[pos, 2], 0.05))
    m = p[:, 2] > base
    if radius is not None:
        m &= np.hypot(p[:, 0], p[:, 1]) <= radius
    if kind == "squash":  # the wall becomes ~1/5 as tall: too short to be a route
        p[m, 2] = base + (p[m, 2] - base) * 0.2
    elif kind == "layback":  # the face is laid back to ~45 degrees: a slab / steep slope
        h = p[m, 2] - base
        p[m, 0] += out_dir[0] * h
        p[m, 1] += out_dir[1] * h
    else:
        raise ValueError(kind)
    c = codes.copy()
    c[pos] = 0
    return p.astype(np.float32), c


def build_blocks(samples: pd.DataFrame, out_root: Path, block_size_m: float = 20.0, voxel_m: float = 0.25, max_points: int = 40000,
                 strip: StripConfig | None = None, exclusion_m: float = 7.5, known_lonlat_by_slug: dict | None = None, aug_kinds=(), crs_by_slug=None,
                 allow_stale_aug: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Write ``<out_root>/<slug>/blocks/samples.{points,labels,feat}.npy`` (``feat`` = the
    per-point channels, ``crag.channels``) and ``channels.json``; return (index, drops).

    Every block is centred on its sample's wall foot (``x, y``)."""
    strip = strip or StripConfig()
    if aug_kinds and CHANNEL_SCALE_VERSION >= 1 and not allow_stale_aug:
        raise ValueError("augmented negatives reshape the wall but keep the original points' channels (hag, vert): refusing without --allow-stale-aug")
    rows, drops, stats = [], [], {}
    for slug, sub in samples.groupby("slug"):
        cloud = AreaCloud(slug)
        tree = cKDTree(sub[["x", "y"]].to_numpy())
        known = None
        if known_lonlat_by_slug and len(known_lonlat_by_slug.get(slug, [])):
            kl = known_lonlat_by_slug[slug]
            known = cKDTree(crs_by_slug[slug].to_work(kl[:, 0], kl[:, 1]))
        pts, labs, feats, offset = [], [], [], 0
        sub = sub.reset_index(drop=True)
        for i, s in sub.iterrows():
            crop = cloud.crop(s["x"], s["y"], block_size_m / 2)
            if len(crop["xyz"]) < 64:
                drops.append({"sample": str(s["sample_id"]), "reason": "no_lidar", "other": "", "source": s["source"]})
                continue
            rng = np.random.default_rng(int(registry.stable_fraction(str(s["sample_id"])) * 2**32))
            idx, local, zmin = voxel_block(crop["xyz"], s["x"], s["y"], voxel_m, max_points, rng)
            nb = [j for j in tree.query_ball_point([s["x"], s["y"]], block_size_m + strip.out_back_m) if j != i]
            codes, info = block_codes(crop, idx, s, sub.loc[nb], known, strip, exclusion_m, rng=rng)
            if codes is None:
                drops.append({"sample": str(s["sample_id"]), "reason": "under_min_points", "other": "", "source": s["source"], "n_center": info["n_center"],
                              "split_strip": bool(s.get("has_opposite_neighbour", False))})
                continue
            feat = point_channels(crop, idx, CHANNELS, stats)
            variants = [(local, codes, s["sample_id"], s["label"], s["source"], s["confidence"])]
            if s["label"] == 1:
                out_dir = np.array([s.get("out_dx", 0.0), s.get("out_dy", 0.0)], float)
                for kind in aug_kinds:
                    r = augment_negative(local, codes, kind, out_dir if np.isfinite(out_dir).all() else np.zeros(2))
                    if r is not None:
                        variants.append((r[0], r[1], f"{s['sample_id']}~{kind}", 0, f"aug_{kind}", 0.7))
            cl = crop["steep"][idx]
            lab_up = crop["xyz"][idx][np.isin(codes, (0, 1)), 2] - float(s["base_z"]) if np.isfinite(s["base_z"]) else np.zeros(0)
            for loc, cod, sid, lab, src, conf in variants:
                pts.append(loc)
                labs.append(cod)
                feats.append(feat)
                n = len(loc)
                rows.append({**s.to_dict(), "sample_id": sid, "label": int(lab), "source": src, "confidence": float(conf),
                             "slug": slug, "tile": "samples", "start": offset, "stop": offset + n, "zmin": zmin,
                             "n_climbable": int((cod == 1).sum()), "n_not_climbable": int((cod == 0).sum()), "n_center": int(np.isin(cod, (0, 1)).sum()),
                             "wall_top_m": info["wall_top_m"], "n_strip": info["n_strip"], "foot_z_local": float(s["base_z"]) - zmin, "n_cliff_block": int(cl.sum()),
                             "block_height_m": float(loc[:, 2].max()), "lab_up_min": float(lab_up.min()) if len(lab_up) else np.nan,
                             "lab_up_med": float(np.median(lab_up)) if len(lab_up) else np.nan, "lab_up_max": float(lab_up.max()) if len(lab_up) else np.nan})
                offset += n
        if not pts:
            continue
        d = Path(out_root) / slug / "blocks"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "samples.points.npy", np.concatenate(pts))
        np.save(d / "samples.labels.npy", np.concatenate(labs).astype(np.int8))
        np.save(d / "samples.feat.npy", np.concatenate(feats))
        log.info("%s: %d sample blocks", slug, sum(1 for r in rows if r["slug"] == slug))
    write_channels_json(Path(out_root), CHANNELS)
    if stats.get("n"):
        log.info("channels: %s", {k: f"{v / stats['n']:.2%}" for k, v in stats.items() if k != "n"})
    index = pd.DataFrame(rows)
    index.attrs["channel_stats"] = stats
    return index, drops


def dataset_dir(name: str) -> Path:
    return FINETUNE_DIR / name


def blocks_root(name: str) -> Path:
    """Where a dataset's block arrays live: its own folder, or the dataset named in
    ``blocks_from.txt`` (the Horseshoe folds share the arrays of the full build; only
    ``index.csv`` differs)."""
    d = dataset_dir(name)
    p = d / "blocks_from.txt"
    return dataset_dir(p.read_text().strip()) if p.exists() else d


def strip_config_dict(cfg: StripConfig | None = None) -> dict:
    return asdict(cfg or StripConfig())
