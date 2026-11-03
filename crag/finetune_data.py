"""Fine-tuning blocks: one ``block_size_m`` block per sample (positive or negative).

Stored label codes (int8), remapped at load time by ``label_mode``:

====  ======================================================  =================
code  meaning                                                  center_only map
====  ======================================================  =================
 1    the block's own positive cylinder (steep points between      1
      the route base and base + route length)
 0    the block's own negative cylinder                            0
 3    another positive's cylinder inside the block               ignore
 4    another negative's cylinder inside the block               ignore
 2    background (everything else)                               ignore
-1    unknown rock: steep points near a known route but not in   ignore
      a positive cylinder - never treated as negative
====  ======================================================  =================

``label_mode``: ``center_only`` (default; each block supervises only its own
sample, so a sample's score is unambiguous), ``all_cylinders`` (3->1, 4->0)
or ``with_background`` (all_cylinders + background 2->0: non-rock terrain
as easy negatives).

Splits are spatial: samples within ``group_m`` of each other form a group
(≈ one crag); groups are dealt to train/val/test by a stable hash, and whole
``test_areas`` (default: Horseshoe Canyon Ranch, the only site with on-site
negatives) go to test. Test groups are written to
``registry/heldout_regions.json`` so pretraining never sees them.

Augmentation negatives (optional, ``aug_squash`` / ``aug_layback``): copies of
positive blocks whose wall is made too short (heights x0.2) or laid back to
~45 degrees, relabelled 0 - labels certain by construction but synthetic.
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

from crag import registry
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


def block_codes(crop: dict, idx: np.ndarray, sample: pd.Series, others: pd.DataFrame, known_tree: cKDTree | None, cyl_m: float, exclusion_m: float,
                max_center_points: int = 48, rng: np.random.Generator | None = None) -> np.ndarray:
    xyz = crop["xyz"][idx]
    steep = crop["steep"][idx]
    codes = np.full(len(idx), 2, np.int8)
    if known_tree is not None:
        unknown_rock = steep & (known_tree.query(xyz[:, :2], distance_upper_bound=exclusion_m)[0] < np.inf)
        codes[unknown_rock] = -1
    for o in others.itertuples(index=False):  # neighbours first, own sample last (wins)
        m = np.hypot(xyz[:, 0] - o.x, xyz[:, 1] - o.y) <= cyl_m
        codes[m & steep] = 3 if o.label == 1 else 4
    m = np.hypot(xyz[:, 0] - sample["x"], xyz[:, 1] - sample["y"]) <= cyl_m
    if sample["label"] == 1:
        zok = np.ones(len(idx), bool)
        if pd.notna(sample.get("base_z")) and pd.notna(sample.get("top_z")):
            zok = (xyz[:, 2] >= sample["base_z"] - 1.0) & (xyz[:, 2] <= sample["top_z"])
        codes[m & ~steep] = -1
        codes[m & steep & zok] = 1
    elif sample["source"] in CLIFF_NEGATIVES:
        codes[m & ~steep] = -1
        codes[m & steep] = 0
    else:  # stepback / talus / plateau: the whole column is not climbable
        codes[m] = 0
    # cap labelled points per sample so big columns (talus) don't outweigh thin route cylinders
    own = np.flatnonzero(np.isin(codes, (0, 1)) & m)
    if max_center_points and len(own) > max_center_points:
        rng = rng or np.random.default_rng(0)
        drop = rng.choice(own, len(own) - max_center_points, replace=False)
        codes[drop] = -1
    return codes


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
                 cyl_m: float = 2.5, exclusion_m: float = 7.5, known_lonlat_by_slug: dict | None = None, aug_kinds=(), crs_by_slug=None) -> pd.DataFrame:
    """Write ``<out_root>/<slug>/blocks/samples.{points,labels,feat}.npy`` and return the index."""
    rows = []
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
                continue
            rng = np.random.default_rng(int(registry.stable_fraction(str(s["sample_id"])) * 2**32))
            idx, local, zmin = voxel_block(crop["xyz"], s["x"], s["y"], voxel_m, max_points, rng)
            nb = [j for j in tree.query_ball_point([s["x"], s["y"]], block_size_m) if j != i]
            codes = block_codes(crop, idx, s, sub.loc[nb], known, cyl_m, exclusion_m, rng=rng)
            inten = (crop["intensity"][idx] / max(np.percentile(crop["intensity"], 99), 1.0)).clip(0, 1).astype(np.float16)
            variants = [(local, codes, s["sample_id"], s["label"], s["source"], s["confidence"])]
            if s["label"] == 1:
                out_dir = np.array([s.get("out_dx", 0.0), s.get("out_dy", 0.0)], float)
                for kind in aug_kinds:
                    r = augment_negative(local, codes, kind, out_dir if np.isfinite(out_dir).all() else np.zeros(2))
                    if r is not None:
                        variants.append((r[0], r[1], f"{s['sample_id']}~{kind}", 0, f"aug_{kind}", 0.7))
            for loc, cod, sid, lab, src, conf in variants:
                pts.append(loc)
                labs.append(cod)
                feats.append(inten)
                n = len(loc)
                rows.append({**s.to_dict(), "sample_id": sid, "label": int(lab), "source": src, "confidence": float(conf),
                             "slug": slug, "tile": "samples", "start": offset, "stop": offset + n, "zmin": zmin,
                             "n_climbable": int((cod == 1).sum()), "n_not_climbable": int((cod == 0).sum()), "n_center": int(np.isin(cod, (0, 1)).sum())})
                offset += n
        if not pts:
            continue
        d = Path(out_root) / slug / "blocks"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "samples.points.npy", np.concatenate(pts))
        np.save(d / "samples.labels.npy", np.concatenate(labs).astype(np.int8))
        np.save(d / "samples.feat.npy", np.concatenate(feats))
        log.info("%s: %d sample blocks", slug, sum(1 for r in rows if r["slug"] == slug))
    return pd.DataFrame(rows)


def dataset_dir(name: str) -> Path:
    return FINETUNE_DIR / name
