"""Torch datasets over the pseudolabelled blocks (pretraining) - OpenPoints-style dicts.

Every item is ``{"pos": (N, 3) float32, "x": (N, C) float32, "y": (N,) int64}``
with exactly ``num_points`` points, which is what OpenPoints ``BaseSeg``
expects (the trainer transposes ``x`` to (B, C, N)).

Coordinates. The S3DIS checkpoints were trained on points voxelised at 4 cm
with ball-query radius 0.1. Our LiDAR is voxelised at ``voxel_m`` (0.25 m),
so positions are multiplied by ``coord_scale = 0.04 / voxel_m`` (0.16): the
pretrained filters then see neighbourhoods with the same number of points
they were trained on (first radius 0.1 = 0.625 m on the ground, doubling per
stage). A *fixed* scale keeps metric meaning - a 30 m cliff is always the
same size - which per-block unit-sphere normalisation would destroy (a
flat block and a cliff block would get different scales).

Features (``feature_keys``): ``pos`` (3, scaled xyz after augmentation),
``heights`` (1, height above the block's lowest point, scaled),
``intensity`` (1, per-tile normalised). The default ``pos,heights`` has the
same 4 channels as the S3DIS checkpoints' ``rgb,height`` so their first layer
loads.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from crag import registry
from crag.paths import AREAS_JSON, HELDOUT_REGIONS_JSON, PRETRAIN_DIR, add_import_paths
from crag.pseudolabels import CLASSES, IGNORE

add_import_paths()
from openpoints.transforms import build_transforms_from_cfg  # noqa: E402

import crag.models  # noqa: E402,F401  (registers PointCloudRotationZ)

log = logging.getLogger(__name__)

FEATURE_DIMS = {"pos": 3, "heights": 1, "intensity": 1}


def feature_dim(feature_keys: str) -> int:
    return sum(FEATURE_DIMS[k.strip()] for k in feature_keys.split(","))


@dataclass
class BlockSpec:
    num_points: int = 8192
    voxel_m: float = 0.25
    coord_scale: float = 0.16
    feature_keys: str = "pos,heights"


def load_heldout_regions(path: Path = HELDOUT_REGIONS_JSON) -> list[dict]:
    if not Path(path).exists():
        return []
    return json.loads(Path(path).read_text()).get("regions", [])


def drop_heldout(index: pd.DataFrame, regions: list[dict]) -> pd.DataFrame:
    """Remove blocks whose centre falls in a held-out lon/lat bbox (fine-tune test crags)."""
    if not regions or index.empty:
        return index
    keep = np.ones(len(index), bool)
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        keep &= ~(index["lon"].between(x0, x1) & index["lat"].between(y0, y1))
    if (~keep).any():
        log.info("dropped %d blocks inside %d held-out regions", (~keep).sum(), len(regions))
    return index[keep].reset_index(drop=True)


def load_block_index(
    splits: dict[str, list[str]] | None = None,
    registry_path: Path = AREAS_JSON,
    split_mode: str = "area",
    val_tile_frac: float = 0.1,
    test_tile_frac: float = 0.1,
    areas: list[str] | None = None,
    include_holdout: bool = False,
) -> pd.DataFrame:
    """All blocks of processed areas, with a ``split`` column.

    ``split_mode="area"`` (default) uses each area's registry split - whole
    areas are held out. ``"tile"`` assigns 200 m tiles by a stable hash
    instead (handy when only one or two areas are downloaded). Areas with
    registry split ``holdout`` are always excluded unless ``include_holdout``.
    """
    reg = registry.load(registry_path)
    frames = []
    for a in registry.areas(reg, areas):
        if a.get("split") == "holdout" and not include_holdout:
            continue  # ground-truth sites never enter pretraining, whatever the split mode
        p = PRETRAIN_DIR / a["slug"] / "index.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        if split_mode == "area":
            df["split"] = a.get("split", "train")
        elif split_mode == "tile":
            u = df["tile"].map(lambda t, s=a["slug"]: registry.stable_fraction(f"{s}/{t}"))
            df["split"] = np.where(u < test_tile_frac, "test", np.where(u < test_tile_frac + val_tile_frac, "val", "train"))
        else:
            raise ValueError(split_mode)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no processed areas under {PRETRAIN_DIR} - run scripts/s1_pseudolabel.py")
    index = pd.concat(frames, ignore_index=True)
    index = drop_heldout(index, load_heldout_regions())
    if splits:  # explicit overrides: {"val": ["smith-rock"], ...}
        for split, slugs in splits.items():
            index.loc[index["slug"].isin(slugs), "split"] = split
    return index


class _Memmaps:
    """Lazily opened, per-worker memory maps of block arrays."""

    def __init__(self, root: Path):
        self.root = root
        self._open: dict[tuple[str, str], tuple] = {}

    def get(self, slug: str, tile: str):
        key = (slug, tile)
        if key not in self._open:
            d = self.root / slug / "blocks"
            self._open[key] = tuple(np.load(d / f"{tile}.{k}.npy", mmap_mode="r") for k in ("points", "labels", "feat"))
            if len(self._open) > 512:
                self._open.pop(next(iter(self._open)))
        return self._open[key]


class BlockDataset(Dataset):
    """Blocks of one split, from a block index (pretraining or fine-tuning layout).

    ``class_names`` name the ``n_<class>`` count columns of the index;
    ``oversample`` maps a class name to the extra sampling weight of blocks
    containing it; ``label_map`` remaps stored label codes (e.g. fine-tuning
    background code 2 -> ignore); ``center_codes`` adds a boolean ``center``
    mask of points whose *stored* code is in it (the fine-tuning sample's own
    cylinder, used for per-sample scores).
    """

    def __init__(
        self,
        index: pd.DataFrame,
        spec: BlockSpec,
        transforms_cfg=None,
        train: bool = True,
        root: Path = PRETRAIN_DIR,
        class_names=CLASSES,
        oversample: dict | None = None,
        label_map: dict | None = None,
        center_codes: tuple[int, ...] | None = None,
    ):
        self.index = index.reset_index(drop=True)
        self.spec = spec
        self.train = train
        self.transform = build_transforms_from_cfg("train" if train else "val", transforms_cfg) if transforms_cfg else None
        self.maps = _Memmaps(Path(root))
        self.keys = [k.strip() for k in spec.feature_keys.split(",")]
        self.class_names = list(class_names)
        self.oversample = dict(oversample or {})
        self.lut = None
        if label_map:
            self.lut = np.arange(-128, 128, dtype=np.int64)  # identity over int8 codes
            for k, v in label_map.items():
                self.lut[int(k) + 128] = int(v)
        self.center_codes = center_codes

    def __len__(self) -> int:
        return len(self.index)

    def sample_weights(self) -> np.ndarray:
        """Per-block sampling weight: 1 + sum of weights of the rare classes it contains."""
        w = np.ones(len(self.index))
        for c, extra in self.oversample.items():
            w += float(extra) * (self.index[f"n_{c}"].to_numpy() > 0)
        return w

    def class_counts(self) -> np.ndarray:
        return self.index[[f"n_{c}" for c in self.class_names]].sum().to_numpy(dtype=np.float64)

    def __getitem__(self, i: int) -> dict:
        r = self.index.iloc[i]
        pts, labs, feat = self.maps.get(r["slug"], r["tile"])
        a, b = int(r["start"]), int(r["stop"])
        # train: from the (seeded) worker stream -> reproducible given the run seed;
        # eval: fixed per item -> identical validation points every epoch
        rng = np.random.default_rng(np.random.randint(0, 2**31)) if self.train else np.random.default_rng(i)
        n, N = b - a, self.spec.num_points
        sel = rng.choice(n, N, replace=n < N) + a
        sel.sort()  # contiguous-ish reads from the memmap
        pos = np.asarray(pts[sel], dtype=np.float32) * self.spec.coord_scale
        raw = np.asarray(labs[sel], dtype=np.int64)
        y = self.lut[raw + 128] if self.lut is not None else raw
        inten = np.asarray(feat[sel], dtype=np.float32)[:, None]
        item = make_item(pos, y, inten, self.keys, self.transform)
        if self.center_codes is not None:
            item["center"] = torch.from_numpy(np.isin(raw, self.center_codes))
        item["index"] = i
        return item


def make_item(pos: np.ndarray, y: np.ndarray, intensity: np.ndarray, keys: list[str], transform=None) -> dict:
    """Augment ``pos`` with OpenPoints transforms, then assemble the feature tensor."""
    data = {"pos": pos, "intensity": intensity, "y": y}
    if transform is not None:
        data = transform(data)
    else:
        data = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in data.items()}
    pos = data["pos"].float()
    feats = []
    for k in keys:
        if k == "pos":
            feats.append(pos)
        elif k == "heights":
            feats.append(pos[:, 2:3] - pos[:, 2:3].min())
        elif k == "intensity":
            feats.append(data["intensity"].float().reshape(-1, 1))
        else:
            raise KeyError(f"unknown feature key {k!r}")
    return {"pos": pos.contiguous(), "x": torch.cat(feats, 1).contiguous(), "y": data["y"].long()}


def weighted_sampler(ds: BlockDataset, num_samples: int, seed: int) -> WeightedRandomSampler:
    g = torch.Generator()
    g.manual_seed(seed)
    w = torch.as_tensor(ds.sample_weights(), dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=num_samples, replacement=True, generator=g)


def ignore_index() -> int:
    return IGNORE
