"""Spatial train/val/test split, patch normalisation and the torch Dataset.

Splitting is by **wall**: every positive cylinder of a wall lands in the same
split, and every negative follows the wall of its nearest positive. Route
cylinders on one wall are 3 m apart and their 5 m input columns overlap, so a
random split would put near-identical patches in train and test.

Example::

    split_walls(samples, SplitConfig())        # adds samples["split"]
    save_samples(path, samples, patches)       # cache
    train_ds = PatchDataset(samples, patches, "train", PatchConfig(), augment=True)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from config import PatchConfig, SplitConfig

SPLITS = ("train", "val", "test")


def split_walls(samples: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """Assign whole walls to splits, aiming for ``val_frac``/``test_frac`` of the positives.

    Walls are shuffled and dealt to the split that is furthest below its
    target share of positive patches, so one huge wall (North Forty) can't
    swallow a whole split by accident. Returns a copy with ``split`` and
    ``split_wall`` columns.
    """
    samples = samples.copy()
    pos = samples[samples["label"] == 1]
    counts = pos.groupby("wall").size()
    rng = np.random.default_rng(cfg.seed)
    walls = list(rng.permutation(counts.index.to_numpy()))
    # deal the biggest walls first (to train) so the small held-out splits
    # get several different walls rather than one large one
    walls.sort(key=lambda w: -counts[w])
    target = {
        "train": 1 - cfg.val_frac - cfg.test_frac,
        "val": cfg.val_frac,
        "test": cfg.test_frac,
    }
    have = dict.fromkeys(SPLITS, 0)
    wall_split = {}
    for w in walls:
        total = sum(have.values()) + counts[w]
        deficit = {s: target[s] * total - have[s] for s in SPLITS}
        s = max(deficit, key=deficit.get)
        wall_split[w] = s
        have[s] += counts[w]

    pos_xy = pos[["x", "y"]].to_numpy()
    nearest = cKDTree(pos_xy).query(samples[["x", "y"]].to_numpy())[1]
    samples["split_wall"] = pos["wall"].to_numpy()[nearest]
    samples.loc[samples["label"] == 1, "split_wall"] = samples.loc[
        samples["label"] == 1, "wall"
    ]
    samples["split"] = samples["split_wall"].map(wall_split)
    return samples


def save_samples(path: Path, samples: pd.DataFrame, patches: list[np.ndarray]) -> None:
    """One ``.npz`` (ragged patches as one array + offsets) and one ``.csv``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0] + [len(p) for p in patches])
    np.savez_compressed(
        path.with_suffix(".npz"), points=np.concatenate(patches), offsets=offsets
    )
    samples.to_csv(path.with_suffix(".csv"), index=False)


def load_samples(path: Path) -> tuple[pd.DataFrame, list[np.ndarray]]:
    path = Path(path)
    with np.load(path.with_suffix(".npz")) as f:
        points, offsets = f["points"], f["offsets"]
    patches = [points[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    return pd.read_csv(path.with_suffix(".csv")), patches


def normalize_patch(pts: np.ndarray, cfg: PatchConfig) -> np.ndarray:
    """Centre and scale a raw (n, 3) column the way the network expects."""
    if cfg.normalize == "unit_sphere":
        # identical to ModelNet40's pc_normalize used for the pretraining
        pts = pts - pts.mean(axis=0)
        return pts / (np.max(np.linalg.norm(pts, axis=1)) + 1e-9)
    if cfg.normalize == "fixed":
        pts = pts - np.array([0.0, 0.0, np.median(pts[:, 2])], dtype=pts.dtype)
        return pts / cfg.fixed_scale_m
    raise ValueError(f"unknown normalize {cfg.normalize!r}")


def resample(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Exactly ``n`` points: subsample without replacement, or pad by repeating."""
    idx = rng.choice(len(pts), n, replace=len(pts) < n)
    return pts[idx]


def augment(pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotation about the vertical axis only (gravity matters for cliffs), scale, jitter."""
    theta = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=pts.dtype)
    pts = pts @ rot.T * rng.uniform(0.9, 1.1)
    return pts + np.clip(rng.normal(0, 0.01, pts.shape), -0.03, 0.03).astype(pts.dtype)


class PatchDataset(Dataset):
    """``(points [3, N] float32, label int64)`` for one split.

    Training draws a fresh random subsample (and augmentation) every time a
    patch is read; evaluation uses a fixed per-patch seed so metrics are
    reproducible.
    """

    def __init__(
        self,
        samples: pd.DataFrame,
        patches: list[np.ndarray],
        split: str | None,
        cfg: PatchConfig,
        augment: bool = False,
    ):
        rows = (
            samples.index if split is None else samples.index[samples["split"] == split]
        )
        self.rows = samples.loc[rows].reset_index()
        self.patches = [patches[i] for i in rows]
        self.cfg = cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patches)

    @property
    def labels(self) -> np.ndarray:
        return self.rows["label"].to_numpy()

    def __getitem__(self, i: int):
        rng = np.random.default_rng() if self.augment else np.random.default_rng(i)
        pts = resample(self.patches[i], self.cfg.num_points, rng)
        pts = normalize_patch(pts, self.cfg)
        if self.augment:
            pts = augment(pts, rng)
        return torch.from_numpy(np.ascontiguousarray(pts.T, dtype=np.float32)), int(
            self.rows.at[i, "label"]
        )


def make_loaders(
    samples: pd.DataFrame,
    patches: list[np.ndarray],
    cfg: PatchConfig,
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Train (shuffled, augmented, last partial batch dropped for BatchNorm) / val / test loaders."""
    loaders = {}
    for split in SPLITS:
        train = split == "train"
        ds = PatchDataset(samples, patches, split, cfg, augment=train)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
    return loaders


def split_summary(samples: pd.DataFrame) -> pd.DataFrame:
    """Patch counts per split and label, plus the walls in each split."""
    table = samples.pivot_table(
        index="split", columns="label", values="x", aggfunc="size", fill_value=0
    )
    table = table.rename(columns={0: "negatives", 1: "positives"}).reindex(SPLITS)
    walls = (
        samples[samples["label"] == 1]
        .groupby("split")["wall"]
        .agg(lambda w: ", ".join(sorted(set(w))))
    )
    table["n_walls"] = (
        samples[samples["label"] == 1]
        .groupby("split")["wall"]
        .nunique()
        .reindex(SPLITS)
    )
    table["walls"] = walls.reindex(SPLITS).fillna("")
    return table
