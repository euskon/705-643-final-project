"""Config -> datasets / model for pretraining (shared by the train and eval scripts)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from crag import models
from crag.datasets import BlockDataset, BlockSpec, feature_dim, load_block_index, weighted_sampler
from crag.paths import CFG_DIR, add_import_paths
from crag.pseudolabels import CLASSES
from crag.train import cfg_get

add_import_paths()
from openpoints.utils import EasyConfig  # noqa: E402

log = logging.getLogger(__name__)


def load_cfg(path: str | Path, overrides: list[str] | None = None) -> EasyConfig:
    """YAML (+ every parent ``default.yaml``) + ``key=value`` overrides."""
    path = Path(path)
    if not path.exists() and (CFG_DIR / path).exists():
        path = CFG_DIR / path
    cfg = EasyConfig()
    cfg.load(str(path), recursive=True)
    if overrides:
        cfg.update(list(overrides))
    cfg.cfg_path = str(path)
    return cfg


def block_spec(cfg) -> BlockSpec:
    return BlockSpec(
        num_points=int(cfg_get(cfg, "dataset.num_points", 4096)),
        voxel_m=float(cfg_get(cfg, "dataset.voxel_m", 0.25)),
        coord_scale=float(cfg_get(cfg, "dataset.coord_scale", 0.067)),
        feature_keys=str(cfg_get(cfg, "dataset.feature_keys", "hag,planarity")),
    )


def select_eval_blocks(index: pd.DataFrame, max_blocks: int, seed: int = 0) -> pd.DataFrame:
    """Deterministic evaluation subset: blocks with the most cliff points first, then a random rest."""
    if not max_blocks or len(index) <= max_blocks:
        return index.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    pri = np.where(index["n_cliff"] >= 200, 0, np.where(index["n_cliff"] > 0, 1, 2))
    order = np.lexsort((rng.random(len(index)), pri))
    keep = np.sort(order[:max_blocks])
    return index.iloc[keep].reset_index(drop=True)


def build_datasets(cfg, areas: list[str] | None = None) -> dict:
    index = load_block_index(split_mode=str(cfg_get(cfg, "dataset.split_mode", "area")), areas=areas)
    index = index[index["split"] != "holdout"]
    from crag.channels import read_channels_json
    from crag.paths import PRETRAIN_DIR

    top = read_channels_json(PRETRAIN_DIR)
    bad = [s for s in index["slug"].unique() if read_channels_json(PRETRAIN_DIR / s) != top]
    if bad:
        raise ValueError(f"areas whose blocks carry other channels than data/pretrain/channels.json ({top}): {bad} - run scripts/s1_add_channels.py --rebuild-blocks")
    spec = block_spec(cfg)
    tcfg = cfg_get(cfg, "datatransforms")
    out = {"index": index}
    for split in ("train", "val", "test"):
        sub = index[index["split"] == split]
        if split != "train":
            sub = select_eval_blocks(sub, int(cfg_get(cfg, "dataset.max_val_blocks", 3000)))
        out[split] = BlockDataset(sub, spec, tcfg, train=split == "train", oversample=dict(cfg_get(cfg, "dataset.oversample", {}) or {}),
                                  window=dict(cfg_get(cfg, "dataset.window", {}) or {}) or None, thin=dict(cfg_get(cfg, "dataset.thin", {}) or {}) or None)
        log.info("%s: %d blocks from %d areas", split, len(sub), sub["slug"].nunique())
    return out


def class_weights(counts: np.ndarray, mode: str | None) -> np.ndarray | None:
    if not mode or mode == "null":
        return None
    freq = counts / counts.sum()
    freq = np.where(freq > 0, freq, 1.0)
    w = 1.0 / freq if mode == "inverse" else 1.0 / np.sqrt(freq)
    w = w / (w * (counts / counts.sum())).sum()  # mean weight over points = 1
    return np.clip(w, 0.05, 50.0).astype(np.float32)


def build_model_from(cfg) -> tuple:
    keys = str(cfg_get(cfg, "dataset.feature_keys", "hag,planarity"))
    n = feature_dim(keys)
    if int(cfg.model.encoder_args.in_channels) != n:
        log.info("in_channels set from feature_keys %r: %s -> %d", keys, cfg.model.encoder_args.in_channels, n)
    cfg.model.encoder_args.in_channels = n
    model = models.build_model(cfg.model)
    report = None
    if cfg_get(cfg, "pretrained_path"):
        report = models.load_pretrained(model, cfg.pretrained_path, tuple(cfg_get(cfg, "pretrained_parts", ["encoder", "decoder"])))
        models.check_first_layer_only(report)
    return model, report


def make_sampler(cfg, train_ds: BlockDataset):
    return weighted_sampler(train_ds, num_samples=int(cfg_get(cfg, "dataset.samples_per_epoch", 20000)), seed=int(cfg_get(cfg, "seed", 42)))


CLASS_NAMES = list(CLASSES)
