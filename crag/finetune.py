"""Fine-tuning: config -> model (pretrained backbone + fresh 2-class head) and datasets.

Strategies (``cfgs/finetune/<strategy>.yaml``, each merged over
``cfgs/finetune/default.yaml``) differ in what is frozen, when it is
unfrozen, LR and LR multipliers - see ``docs/step4-finetuning-strategy.md``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import WeightedRandomSampler

from crag import models
from crag.datasets import BlockDataset, feature_dim
from crag.finetune_data import FT_CLASSES, LABEL_MODES, dataset_dir
from crag.paths import PROJECT_DIR, add_import_paths
from crag.pretrain import block_spec, load_cfg
from crag.train import cfg_get

add_import_paths()
from openpoints.utils import EasyConfig  # noqa: E402

log = logging.getLogger(__name__)
HARD_NEGATIVE_SOURCES = ("band", "handgathered_nc")


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_DIR / p


def architecture(cfg) -> EasyConfig:
    """Model section from ``init.arch``: a pretraining run dir (its config.json) or a YAML."""
    src = _abs(cfg_get(cfg, "init.arch", "cfgs/pretrain/pointnext-s.yaml"))
    if src.is_dir():
        mcfg = EasyConfig()
        mcfg.update(json.loads((src / "config.json").read_text())["model"])
    else:
        mcfg = load_cfg(src).model
    mcfg.cls_args.num_classes = int(cfg_get(cfg, "num_classes", 2))
    return mcfg


def build_model(cfg):
    cfg.model = architecture(cfg)
    keys = str(cfg_get(cfg, "dataset.feature_keys", "pos,heights"))
    if int(cfg.model.encoder_args.in_channels) != feature_dim(keys):
        raise ValueError(f"architecture in_channels={cfg.model.encoder_args.in_channels} != feature_dim({keys!r})")
    model = models.build_model(cfg.model)
    report = None
    weights = cfg_get(cfg, "init.weights")
    if weights:
        p = _abs(weights)
        if p.is_dir():
            p = p / "backbone.pt"
        report = models.load_pretrained(model, p, tuple(cfg_get(cfg, "init.parts", ["encoder", "decoder"])))
    models.set_frozen(model, tuple(cfg_get(cfg, "freeze", []) or []))
    return model, report


def load_index(cfg) -> pd.DataFrame:
    name = str(cfg_get(cfg, "dataset.name", "v1"))
    index = pd.read_csv(dataset_dir(name) / "index.csv")
    srcs = cfg_get(cfg, "dataset.exclude_sources", []) or []
    if srcs:
        index = index[~index["source"].isin(srcs)]
    index = index[index["confidence"] >= float(cfg_get(cfg, "dataset.min_confidence", 0.0))]
    return index.reset_index(drop=True)


def build_datasets(cfg, index: pd.DataFrame | None = None) -> dict:
    index = load_index(cfg) if index is None else index
    spec = block_spec(cfg)
    mode = str(cfg_get(cfg, "dataset.label_mode", "center_only"))
    root = dataset_dir(str(cfg_get(cfg, "dataset.name", "v1")))
    out = {"index": index}
    for split in ("train", "val", "test"):
        sub = index[index["split"] == split]
        if split != "train":
            sub = sub[~sub["source"].str.startswith("aug_")]  # evaluate on real samples only
            sub = sub.sample(frac=1.0, random_state=0)  # mixed classes even when validation is truncated
        if len(sub) and sub["label"].nunique() < 2:
            log.warning("%s split has only label %s samples - metrics will be degenerate", split, sub["label"].iloc[0])
        out[split] = BlockDataset(sub, spec, cfg_get(cfg, "datatransforms"), train=split == "train", root=root, class_names=FT_CLASSES,
                                  label_map=LABEL_MODES[mode], center_codes=(0, 1))
        log.info("%s: %d samples (%d pos / %d neg) from %d areas", split, len(sub), int((sub["label"] == 1).sum()), int((sub["label"] == 0).sum()), sub["slug"].nunique())
    return out


def balanced_sampler(ds: BlockDataset, num_samples: int, seed: int) -> WeightedRandomSampler:
    """Equal total weight per class, and within a class proportional to confidence."""
    lab = ds.index["label"].to_numpy()
    conf = ds.index["confidence"].to_numpy(dtype=float)
    w = np.zeros(len(lab))
    for c in (0, 1):
        m = lab == c
        if m.any():
            w[m] = conf[m] / conf[m].sum()
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=num_samples, replacement=True, generator=g)


class SampleScoreHook:
    """Per-sample score = mean P(climbable) over the sample's own cylinder points."""

    def __init__(self, ds: BlockDataset, threshold: float = 0.5):
        self.ds, self.threshold = ds, threshold

    def start(self):
        return {"idx": [], "score": []}

    def update(self, st, batch, logits):
        p = torch.softmax(logits.float(), 1)[:, 1]
        c = batch["center"].to(p.device)
        s = (p * c).sum(1) / c.sum(1).clamp(min=1)
        s[c.sum(1) == 0] = float("nan")
        st["idx"] += batch["index"].tolist()
        st["score"] += s.cpu().tolist()

    def finish(self, st) -> dict:
        idx = np.asarray(st["idx"])
        score = np.asarray(st["score"], dtype=float)
        rows = self.ds.index.iloc[idx]
        self.last_predictions = rows.assign(score=score)[["sample_id", "slug", "source", "label", "confidence", "split", "score", "lon", "lat"]]
        return sample_metrics(self.last_predictions, self.threshold)


def sample_metrics(pred: pd.DataFrame, threshold: float = 0.5) -> dict:
    p = pred[pred["score"].notna()]
    y, s = p["label"].to_numpy(), p["score"].to_numpy()
    out = {"sample_n": int(len(p)), "sample_n_pos": int((y == 1).sum()), "sample_n_neg": int((y == 0).sum())}
    if len(np.unique(y)) == 2:
        out.update(
            sample_auroc=float(roc_auc_score(y, s)),
            sample_ap=float(average_precision_score(y, s)),
            sample_bal_acc=float(balanced_accuracy_score(y, s >= threshold)),
            sample_f1=float(f1_score(y, s >= threshold)),
        )
        hard = p["source"].isin(HARD_NEGATIVE_SOURCES) | (p["label"] == 1)
        if (p.loc[hard, "label"] == 0).any():
            out["sample_auroc_hard"] = float(roc_auc_score(p.loc[hard, "label"], p.loc[hard, "score"]))
    for src, g in p.groupby("source"):
        # positives: recall; negatives: specificity (fraction correctly scored below threshold)
        ok = (g["score"] >= threshold) if g["label"].iloc[0] == 1 else (g["score"] < threshold)
        out[f"acc_{src}"] = float(ok.mean())
    return out
