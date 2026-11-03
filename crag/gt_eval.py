"""Score a segmentation checkpoint on the hand-gathered Horseshoe Canyon Ranch waypoints.

For each ``Climb`` (1) / ``Nc`` (0) waypoint: cut the same kind of block the
model was trained on, centred on the waypoint, predict per-point class
probabilities, and summarise the points within ``radius_m`` (horizontal) of
the waypoint as one score (90th percentile, robust to a few stray points;
the waypoint is where the person stood, a few metres off the face).

Reported: AUROC / average precision of Climb vs Nc for the model's
"positive" probability, plus the surface-normal heuristic (fraction of steep
points) as a reference.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from crag.crop import AreaCloud, voxel_block
from crag.datasets import BlockSpec, make_item
from crag.paths import add_import_paths
from crag.train import model_forward

add_import_paths()
from config import GroundTruthConfig  # noqa: E402
from groundtruth import load_waypoints  # noqa: E402
from openpoints.transforms import build_transforms_from_cfg  # noqa: E402

log = logging.getLogger(__name__)
GT_SLUG = "horseshoe-canyon-ranch"


def _safe(fn, y, s) -> float:
    return float(fn(y, s)) if len(np.unique(y)) == 2 else float("nan")


@torch.no_grad()
def score_waypoints(
    model,
    spec: BlockSpec,
    transforms_cfg,
    positive_classes: list[int],
    rock_classes: list[int] | None = None,
    block_size_m: float = 20.0,
    radius_m: float = 5.0,
    slug: str = GT_SLUG,
    batch_size: int = 8,
) -> tuple[dict, pd.DataFrame]:
    device = next(model.parameters()).device
    gt = load_waypoints(GroundTruthConfig())
    gt = gt[gt["label"] >= 0].reset_index(drop=True)
    cloud = AreaCloud(slug)
    tf = build_transforms_from_cfg("val", transforms_cfg) if transforms_cfg else None
    keys = [k.strip() for k in spec.feature_keys.split(",")]
    model.eval()
    rows, batch, meta = [], [], []

    def flush():
        if not batch:
            return
        b = {k: torch.stack([it[k] for it in batch]) for k in ("pos", "x")}
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            prob = torch.softmax(model_forward(model, b, device).float(), 1).cpu().numpy()
        for p, (r, near, steep_frac) in zip(prob, meta):
            pos_p = p[positive_classes].sum(0)[near]
            rock_p = p[rock_classes].sum(0)[near] if rock_classes else pos_p
            rows.append({**r, "n_near": int(near.sum()), "score": float(np.quantile(pos_p, 0.9)) if near.any() else np.nan,
                         "rock_score": float(np.quantile(rock_p, 0.9)) if near.any() else np.nan, "heuristic_steep_frac": steep_frac})
        batch.clear()
        meta.clear()

    for i, w in gt.iterrows():
        crop = cloud.crop(w["x"], w["y"], block_size_m / 2)
        if len(crop["xyz"]) < 64:
            rows.append({"gt_id": w["gt_id"], "label": int(w["label"]), "name": w["name"], "n_near": 0, "score": np.nan, "rock_score": np.nan, "heuristic_steep_frac": np.nan})
            continue
        rng = np.random.default_rng(i)
        idx, local, _ = voxel_block(crop["xyz"], w["x"], w["y"], spec.voxel_m, 40000, rng)
        sel = rng.choice(len(idx), spec.num_points, replace=len(idx) < spec.num_points)
        loc = local[sel]
        near = np.hypot(loc[:, 0], loc[:, 1]) <= radius_m
        d_all = np.hypot(crop["xyz"][:, 0] - w["x"], crop["xyz"][:, 1] - w["y"]) <= radius_m
        steep_frac = float(crop["steep"][d_all].mean()) if d_all.any() else np.nan
        inten = (crop["intensity"][idx][sel] / max(np.percentile(crop["intensity"], 99), 1.0)).clip(0, 1)[:, None]
        item = make_item(loc * spec.coord_scale, np.zeros(len(loc), np.int64), inten.astype(np.float32), keys, tf)
        batch.append(item)
        meta.append(({"gt_id": w["gt_id"], "label": int(w["label"]), "name": w["name"]}, near, steep_frac))
        if len(batch) == batch_size:
            flush()
    flush()
    df = pd.DataFrame(rows)
    ok = df["score"].notna()
    y = df.loc[ok, "label"].to_numpy()
    metrics = {
        "gt/n": int(ok.sum()),
        "gt/n_climb": int((y == 1).sum()),
        "gt/n_nc": int((y == 0).sum()),
        "gt/auroc_climb_vs_nc": _safe(roc_auc_score, y, df.loc[ok, "score"]),
        "gt/ap_climb": _safe(average_precision_score, y, df.loc[ok, "score"]),
        "gt/auroc_rock_score": _safe(roc_auc_score, y, df.loc[ok, "rock_score"]),
        "gt/mean_score_climb": float(df.loc[ok & (df["label"] == 1), "score"].mean()),
        "gt/mean_score_nc": float(df.loc[ok & (df["label"] == 0), "score"].mean()),
    }
    h = ok & df["heuristic_steep_frac"].notna()
    metrics["gt/auroc_heuristic_steep_frac"] = _safe(roc_auc_score, df.loc[h, "label"], df.loc[h, "heuristic_steep_frac"])
    return metrics, df
