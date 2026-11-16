"""Score a segmentation checkpoint on the hand-gathered Horseshoe Canyon Ranch waypoints.

The waypoints are the saved, judged positions (``presnapped``; plan 4): the wall
check only confirms them. Each is then centred on its wall foot and scored
exactly like a fine-tuning sample: the same block, the same face strip and the
same 4 x 12 point pick (``crag.finetune_data``), the run's own channels and
scale. Score = mean "positive" probability over the picked points.

Reported: AUROC / AP of Climb vs Nc, also without the judges' ``unresolved``
points and on high + medium confidence only, plus the surface-normal
heuristic (share of steep points) as a reference.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from crag import registry
from crag.channels import CHANNELS, point_channels
from crag.crop import AreaCloud, voxel_block
from crag.datasets import BlockSpec, make_item, resample_keep
from crag.finetune_data import StripConfig, face_strip_mask, pick_points
from crag.groundtruth import load_waypoints
from crag.negatives import centre_on_foot, snap_routes, waypoint_frame
from crag.paths import add_import_paths
from crag.train import model_forward

add_import_paths()
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
    cloud = AreaCloud(slug)
    reg_area = next((a for a in registry.load()["areas"] if a["slug"] == slug), None)
    # waypoints in the same metric CRS as the area's tiles
    crs = registry.area_crs(reg_area) if reg_area else None
    gt = load_waypoints(to_work=crs.to_work if crs else None)
    gt = gt[gt["label"] >= 0].reset_index(drop=True)
    n_all = len(gt)
    strip = StripConfig()
    if crs is not None:
        chk = snap_routes(cloud, waypoint_frame(gt), crs)
        ok = (chk["status"] == "ok").to_numpy()
        gt = gt[ok].assign(x=chk.loc[ok, "snap_x"].to_numpy(), y=chk.loc[ok, "snap_y"].to_numpy()).reset_index(drop=True)
    n_off_wall = n_all - len(gt)
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
            rows.append({**r, "n_near": int(near.sum()), "score": float(pos_p.mean()) if near.any() else np.nan,
                         "rock_score": float(rock_p.mean()) if near.any() else np.nan, "heuristic_steep_frac": steep_frac})
        batch.clear()
        meta.clear()

    for i, w in gt.iterrows():
        base = {"gt_id": w["gt_id"], "label": int(w["label"]), "name": w["name"], "judge_result": w.get("judge_result", ""), "judge_confidence": w.get("judge_confidence", "")}
        f = centre_on_foot(cloud, None, np.array([w["x"], w["y"]]))
        crop = cloud.crop(f["foot_x"], f["foot_y"], block_size_m / 2) if f["status"] == "ok" else {"xyz": np.zeros((0, 3))}
        if len(crop["xyz"]) < 64:
            rows.append({**base, "n_near": 0, "score": np.nan, "rock_score": np.nan, "heuristic_steep_frac": np.nan})
            continue
        rng = np.random.default_rng(i)
        idx, local, _ = voxel_block(crop["xyz"], f["foot_x"], f["foot_y"], spec.voxel_m, 40000, rng)
        xyz = crop["xyz"][idx]
        m, _ = face_strip_mask(xyz, crop["steep"][idx], (f["foot_x"], f["foot_y"]), f["base_z"], (f["out_dx"], f["out_dy"]), strip, wall_height_m=f.get("wall_height_m"))
        own = np.flatnonzero(m)
        own = own[pick_points(xyz[own, 2] - f["base_z"], strip, rng)]
        keep = np.zeros(len(idx), bool)
        keep[own] = True
        sel = resample_keep(keep, spec.num_points, rng)
        loc, near = local[sel], keep[sel]
        d_all = np.hypot(crop["xyz"][:, 0] - f["foot_x"], crop["xyz"][:, 1] - f["foot_y"]) <= radius_m
        steep_frac = float(crop["steep"][d_all].mean()) if d_all.any() else np.nan
        ch = point_channels(crop, idx[sel], CHANNELS)
        cols = {k: ch[:, j].astype(np.float32) for j, k in enumerate(CHANNELS)}
        item = make_item(loc * spec.coord_scale, np.zeros(len(loc), np.int64), cols, keys, tf)
        if int(item["x"].shape[1]) != int(sum(1 if k not in ("pos",) else 3 for k in keys)):
            raise AssertionError("channel count differs from the run's feature_keys")
        batch.append(item)
        meta.append((base, near, steep_frac))
        if len(batch) == batch_size:
            flush()
    flush()
    df = pd.DataFrame(rows)
    ok = df["score"].notna()
    y = df.loc[ok, "label"].to_numpy()
    metrics = {
        "gt/n": int(ok.sum()),
        "gt/n_off_wall": int(n_off_wall),
        "gt/n_climb": int((y == 1).sum()),
        "gt/n_nc": int((y == 0).sum()),
        "gt/auroc_climb_vs_nc": _safe(roc_auc_score, y, df.loc[ok, "score"]),
        "gt/ap_climb": _safe(average_precision_score, y, df.loc[ok, "score"]),
        "gt/auroc_rock_score": _safe(roc_auc_score, y, df.loc[ok, "rock_score"]),
        "gt/mean_score_climb": float(df.loc[ok & (df["label"] == 1), "score"].mean()),
        "gt/mean_score_nc": float(df.loc[ok & (df["label"] == 0), "score"].mean()),
    }
    for tag, m in (("no_unresolved", df["judge_result"].astype(str) != "unresolved"), ("high_medium", df["judge_confidence"].astype(str) != "low")):
        mm = ok & m
        metrics[f"gt/auroc_climb_vs_nc_{tag}"] = _safe(roc_auc_score, df.loc[mm, "label"].to_numpy(), df.loc[mm, "score"])
        metrics[f"gt/n_{tag}"] = int(mm.sum())
    h = ok & df["heuristic_steep_frac"].notna()
    metrics["gt/auroc_heuristic_steep_frac"] = _safe(roc_auc_score, df.loc[h, "label"], df.loc[h, "heuristic_steep_frac"])
    return metrics, df
