#!/usr/bin/env python
"""How many **real** (distinct) points each grouping stage of the model sees (plan 2, step 5).

Uses the model's own CUDA operations (furthest point sampling + ball query) on blocks built
exactly as in training (validation mode, so repeatable), with the radius, ``nsample`` and
strides of the model config. Reports the median and 10th percentile of distinct points per
ball, per stage, overall and per area; exits with an error if a stage's median is under its
target (stage 1: 12, later stages: 24).

    python scripts/check_neighbourhoods.py --cfg cfgs/pretrain/pointnext-s.yaml --dataset pretrain
    python scripts/check_neighbourhoods.py --cfg cfgs/finetune/default.yaml --arch cfgs/pretrain/pointnext-s.yaml --dataset finetune:v4
    python scripts/check_neighbourhoods.py --cfg cfgs/pretrain/pointnext-s.yaml --dataset pretrain dataset.coord_scale=0.16 dataset.num_points=8192
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from crag.paths import PROJECT_DIR, RUNS_DIR, add_import_paths
from crag.pretrain import block_spec, load_cfg
from crag.train import cfg_get

add_import_paths()
from openpoints.models.layers.group import ball_query  # noqa: E402
from openpoints.models.layers.subsample import furthest_point_sample  # noqa: E402


def stages(model_cfg) -> list[tuple[list[float], list[int], int]]:
    """[(radii, nsamples, stride)] per grouping stage, as the encoder builds them."""
    e = model_cfg.encoder_args
    out = []
    if "PointNext" in e.NAME:
        r = float(e.radius)
        for k, s in enumerate(e.strides):
            if s == 1:
                continue
            out.append(([r], [int(e.nsample)], int(s)))
            r *= 2  # radius_scaling 2 per downsampling stage
    else:
        rad = e.radius
        ns = e.num_samples
        for k, s in enumerate(e.strides):
            rk = rad[k] if isinstance(rad, (list, tuple)) else float(rad) * 2**k
            nk = ns[k] if isinstance(ns, (list, tuple)) else int(ns)
            rk = list(rk) if isinstance(rk, (list, tuple)) else [float(rk)]
            nk = list(nk) if isinstance(nk, (list, tuple)) else [int(nk)] * len(rk)
            out.append((rk, nk, int(s)))
    return out


@torch.no_grad()
def count(pos: torch.Tensor, st) -> list[list[np.ndarray]]:
    """Distinct points per ball for every stage (and every radius of an MSG stage)."""
    res, xyz = [], pos
    for radii, nss, stride in st:
        n_next = max(xyz.shape[1] // stride, 1)
        idx = furthest_point_sample(xyz.contiguous(), n_next).long()
        new = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        per = []
        for r, ns in zip(radii, nss):
            g = ball_query(r, ns, xyz.contiguous(), new.contiguous()).long()  # (B, M, ns) - padded with the first hit
            srt = torch.sort(g, -1).values
            per.append(((srt[..., 1:] != srt[..., :-1]).sum(-1) + 1).flatten().cpu().numpy())
        res.append(per)
        xyz = new
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--arch", help="model config (fine-tuning configs take it from init.arch)")
    ap.add_argument("--dataset", default="pretrain", help="pretrain | finetune:<name>")
    ap.add_argument("--n-blocks", type=int, default=200)
    ap.add_argument("--min-stage1", type=float, default=12)
    ap.add_argument("--min-later", type=float, default=24)
    ap.add_argument("--out", type=Path)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_cfg(args.cfg, args.overrides)
    model_cfg = load_cfg(args.arch).model if args.arch else cfg.model
    spec = block_spec(cfg)
    if args.dataset == "pretrain":
        from crag.datasets import BlockDataset, load_block_index

        index = load_block_index()
        index = index[index["split"] == "val"]
        k = max(args.n_blocks // index["slug"].nunique(), 2)
        ds_index = pd.concat([g.sample(min(len(g), k), random_state=0) for _, g in index.groupby("slug")])
        ds = BlockDataset(ds_index, spec, None, train=False)
    else:
        from crag import finetune

        cfg.dataset.name = args.dataset.split(":", 1)[1]
        idx = finetune.load_index(cfg)
        k = max(args.n_blocks // idx["slug"].nunique(), 2)
        idx = pd.concat([g.sample(min(len(g), k), random_state=0) for _, g in idx.groupby("slug")])
        ds = finetune.build_datasets(cfg, idx.assign(split="val"))["val"]
    st = stages(model_cfg)
    rows = []
    for i in range(len(ds)):
        it = ds[i]
        pos = (it["pos"] - it["pos"].min(0).values * torch.tensor([0, 0, 1.0])).unsqueeze(0).cuda()
        for k, per in enumerate(count(pos, st)):
            for j, c in enumerate(per):
                rows.append({"slug": ds.index.iloc[i]["slug"], "stage": k + 1, "radius_m": st[k][0][j] / spec.coord_scale, "median": float(np.median(c)), "p10": float(np.quantile(c, 0.1))})
    df = pd.DataFrame(rows)
    overall = df.groupby(["stage", "radius_m"])[["median", "p10"]].median().round(1)
    per_area = df.groupby(["slug", "stage"])["median"].median().unstack().round(1)
    print(f"coord_scale {spec.coord_scale}, num_points {spec.num_points}, {len(ds)} blocks")
    print(overall.to_string())
    print("\nper area (median distinct points per ball, by stage):")
    print(per_area.to_string())
    out = args.out or (RUNS_DIR / "checks" / f"neighbourhoods_{Path(args.cfg).stem}_{args.dataset.replace(':', '-')}_cs{spec.coord_scale}_n{spec.num_points}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"coord_scale": spec.coord_scale, "num_points": spec.num_points, "overall": overall.reset_index().to_dict("records"),
                               "per_area": per_area.reset_index().to_dict("records")}, indent=1))
    fail = [(s, r, m) for (s, r), m in overall["median"].items() if m < (args.min_stage1 if s == 1 else args.min_later)]
    if fail:
        print("\nBELOW TARGET:", fail)
        sys.exit(1)
    print("\nall stages at or above target")


if __name__ == "__main__":
    main()
