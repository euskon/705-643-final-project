#!/usr/bin/env python
"""2D model - one table of every scored 2D run and baseline, next to the committed 3D results.

    python scripts/model2d_compare.py
    python scripts/model2d_compare.py --out runs/model2d/comparison.csv

Rows: every ``runs/model2d/<group>/<run>/eval_best.json``, then the 2D
configurations averaged over seeds (mean +- std), then the 3D runs from
``docs/results/v3/`` (read only). Columns follow the 3D tables: validation
and test AUROC (all samples and hard negatives only) and the Horseshoe Canyon
numbers. Runs are sorted by validation AUROC: choose on validation, report test.

The ``pretrain`` column says what came before fine-tuning, from the run's
folder: ``none`` (``no_pseudolabel/``: ImageNet weights, or random init for
scratch), ``pseudolabel`` (``pseudolabel/``: + pretraining on the LiDAR terrain
pseudolabels) or ``baseline`` (``baselines/``: no training at all).

2D and 3D share one split: the 3D numbers come from the teammate's build of
fine-tuning set v3, and the 2D images are rasterized from that same build's
``data/finetune/v3/index.csv`` (committed on dev-zach), so every sample and its
train/val/test assignment match. Do not rebuild v3 locally before rasterizing:
a rebuild can snap some routes to different wall points and change splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from crag import model2d_train as m2
from crag.paths import PROJECT_DIR

RESULTS_3D = PROJECT_DIR / "docs" / "results" / "v3"
COLS = ["val_auroc", "val_auroc_hard", "test_auroc", "test_auroc_hard", "hcr_auroc", "hcr_hand_auroc", "hcr_bal_acc_valcut"]
GROUP_TO_PRETRAIN = {**{g: stage for stage, g in m2.PRETRAIN_GROUPS.items()}, m2.BASELINES_DIR.name: "baseline"}


def rows_2d() -> pd.DataFrame:
    rows = []
    for p in sorted(m2.RUNS_DIR.glob("*/*/eval_best.json")):
        group = p.parent.parent.name
        if group not in GROUP_TO_PRETRAIN:
            continue
        r = json.loads(p.read_text())
        s, h = r["splits"], r.get("horseshoe", {})
        rows.append({
            "model": "2D", "pretrain": r.get("pretrain", GROUP_TO_PRETRAIN[group]),
            "run": r.get("run", p.parent.name), "input": r.get("input"), "strategy": r.get("strategy"),
            "position": r.get("position_channel"), "seed": r.get("seed"), "epoch": r.get("epoch"),
            "val_auroc": s.get("val", {}).get("sample_auroc"), "val_auroc_hard": s.get("val", {}).get("sample_auroc_hard"),
            "test_auroc": s.get("test", {}).get("sample_auroc"), "test_auroc_hard": s.get("test", {}).get("sample_auroc_hard"),
            "hcr_auroc": h.get("auroc"), "hcr_hand_auroc": h.get("hand_auroc"), "hcr_bal_acc_valcut": h.get("bal_acc_valcut"),
        })
    return pd.DataFrame(rows)


def rows_3d() -> pd.DataFrame:
    comp = RESULTS_3D / "finetune_comparison.csv"
    if not comp.exists():
        return pd.DataFrame()
    c = pd.read_csv(comp).rename(columns={"val_sample_auroc": "val_auroc", "val_sample_auroc_hard": "val_auroc_hard",
                                          "test_sample_auroc": "test_auroc", "test_sample_auroc_hard": "test_auroc_hard"})
    c = c[["run", "strategy", "epoch", "val_auroc", "val_auroc_hard", "test_auroc", "test_auroc_hard"]].assign(model="3D")
    hcr = RESULTS_3D / "horseshoe_confidence_and_cutoffs.txt"
    if hcr.exists():
        lines = hcr.read_text().splitlines()
        end = next((i for i, l in enumerate(lines[1:], 1) if not l.strip() or l.startswith("split")), len(lines))
        h = pd.read_csv(pd.io.common.StringIO("\n".join(lines[:end])), sep=r"\s+")
        h = h.rename(columns={"hcr": "hcr_auroc", "hard": "hcr_hand_auroc", "balacc_valcut": "hcr_bal_acc_valcut"})
        c = c.merge(h[["run", "hcr_auroc", "hcr_hand_auroc", "hcr_bal_acc_valcut"]], on="run", how="left")
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=m2.RUNS_DIR / "comparison.csv")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    d2, d3 = rows_2d(), rows_3d()
    if len(d2):
        print("2D runs and baselines (sorted by validation AUROC):")
        print(d2.sort_values("val_auroc", ascending=False).round(3).to_string(index=False))
        trained = d2[d2["strategy"] != "baseline"]
        if len(trained):
            g = trained.groupby(["pretrain", "input", "strategy", "position"])[COLS]
            agg = g.mean().round(3).astype(str) + " +- " + g.std().fillna(0).round(3).astype(str)
            agg.insert(0, "seeds", g.size())
            print("\n2D configurations, mean +- std over seeds:")
            print(agg.reset_index().to_string(index=False))
    if len(d3):
        print("\n3D runs (docs/results/v3, same split as the 2D runs):")
        print(d3.sort_values("val_auroc", ascending=False).round(3).to_string(index=False))
    allr = pd.concat([d2, d3], ignore_index=True)
    if len(allr):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        allr.to_csv(args.out, index=False)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
