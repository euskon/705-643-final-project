#!/usr/bin/env python
"""2D model - one table of every scored 2D run and baseline, next to the committed 3D results.

    python scripts/model2d_compare.py                  # v3 -> runs/model2d/v3/comparison.csv
    python scripts/model2d_compare.py --data v4        # v4 -> runs/model2d/v4/comparison.csv

Rows: every ``runs/model2d/<data>/<group>/<run>/eval_best.json``, then the 2D
configurations averaged over seeds (mean +- std), then the 3D runs from
``docs/results/<data>/`` (read only). Columns follow the 3D tables: validation
and test AUROC (all samples and hard negatives only) and the Horseshoe Canyon
numbers. Runs are sorted by validation AUROC: choose on validation, report test.

V4 adds the columns of the V4 3D table: ``rrg_auroc`` (test AUROC on Red River
Gorge only), ``hcr_hand_auroc_hm`` (hand points without the judges' low-confidence
ones) and ``pair_acc`` (close Climb / Nc pairs). V4's 3D table has no all-areas test
AUROC, so those cells stay empty for 3D. V3 and V4 numbers are not comparable
(different samples and labels).

The ``pretrain`` column says what came before fine-tuning, from the run's
folder: ``none`` (``no_pseudolabel/``: ImageNet weights, or random init for
scratch), ``pseudolabel`` (``pseudolabel/``: + pretraining on the LiDAR terrain
pseudolabels) or ``baseline`` (``baselines/``: no training at all).

2D and 3D share one split: the 3D numbers come from the teammate's build of
the fine-tuning set, and the 2D images are rasterized from that same build's
``data/finetune/<data>/index.csv`` (committed on dev-zach), so every sample and its
train/val/test assignment match. Do not rebuild it locally before rasterizing:
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

RRG = "red-river-gorge"
COLS = ["val_auroc", "val_auroc_hard", "test_auroc", "test_auroc_hard", "rrg_auroc", "hcr_auroc", "hcr_hand_auroc",
        "hcr_hand_auroc_hm", "hcr_bal_acc_valcut", "pair_acc"]
GROUP_TO_PRETRAIN = {**{g: stage for stage, g in m2.PRETRAIN_GROUPS.items()}, m2.BASELINES: "baseline"}


def rows_2d(data: str) -> pd.DataFrame:
    rows = []
    for p in sorted(m2.runs_root(data).glob("*/*/eval_best.json")):
        group = p.parent.parent.name
        if group not in GROUP_TO_PRETRAIN:
            continue
        r = json.loads(p.read_text())
        s, h = r["splits"], r.get("horseshoe", {})
        test = s.get("test", {})
        rows.append({
            "model": "2D", "pretrain": r.get("pretrain", GROUP_TO_PRETRAIN[group]),
            "run": r.get("run", p.parent.name), "input": r.get("input"), "strategy": r.get("strategy"),
            "position": r.get("position_channel"), "seed": r.get("seed"), "epoch": r.get("epoch"),
            "val_auroc": s.get("val", {}).get("sample_auroc"), "val_auroc_hard": s.get("val", {}).get("sample_auroc_hard"),
            "test_auroc": test.get("sample_auroc"), "test_auroc_hard": test.get("sample_auroc_hard"),
            "rrg_auroc": r.get("per_area", {}).get("test", {}).get(RRG, {}).get("sample_auroc"),
            "hcr_auroc": h.get("auroc"), "hcr_hand_auroc": h.get("hand_auroc"), "hcr_hand_auroc_hm": h.get("hand_auroc_high_medium"),
            "hcr_bal_acc_valcut": h.get("bal_acc_valcut"), "pair_acc": test.get("pair_acc"),
        })
    return pd.DataFrame(rows)


def rows_3d(data: str) -> pd.DataFrame:
    results = PROJECT_DIR / "docs" / "results" / data
    comp = results / "finetune_comparison.csv"
    if not comp.exists():
        return pd.DataFrame()
    c = pd.read_csv(comp)
    if "val_sample_auroc" not in c:  # V4 table: run, ep, val, val_hard, rrg, hcr (hand Climb vs Nc), hcr_hm, pair_acc, ...
        c = c.rename(columns={"ep": "epoch", "val": "val_auroc", "val_hard": "val_auroc_hard", "rrg": "rrg_auroc",
                              "hcr": "hcr_hand_auroc", "hcr_hm": "hcr_hand_auroc_hm"})
        return c[[k for k in ["run", "epoch", *COLS] if k in c]].assign(model="3D")
    c = c.rename(columns={"val_sample_auroc": "val_auroc", "val_sample_auroc_hard": "val_auroc_hard",
                          "test_sample_auroc": "test_auroc", "test_sample_auroc_hard": "test_auroc_hard"})
    c = c[["run", "strategy", "epoch", "val_auroc", "val_auroc_hard", "test_auroc", "test_auroc_hard"]].assign(model="3D")
    hcr = results / "horseshoe_confidence_and_cutoffs.txt"
    if hcr.exists():
        lines = hcr.read_text().splitlines()
        end = next((i for i, l in enumerate(lines[1:], 1) if not l.strip() or l.startswith("split")), len(lines))
        h = pd.read_csv(pd.io.common.StringIO("\n".join(lines[:end])), sep=r"\s+")
        h = h.rename(columns={"hcr": "hcr_auroc", "hard": "hcr_hand_auroc", "balacc_valcut": "hcr_bal_acc_valcut"})
        c = c.merge(h[["run", "hcr_auroc", "hcr_hand_auroc", "hcr_bal_acc_valcut"]], on="run", how="left")
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="v3", help="dataset version: runs/model2d/<data>/ and docs/results/<data>/")
    ap.add_argument("--out", type=Path, help="default: runs/model2d/<data>/comparison.csv")
    args = ap.parse_args()
    out = args.out or m2.runs_root(args.data) / "comparison.csv"
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)

    d2, d3 = rows_2d(args.data), rows_3d(args.data)
    if len(d2):
        d2 = d2.dropna(axis=1, how="all")
        print(f"2D runs and baselines on {args.data} (sorted by validation AUROC):")
        print(d2.sort_values("val_auroc", ascending=False).round(3).to_string(index=False))
        trained = d2[d2["strategy"] != "baseline"]
        if len(trained):
            cols = [c for c in COLS if c in trained]
            g = trained.groupby(["pretrain", "input", "strategy", "position"])[cols]
            agg = g.mean().round(3).astype(str) + " +- " + g.std().fillna(0).round(3).astype(str)
            agg.insert(0, "seeds", g.size())
            print("\n2D configurations, mean +- std over seeds:")
            print(agg.reset_index().to_string(index=False))
    if len(d3):
        print(f"\n3D runs (docs/results/{args.data}, same split as the 2D runs):")
        print(d3.sort_values("val_auroc", ascending=False).round(3).to_string(index=False))
    allr = pd.concat([d2, d3], ignore_index=True)
    if len(allr):
        out.parent.mkdir(parents=True, exist_ok=True)
        allr.to_csv(out, index=False)
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
