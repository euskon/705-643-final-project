#!/usr/bin/env python
"""Table of the V4 check runs (``runs/finetune/chk-*``), averaged over seeds, next to the leak
test's group B baseline (plain wall geometry). Writes ``runs/finetune/v4_checks.csv``.

    python scripts/v4_summary.py [--prefix chk-] [--dataset v4]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from crag.finetune_data import dataset_dir
from crag.paths import RUNS_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="chk-")
    ap.add_argument("--dataset", default="v4")
    args = ap.parse_args()
    rows = []
    for f in sorted((RUNS_DIR / "finetune").glob(f"{args.prefix}*/eval_best.json")):
        r = json.loads(f.read_text())
        t = r.get("test", {})
        rows.append({"run": re.sub(r"-s\d+$", "", f.parent.name), "seed": f.parent.name.rsplit("-s", 1)[-1], "epoch": r.get("epoch"),
                     "val": r.get("val", {}).get("sample_auroc"), "rrg_test": t.get("per_area", {}).get("red-river-gorge", {}).get("sample_auroc"),
                     "hcr_climb_vs_nc": t.get("hcr_climb_vs_nc", {}).get("sample_auroc"), "hcr_high_medium": t.get("hcr_climb_vs_nc_high_medium", {}).get("sample_auroc"),
                     "hcr_pair_acc": t.get("hcr_climb_vs_nc", {}).get("pair_acc"), "hcr_auroc_near": t.get("hcr_climb_vs_nc", {}).get("auroc_near")})
    if not rows:
        sys.exit("no check runs evaluated yet")
    df = pd.DataFrame(rows)
    mean = df.groupby("run").mean(numeric_only=True)
    mean["n_seeds"] = df.groupby("run").size()
    lt = dataset_dir(args.dataset) / "leak_test.json"
    if lt.exists():
        b = json.loads(lt.read_text())["group_B"]
        base = {"val": b.get("val", {}).get("auroc"), "rrg_test": b.get("rrg_test", {}).get("auroc"), "hcr_climb_vs_nc": b.get("hcr_climb_vs_nc", {}).get("auroc")}
        mean.loc["BASELINE wall geometry (group B)", list(base)] = list(base.values())
        for k, v in base.items():
            if v is not None:
                mean[f"{k}_gap_over_B"] = mean[k] - v
    out = RUNS_DIR / "finetune" / "v4_checks.csv"
    mean.round(3).to_csv(out)
    with pd.option_context("display.width", 220):
        print(df.round(3).to_string(index=False))
        print("\nmean over seeds:\n", mean.round(3).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
