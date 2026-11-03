#!/usr/bin/env python
"""Step 4 - compare fine-tuning runs (reads every runs/finetune/*/eval_best.json).

Model selection uses the **val** columns; the **test** columns (Horseshoe
Canyon Ranch with its on-site Climb / Nc labels, plus held-out crag groups)
are reported once the strategy is chosen.

    python scripts/s4_compare.py
    python scripts/s4_compare.py --wandb online
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from crag import wandb_utils
from crag.paths import RUNS_DIR

COLS = ["sample_auroc", "sample_ap", "sample_bal_acc", "sample_auroc_hard", "loss", "iou_climbable"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RUNS_DIR / "finetune")
    ap.add_argument("--ckpt", default="best")
    wandb_utils.add_cli(ap, default="disabled")
    args = ap.parse_args()

    rows = []
    for f in sorted(args.runs.glob(f"*/eval_{args.ckpt}.json")):
        r = json.loads(f.read_text())
        row = {"run": f.parent.name, "strategy": r.get("strategy"), "epoch": r.get("epoch")}
        for split in ("val", "test"):
            for c in COLS:
                row[f"{split}_{c}"] = r.get(split, {}).get(c)
        rows.append(row)
    if not rows:
        sys.exit(f"no eval_{args.ckpt}.json under {args.runs} - run s4_eval_finetune.py first")
    df = pd.DataFrame(rows).sort_values("val_sample_auroc", ascending=False)
    out = args.runs / "comparison.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    print(f"\nselected by val AUROC: {df.iloc[0]['run']}  (wrote {out})")
    run = wandb_utils.init(args.wandb, args.wandb_project, job_type="analysis", name="finetune-comparison", group="finetune-eval")
    if run is not None:
        import wandb

        run.log({"finetune/comparison": wandb.Table(dataframe=df)})
        run.finish()


if __name__ == "__main__":
    main()
