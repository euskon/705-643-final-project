#!/usr/bin/env python
"""2D model, step 3 - score a trained run (its best checkpoint) on val and test.

    python scripts/model2d_eval.py --run runs/model2d/no_pseudolabel/gradient-full_discriminative-s0

Writes ``<run>/predictions_best.csv`` (one score per sample, the same columns
as the 3D ``predictions_best.csv``) and ``<run>/eval_best.json``: per-sample
metrics per split and per area (the 3D metric code), the cutoff with the best
validation balanced accuracy, and the Horseshoe Canyon table (AUROC with 95%
range, hand-gathered climbable vs not-climbable AUROC, balanced accuracy at
0.5 and at the validation cutoff).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from crag import model2d_train as m2
from crag.cli import setup_logging

log = logging.getLogger("model2d_eval")


def evaluate_run(run_dir: Path, device: str = "cuda") -> dict:
    cfg, pipe, model = m2.load_run(run_dir, device)
    images, index, _ = m2.load_data(cfg)
    parts = []
    for split in ("val", "test"):
        sub = index[index["split"] == split].reset_index(drop=True)
        x = torch.from_numpy(np.ascontiguousarray(images[sub["row"].to_numpy()])).to(device)
        parts.append(m2.predictions(sub, m2.predict(pipe, model, x)))
    res = m2.write_eval(run_dir, pd.concat(parts, ignore_index=True), {"run": run_dir.name, "pretrain": cfg.get("pretrain", "none"), "strategy": cfg["strategy"],
                        "input": cfg["data"]["input"], "position_channel": cfg["data"]["position_channel"], "seed": cfg["seed"], "epoch": cfg["best_epoch"]})
    s, h = res["splits"], res.get("horseshoe", {})
    log.info("%s | val AUROC %.3f hard %.3f | test AUROC %.3f hard %.3f | Horseshoe AUROC %.3f (%.3f-%.3f), hand Climb vs Nc %.3f",
             run_dir.name, s["val"].get("sample_auroc", np.nan), s["val"].get("sample_auroc_hard", np.nan), s["test"].get("sample_auroc", np.nan),
             s["test"].get("sample_auroc_hard", np.nan), h.get("auroc", np.nan), h.get("auroc_lo", np.nan), h.get("auroc_hi", np.nan), h.get("hand_auroc", np.nan))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    setup_logging()
    evaluate_run(args.run)


if __name__ == "__main__":
    main()
