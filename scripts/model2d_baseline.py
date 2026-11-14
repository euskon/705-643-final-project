#!/usr/bin/env python
"""2D model - geometric baselines with no training, scored exactly like the trained runs.

A model has to beat these to show it learned more than "steep = climbable".
Each score is computed on the bare-earth height of the image, within the 3D
cylinder radius (2.5 m) of the centre:

* ``slope_mean``: mean steepness (degrees / 90);
* ``slope_max``: steepest cell (degrees / 90);
* ``relief``: highest minus lowest height in the whole 20 m image (metres).

    python scripts/model2d_baseline.py                 # data/raster/v3 -> runs/model2d/baselines/<name>/

Writes ``predictions_best.csv`` and ``eval_best.json`` per baseline, like
``scripts/model2d_eval.py``, so ``scripts/model2d_compare.py`` lists them too.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from crag import model2d_train as m2
from crag.cli import setup_logging
from crag.model2d_raster import CHANNELS, slope_deg

log = logging.getLogger("model2d_baseline")
CYLINDER_M = 2.5  # crag.finetune_data.build_blocks(cyl_m=...)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="v3", help="data/raster/<name>")
    args = ap.parse_args()
    setup_logging()

    images, index, meta = m2.load_data({"data": {"name": args.data}})
    index = index[index["split"].isin(["val", "test"])].reset_index(drop=True)
    res_m, n = meta["config"]["res_m"], meta["shape"][-1]
    dem = np.asarray(images[index["row"].to_numpy(), CHANNELS.index("dem")])
    slope = slope_deg(dem, res_m)
    c = (np.arange(n) + 0.5 - n / 2) * res_m
    centre = np.hypot(*np.meshgrid(c, c)) <= CYLINDER_M
    scores = {
        "slope_mean": slope[:, centre].mean(1) / 90.0,
        "slope_max": slope[:, centre].max(1) / 90.0,
        "relief": dem.max((1, 2)) - dem.min((1, 2)),
    }
    for name, s in scores.items():
        run_dir = m2.BASELINES_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)
        res = m2.write_eval(run_dir, m2.predictions(index, s), {"run": f"baseline-{name}", "pretrain": "baseline", "strategy": "baseline", "input": name})
        h = res.get("horseshoe", {})
        log.info("%-12s val AUROC %.3f | test AUROC %.3f hard %.3f | Horseshoe AUROC %.3f, hand Climb vs Nc %.3f", name,
                 res["splits"]["val"].get("sample_auroc", np.nan), res["splits"]["test"].get("sample_auroc", np.nan),
                 res["splits"]["test"].get("sample_auroc_hard", np.nan), h.get("auroc", np.nan), h.get("hand_auroc", np.nan))


if __name__ == "__main__":
    main()
