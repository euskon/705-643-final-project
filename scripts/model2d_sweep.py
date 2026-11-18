#!/usr/bin/env python
"""2D model - train and score every input x strategy x seed, then print the comparison.

    python scripts/model2d_sweep.py                                       # 2 inputs x 3 strategies x 3 seeds = 18 runs
    python scripts/model2d_sweep.py --inputs gradient --strategies full_discriminative --seeds 0,1,2
    python scripts/model2d_sweep.py --extra data.position_channel=true    # the same grid with the position channel
    python scripts/model2d_sweep.py --extra pretrain=pseudolabel          # -> runs/model2d/v3/pseudolabel/ (next step)
    python scripts/model2d_sweep.py --data v4                             # data/raster/v4 -> runs/model2d/v4/

Runs that already have ``eval_best.json`` are skipped, so an interrupted sweep
resumes where it stopped. Each run is its own process (a crash stops only that run).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crag import model2d_train as m2
from crag.cli import parse_list

SCRIPTS = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default="gradient,inbetween")
    ap.add_argument("--strategies", default="full_discriminative,linear_probe,scratch")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data", default="v3", help="dataset version: data/raster/<data> -> runs/model2d/<data>/")
    ap.add_argument("--extra", nargs="*", default=[], help="key=value overrides for every run")
    args = ap.parse_args()

    failed = []
    for inp in parse_list(args.inputs):
        for strat in parse_list(args.strategies):
            for seed in parse_list(args.seeds):
                ov = [f"data.name={args.data}", f"data.input={inp}", f"seed={seed}", *args.extra]
                cfg = m2.load_cfg(strat, ov)
                name = m2.run_name(cfg)
                if (m2.run_dir(cfg) / "eval_best.json").exists():
                    print(f"skip {name} (done)")
                    continue
                print(f"=== {name}", flush=True)
                if subprocess.call([sys.executable, str(SCRIPTS / "model2d_train.py"), "--strategy", strat, *ov]) != 0:
                    failed.append(name)
    subprocess.call([sys.executable, str(SCRIPTS / "model2d_compare.py"), "--data", args.data])
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
