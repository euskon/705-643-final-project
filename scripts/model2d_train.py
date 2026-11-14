#!/usr/bin/env python
"""2D model, step 2 - train one ResNet run on the top-down images, then score it.

    python scripts/model2d_train.py --strategy full_discriminative
    python scripts/model2d_train.py --strategy scratch data.input=inbetween seed=1
    python scripts/model2d_train.py --strategy linear_probe data.position_channel=false
    python scripts/model2d_train.py --strategy full_discriminative epochs=2 --name smoke   # quick pipeline check

Config = ``cfgs/model2d/default.yaml`` + ``cfgs/model2d/<strategy>.yaml`` +
``key=value`` overrides. Writes ``runs/model2d/<group>/<name>/`` (group from
``pretrain``: ``no_pseudolabel`` or ``pseudolabel``): ``config.yaml``,
``log.csv`` (one row per epoch), ``best.pt`` (lowest validation loss), then
``predictions_best.csv`` and ``eval_best.json`` (``scripts/model2d_eval.py``).
The default name is ``<input>-<strategy>[-nopos]-s<seed>``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crag import model2d_train as m2
from crag.cli import setup_logging

log = logging.getLogger("model2d_train")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="full_discriminative", help="cfgs/model2d/<strategy>.yaml (or a path)")
    ap.add_argument("--name", help="run name -> runs/model2d/<group>/<name>")
    ap.add_argument("--no-eval", action="store_true", help="skip scoring val/test after training")
    ap.add_argument("overrides", nargs="*", help="key=value, e.g. data.input=inbetween seed=1")
    args = ap.parse_args()
    setup_logging()

    cfg = m2.load_cfg(args.strategy, args.overrides)
    run_dir = m2.run_dir(cfg, args.name)
    res = m2.train(cfg, run_dir)
    log.info("best epoch %d (val loss %.4f) after %d epochs -> %s", res["best_epoch"], res["best_val_loss"], res["epochs_run"], run_dir)
    if not args.no_eval:
        from scripts.model2d_eval import evaluate_run  # noqa: E402

        evaluate_run(run_dir)


if __name__ == "__main__":
    main()
