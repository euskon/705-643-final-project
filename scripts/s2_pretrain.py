#!/usr/bin/env python
"""Step 2 - pretrain an OpenPoints backbone on the pseudolabelled blocks.

    # PointNeXt-S from the S3DIS checkpoint (default), tracked in W&B:
    python scripts/s2_pretrain.py --cfg cfgs/pretrain/pointnext-s.yaml

    # the second backbone (PointNet++ from its S3DIS checkpoint):
    python scripts/s2_pretrain.py --cfg cfgs/pretrain/pointnet2.yaml

    # any config value can be overridden as key=value:
    python scripts/s2_pretrain.py --cfg cfgs/pretrain/pointnext-s.yaml \\
        batch_size=4 grad_accum_steps=8 dataset.num_points=16384 epochs=200

    # quick end-to-end smoke test (no W&B, a few steps):
    python scripts/s2_pretrain.py --cfg cfgs/pretrain/pointnext-s.yaml --smoke

Outputs go to ``runs/pretrain/<run-name>/`` (default run name = config name):
``best.pt`` (saved whenever val loss improves by > checkpoint.min_delta),
``backbone.pt`` (encoder+decoder of best.pt), ``last.pt`` (resume point),
``metrics.jsonl``, ``config.json``, ``summary.json``. Re-running the same
command resumes from ``last.pt`` (``resume=none`` to start over).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crag import pretrain, wandb_utils
from crag.cli import parse_list, setup_logging
from crag.paths import RUNS_DIR
from crag.train import Trainer, cfg_get, seed_everything

log = logging.getLogger("pretrain")

SMOKE = [
    "epochs=2", "max_steps=6", "limit_val_batches=3", "dataset.samples_per_epoch=64",
    "dataset.max_val_blocks=24", "batch_size=4", "val_batch_size=4", "grad_accum_steps=2",
    "dataloader.num_workers=2", "log_every=2", "warmup_epochs=0", "resume=none",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default="cfgs/pretrain/pointnext-s.yaml")
    ap.add_argument("--run-name", help="run directory / W&B name (default: config file name)")
    ap.add_argument("--areas", help="restrict to these area slugs (default: every processed area)")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline (implies --wandb disabled unless given)")
    wandb_utils.add_cli(ap, default="online")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()
    setup_logging()

    overrides = (SMOKE if args.smoke else []) + list(args.overrides)
    cfg = pretrain.load_cfg(args.cfg, overrides)
    if args.smoke and "--wandb" not in sys.argv:
        args.wandb = "disabled"
    seed_everything(int(cfg_get(cfg, "seed", 42)))
    run_name = args.run_name or Path(cfg.cfg_path).stem + ("-smoke" if args.smoke else "")
    run_dir = RUNS_DIR / "pretrain" / run_name

    data = pretrain.build_datasets(cfg, parse_list(args.areas))
    if len(data["train"]) == 0:
        sys.exit("training split is empty - process more areas (s1_pseudolabel.py) or use dataset.split_mode=tile")
    if len(data["val"]) == 0:
        sys.exit("validation split is empty - give some areas split=val in registry/areas.json, or dataset.split_mode=tile")
    weights = pretrain.class_weights(data["train"].class_counts(), cfg_get(cfg, "class_weights"))
    log.info("class weights: %s", None if weights is None else dict(zip(pretrain.CLASS_NAMES, weights.round(3).tolist())))
    model, report = pretrain.build_model_from(cfg)
    cfg.pretrained_report = report
    trainer = Trainer(
        cfg, model, data["train"], data["val"], pretrain.CLASS_NAMES, run_dir,
        train_sampler=pretrain.make_sampler(cfg, data["train"]), class_weights=weights,
    )
    summary = trainer.fit(args.wandb, args.wandb_project, run_name=run_name, job_type="pretrain")
    log.info("done: %s", summary)


if __name__ == "__main__":
    main()
