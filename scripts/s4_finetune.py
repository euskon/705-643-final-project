#!/usr/bin/env python
"""Step 4 - fine-tune a pretrained backbone with a fresh 2-class head (climbable / not).

    # one strategy (see cfgs/finetune/*.yaml and docs/step4-finetuning-strategy.md):
    python scripts/s4_finetune.py --strategy full_discriminative

    # another backbone / dataset / any override:
    python scripts/s4_finetune.py --strategy linear_probe \\
        init.arch=runs/pretrain/pointnet2 init.weights=runs/pretrain/pointnet2/backbone.pt dataset.name=v1

    # every strategy, then a comparison table:
    bash scripts/s4_sweep.sh

    # smoke test (a few steps, no W&B):
    python scripts/s4_finetune.py --strategy linear_probe --smoke

Outputs ``runs/finetune/<run-name>/`` (default ``<strategy>``): ``best.pt``
(saved when val loss improves by > checkpoint.min_delta), ``last.pt``,
``metrics.jsonl`` (per-point and per-sample validation metrics each epoch).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crag import finetune, wandb_utils
from crag.cli import setup_logging
from crag.finetune_data import FT_CLASSES
from crag.paths import CFG_DIR, RUNS_DIR
from crag.pretrain import load_cfg
from crag.train import Trainer, cfg_get, seed_everything

log = logging.getLogger("finetune")

SMOKE = [
    "epochs=2", "max_steps=6", "limit_val_batches=4", "dataset.samples_per_epoch=48", "batch_size=4", "val_batch_size=4",
    "grad_accum_steps=1", "dataloader.num_workers=2", "log_every=2", "warmup_epochs=0", "resume=none",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="full_discriminative", help="name of a cfgs/finetune/<strategy>.yaml (or a path)")
    ap.add_argument("--run-name")
    ap.add_argument("--smoke", action="store_true")
    wandb_utils.add_cli(ap, default="online")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()
    setup_logging()

    path = Path(args.strategy)
    if not path.suffix:
        path = CFG_DIR / "finetune" / f"{args.strategy}.yaml"
    cfg = load_cfg(path, (SMOKE if args.smoke else []) + list(args.overrides))
    if args.smoke and "--wandb" not in sys.argv:
        args.wandb = "disabled"
    cfg.strategy = path.stem
    seed_everything(int(cfg_get(cfg, "seed", 42)))
    run_name = args.run_name or path.stem + ("-smoke" if args.smoke else "")
    run_dir = RUNS_DIR / "finetune" / run_name

    data = finetune.build_datasets(cfg)
    for split in ("train", "val"):
        if len(data[split]) == 0:
            sys.exit(f"{split} split is empty - rebuild the dataset (s3_build_finetune_set.py) with other --test-areas/--val-frac")
    model, report = finetune.build_model(cfg)
    cfg.pretrained_report = report
    trainer = Trainer(
        cfg, model, data["train"], data["val"], FT_CLASSES, run_dir,
        train_sampler=finetune.balanced_sampler(data["train"], int(cfg_get(cfg, "dataset.samples_per_epoch", 4000)), int(cfg_get(cfg, "seed", 42))),
        val_hook=finetune.SampleScoreHook(data["val"]),
    )
    summary = trainer.fit(args.wandb, args.wandb_project, run_name=run_name, job_type="finetune")
    log.info("done: %s", summary)


if __name__ == "__main__":
    main()
