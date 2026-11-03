#!/usr/bin/env python
"""Step 4 - evaluate a fine-tuned checkpoint (best by default) on val and test.

Per-point metrics (loss, IoU) and per-sample metrics (AUROC, AP, balanced
accuracy, F1, AUROC on hard negatives only, accuracy per sample source),
overall and per area, plus a predictions CSV for maps / error analysis.

    python scripts/s4_eval_finetune.py --run runs/finetune/full_discriminative
    python scripts/s4_eval_finetune.py --run runs/finetune/linear_probe --splits test --wandb online

Writes ``<run>/eval_<ckpt>.json`` and ``<run>/predictions_<ckpt>.csv``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from crag import finetune, models, wandb_utils
from crag.cli import setup_logging
from crag.finetune_data import FT_CLASSES
from crag.paths import PROJECT_DIR, add_import_paths
from crag.train import cfg_get, run_eval

add_import_paths()
from openpoints.utils import EasyConfig  # noqa: E402

log = logging.getLogger("eval_finetune")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    wandb_utils.add_cli(ap, default="disabled")
    args = ap.parse_args()
    setup_logging()

    run = args.run if args.run.is_absolute() else PROJECT_DIR / args.run
    ck = torch.load(run / args.ckpt, map_location="cpu", weights_only=False)
    cfg = EasyConfig()
    cfg.update(ck["cfg"])
    model = models.build_model(cfg.model)
    model.load_state_dict(ck["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = finetune.build_datasets(cfg)
    ign = int(cfg_get(cfg, "ignore_index", -1))
    results = {"checkpoint": str(run / args.ckpt), "epoch": ck.get("epoch"), "strategy": cfg_get(cfg, "strategy"), "train_best_val_loss": ck.get("best_val_loss")}
    preds = []
    for split in [s.strip() for s in args.splits.split(",")]:
        ds = data[split]
        if len(ds) == 0:
            log.warning("split %s empty", split)
            continue
        hook = finetune.SampleScoreHook(ds, args.threshold)
        res = run_eval(model, DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers), FT_CLASSES, ign, device, hook=hook)
        res.pop("confusion", None)
        p = hook.last_predictions
        preds.append(p)
        res["per_area"] = {slug: finetune.sample_metrics(g, args.threshold) for slug, g in p.groupby("slug")}
        results[split] = res
        log.info("%s: point loss %.4f  IoU(climbable) %.3f | sample AUROC %.3f  AP %.3f  bal.acc %.3f  hard-neg AUROC %s",
                 split, res["loss"], res.get("iou_climbable", float("nan")), res.get("sample_auroc", float("nan")), res.get("sample_ap", float("nan")),
                 res.get("sample_bal_acc", float("nan")), round(res["sample_auroc_hard"], 3) if "sample_auroc_hard" in res else "n/a")
        log.info("  per-source accuracy: %s", {k[4:]: round(v, 3) for k, v in res.items() if k.startswith("acc_")})
    stem = Path(args.ckpt).stem
    if preds:
        pd.concat(preds).to_csv(run / f"predictions_{stem}.csv", index=False)
    (run / f"eval_{stem}.json").write_text(json.dumps(results, indent=1, default=float))
    log.info("wrote %s", run / f"eval_{stem}.json")
    wb = wandb_utils.init(args.wandb, args.wandb_project, name=f"{run.name}-eval", job_type="eval", group="finetune-eval", config={"run": run.name, "strategy": results["strategy"]})
    if wb is not None:
        wb.summary.update({f"{s}/{k}": v for s in ("val", "test") if s in results for k, v in results[s].items() if isinstance(v, (int, float))})
        wb.finish()


if __name__ == "__main__":
    main()
