#!/usr/bin/env python
"""Step 2 - evaluate a pretrained backbone at its best checkpoint.

Reports, for the checkpoint's own config:

* per-point pseudolabel metrics (loss, OA, mIoU, per-class IoU/recall,
  confusion matrix) on the held-out **val** and **test** areas, overall and
  per area;
* a real-label check on the hand-gathered Horseshoe Canyon Ranch waypoints
  (never used in pretraining): AUROC of P(climbable) for Climb vs Nc, next
  to the surface-normal heuristic (needs that area's tiles + pseudolabels).

    python scripts/s2_eval_pretrain.py --run runs/pretrain/pointnext-s
    python scripts/s2_eval_pretrain.py --run runs/pretrain/pointnet2 --ckpt last.pt --no-gt
    python scripts/s2_eval_pretrain.py --run runs/pretrain/pointnext-s --wandb online

Writes ``<run>/eval_<ckpt>.json`` and ``<run>/gt_scores_<ckpt>.csv``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from crag import models, pretrain, wandb_utils
from crag.cli import setup_logging
from crag.datasets import BlockDataset
from crag.paths import PROJECT_DIR, add_import_paths
from crag.train import cfg_get, run_eval

add_import_paths()
from openpoints.utils import EasyConfig  # noqa: E402

log = logging.getLogger("eval_pretrain")


def load_checkpoint_model(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = EasyConfig()
    cfg.update(ck["cfg"])
    model = models.build_model(cfg.model)
    model.load_state_dict(ck["model"])
    return model, cfg, ck


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="run directory (runs/pretrain/<name>)")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--max-blocks", type=int, default=0, help="cap blocks per split (0 = config's max_val_blocks)")
    ap.add_argument("--no-gt", action="store_true", help="skip the hand-gathered ground-truth check")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    wandb_utils.add_cli(ap, default="disabled")
    args = ap.parse_args()
    setup_logging()

    run = args.run if args.run.is_absolute() else PROJECT_DIR / args.run
    ckpt = run / args.ckpt
    model, cfg, ck = load_checkpoint_model(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.max_blocks:
        cfg.dataset.max_val_blocks = args.max_blocks
    log.info("checkpoint %s: epoch %s, best val loss %.4f", ckpt, ck.get("epoch"), ck.get("best_val_loss", float("nan")))

    results = {"checkpoint": str(ckpt), "epoch": ck.get("epoch"), "train_best_val_loss": ck.get("best_val_loss")}
    data = pretrain.build_datasets(cfg)
    ign = int(cfg_get(cfg, "ignore_index", -1))
    for split in [s.strip() for s in args.splits.split(",")]:
        ds = data[split]
        if len(ds) == 0:
            log.warning("split %s is empty", split)
            continue
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
        res = run_eval(model, dl, pretrain.CLASS_NAMES, ign, device)
        results[split] = res
        log.info("%s: loss %.4f  OA %.3f  mIoU %.3f  %s", split, res["loss"], res["oa"], res["miou"],
                 "  ".join(f"{c}={res[f'iou_{c}']:.3f}" for c in pretrain.CLASS_NAMES))
        per_area = {}
        for slug, sub in ds.index.groupby("slug"):
            sds = BlockDataset(sub, ds.spec, cfg_get(cfg, "datatransforms"), train=False)
            r = run_eval(model, DataLoader(sds, batch_size=args.batch_size, num_workers=args.num_workers), pretrain.CLASS_NAMES, ign, device)
            per_area[slug] = {k: r[k] for k in ("loss", "oa", "miou", "iou_cliff", "iou_climbable")}
        results[f"{split}_per_area"] = per_area

    if not args.no_gt:
        from crag.gt_eval import score_waypoints

        try:
            gt_metrics, gt_df = score_waypoints(
                model, pretrain.block_spec(cfg), cfg_get(cfg, "datatransforms"),
                positive_classes=[pretrain.CLASS_NAMES.index("climbable")],
                rock_classes=[pretrain.CLASS_NAMES.index("cliff"), pretrain.CLASS_NAMES.index("climbable")],
            )
            results["ground_truth"] = gt_metrics
            gt_df.to_csv(run / f"gt_scores_{Path(args.ckpt).stem}.csv", index=False)
            log.info("ground truth: %s", {k: round(v, 3) if isinstance(v, float) else v for k, v in gt_metrics.items()})
        except Exception as exc:
            log.warning("ground-truth check skipped: %s", exc)

    out = run / f"eval_{Path(args.ckpt).stem}.json"
    out.write_text(json.dumps(results, indent=1, default=float))
    log.info("wrote %s", out)
    wb = wandb_utils.init(args.wandb, args.wandb_project, name=f"{run.name}-eval", job_type="eval", group="pretrain-eval", config={"run": run.name, "ckpt": args.ckpt})
    if wb is not None:
        flat = {f"{s}/{k}": v for s in ("val", "test") if s in results for k, v in results[s].items() if isinstance(v, (int, float))}
        flat.update(results.get("ground_truth", {}))
        wb.summary.update(flat)
        wb.finish()


if __name__ == "__main__":
    main()
