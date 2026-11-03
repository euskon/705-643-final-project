#!/usr/bin/env python
"""Step 1d - turn downloaded LiDAR tiles into pseudolabelled pretraining blocks.

For every area with tiles in ``data/lidar/<slug>/`` this runs the existing
labeling heuristics (normals -> cliff points -> MP route cylinders) and writes
per-point classes, cut into voxel-downsampled blocks, to
``data/pretrain/<slug>/`` (see ``crag/pseudolabels.py`` for the classes).
Areas whose tiles and config are unchanged are skipped, so this is safe to
re-run after downloading more tiles. A dataset summary is written to
``data/pretrain/summary.csv`` and (optionally) logged to W&B.

    python scripts/s1_pseudolabel.py                         # all areas
    python scripts/s1_pseudolabel.py --areas horseshoe-canyon-ranch --force
    python scripts/s1_pseudolabel.py --workers 4 --wandb online
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from crag import registry, wandb_utils
from crag.cli import parse_list, setup_logging
from crag.paths import AREAS_JSON, PRETRAIN_DIR
from crag.pseudolabels import CLASSES, PseudoLabelConfig, process_area

log = logging.getLogger("pseudolabel")


def _run(area: dict, cfg: PseudoLabelConfig, force: bool) -> dict:
    setup_logging()
    return process_area(area, cfg, force=force)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=AREAS_JSON)
    ap.add_argument("--areas", help="comma-separated slugs (default: all enabled)")
    ap.add_argument("--force", action="store_true", help="recompute even if up to date")
    ap.add_argument("--workers", type=int, default=1, help="areas processed in parallel (each needs a few GB RAM)")
    ap.add_argument("--block-size-m", type=float, default=20.0)
    ap.add_argument("--voxel-m", type=float, default=0.25)
    ap.add_argument("--tile-size-m", type=float, default=200.0, help="must match s1_download_lidar.py")
    wandb_utils.add_cli(ap, default="disabled")
    args = ap.parse_args()
    setup_logging()

    cfg = PseudoLabelConfig(block_size_m=args.block_size_m, voxel_m=args.voxel_m, tile_size_m=args.tile_size_m)
    reg = registry.load(args.registry)
    areas = registry.areas(reg, parse_list(args.areas))
    summaries = []
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            futs = {pool.submit(_run, a, cfg, args.force): a["slug"] for a in areas}
            for fut in as_completed(futs):
                try:
                    summaries.append(fut.result())
                except Exception:
                    log.exception("%s failed", futs[fut])
    else:
        for a in areas:
            try:
                summaries.append(process_area(a, cfg, force=args.force))
            except Exception:
                log.exception("%s failed", a["slug"])
    # summary over *every* processed area on disk, not just this invocation
    rows = []
    for a in reg["areas"]:
        st = PRETRAIN_DIR / a["slug"] / "state.json"
        if st.exists():
            import json

            rows.append(json.loads(st.read_text())["summary"])
    df = pd.DataFrame(rows)
    if len(df):
        PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PRETRAIN_DIR / "summary.csv", index=False)
        print(df.to_string(index=False))
        tot = df[[f"pts_{c}" for c in CLASSES] + ["pts_ignore"]].sum()
        print("\nclass balance (points):")
        print((tot / tot.sum()).round(4).to_string())
    run = wandb_utils.init(args.wandb, args.wandb_project, job_type="dataset", name="pretrain-dataset", config={"pseudolabel": cfg.digest(), "block_size_m": cfg.block_size_m, "voxel_m": cfg.voxel_m})
    if run is not None and len(df):
        import wandb

        run.log({"pretrain_dataset/areas": wandb.Table(dataframe=df)})
        run.summary.update({"blocks": int(df["blocks"].sum()), "points": int(df["points"].sum()), "areas": len(df), "route_cylinders": int(df["route_cylinders_ok"].sum())})
        run.finish()


if __name__ == "__main__":
    main()
