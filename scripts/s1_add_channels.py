#!/usr/bin/env python
"""Step 1e - per-point input channels for the pretraining blocks (plan 3).

1. For every tile of the chosen areas, write ``stage/<tile>.hag.npy`` (height above the
   lowest bare-earth ground within 10 m, from the tile + its 10 m border). Existing files
   of the right length are skipped unless ``--force``. The pseudolabels are untouched.
2. With ``--rebuild-blocks``: rewrite the blocks of each area **in place**, one area at a
   time, storing the channels of ``crag.channels.PRETRAIN_CHANNELS`` (labels and block
   cuts unchanged; ``index.csv`` is compared before / after).

    python scripts/s1_add_channels.py --workers 3                     # every processed area (incl. Horseshoe)
    python scripts/s1_add_channels.py --areas red-river-gorge --rebuild-blocks
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from crag import registry
from crag.cli import parse_list, setup_logging
from crag.paths import AREAS_JSON, PRETRAIN_DIR
from crag.pseudolabels import CLASSES, PseudoLabelConfig, _TileCache, add_hag_tile, rebuild_blocks

log = logging.getLogger("add_channels")


def run_area(area: dict, force: bool, rebuild: bool) -> dict:
    setup_logging()
    cfg = PseudoLabelConfig()
    slug = area["slug"]
    stage = PRETRAIN_DIR / slug / "stage"
    tids = sorted(p.name[: -len(".npz")] for p in stage.glob("E*_N*.npz"))
    cache = _TileCache(slug, cfg.tile_size_m)
    c = Counter(add_hag_tile(slug, t, cache, cfg, force) for t in tids)
    out = {"slug": slug, "tiles": len(tids), **c}
    if rebuild:
        ip = PRETRAIN_DIR / slug / "index.csv"
        before = pd.read_csv(ip) if ip.exists() else None
        out.update(rebuild_blocks(area, cfg))
        after = pd.read_csv(ip)
        if before is not None:
            cols = ["tile", "block", "start", "stop"] + [f"n_{k}" for k in CLASSES]
            b0, a0 = before.sort_values(["tile", "block"])[cols].to_numpy(), after.sort_values(["tile", "block"])[cols].to_numpy()
            same = len(before) == len(after) and (b0 == a0).all()
            out["index_unchanged"] = bool(same)
            if not same:
                log.warning("%s: index.csv changed after the rebuild (%d -> %d blocks)", slug, len(before), len(after))
    log.info("%s: %s", slug, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=AREAS_JSON)
    ap.add_argument("--areas", help="comma-separated slugs (default: every area with stage files)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rebuild-blocks", action="store_true")
    args = ap.parse_args()
    setup_logging()
    reg = registry.load(args.registry)
    wanted = parse_list(args.areas)
    areas = [a for a in reg["areas"] if (PRETRAIN_DIR / a["slug"] / "stage").exists() and (not wanted or a["slug"] in wanted)]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run_area, a, args.force, args.rebuild_blocks): a["slug"] for a in areas}
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except Exception:
                log.exception("%s failed", futs[f])
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if len(rows) < len(areas):
        sys.exit(1)
    if args.rebuild_blocks:
        from crag.channels import PRETRAIN_CHANNELS, read_channels_json, write_channels_json

        every = [a["slug"] for a in reg["areas"] if (PRETRAIN_DIR / a["slug"] / "index.csv").exists()]
        stale = [s for s in every if read_channels_json(PRETRAIN_DIR / s) != PRETRAIN_CHANNELS]
        if stale:
            log.warning("areas without channel blocks yet (data/pretrain/channels.json not written): %s", stale)
        else:
            write_channels_json(PRETRAIN_DIR, PRETRAIN_CHANNELS)
            log.info("all %d areas carry channels -> %s", len(every), PRETRAIN_DIR / "channels.json")


if __name__ == "__main__":
    main()
