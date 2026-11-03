#!/usr/bin/env python
"""Step 3c - build the fine-tuning dataset: verified positives + verified hard negatives.

Input: ``data/routes/verified_routes.csv`` (from ``s3_collect_routes.py``):
OSM route starts matched to Mountain Project + the hand-gathered Horseshoe
Canyon Ranch waypoints. For each area this script

1. makes sure LiDAR exists around every route (downloads missing tiles and
   computes normals / cliff points for them),
2. snaps route starts onto the cliff (positives),
3. generates negatives (``--strategies``; see ``crag/negatives.py``) and
   verifies them (exclusion radius, documentation coverage, geometry),
4. cuts one block per sample, assigns spatial train/val/test splits
   (``--test-areas``, default Horseshoe Canyon Ranch) and writes the test
   regions to ``registry/heldout_regions.json`` so pretraining never sees them,
5. validates the band strategy against the hand labels at Horseshoe Canyon
   (precision vs documentation coverage), writes a GeoJSON for visual review
   and a review sheet, and logs everything to W&B.

    python scripts/s3_build_finetune_set.py
    python scripts/s3_build_finetune_set.py --name v1 --strategies handgathered_nc,band,stepback --aug squash,layback
    python scripts/s3_build_finetune_set.py --areas red-river-gorge,horseshoe-canyon-ranch --wandb online

Output: ``data/finetune/<name>/`` with ``index.csv`` (one row per sample
block), ``samples.csv``, ``positives_all.csv``, ``report.json``,
``samples.geojson``, ``review_negatives.csv`` and the block arrays.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from crag import finetune_data, registry, tiles, wandb_utils
from crag.cli import parse_list, setup_logging
from crag.crop import AreaCloud
from crag.groundtruth import load_waypoints
from crag.negatives import NegativeConfig, build_area_samples, validate_on_handgathered
from crag.paths import AREAS_JSON, PRETRAIN_DIR, ROUTES_DIR
from crag.pseudolabels import PseudoLabelConfig, area_pins, process_area

log = logging.getLogger("build_finetune")


def geojson(samples: pd.DataFrame, path: Path) -> None:
    feats = [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(r.lon, 7), round(r.lat, 7)]},
         "properties": {"sample_id": str(r.sample_id), "label": int(r.label), "source": r.source, "confidence": float(r.confidence), "split": r.split, "slug": r.slug}}
        for r in samples.itertuples(index=False)
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="v1", help="dataset name -> data/finetune/<name>")
    ap.add_argument("--routes", type=Path, default=ROUTES_DIR / "verified_routes.csv")
    ap.add_argument("--registry", type=Path, default=AREAS_JSON)
    ap.add_argument("--areas", help="comma-separated slugs (default: every area with verified routes)")
    ap.add_argument("--strategies", default="handgathered_nc,band,stepback,far_cliff")
    ap.add_argument("--aug", default="", help="augmentation negatives from positive blocks: squash,layback")
    ap.add_argument("--band-min-m", type=float, default=10.0)
    ap.add_argument("--band-max-m", type=float, default=40.0)
    ap.add_argument("--min-coverage", type=float, default=0.6)
    ap.add_argument("--block-size-m", type=float, default=20.0)
    ap.add_argument("--group-m", type=float, default=150.0, help="samples closer than this share a split group")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--test-areas", default="horseshoe-canyon-ranch")
    ap.add_argument("--val-areas", default="")
    ap.add_argument("--no-download", action="store_true", help="use only LiDAR already on disk")
    ap.add_argument("--no-validate", action="store_true", help="skip the Horseshoe Canyon precision check")
    ap.add_argument("--workers", type=int, default=4)
    wandb_utils.add_cli(ap, default="disabled")
    args = ap.parse_args()
    setup_logging()

    reg = registry.load(args.registry)
    routes = pd.read_csv(args.routes)
    slugs = parse_list(args.areas) or sorted(routes["slug"].unique())
    cfg = NegativeConfig(band_min_m=args.band_min_m, band_max_m=args.band_max_m, min_coverage=args.min_coverage)
    strategies = tuple(parse_list(args.strategies) or ())
    out = finetune_data.dataset_dir(args.name)
    out.mkdir(parents=True, exist_ok=True)
    pl_cfg = PseudoLabelConfig()
    all_pos, all_neg, known_by_slug, crs_by_slug = [], [], {}, {}
    for area in registry.areas(reg, slugs, enabled_only=False):
        slug = area["slug"]
        r = routes[routes["slug"] == slug]
        if r.empty:
            continue
        if not (area.get("usgs") or {}).get("datasets"):
            log.warning("%s: %d routes but no USGS LiDAR coverage - skipped", slug, len(r))
            continue
        crs = registry.area_crs(area)
        crs_by_slug[slug] = crs
        if not args.no_download:
            plan = tiles.plan_tiles_for_points(area, r[["lon", "lat"]].to_numpy(), radius_m=args.band_max_m + 60, tile_size_m=pl_cfg.tile_size_m)
            res = tiles.download_area(area, plan, workers=args.workers)
            log.info("%s: LiDAR tiles %s", slug, pd.Series([x["status"] for x in res]).value_counts().to_dict())
            process_area(area, pl_cfg)  # normals / cliff points for any new tile (incremental)
        osm_all = ROUTES_DIR / slug / "osm_routes.csv"
        known = [r.loc[r["label"] == 1, ["lon", "lat"]].to_numpy()]
        if osm_all.exists():
            o = pd.read_csv(osm_all)
            if len(o):
                known.append(o[["lon", "lat"]].to_numpy())
        known_ll = np.vstack(known)
        known_by_slug[slug] = known_ll
        cov_path = ROUTES_DIR / slug / "coverage.csv"
        coverage = dict(pd.read_csv(cov_path)[["wall", "coverage"]].values) if cov_path.exists() else {}
        bi_path = PRETRAIN_DIR / slug / "index.csv"
        block_index = pd.read_csv(bi_path) if bi_path.exists() else None
        _, pins = area_pins(area, crs)
        pos, neg = build_area_samples(AreaCloud(slug), r.reset_index(drop=True), known_ll, crs, coverage, cfg, block_index, pins, strategies)
        pos["slug"], neg["slug"] = slug, slug
        all_pos.append(pos)
        all_neg.append(neg)
        log.info("%s: %d/%d positives snapped, negatives %s", slug, int((pos["status"] == "ok").sum()), len(pos), neg["source"].value_counts().to_dict() if len(neg) else {})

    pos_all = pd.concat(all_pos, ignore_index=True)
    pos_all.to_csv(out / "positives_all.csv", index=False)
    ok = pos_all[pos_all["status"] == "ok"].copy()
    P = pd.DataFrame({
        "sample_id": ok["route_uid"], "slug": ok["slug"], "label": 1, "source": ok["source"], "confidence": np.where(ok["match_type"] == "fuzzy", 0.9, 1.0),
        "x": ok["snap_x"], "y": ok["snap_y"], "lon": ok["snap_lon"], "lat": ok["snap_lat"], "base_z": ok["base_z"], "top_z": ok["top_z"],
        "out_dx": ok["out_dx"], "out_dy": ok["out_dy"], "name": ok["name"], "mp_route_id": ok["mp_route_id"], "match_type": ok["match_type"],
    })
    neg_all = pd.concat(all_neg, ignore_index=True) if all_neg else pd.DataFrame()
    if len(neg_all):
        neg_all["sample_id"] = [f"{s}:neg:{src}:{i}" for i, (s, src) in enumerate(zip(neg_all["slug"], neg_all["source"]))]
    samples = pd.concat([P, neg_all], ignore_index=True)
    samples["group"] = finetune_data.assign_groups(samples, args.group_m)
    samples["split"] = finetune_data.assign_splits(samples, args.val_frac, args.test_frac, tuple(parse_list(args.test_areas) or ()), tuple(parse_list(args.val_areas) or ()))
    samples.to_csv(out / "samples.csv", index=False)
    finetune_data.write_heldout_regions(samples[~samples["slug"].isin([a["slug"] for a in reg["areas"] if a.get("split") == "holdout"])])

    index = finetune_data.build_blocks(samples, out, args.block_size_m, pl_cfg.voxel_m, known_lonlat_by_slug=known_by_slug,
                                       aug_kinds=tuple(parse_list(args.aug) or ()), crs_by_slug=crs_by_slug)
    index.to_csv(out / "index.csv", index=False)
    for split, g in index[~index["source"].str.startswith("aug_")].groupby("split"):
        if g["label"].nunique() < 2:
            log.warning("split %r has only label-%d samples - adjust --test-areas / --val-frac / --group-m", split, g["label"].iloc[0])
    empty = index[index["n_center"] == 0]
    if len(empty):
        log.warning("%d sample blocks have no labelled cylinder points (dropped from training by the loss mask): %s", len(empty), empty["source"].value_counts().to_dict())
    geojson(index.drop_duplicates("sample_id"), out / "samples.geojson")

    # review sheet: a stratified random sample of negatives for a human look (imagery / site visit)
    negs = index[index["label"] == 0]
    if len(negs):
        rev = pd.concat([g.sample(min(len(g), 15), random_state=0) for _, g in negs.groupby("source")])
        rev = rev[["sample_id", "slug", "source", "confidence", "lat", "lon"]].assign(
            google_maps=lambda d: [f"https://www.google.com/maps/@{a:.6f},{b:.6f},60m/data=!3m1!1e3" for a, b in zip(d["lat"], d["lon"])], review_label="", notes="")
        rev.to_csv(out / "review_negatives.csv", index=False)

    report = {
        "name": args.name,
        "negative_config": asdict(cfg),
        "strategies": list(strategies),
        "aug": args.aug,
        "positives_snapped": int(len(ok)),
        "positives_total": int(len(pos_all)),
        "snap_status": pos_all["status"].value_counts().to_dict(),
        "samples_by_split_label": index.groupby(["split", "label"]).size().rename("n").reset_index().to_dict("records"),
        "samples_by_source": index["source"].value_counts().to_dict(),
        "samples_by_area": index.groupby("slug")["label"].agg(["count", "sum"]).rename(columns={"sum": "positives"}).reset_index().to_dict("records"),
        "negative_features_by_source": (neg_all.groupby("source")[[c for c in ("cliff_height_m", "n_steep", "multi_frac", "dist_to_route_m", "confidence") if c in neg_all]].mean().round(3).reset_index().to_dict("records") if len(neg_all) else []),
    }
    if not args.no_validate and (Path(PRETRAIN_DIR) / finetune_data_hand_slug() / "stage").exists():
        hand_area = next((a for a in reg["areas"] if a["slug"] == finetune_data_hand_slug()), None)
        if hand_area:
            v = validate_on_handgathered(AreaCloud(hand_area["slug"]), load_waypoints(to_work=registry.area_crs(hand_area).to_work), registry.area_crs(hand_area), cfg, repeats=5)
            report["band_validation_horseshoe"] = v
            log.info("band-negative precision vs documentation coverage (Horseshoe Canyon):\n%s", pd.DataFrame(v["by_coverage"]).to_string(index=False))
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(index.groupby(["split", "source"]).size().unstack(0).fillna(0).astype(int).to_string())
    log.info("fine-tuning dataset: %s (%d sample blocks)", out, len(index))

    run = wandb_utils.init(args.wandb, args.wandb_project, job_type="data", name=f"finetune-dataset-{args.name}", config={**asdict(cfg), "strategies": list(strategies), "aug": args.aug})
    if run is not None:
        import wandb

        run.log({"finetune_dataset/by_split_source": wandb.Table(dataframe=index.groupby(["split", "source", "label"]).size().rename("n").reset_index())})
        if "band_validation_horseshoe" in report:
            run.log({"finetune_dataset/band_precision_vs_coverage": wandb.Table(dataframe=pd.DataFrame(report["band_validation_horseshoe"]["by_coverage"]))})
        run.summary.update({"positives": int((index["label"] == 1).sum()), "negatives": int((index["label"] == 0).sum())})
        run.finish()


def finetune_data_hand_slug() -> str:
    return "horseshoe-canyon-ranch"


if __name__ == "__main__":
    main()
