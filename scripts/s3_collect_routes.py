#!/usr/bin/env python
"""Step 3b - collect route-level GPS (OSM / openclimbing.org) and cross-reference Mountain Project.

For every registry area: query OSM for route starts inside the area's bbox,
match them to the area's Mountain Project routes (``crag/crossref.py``), and
keep only locations of documented routes. The hand-gathered Horseshoe Canyon
Ranch waypoints are added as their own source (Climb = verified positive,
Nc = verified negative).

    python scripts/s3_collect_routes.py                    # all registry areas
    python scripts/s3_collect_routes.py --areas red-river-gorge --keep-unmatched

Outputs (``data/routes/``):

* ``<slug>/osm_routes.csv`` - every OSM route feature in the bbox
* ``<slug>/matched.csv``    - the same + MP match columns
* ``<slug>/coverage.csv``   - per MP wall: documented vs located routes
* ``verified_routes.csv``   - all areas: the route locations used for fine-tuning
* ``collection_summary.csv``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from crag import crossref, mp, openclimbing, registry, wandb_utils
from crag.cli import parse_list, setup_logging
from crag.groundtruth import load_waypoints
from crag.paths import AREAS_JSON, ROUTES_DIR

log = logging.getLogger("collect_routes")
HAND_SLUG = "horseshoe-canyon-ranch"

COLUMNS = ["route_uid", "slug", "source", "label", "name", "lat", "lon", "lat2", "lon2", "grade", "length_m",
           "mp_route_id", "mp_name", "mp_wall", "match_type", "is_boulder", "osm_type", "osm_id"]


def handgathered_rows() -> pd.DataFrame:
    gt = load_waypoints()
    gt = gt[gt["label"] >= 0]
    out = pd.DataFrame(
        {
            "route_uid": "hand:" + gt["gt_id"],
            "slug": HAND_SLUG,
            "source": np.where(gt["label"] == 1, "handgathered", "handgathered_nc"),
            "label": gt["label"].astype(int),
            "name": gt["name"].where(gt["name"] != "?", None),
            "lat": gt["lat"],
            "lon": gt["lon"],
            "lat2": np.nan,
            "lon2": np.nan,
            "grade": None,
            "length_m": np.nan,
            "mp_route_id": pd.to_numeric(gt["route_id"], errors="coerce"),
            "mp_name": None,
            "mp_wall": None,
            "match_type": np.where(gt["route_id"].notna(), "hand_mp_url", "hand"),
            "is_boulder": False,
            "osm_type": None,
            "osm_id": None,
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=AREAS_JSON)
    ap.add_argument("--areas", help="comma-separated slugs (default: all enabled)")
    ap.add_argument("--keep-unmatched", action="store_true", help="also keep OSM routes with no MP match (flagged)")
    ap.add_argument("--include-boulders", action="store_true")
    ap.add_argument("--max-pin-dist-m", type=float, default=2000.0)
    ap.add_argument("--fuzzy-min", type=float, default=0.88)
    ap.add_argument("--no-handgathered", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="re-query Overpass")
    wandb_utils.add_cli(ap, default="disabled")
    args = ap.parse_args()
    setup_logging()

    reg = registry.load(args.registry)
    verified, summary, failed = [], [], []
    for area in registry.areas(reg, parse_list(args.areas)):
        slug = area["slug"]
        if not area.get("bbox"):
            log.warning("%s: no bbox (run s1_build_registry.py)", slug)
            continue
        out_dir = ROUTES_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        pins = np.array([[c["lon"], c["lat"]] for c in area.get("crags", []) if not c.get("default_pin")] or [[area["lon"], area["lat"]]])
        try:
            osm, osm_crags = openclimbing.routes_in_bbox(area["bbox"], refresh=args.refresh, points=pins)
        except Exception as exc:  # one unreachable area must not lose the others; re-run retries it
            log.error("%s: OSM query failed (%s) - skipped; re-run to retry (finished queries are cached)", slug, exc)
            failed.append(slug)
            summary.append({"slug": slug, "status": "failed"})
            continue
        osm.to_csv(out_dir / "osm_routes.csv", index=False)
        osm_crags.to_csv(out_dir / "osm_crags.csv", index=False)
        row = {"slug": slug, "osm_routes": len(osm)}
        if osm.empty:
            summary.append(row)
            continue
        if not args.include_boulders:
            osm = osm[~osm["is_boulder"]].reset_index(drop=True)
        row["osm_non_boulder"] = len(osm)
        try:
            mpr = mp.load_routes(area["mp_id"], area_pin=(area["lat"], area["lon"]))
        except Exception as exc:
            log.warning("%s: MP routes unavailable (%s) - cannot cross-reference", slug, exc)
            mpr = pd.DataFrame(columns=["route", "route_id", "wall", "location", "rating", "length_m", "lat", "lon"])
        matched = crossref.match_routes(osm, mpr, args.max_pin_dist_m, args.fuzzy_min)
        matched.to_csv(out_dir / "matched.csv", index=False)
        if len(mpr):
            crossref.crag_coverage(matched, mpr).to_csv(out_dir / "coverage.csv", index=False)
        vc = matched["match_type"].value_counts().to_dict()
        row.update({f"match_{k}": v for k, v in vc.items()})
        row["match_rate"] = round(1 - vc.get("none", 0) / len(matched), 3) if len(matched) else float("nan")
        keep = matched if args.keep_unmatched else matched[matched["match_type"] != "none"]
        v = pd.DataFrame(
            {
                "route_uid": "osm:" + keep["osm_type"] + "/" + keep["osm_id"].astype(str),
                "slug": slug,
                "source": "osm",
                "label": 1,
                "name": keep["name"],
                "lat": keep["lat"],
                "lon": keep["lon"],
                "lat2": keep["lat2"],
                "lon2": keep["lon2"],
                "grade": keep["grade"].fillna(keep["mp_rating"]),
                "length_m": keep["length_m"].fillna(keep["mp_length_m"]),
                "mp_route_id": keep["mp_route_id"],
                "mp_name": keep["mp_name"],
                "mp_wall": keep["mp_wall"],
                "match_type": keep["match_type"],
                "is_boulder": keep["is_boulder"],
                "osm_type": keep["osm_type"],
                "osm_id": keep["osm_id"],
            }
        )
        verified.append(v)
        row["verified"] = len(v)
        summary.append(row)
        log.info("%s: %s", slug, row)
    if not args.no_handgathered:
        h = handgathered_rows()
        verified.append(h)
        summary.append({"slug": HAND_SLUG + " (hand)", "verified": int((h["label"] == 1).sum()), "verified_negatives": int((h["label"] == 0).sum())})
    allv = pd.concat(verified, ignore_index=True)[COLUMNS] if verified else pd.DataFrame(columns=COLUMNS)
    allv = allv.drop_duplicates("route_uid").reset_index(drop=True)
    ROUTES_DIR.mkdir(parents=True, exist_ok=True)
    allv.to_csv(ROUTES_DIR / "verified_routes.csv", index=False)
    sm = pd.DataFrame(summary).fillna(0)
    sm.to_csv(ROUTES_DIR / "collection_summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(sm.sort_values("verified", ascending=False).to_string(index=False))
    log.info("verified route locations: %d positives, %d negatives -> %s", int((allv["label"] == 1).sum()), int((allv["label"] == 0).sum()), ROUTES_DIR / "verified_routes.csv")
    run = wandb_utils.init(args.wandb, args.wandb_project, job_type="data", name="route-collection")
    if run is not None:
        import wandb

        run.log({"routes/summary": wandb.Table(dataframe=sm.astype(str)), "routes/match_types": wandb.Table(dataframe=allv["match_type"].value_counts().rename_axis("type").reset_index())})
        run.summary.update({"verified_positives": int((allv["label"] == 1).sum()), "verified_negatives": int((allv["label"] == 0).sum())})
        run.finish()
    if failed:
        log.error("%d area(s) failed and are missing from verified_routes.csv: %s", len(failed), ",".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
