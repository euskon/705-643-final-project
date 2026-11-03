#!/usr/bin/env python
"""Step 1b - build / complete ``registry/areas.json``.

Seeds the registry with the top-N areas from ``registry/mp_ranked.csv``
(output of ``s1_rank_areas.py``) plus the always-kept holdout areas, then
fills every *missing* field of every entry:

* MP GPS + popularity (cached area page),
* crags (name + coordinates of every MP sub-area pin that holds routes,
  from the route-finder export - one cached request per 1000 routes),
* bounding box enclosing the crags (+ margin) and the area's UTM zone,
* the ranked USGS 3DEP EPT dataset(s) covering the crags,
* a deterministic pretraining split.

Existing values are never overwritten (use ``--recompute FIELD,...`` to
force), so hand edits survive. The file is saved after every area, so the
script can be interrupted and re-run.

    python scripts/s1_build_registry.py                 # top 50 + holdout
    python scripts/s1_build_registry.py --areas red-river-gorge --recompute usgs
    python scripts/s1_build_registry.py --no-routes     # skip MP route exports (fast)

To add an area by hand, append ``{"slug": ..., "name": ..., "mp_id": ...}``
to ``registry/areas.json`` and re-run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from crag import ept_index, mp, registry
from crag.cli import parse_list, setup_logging
from crag.geo import haversine_m, utm_epsg
from crag.paths import AREAS_JSON, REGISTRY_DIR

log = logging.getLogger("build_registry")

HOLDOUT_MP_IDS = {105903004: "hand-gathered ground truth (Horseshoe Canyon Ranch)"}
DESCRIPTION = (
    "Climbing-area registry for pseudolabelled pretraining. One entry per area; "
    "add {slug, name, mp_id} and run scripts/s1_build_registry.py to fill the rest. "
    "See docs/registry-schema.md."
)


def seed_from_ranked(reg: dict, ranked_csv: Path, top: int) -> None:
    if not ranked_csv.exists():
        log.warning("%s not found - run s1_rank_areas.py first (only existing entries will be completed)", ranked_csv)
        return
    ranked = pd.read_csv(ranked_csv)
    ranked = ranked[ranked["rank"].notna()].sort_values("rank")
    chosen = pd.concat([ranked.head(top), ranked[ranked["mp_id"].isin(HOLDOUT_MP_IDS)]]).drop_duplicates("mp_id")
    have = {a.get("mp_id") for a in reg["areas"]}
    slugs = {a["slug"] for a in reg["areas"]}
    for r in chosen.itertuples():
        if int(r.mp_id) in have:
            continue
        slug = registry.slugify(r.name)
        if slug in slugs:
            slug = f"{slug}-{registry.slugify(str(r.state))}"
        entry = {"slug": slug, "name": r.name, "mp_id": int(r.mp_id), "mp_rank": int(r.rank)}
        if int(r.mp_id) in HOLDOUT_MP_IDS:
            entry["split"] = "holdout"
            entry["notes"] = HOLDOUT_MP_IDS[int(r.mp_id)]
        reg["areas"].append(entry)
        slugs.add(slug)


def compute_bbox(area: dict, margin_m: float, max_crag_km: float) -> list[float]:
    crags = [c for c in area.get("crags", []) if not c.get("default_pin")]
    if crags:
        ll = np.array([[c["lon"], c["lat"]] for c in crags])
        d = haversine_m(area["lon"], area["lat"], ll[:, 0], ll[:, 1])
        far = d > max_crag_km * 1000
        if far.any():
            log.warning("%s: ignoring %d crag pins > %.0f km from the area pin for the bbox", area["slug"], far.sum(), max_crag_km)
        ll = ll[~far] if (~far).any() else np.array([[area["lon"], area["lat"]]])
    else:
        ll = np.array([[area["lon"], area["lat"]]])
        margin_m = max(margin_m, 2000.0)
    dlat = margin_m / 111_320.0
    dlon = margin_m / (111_320.0 * np.cos(np.radians(ll[:, 1].mean())))
    return [
        round(float(ll[:, 0].min() - dlon), 6),
        round(float(ll[:, 1].min() - dlat), 6),
        round(float(ll[:, 0].max() + dlon), 6),
        round(float(ll[:, 1].max() + dlat), 6),
    ]


def complete_area(area: dict, args, recompute: set[str]) -> None:
    need = lambda k: k in recompute or area.get(k) in (None, [], {})  # noqa: E731
    if area.get("mp_id") and (need("lat") or need("popularity") or need("mp_url")):
        info = mp.parse_area_page(mp.fetch_area_page(area["mp_id"]))
        area.setdefault("state", info["state"])
        if need("mp_url"):
            area["mp_url"] = info["url"] or mp.area_url(area["mp_id"])
        if need("lat"):
            area["lat"], area["lon"] = info["lat"], info["lon"]
        if need("popularity"):
            area["popularity"] = {
                "mp_page_views_total": info["page_views_total"],
                "mp_page_views_month": info["page_views_month"],
                "retrieved": dt.date.today().isoformat(),
            }
    if need("utm_epsg"):
        area["utm_epsg"] = utm_epsg(area["lon"], area["lat"])
    if need("crags") and not args.no_routes and area.get("mp_id"):
        routes = mp.load_routes(area["mp_id"], area_pin=(area["lat"], area["lon"]))
        crags = mp.crags_from_routes(routes)
        area["n_routes"] = int(len(routes))
        area["crags"] = [
            {
                "name": c.name,
                "lat": round(float(c.lat), 6),
                "lon": round(float(c.lon), 6),
                "n_routes": int(c.n_routes),
                "default_pin": bool(c.default_pin),
                "location": c.location,
            }
            for c in crags.itertuples()
        ]
    if need("bbox") or "crags" in recompute:
        area["bbox"] = compute_bbox(area, args.bbox_margin_m, args.max_crag_km)
    if need("usgs") or "crags" in recompute:
        from crag.tiles import crag_points

        pts = crag_points(area)
        ds = ept_index.rank_datasets(pts[:, :2], area["utm_epsg"], radius_m=args.crag_radius_m)
        area["usgs"] = {"datasets": ds, "selected": dt.date.today().isoformat()}
        if not ds:
            log.warning("%s: no USGS EPT coverage found", area["slug"])
    area.setdefault("enabled", True)
    area.setdefault("notes", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=AREAS_JSON)
    ap.add_argument("--ranked", type=Path, default=REGISTRY_DIR / "mp_ranked.csv")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--areas", help="comma-separated slugs to (re)process; default all")
    ap.add_argument("--recompute", default="", help="fields to overwrite, e.g. usgs,bbox,crags")
    ap.add_argument("--no-routes", action="store_true", help="don't fetch MP route exports (crags stay empty)")
    ap.add_argument("--crag-radius-m", type=float, default=150.0, help="LiDAR footprint radius around crags")
    ap.add_argument("--bbox-margin-m", type=float, default=500.0)
    ap.add_argument("--max-crag-km", type=float, default=60.0, help="crag pins further than this from the area pin are treated as mistakes")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()
    setup_logging()

    reg = registry.load(args.registry)
    reg["description"] = DESCRIPTION
    seed_from_ranked(reg, args.ranked, args.top)
    recompute = set(parse_list(args.recompute) or [])
    todo = registry.areas(reg, parse_list(args.areas), enabled_only=False)
    for i, area in enumerate(todo, 1):
        log.info("[%d/%d] %s", i, len(todo), area["slug"])
        try:
            complete_area(area, args, recompute)
        except Exception:
            log.exception("%s: failed (left incomplete; re-run to retry)", area["slug"])
        registry.save(reg, args.registry)  # checkpoint after every area
    registry.assign_splits(reg, args.val_frac, args.test_frac)
    registry.save(reg, args.registry)

    rows = [
        {
            "slug": a["slug"],
            "split": a.get("split"),
            "views": (a.get("popularity") or {}).get("mp_page_views_total"),
            "routes": a.get("n_routes"),
            "crags": len(a.get("crags", [])),
            "usgs": (a.get("usgs", {}).get("datasets") or [{}])[0].get("name"),
        }
        for a in reg["areas"]
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    log.info("registry: %s (%d areas)", args.registry, len(reg["areas"]))


if __name__ == "__main__":
    main()
