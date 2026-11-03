#!/usr/bin/env python
"""Step 3a - find the US climbing areas with the most route-level GPS on openclimbing.org / OSM.

One Overpass query fetches every ``climbing=route_bottom`` / ``climbing=route``
feature in the USA; points closer than ``--eps-m`` are grouped into clusters
(≈ climbing areas). Each cluster is ranked by its number of route locations
and matched to the registry. With ``--resolve-mp`` the script fetches one
Mountain Project route page per top cluster (via a route's
``climbing:url:mountainproject`` tag) to get its MP area chain, so the area
can be added to ``registry/areas.json`` with its ``mp_id``.

    python scripts/s3_survey_openclimbing.py
    python scripts/s3_survey_openclimbing.py --top 20 --resolve-mp --wandb online

Writes ``data/routes/us_survey.csv`` (clusters) and ``data/routes/us_routes.csv``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from crag import mp, openclimbing, registry, wandb_utils
from crag.cli import setup_logging
from crag.paths import AREAS_JSON, ROUTES_DIR

log = logging.getLogger("survey")


def registry_match(reg: dict, lon: float, lat: float) -> str | None:
    for a in reg["areas"]:
        bb = a.get("bbox")
        if bb and bb[0] <= lon <= bb[2] and bb[1] <= lat <= bb[3]:
            return a["slug"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eps-m", type=float, default=1500.0, help="cluster linkage distance")
    ap.add_argument("--min-routes", type=int, default=5)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--resolve-mp", action="store_true", help="look up MP area chains for the top clusters (1 MP request each, 60 s apart)")
    ap.add_argument("--include-boulders", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    wandb_utils.add_cli(ap, default="disabled")
    args = ap.parse_args()
    setup_logging()

    routes, _ = openclimbing.parse_elements(openclimbing.overpass(openclimbing.us_survey_query(), refresh=args.refresh))
    log.info("US: %d route features (%s)", len(routes), routes["osm_kind"].value_counts().to_dict())
    if not args.include_boulders:
        routes = routes[~routes["is_boulder"]].reset_index(drop=True)
    routes["cluster"] = openclimbing.cluster_points(routes[["lon", "lat"]].to_numpy(), args.eps_m)
    ROUTES_DIR.mkdir(parents=True, exist_ok=True)
    routes.to_csv(ROUTES_DIR / "us_routes.csv", index=False)

    reg = registry.load(AREAS_JSON)
    rows = []
    for cid, g in routes.groupby("cluster"):
        if len(g) < args.min_routes:
            continue
        lon, lat = float(g["lon"].median()), float(g["lat"].median())
        crags = g["osm_crag"].dropna().value_counts()
        rows.append(
            {
                "cluster": int(cid),
                "n_routes": len(g),
                "n_route_bottom": int((g["osm_kind"] == "route_bottom").sum()),
                "n_route_ways": int((g["osm_type"] == "way").sum()),
                "n_named": int(g["name"].notna().sum()),
                "n_mp_url": int(g["mp_route_id_tag"].notna().sum()),
                "n_with_grade": int(g["grade"].notna().sum()),
                "lon": round(lon, 5),
                "lat": round(lat, 5),
                "bbox": [round(g["lon"].min(), 5), round(g["lat"].min(), 5), round(g["lon"].max(), 5), round(g["lat"].max(), 5)],
                "top_crags": "; ".join(f"{k} ({v})" for k, v in crags.head(4).items()),
                "registry_slug": registry_match(reg, lon, lat),
                "sample_mp_route_id": int(g["mp_route_id_tag"].dropna().iloc[0]) if g["mp_route_id_tag"].notna().any() else None,
            }
        )
    df = pd.DataFrame(rows).sort_values("n_routes", ascending=False).reset_index(drop=True)
    if args.resolve_mp:
        chains = []
        for r in df.head(args.top).itertuples():
            if r.sample_mp_route_id is None or pd.isna(r.sample_mp_route_id):
                chains.append(None)
                continue
            page = mp._polite_get(f"{mp.BASE}/route/{int(r.sample_mp_route_id)}")
            crumbs = mp.parse_area_page(page)["breadcrumb"]
            chains.append(" > ".join(f"{n} [{i}]" for i, n in crumbs))
        df.loc[: len(chains) - 1, "mp_area_chain"] = chains
    df.to_csv(ROUTES_DIR / "us_survey.csv", index=False)
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(df.head(args.top)[["n_routes", "n_route_bottom", "n_mp_url", "lat", "lon", "registry_slug", "top_crags"]].to_string())
    log.info("wrote %s (%d clusters with >= %d routes)", ROUTES_DIR / "us_survey.csv", len(df), args.min_routes)
    run = wandb_utils.init(args.wandb, args.wandb_project, job_type="data", name="openclimbing-us-survey")
    if run is not None:
        import wandb

        run.log({"survey/clusters": wandb.Table(dataframe=df.astype({"bbox": str}))})
        run.summary.update({"us_route_features": len(routes), "clusters": len(df)})
        run.finish()


if __name__ == "__main__":
    main()
