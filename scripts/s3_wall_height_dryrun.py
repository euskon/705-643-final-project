#!/usr/bin/env python
"""Step 3 - dry run of the wall check: what each minimum wall height and snap method costs.

Reads ``data/routes/verified_routes.csv`` (all areas). For Horseshoe Canyon Ranch it reads
the **original** waypoints in ``data/handgathered/original/``, so it always measures the
automatic method, never the saved (snapped, judged) points.

    python scripts/s3_wall_height_dryrun.py
    python scripts/s3_wall_height_dryrun.py --thresholds 6 --methods nearest,toward_wall --sample-per-area 250

Per threshold and method, per area group and label: share passing, median distance moved,
share that skipped a nearer shorter wall, share slid more than 3 m along the wall, squashed
pairs, and at Horseshoe the Climb/Nc pairs within 1 / 2.5 / 5 m and Nc points landing on a
Climb point. Also the route-length check for both height measures. Writes
``data/routes/wall_height_dryrun.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from crag import registry
from crag.cli import setup_logging
from crag.crop import AreaCloud
from crag.groundtruth import GroundTruthConfig, load_waypoints, snap_one
from crag.labeling import LabelConfig
from crag.negatives import local_pc, spacing_report
from crag.paths import AREAS_JSON, PROJECT_DIR, ROUTES_DIR
from crag.walls import grid_for

HCR = "horseshoe-canyon-ranch"
ORIGINAL = PROJECT_DIR / "data" / "handgathered" / "original"


def original_waypoints() -> pd.DataFrame:
    cfg = GroundTruthConfig(paths=tuple(sorted(ORIGINAL.glob("Waypoints_*.csv"))))
    gt = load_waypoints(cfg)
    gt = gt[gt["label"] >= 0]
    return pd.DataFrame({"route_uid": "hand:" + gt["gt_id"], "slug": HCR, "label": gt["label"].astype(int), "lat": gt["lat"], "lon": gt["lon"],
                         "length_m": np.nan, "name": gt["name"]})


def group_of(slug: str, label: int) -> str:
    if slug == HCR:
        return "Horseshoe Climb" if label == 1 else "Horseshoe Nc"
    return "Red River Gorge" if slug == "red-river-gorge" else "other areas"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routes", type=Path, default=ROUTES_DIR / "verified_routes.csv")
    ap.add_argument("--sample-per-area", type=int, default=250)
    ap.add_argument("--thresholds", default="4,6,8,10,12")
    ap.add_argument("--methods", default="nearest,toward_wall")
    ap.add_argument("--out", type=Path, default=ROUTES_DIR / "wall_height_dryrun.json")
    args = ap.parse_args()
    setup_logging()

    routes = pd.read_csv(args.routes)
    routes = routes[(routes["slug"] != HCR) & (routes["label"] == 1)]
    routes = pd.concat([g.sample(min(len(g), args.sample_per_area), random_state=0) for _, g in routes.groupby("slug")])
    pts = pd.concat([original_waypoints(), routes[["route_uid", "slug", "label", "lat", "lon", "length_m", "name"]]], ignore_index=True)
    reg = registry.load(AREAS_JSON)
    areas = {a["slug"]: a for a in reg["areas"]}
    thresholds = [float(t) for t in args.thresholds.split(",")]
    methods = args.methods.split(",")
    lab = LabelConfig()
    rows = []
    for slug, sub in pts.groupby("slug"):
        area = areas.get(slug)
        if area is None or not (area.get("usgs") or {}).get("datasets"):
            continue
        crs, cloud = registry.area_crs(area), AreaCloud(slug)
        for r in sub.itertuples(index=False):
            xy = np.asarray(crs.to_work(r.lon, r.lat), float)
            pc, _ = local_pc(cloud, xy[0], xy[1], 40.0)
            if pc is None:
                continue
            grid = grid_for(cloud, xy[0], xy[1])
            for th in thresholds:
                for m in methods:
                    s = snap_one(pc, xy, lab, replace(GroundTruthConfig(), min_wall_height_m=th, snap_method=m), grid)
                    rows.append({"route_uid": r.route_uid, "slug": slug, "label": r.label, "length_m": r.length_m, "x": xy[0], "y": xy[1], "threshold": th, "method": m,
                                 "group": group_of(slug, r.label), **s})
    df = pd.DataFrame(rows)
    report = {"n_points": int(pts["route_uid"].nunique()), "runs": []}
    for (th, m), d in df.groupby(["threshold", "method"]):
        ok = d[d["status"] == "ok"]
        run = {"threshold": th, "method": m, "groups": {}}
        for g, dd in d.groupby("group"):
            o = dd[dd["status"] == "ok"]
            run["groups"][g] = {"n": int(len(dd)), "pass": round(len(o) / len(dd), 3), "too_short": int((dd["status"] == "too_short").sum()),
                                "off_wall": int((dd["status"] == "off_wall").sum()), "median_move_m": round(float(o["snap_dist_m"].median()), 2),
                                "skipped_shorter": round(float(o["skipped_shorter_wall"].mean()), 3), "slid_gt_3m": round(float((o["along_shift_m"] > 3).mean()), 3)}
        sq = spacing_report(ok)
        run["squashed_pairs"] = {k: int(v) for k, v in sq[sq["squashed_pair"]].groupby("slug").size().items()}
        h = ok[ok["slug"] == HCR]
        xy = h[["snap_x", "snap_y"]].to_numpy(float)
        climb, nc = xy[h["label"].to_numpy() == 1], xy[h["label"].to_numpy() == 0]
        if len(climb) and len(nc):
            dn = cKDTree(climb).query(nc)[0]
            run["hcr"] = {"nc_on_climb_lt_0.5m": int((dn < 0.5).sum()), **{f"nc_within_{r}m_of_climb": int((dn <= r).sum()) for r in (1, 2.5, 5)},
                          "pairs_within_1m": int(len(cKDTree(xy).query_pairs(1.0))), "n_nc": int(len(nc)), "n_climb": int(len(climb))}
        L = ok[ok["length_m"] > 0]
        if len(L) > 10:
            run["route_length"] = {"n": int(len(L)),
                                   **{f"{k}_spearman": round(float(spearmanr(L[c], L["length_m"]).statistic), 3) for k, c in (("grid", "wall_height_m"), ("column", "wall_height_column_m"))},
                                   **{f"{k}_share_ge_0.8_len": round(float((L[c] >= 0.8 * L["length_m"]).mean()), 3) for k, c in (("grid", "wall_height_m"), ("column", "wall_height_column_m"))}}
        report["runs"].append(run)
        print(f"\n=== minimum {th:g} m, snap {m}")
        print(pd.DataFrame(run["groups"]).T.to_string())
        print("squashed pairs:", run["squashed_pairs"], "| Horseshoe:", run.get("hcr"), "| route length:", run.get("route_length"))
    raw = original_waypoints()
    crs = registry.area_crs(areas[HCR])
    rxy = crs.to_work(raw["lon"].to_numpy(), raw["lat"].to_numpy())
    report["hcr_raw_pairs_within_1m"] = int(len(cKDTree(rxy).query_pairs(1.0)))
    print("\nHorseshoe raw GPS pairs within 1 m:", report["hcr_raw_pairs_within_1m"])
    args.out.write_text(json.dumps(report, indent=1))
    df.drop(columns=[c for c in df if c in ("ground_z",)]).to_csv(args.out.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
