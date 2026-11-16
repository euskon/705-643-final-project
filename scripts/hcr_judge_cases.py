#!/usr/bin/env python
"""Build the judge cases for the Horseshoe Canyon Ranch waypoints (plan 4, step 6.1-6.2).

For every labelled point of the working files (after ``snap_hcr_waypoints.py``): one
picture and one JSON under ``data/handgathered/review/judge/cases/``, the batches per wall
section (walking order) and ``PROMPT.md``, the judges' instructions.

    python scripts/hcr_judge_cases.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from crag import hcr_points as H
from crag import registry
from crag.cli import setup_logging
from crag.crop import AreaCloud
from crag.groundtruth import GroundTruthConfig
from crag.labeling import LabelConfig
from crag.negatives import local_pc
from crag.walls import grid_from_crop


def load_working(crs) -> pd.DataFrame:
    frames = []
    for f in H.FILES:
        df = pd.read_csv(H.HAND_DIR / f, dtype={"route_index": str, "name": str})
        if "raw_lat" not in df:
            raise SystemExit(f"{f} has no snap columns - run scripts/snap_hcr_waypoints.py first")
        df["gt_id"] = [f"{Path(f).stem}#{i}" for i in range(len(df))]
        df["source_file"] = f
        frames.append(df)
    p = pd.concat(frames, ignore_index=True)
    raw = crs.to_work(p["raw_lon"].to_numpy(), p["raw_lat"].to_numpy())
    p["raw_x"], p["raw_y"] = raw[:, 0], raw[:, 1]
    has = p["auto_lat"].notna().to_numpy()
    p["auto_x"], p["auto_y"] = np.nan, np.nan
    if has.any():
        a = crs.to_work(p.loc[has, "auto_lon"].to_numpy(), p.loc[has, "auto_lat"].to_numpy())
        p.loc[has, "auto_x"], p.loc[has, "auto_y"] = a[:, 0], a[:, 1]
    return p


def section_slug(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def case_context(cloud, row):
    """(pc, crop, grid) around one point's raw position."""
    pc, crop = local_pc(cloud, row.raw_x, row.raw_y, 50.0)
    return pc, crop, grid_from_crop(crop)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default=H.HCR_SLUG)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", help="comma-separated gt_ids (debugging)")
    args = ap.parse_args()
    setup_logging()
    H.require_hcr(args.area)
    H.protect_judging([H.HAND_DIR / f for f in H.FILES], args.force)
    area = next(a for a in registry.load()["areas"] if a["slug"] == args.area)
    crs, cloud, g, lab = registry.area_crs(area), AreaCloud(args.area), GroundTruthConfig(), LabelConfig()
    p = load_working(crs)
    L = p[p["comment"] != "?"].sort_values("time").reset_index(drop=True)
    cases_dir = H.JUDGE_DIR / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    (H.JUDGE_DIR / "batches").mkdir(exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    for i, r in enumerate(L.itertuples(index=False)):
        if only and r.gt_id not in only:
            continue
        raw = np.array([r.raw_x, r.raw_y])
        auto = np.array([r.auto_x, r.auto_y])
        pc, crop, grid = case_context(cloud, r)
        cands, frame = H.build_candidates(pc, grid, raw, auto if np.isfinite(auto).all() else None, lab, g)
        near = L[(np.hypot(L["raw_x"] - raw[0], L["raw_y"] - raw[1]) <= 35) & (L["gt_id"] != r.gt_id)]
        nb = [{"label": n.comment, "name": str(n.name), "raw_e": n.raw_x - raw[0], "raw_n": n.raw_y - raw[1],
               "auto_e": None if np.isnan(n.auto_x) else n.auto_x - raw[0], "auto_n": None if np.isnan(n.auto_y) else n.auto_y - raw[1]} for n in near.itertuples(index=False)]
        title = f"{r.gt_id}   {r.comment}   '{r.name}'   {r.area_name}   {r.time}   automatic snap: {r.snap_status}" + (f", moved {r.snap_dist_m:.1f} m" if r.snap_status == "ok" else "")
        H.draw_case(cases_dir / f"{r.gt_id}.png", title, grid, crop, raw, auto, cands, frame, nb)
        walk = []
        for k in range(max(i - 3, 0), min(i + 4, len(L))):
            if k == i:
                continue
            n = L.iloc[k]
            if n["area_name"] != r.area_name:
                continue
            walk.append({"order": k - i, "gt_id": n["gt_id"], "label": n["comment"], "name": str(n["name"]), "time": n["time"],
                         "raw_east_m": round(n["raw_x"] - raw[0], 1), "raw_north_m": round(n["raw_y"] - raw[1], 1),
                         "auto_east_m": None if np.isnan(n["auto_x"]) else round(n["auto_x"] - raw[0], 1),
                         "auto_north_m": None if np.isnan(n["auto_y"]) else round(n["auto_y"] - raw[1], 1)})
        case = {
            "gt_id": r.gt_id,
            "point": {"label": r.comment, "name": str(r.name), "mp_url": None if pd.isna(r.mp_url) else r.mp_url, "area_name": r.area_name, "route_index": r.route_index,
                      "time": r.time, "raw_lat": r.raw_lat, "raw_lon": r.raw_lon, "automatic_snap_status": r.snap_status,
                      "automatic_snap_moved_m": None if pd.isna(r.snap_dist_m) else round(float(r.snap_dist_m), 1),
                      "flags": [] if pd.isna(r.snap_flags) else str(r.snap_flags).split(";"), "standing_height_m": round(frame["ground_z"], 1)},
            "candidates": [{"id": c["id"], "roles": c["roles"], "east_m": round(float(c["xy"][0] - raw[0]), 2), "north_m": round(float(c["xy"][1] - raw[1]), 2),
                            "dist_from_raw_m": round(c["dist_from_raw_m"], 1), "along_shift_m": round(c["along_shift_m"], 1), "wall_height_grid_m": round(c["grid_h"], 1),
                            "wall_height_column_m": round(c["col_h"], 1), "foot_minus_standing_m": None if c["foot_minus_standing_m"] is None else round(c["foot_minus_standing_m"], 1),
                            "cliff_points_within_2.5m": int(c["n_steep"]), "wall_piece": int(c["piece"])} for c in cands],
            "walls_within_reach": int(frame["n_walls_within_reach"]),
            "walking_order": walk,
            "easy": bool(r.snap_status == "ok" and r.snap_dist_m < 1.0 and frame["n_walls_within_reach"] == 1),
            "raw_x": float(raw[0]), "raw_y": float(raw[1]), "along": [float(v) for v in frame["along"]],
        }
        (cases_dir / f"{r.gt_id}.json").write_text(H.dumps(case))
    for sec, sub in L.groupby("area_name"):
        (H.JUDGE_DIR / "batches" / f"{section_slug(sec)}.json").write_text(H.dumps(sub.sort_values("time")["gt_id"].tolist()))
    (H.JUDGE_DIR / "PROMPT.md").write_text(H.PROMPT)
    print(L.groupby("area_name").size().to_string())
    print(f"wrote {len(L)} cases to {cases_dir}")


if __name__ == "__main__":
    main()
