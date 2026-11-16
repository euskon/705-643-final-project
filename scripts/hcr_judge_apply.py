#!/usr/bin/env python
"""Check and apply the judges' decisions on the Horseshoe Canyon Ranch waypoints (plan 4, step 6.4).

    python scripts/hcr_judge_apply.py --phase check    # validate a + b, agreement, calibration, tie-break batches
    python scripts/hcr_judge_apply.py --phase apply    # (after the tie-break judges) final positions -> working CSVs

``check`` stops (exit 2) and applies nothing when a judge keeps the automatic snap on fewer
than ``--calibration`` (90%) of the easy points. ``apply`` refuses to overwrite earlier
judge decisions without ``--force``. Then refresh the routes copy:
``python scripts/s3_collect_routes.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from crag import hcr_points as H
from crag import registry
from crag.cli import setup_logging
from crag.crop import AreaCloud
from crag.negatives import local_pc
from crag.walls import grid_from_crop

SECTIONS = ("west_side", "north_forty", "east_side")


def load_cases() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((H.JUDGE_DIR / "cases").glob("*.json"))}


def load_judge(judge: str, cases: dict, check_custom, sections=SECTIONS, required: bool = True) -> dict[str, dict]:
    out = {}
    for sec in sections:
        f = H.JUDGE_DIR / f"decisions_{sec}_{judge}.json"
        if not f.exists():
            if required:
                raise SystemExit(f"missing {f}")
            continue
        batch = H.JUDGE_DIR / "batches" / (f"{sec}_tiebreak.json" if judge == "c" else f"{sec}.json")
        ids = json.loads(batch.read_text())
        ids = [x["gt_id"] if isinstance(x, dict) else x for x in ids]
        try:
            out.update(H.validate_decisions(json.loads(f.read_text()), {i: cases[i] for i in ids}, check_custom))
        except ValueError as exc:
            raise SystemExit(f"{f.name} rejected: {exc}") from None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default=H.HCR_SLUG)
    ap.add_argument("--phase", choices=("check", "apply"), required=True)
    ap.add_argument("--calibration", type=float, default=0.9)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    setup_logging()
    H.require_hcr(args.area)
    area = next(a for a in registry.load()["areas"] if a["slug"] == args.area)
    crs, cloud = registry.area_crs(area), AreaCloud(args.area)
    cases = load_cases()

    def ctx(case):
        pc, crop = local_pc(cloud, case["raw_x"], case["raw_y"], 50.0)
        return grid_from_crop(crop), (case["raw_x"], case["raw_y"]), pc, case["point"]["standing_height_m"]

    check_custom = H.custom_checker(ctx)
    a, b = load_judge("a", cases, check_custom), load_judge("b", cases, check_custom)
    agree = {g: H.same_decision(cases[g], a[g], b[g]) for g in cases}
    cal = {j: H.calibration(cases, d) for j, d in (("a", a), ("b", b))}
    print(f"agreement a/b: {sum(agree.values())}/{len(agree)} = {np.mean(list(agree.values())):.2f}")
    for j, (rate, changed) in cal.items():
        print(f"calibration judge {j}: kept the automatic snap on {rate:.0%} of {sum(c['easy'] for c in cases.values())} easy points")
    bad = {j: ch for j, (rate, ch) in cal.items() if rate < args.calibration}
    if bad:
        for j, ch in bad.items():
            print(f"\njudge {j} is below {args.calibration:.0%}; easy cases it changed:")
            for g in ch:
                d = (a if j == "a" else b)[g]
                print(f"  {g}: choice {d['choice']} ({d['confidence']}) - {d.get('reason', '')}")
        print("\nnothing applied: fix the instructions or pictures first")
        sys.exit(2)
    sec_of = {g: c["point"]["area_name"].strip().lower().replace(" ", "_") for g, c in cases.items()}
    if args.phase == "check":
        for sec in SECTIONS:
            dis = [{"gt_id": g, "judge_a": {k: a[g].get(k) for k in ("choice", "custom_east_m", "custom_north_m", "confidence", "reason")},
                    "judge_b": {k: b[g].get(k) for k in ("choice", "custom_east_m", "custom_north_m", "confidence", "reason")}}
                   for g in sorted(cases, key=lambda g: cases[g]["point"]["time"]) if sec_of[g] == sec and not agree[g]]
            f = H.JUDGE_DIR / "batches" / f"{sec}_tiebreak.json"
            if dis:
                f.write_text(H.dumps(dis))
            elif f.exists():
                f.unlink()
            print(f"{sec}: {len(dis)} disagreements" + (f" -> {f.name}" if dis else ""))
        return

    c = load_judge("c", cases, check_custom, sections=[s for s in SECTIONS if (H.JUDGE_DIR / "batches" / f"{s}_tiebreak.json").exists()], required=False)
    paths = [H.HAND_DIR / f for f in H.FILES]
    H.protect_judging(paths, args.force)
    frames = []
    for f in H.FILES:
        df = pd.read_csv(H.HAND_DIR / f, dtype={"route_index": str, "name": str})
        df["gt_id"] = [f"{Path(f).stem}#{i}" for i in range(len(df))]
        df["source_file"] = f
        frames.append(df)
    p = pd.concat(frames, ignore_index=True)
    for col in list(H.JUDGE_COLS) + ["snap_flags", "snap_status"]:
        p[col] = p[col].astype(object).where(p[col].notna(), "")
    final_xy = np.full((len(p), 2), np.nan)
    raw_xy = crs.to_work(p["raw_lon"].to_numpy(), p["raw_lat"].to_numpy())
    along = np.full((len(p), 2), np.nan)
    moved = []
    for i, g in enumerate(p["gt_id"]):
        if g not in cases:
            continue
        case = cases[g]
        along[i] = case["along"]
        r = H.resolve(case, a[g], b[g], c.get(g))
        deciders = [a[g], b[g]] if r["judge_result"] == "agreed" else [c[g], a[g] if H.same_decision(case, c[g], a[g]) else b[g]] if r["judge_result"] == "majority" else []
        conf = min((d["confidence"] for d in deciders), key=H.CONFIDENCE_ORDER.get) if deciders else "low"
        flags = [x for x in str(p.at[i, "snap_flags"]).split(";") if x and not x.startswith("judge_") and x != "order_reversed"]
        had_auto = p.at[i, "snap_status"] in ("ok", "judged") or pd.notna(p.at[i, "auto_lat"])
        status = r["snap_status"]
        if r["judge_result"] == "unresolved" and not had_auto:
            status = p.at[i, "snap_status"]  # too_short / off_wall stays what it was
        if r["offset"] is not None and status in ("ok", "judged"):
            final_xy[i] = raw_xy[i] + r["offset"]
            if status == "judged":
                ll = crs.to_lonlat(final_xy[i:i + 1, 0], final_xy[i:i + 1, 1])
                p.at[i, "lon"], p.at[i, "lat"] = float(ll[0, 0]), float(ll[0, 1])
                flags.append("judge_moved_from_auto")
                if pd.notna(p.at[i, "auto_lat"]):
                    auto = crs.to_work(np.array([p.at[i, "auto_lon"]]), np.array([p.at[i, "auto_lat"]]))[0]
                    moved.append(float(np.linalg.norm(final_xy[i] - auto)))
            else:
                p.at[i, "lat"], p.at[i, "lon"] = p.at[i, "auto_lat"], p.at[i, "auto_lon"]
        if status == "judge_none":
            flags.append("judge_none")
            p.at[i, "lat"], p.at[i, "lon"] = p.at[i, "raw_lat"], p.at[i, "raw_lon"]
        if r["judge_result"] == "unresolved":
            flags.append("judge_unresolved")
        if conf == "low":
            flags.append("judge_low_confidence")
        p.at[i, "snap_status"], p.at[i, "presnapped"] = status, status in ("ok", "judged")
        p.at[i, "judge_a_choice"], p.at[i, "judge_b_choice"] = str(a[g]["choice"]), str(b[g]["choice"])
        p.at[i, "judge_reason_a"], p.at[i, "judge_reason_b"] = a[g].get("reason", ""), b[g].get("reason", "")
        if g in c:
            p.at[i, "judge_c_choice"], p.at[i, "judge_reason_c"] = str(c[g]["choice"]), c[g].get("reason", "")
        p.at[i, "judge_confidence"], p.at[i, "judge_result"] = conf, r["judge_result"]
        p.at[i, "snap_flags"] = ";".join(flags)
    times = pd.to_datetime(p["time"]).astype("int64").to_numpy()
    rev = np.zeros(len(p), bool)
    for sec, idx in p[p["gt_id"].isin(cases)].groupby("area_name").groups.items():
        idx = np.asarray(list(idx))
        rev[idx] = H.order_reversed(final_xy[idx], raw_xy[idx], along[idx], times[idx])
    for i in np.flatnonzero(rev):
        p.at[i, "snap_flags"] = ";".join([x for x in str(p.at[i, "snap_flags"]).split(";") if x] + ["order_reversed"])
    for f, sub in p.groupby("source_file", sort=False):
        H.write_atomic(sub.drop(columns=["gt_id", "source_file"]), H.HAND_DIR / f)
    L = p[p["gt_id"].isin(cases)]
    print("\nfinal status by label:\n", L.groupby(["comment", "snap_status"]).size().unstack(fill_value=0).to_string())
    print("\njudge result:\n", L["judge_result"].value_counts().to_string())
    print("\nconfidence:\n", L["judge_confidence"].value_counts().to_string())
    print(f"\nmoved from the automatic snap: {len(moved)} points, median {np.median(moved) if moved else 0:.1f} m, max {max(moved) if moved else 0:.1f} m")
    fl = L["snap_flags"].str.split(";").explode()
    print("\nflags:\n", fl[fl != ""].value_counts().to_string())
    kept_rev = L[rev[L.index] & L["judge_result"].isin(["agreed", "majority"])]
    if len(kept_rev):
        print(f"\nwalking order reversed for {len(kept_rev)} agreed / majority points (kept): {kept_rev['gt_id'].tolist()}")
    summary = {"agreement": float(np.mean(list(agree.values()))), "calibration": {j: r for j, (r, _) in cal.items()}, "judge_result": L["judge_result"].value_counts().to_dict(),
               "status": L.groupby(["comment", "snap_status"]).size().rename("n").reset_index().to_dict("records"), "moved_from_auto": len(moved),
               "moved_median_m": float(np.median(moved)) if moved else 0.0, "flags": fl[fl != ""].value_counts().to_dict(), "confidence": L["judge_confidence"].value_counts().to_dict()}
    (H.JUDGE_DIR / "summary.json").write_text(H.dumps(summary))


if __name__ == "__main__":
    main()
