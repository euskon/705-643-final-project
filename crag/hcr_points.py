"""One-time clean-up of the hand-gathered Horseshoe Canyon Ranch waypoints.

Three steps, each a thin script over the functions here (Horseshoe only; they
refuse any other area):

1. ``scripts/snap_hcr_waypoints.py``  - automatic snap of the **originals**
   (``data/handgathered/original``) with the pipeline's wall check, written to
   the working files at the original paths (same rows, same order).
2. ``scripts/hcr_judge_cases.py``     - one picture + one JSON per labelled
   point, plus the judges' instructions, for Claude subagent judges.
3. ``scripts/hcr_judge_apply.py``     - validate the judges' decisions, check
   agreement and calibration, settle disagreements with a tie-break judge,
   write the final positions.

After that ``lat/lon`` in the working files are final (``presnapped``); the
pipeline only confirms them (``crag.negatives.snap_routes``). There is no hand
review: the judges' decision is final. See
``research-prompts/fable-investigation-project-issues-resolution-plan4.md``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from crag.groundtruth import JUDGE_COLS, LABELS, SNAP_FILE_COLS, GroundTruthConfig, _spot, snap_one
from crag.labeling import LabelConfig
from crag.paths import PROJECT_DIR
from crag.walls import WallGrid, wall_frame_at, wall_height_at

log = logging.getLogger(__name__)

HCR_SLUG = "horseshoe-canyon-ranch"
HAND_DIR = PROJECT_DIR / "data" / "handgathered"
ORIGINAL_DIR = HAND_DIR / "original"
JUDGE_DIR = HAND_DIR / "review" / "judge"
FILES = ("Waypoints_2026-10-24.csv", "Waypoints_2026-10-25.csv")
NEAR_OPPOSITE_M = 5.0
CONFIDENCE_ORDER = {"high": 2, "medium": 1, "low": 0}


def require_hcr(slug: str) -> None:
    if slug != HCR_SLUG:
        raise SystemExit(f"this step is for {HCR_SLUG} only (got {slug!r}); other areas use the automatic wall check on every build")


def judged_rows(path: Path) -> int:
    """How many rows of a working file already carry a judge decision."""
    if not Path(path).exists():
        return 0
    df = pd.read_csv(path, dtype=str)
    cols = [c for c in JUDGE_COLS if c in df]
    return int(df[cols].notna().any(axis=1).sum()) if cols else 0


def protect_judging(paths, force: bool) -> None:
    n = sum(judged_rows(p) for p in paths)
    if n and not force:
        raise SystemExit(f"{n} rows already carry judge decisions and would be lost; rerun with --force to overwrite them")


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = Path(path).with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# step 1: automatic snap
# ---------------------------------------------------------------------------


def snap_waypoint_files(snap_fn, to_work, to_lonlat, original_dir: Path = ORIGINAL_DIR, out_dir: Path = HAND_DIR, files=FILES, cfg: GroundTruthConfig | None = None,
                        force: bool = False) -> pd.DataFrame:
    """Snap every labelled row of the original files and write the working files.

    ``snap_fn(xy) -> dict`` is the wall check (``snap_one`` bound to the area's cloud). Rows,
    order and the original columns are kept, so every ``gt_id`` (``<file>#<row>``) stays the
    same. Returns all rows of all files (with ``gt_id``, ``x, y`` final, ``raw_x, raw_y``)."""
    cfg = cfg or GroundTruthConfig()
    out_dir = Path(out_dir)
    protect_judging([out_dir / f for f in files], force)
    frames = []
    for f in files:
        df = pd.read_csv(Path(original_dir) / f, dtype={"route_index": str, "name": str})
        unknown = set(df["comment"]) - set(LABELS)
        if unknown:
            raise ValueError(f"unexpected labels in {f}: {unknown}")
        df["gt_id"] = [f"{Path(f).stem}#{i}" for i in range(len(df))]
        df["source_file"] = f
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    raw = to_work(allp["lon"].to_numpy(), allp["lat"].to_numpy())
    allp["raw_lat"], allp["raw_lon"], allp["raw_x"], allp["raw_y"] = allp["lat"], allp["lon"], raw[:, 0], raw[:, 1]
    rows = []
    for i, r in enumerate(allp.itertuples(index=False)):
        if r.comment == "?":
            rows.append({"snap_status": "unlabelled", "presnapped": False})
            continue
        s = snap_fn(raw[i])
        ok = s["status"] == "ok"
        rows.append({"snap_status": s["status"], "presnapped": ok, "x": s.get("snap_x", np.nan), "y": s.get("snap_y", np.nan), "snap_dist_m": s.get("snap_dist_m", np.nan),
                     "along_shift_m": s.get("along_shift_m", np.nan), "wall_height_m": s.get("wall_height_m", np.nan),
                     "wall_height_column_m": s.get("wall_height_column_m", np.nan), "nearest_wall_height_m": s.get("nearest_wall_height_m", np.nan),
                     "skipped_shorter_wall": bool(s.get("skipped_shorter_wall", False)), "out_dx": s.get("out_dx", np.nan), "out_dy": s.get("out_dy", np.nan),
                     "anchor_x": s.get("anchor_x", np.nan), "anchor_y": s.get("anchor_y", np.nan)})
    S = pd.DataFrame(rows)
    for c in S:
        allp[c] = S[c].to_numpy()
    ok = allp["snap_status"].eq("ok").to_numpy()
    ll = to_lonlat(allp.loc[ok, "x"].to_numpy(float), allp.loc[ok, "y"].to_numpy(float))
    allp["auto_lat"], allp["auto_lon"] = np.nan, np.nan
    allp.loc[ok, "auto_lon"], allp.loc[ok, "auto_lat"] = ll[:, 0], ll[:, 1]
    allp.loc[ok, "lat"], allp.loc[ok, "lon"] = allp.loc[ok, "auto_lat"], allp.loc[ok, "auto_lon"]
    allp["x"] = np.where(ok, allp["x"], allp["raw_x"])
    allp["y"] = np.where(ok, allp["y"], allp["raw_y"])
    allp["snap_flags"] = [";".join(f) for f in snap_flags(allp, cfg)]
    allp["snap_min_wall_height_m"], allp["snap_method"] = cfg.min_wall_height_m, cfg.snap_method
    for c in JUDGE_COLS:
        allp[c] = ""
    for f, sub in allp.groupby("source_file", sort=False):
        orig_cols = list(pd.read_csv(Path(original_dir) / f, nrows=0).columns)
        write_atomic(sub[orig_cols + list(SNAP_FILE_COLS) + list(JUDGE_COLS)], out_dir / f)
    return allp


def snap_flags(p: pd.DataFrame, cfg: GroundTruthConfig) -> list[list[str]]:
    """Per-row notes (information, not errors) from the automatic snap."""
    ok = p["snap_status"].eq("ok").to_numpy()
    flags = [[] for _ in range(len(p))]
    xy, raw, lab = p[["x", "y"]].to_numpy(float), p[["raw_x", "raw_y"]].to_numpy(float), p["comment"].to_numpy()
    for i in np.flatnonzero(~ok):
        if p["snap_status"].iloc[i] in ("too_short", "off_wall"):
            flags[i].append(p["snap_status"].iloc[i])
    idx = np.flatnonzero(ok)
    for i in idx:
        d = p["snap_dist_m"].iloc[i]
        if d > 10:
            flags[i].append("moved_gt_10m")
        elif d > 5:
            flags[i].append("moved_gt_5m")
        if p["skipped_shorter_wall"].iloc[i]:
            flags[i].append("skipped_shorter_wall")
        if p["along_shift_m"].iloc[i] > cfg.max_along_shift_m:
            flags[i].append("slid_along_wall")
    if len(idx) > 1:
        tree = cKDTree(xy[idx])
        for a, b in tree.query_pairs(NEAR_OPPOSITE_M):
            i, j = idx[a], idx[b]
            if lab[i] != lab[j]:
                for k in (i, j):
                    if "near_opposite_label" not in flags[k]:
                        flags[k].append("near_opposite_label")
        along = np.column_stack([-p["out_dy"].to_numpy(float), p["out_dx"].to_numpy(float)])
        for a, b in cKDTree(raw[idx]).query_pairs(20.0):
            i, j = idx[a], idx[b]
            d = np.linalg.norm(raw[i] - raw[j])
            if d < 2.0:
                continue
            if np.linalg.norm(xy[i] - xy[j]) < 0.5 * abs((raw[i] - raw[j]) @ along[i]):
                for k in (i, j):
                    if "squashed_pair" not in flags[k]:
                        flags[k].append("squashed_pair")
    return flags


# ---------------------------------------------------------------------------
# step 2: judge cases
# ---------------------------------------------------------------------------


def wall_pieces(grid: WallGrid) -> np.ndarray:
    """Connected wall-mask pieces (0 = no wall), 8-connected."""
    return ndimage.label(grid.wall, structure=np.ones((3, 3), bool))[0]


def piece_at(grid: WallGrid, pieces: np.ndarray, xy) -> int:
    """Id of the wall piece nearest to ``xy`` (0 if none within 3 m)."""
    iy, ix = grid.cell(np.atleast_2d(xy))
    iy, ix = int(iy[0]), int(ix[0])
    y0, x0 = max(iy - 3, 0), max(ix - 3, 0)
    win = pieces[y0:iy + 4, x0:ix + 4]
    ys, xs = np.nonzero(win)
    if not len(ys):
        return 0
    k = int(np.argmin((ys + y0 - iy) ** 2 + (xs + x0 - ix) ** 2))
    return int(win[ys[k], xs[k]])


def passing_spots(pc, grid: WallGrid, raw_xy, lab: LabelConfig, g: GroundTruthConfig) -> tuple[list[dict], float]:
    """Every candidate cliff spot within the search radius that passes the anchor checks except height."""
    gi = pc.column(raw_xy, g.ground_radius_m)
    ground_z = float(np.quantile(pc.xyz[gi, 2], 0.1)) if len(gi) else np.nan
    near = pc.steep_tree_xy.query_ball_point(raw_xy, g.snap_radius_m) if len(pc.steep_idx) else []
    if not near:
        return [], ground_z
    cand = pc.xyz[pc.steep_idx[near], :2]
    cand = cand[np.argsort(np.linalg.norm(cand - raw_xy, axis=1))]
    _, first = np.unique(np.round(cand / 1.0).astype(np.int64), axis=0, return_index=True)
    out = []
    for c in cand[np.sort(first)]:
        sp = _spot(pc, c, lab, g, grid, ground_z, lab.min_steep_points)
        if sp is not None:
            out.append(sp)
    return out, ground_z


def build_candidates(pc, grid: WallGrid, raw_xy, auto_xy, lab: LabelConfig, g: GroundTruthConfig, max_n: int = 8) -> tuple[list[dict], dict]:
    """Numbered candidate spots for one point (see plan 4, section 7.1) and the anchor frame."""
    pieces = wall_pieces(grid)
    spots, ground_z = passing_spots(pc, grid, raw_xy, lab, g)
    tall = [s for s in spots if s["h"] >= g.min_wall_height_m]
    for s in spots:
        s["piece"] = piece_at(grid, pieces, s["xy"])
    cands: list[dict] = []

    def add(xy, role, sp=None):
        xy = np.asarray(xy, float)
        for c in cands:
            if np.linalg.norm(c["xy"] - xy) < 1.0:
                c["roles"].append(role)
                return
        if len(cands) >= max_n:
            return
        sp = sp or _spot(pc, xy, lab, g, grid, np.nan, 1) or {"n_steep": 0, "base_z": float("nan"), "col_h": 0.0, "grid_h": 0.0, "h": 0.0}
        cands.append({"xy": xy, "roles": [role], "n_steep": sp["n_steep"], "base_z": sp["base_z"], "col_h": sp["col_h"], "grid_h": sp["grid_h"],
                      "piece": piece_at(grid, pieces, xy)})

    s_auto = snap_one(pc, raw_xy, lab, g, grid)
    if auto_xy is not None and np.isfinite(auto_xy).all():
        add(auto_xy, "automatic snap")
    anchor = None
    if s_auto["status"] == "ok":
        anchor = np.array([s_auto["anchor_x"], s_auto["anchor_y"]])
        add(anchor, "anchor (nearest wall spot that passes every check)")
    if tall:
        add(tall[0]["xy"], "nearest spot that passes every check", tall[0])
        seen = {c["piece"] for c in cands}
        for p in sorted({s["piece"] for s in tall} - seen):
            on = [s for s in tall if s["piece"] == p]
            if anchor is not None:  # the spot on that wall straight in front (smallest offset along the anchor's wall line)
                _, along = wall_frame_at(grid, anchor)
                on.sort(key=lambda s: abs((s["xy"] - raw_xy) @ along))
            add(on[0]["xy"], f"another wall within {g.snap_radius_m:.0f} m (piece {p})", on[0])
        best = max(tall, key=lambda s: s["h"])
        add(best["xy"], f"tallest wall within {g.snap_radius_m:.0f} m", best)
    frame_at = anchor if anchor is not None else raw_xy
    out, along = wall_frame_at(grid, frame_at)
    for k, c in enumerate(cands, 1):
        c["id"] = k
        c["dist_from_raw_m"] = float(np.linalg.norm(c["xy"] - raw_xy))
        c["along_shift_m"] = float(abs((c["xy"] - raw_xy) @ along))
        c["foot_minus_standing_m"] = float(c["base_z"] - ground_z) if np.isfinite(c["base_z"]) and np.isfinite(ground_z) else None
    n_pieces = len({s["piece"] for s in tall if s["piece"]})
    return cands, {"out": out, "along": along, "ground_z": ground_z, "n_walls_within_reach": n_pieces, "anchor": anchor, "pieces": pieces}


def hillshade(dem: np.ndarray, res: float, az: float = 315.0, alt: float = 45.0) -> np.ndarray:
    gy, gx = np.gradient(dem, res)
    slope, aspect = np.pi / 2 - np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    a, h = np.radians(az), np.radians(alt)
    return np.clip(np.sin(h) * np.sin(slope) + np.cos(h) * np.cos(slope) * np.cos(a - aspect), 0, 1)


def draw_case(path: Path, title: str, grid: WallGrid, crop: dict, raw_xy, auto_xy, cands: list[dict], frame: dict, neighbours: list[dict], half: float = 30.0) -> None:
    """The judge's picture: map (left) and elevation profile across the wall (right). Axes are metres east / north of the raw GPS point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, pr) = plt.subplots(1, 2, figsize=(13.5, 7.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ny, nx = grid.dem.shape
    ext = [grid.x0 - raw_xy[0], grid.x0 + nx * grid.res - raw_xy[0], grid.y0 - raw_xy[1], grid.y0 + ny * grid.res - raw_xy[1]]
    ax.imshow(hillshade(grid.dem, grid.res), extent=ext, origin="lower", cmap="gray", vmin=0, vmax=1)
    xs, ys = np.linspace(ext[0], ext[1], nx, endpoint=False) + grid.res / 2, np.linspace(ext[2], ext[3], ny, endpoint=False) + grid.res / 2
    lv = np.arange(np.floor(grid.dem.min() / 2) * 2, grid.dem.max() + 2, 2.0)
    ax.contour(xs, ys, grid.dem, levels=lv, colors="#8a6d3b", linewidths=0.35, alpha=0.7)
    ax.imshow(np.ma.masked_where(~grid.wall, grid.wall), extent=ext, origin="lower", cmap="Reds", alpha=0.35, vmin=0, vmax=1.3)
    st = crop["steep"]
    base = next((c["base_z"] for c in cands if np.isfinite(c["base_z"])), frame["ground_z"])
    rel = crop["xyz"][st] - [raw_xy[0], raw_xy[1], base]
    sc = ax.scatter(rel[:, 0], rel[:, 1], c=np.clip(rel[:, 2], -2, 30), s=5, cmap="viridis", vmin=-2, vmax=30, linewidths=0)
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01, label="cliff points: height above the wall foot (m)")
    for n in neighbours:
        col = "#1f77b4" if n["label"] == "Climb" else "#d62728"
        ax.plot(n["raw_e"], n["raw_n"], "x", color=col, ms=6, mew=1.2)
        if n.get("auto_e") is not None:
            ax.plot([n["raw_e"], n["auto_e"]], [n["raw_n"], n["auto_n"]], "-", color=col, lw=0.6)
            ax.plot(n["auto_e"], n["auto_n"], "o", mfc="none", color=col, ms=6, mew=1.0)
        ax.annotate(f"{n['label']} {n['name'][:16]}", (n["raw_e"], n["raw_n"]), fontsize=6, color=col, xytext=(3, 3), textcoords="offset points")
    ax.plot(0, 0, "x", color="yellow", ms=14, mew=3)
    ax.plot(0, 0, "x", color="black", ms=11, mew=1.2)
    if auto_xy is not None and np.isfinite(auto_xy).all():
        ax.plot(auto_xy[0] - raw_xy[0], auto_xy[1] - raw_xy[1], "o", mfc="none", color="yellow", ms=15, mew=2)
    for c in cands:
        e, n = c["xy"] - raw_xy
        ax.annotate(str(c["id"]), (e, n), fontsize=11, fontweight="bold", color="white", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="black", ec="yellow", lw=1))
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.plot([-half + 3, -half + 13], [-half + 3, -half + 3], "-", color="white", lw=3)
    ax.text(-half + 8, -half + 4, "10 m", color="white", ha="center", fontsize=8)
    ax.annotate("N", xy=(half - 4, half - 3), xytext=(half - 4, half - 9), color="white", ha="center", fontsize=10, arrowprops=dict(arrowstyle="->", color="white"))
    ax.set_xlabel("metres east of the raw GPS point")
    ax.set_ylabel("metres north of the raw GPS point")
    ax.set_title("map: yellow x = raw GPS, yellow o = automatic snap, numbers = candidates,\nred shade = wall mask, blue/red x-o = neighbours (Climb/Nc) raw -> snap", fontsize=8)
    # profile through the raw point along the wall's outward direction (left = into the hill, right = out into the air)
    out = frame["out"]
    t = np.arange(-half, half + 0.5, 0.5)
    line = raw_xy[None, :] + t[:, None] * out[None, :]
    pr.plot(t, grid.dem_at(line), "-", color="#8a6d3b", lw=1.5, label="bare-earth ground grid")
    pr.fill_between(t, grid.dem_at(line).min() - 2, grid.dem_at(line), where=grid.is_wall(line), color="red", alpha=0.15, label="wall mask")
    r2 = crop["xyz"][:, :2] - raw_xy
    m = np.abs(r2 @ frame["along"]) <= 2.5
    pr.scatter((r2 @ out)[m & ~st], crop["xyz"][m & ~st, 2], s=1, color="#bbbbbb", linewidths=0, label="all points within 2.5 m of the line")
    pr.scatter((r2 @ out)[m & st], crop["xyz"][m & st, 2], s=6, color="#2ca02c", linewidths=0, label="cliff points")
    if np.isfinite(frame["ground_z"]):
        pr.axhline(frame["ground_z"], color="black", lw=0.6, ls=":")
        pr.plot(0, frame["ground_z"], "x", color="black", ms=11, mew=2, label="raw GPS (standing height)")
    for c in cands:
        if abs((c["xy"] - raw_xy) @ frame["along"]) <= 4 and np.isfinite(c["base_z"]):
            to = (c["xy"] - raw_xy) @ out
            pr.plot([to, to], [c["base_z"], c["base_z"] + c["grid_h"]], "-", color="black", lw=1)
            pr.annotate(str(c["id"]), (to, c["base_z"]), fontsize=10, fontweight="bold", color="white", ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.15", fc="black", ec="yellow", lw=1))
    pr.set_xlabel("metres from the raw GPS point along the wall's outward direction  (left: into the hill, right: out into the air)")
    pr.set_ylabel("elevation (m)")
    pr.set_title("profile across the wall through the raw point; black bars = candidate foot to wall top (ground grid)", fontsize=8)
    pr.legend(fontsize=6, loc="upper right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)


PROMPT = """# Judging the Horseshoe Canyon Ranch waypoints

You are checking where hand-recorded GPS waypoints belong on the rock. For each case you
get a picture (`cases/<id>.png`) and numbers (`cases/<id>.json`). Decide which candidate
spot is the foot of the rock the label is about.

## Facts

* Each waypoint is where a person **stood at the base of a wall** and recorded a label:
  `Climb` (a climbing route starts here; often with the route name) or `Nc` (this rock is
  not climbable).
* Handheld GPS under a cliff is off by **3-10 m**.
* **East Side** points sit **7-12 m out on the downhill side** of the wall (a known offset).
* Walls **under 6 m** tall don't count (bouldering height). Heights are measured on the
  bare-earth ground grid.
* A `Climb` and an `Nc` point may be close together. That is real (a route beside a blank
  or loose stretch), not an error.
* **Nobody will check your answer by hand.** Mark real uncertainty as low confidence.

## The picture

* Left, the map (60 m x 60 m, north up, axes in metres east / north of the raw GPS point):
  grey hillshade of the ground, brown 2 m contour lines, **red shade = wall mask** (steep
  ground with relief), coloured dots = cliff LiDAR points (colour = height above the wall
  foot), **yellow x = the raw GPS point**, **yellow o = the automatic snap**, **numbered
  black circles = candidate spots**, blue (Climb) / red (Nc) x = neighbouring waypoints'
  raw points, joined by a line to their own automatic snap (o).
* Right, the profile: a cut across the wall through the raw point. Left is into the hill,
  right is out into the air. Brown line = ground, green dots = cliff points, black bars =
  candidates (foot to wall top). The person stood at the black x.

## The numbers (`<id>.json`)

* `point`: label, name, wall section (`area_name`), time, the automatic snap's status and flags.
* `candidates`: for each numbered spot: what it is (`roles`), distance from the raw point,
  how far it is along the wall from straight in front (`along_shift_m`), wall height on the
  ground grid (`wall_height_grid_m`) and in the upright column, foot height minus the
  standing height, cliff points within 2.5 m, and which wall piece it is on (`wall_piece`).
* `walking_order`: the three waypoints recorded before and after this one (by time), with
  their labels, names and where their own automatic snaps are relative to this raw point.

## What to decide, per case

* a candidate number: that spot is the foot of the rock this label is about; or
* `"none"`: no candidate fits and no better spot exists - the point should be dropped; or
* `"custom"` with `custom_east_m` and `custom_north_m` (metres east / north of the raw GPS
  point, read from the map axes). A custom spot **must** lie within 2 m of the wall mask
  (red shade) on a wall at least 6 m tall, or it is rejected. Use it rarely.

## How to decide

* Prefer the wall the person faced, on the downhill side of it.
* The foot should be near the standing height (the black x in the profile).
* Keep the walking order: consecutive waypoints should progress along the wall, unless the
  times show the person turned back.
* A route and its neighbours should end up on the same stretch of wall.
* Keep neighbouring points apart about as far as their raw points are apart along the wall;
  do not put two points on the same spot unless their raw points are that close.
* Never pick a wall just because it is taller.
* If the automatic snap (candidate 1) is reasonable, keep it.
* If unsure, say so with low confidence rather than guessing.

## Output

Read your batch file (a JSON list of case ids in walking order). For every id read
`cases/<id>.png` (with the Read tool) and `cases/<id>.json`. Then write **one** file, the
decisions file named in your instructions, as JSON:

```json
[
  {"gt_id": "Waypoints_2026-10-24#0", "choice": 1, "confidence": "high", "reason": "one or two sentences"},
  {"gt_id": "Waypoints_2026-10-24#1", "choice": "custom", "custom_east_m": -3.5, "custom_north_m": 6.0, "confidence": "low", "reason": "..."},
  {"gt_id": "Waypoints_2026-10-24#2", "choice": "none", "confidence": "medium", "reason": "..."}
]
```

`confidence` is `high`, `medium` or `low`. One entry per case, every case answered once.
**Read the case files; write only that one decisions file; change nothing else.**
"""


# ---------------------------------------------------------------------------
# step 3: check and apply
# ---------------------------------------------------------------------------


def validate_decisions(decisions: list[dict], cases: dict[str, dict], check_custom) -> dict[str, dict]:
    """Decisions of one judge keyed by ``gt_id``; raises ValueError with a clear message.

    ``cases`` maps ``gt_id`` to its case JSON; ``check_custom(case, east, north) -> str | None``
    returns why a custom spot is not allowed (off the wall mask, wall too short)."""
    out = {}
    for d in decisions:
        gid = d.get("gt_id")
        if gid not in cases:
            raise ValueError(f"unknown case {gid!r}")
        if gid in out:
            raise ValueError(f"case {gid} answered twice")
        ch = d.get("choice")
        if d.get("confidence") not in CONFIDENCE_ORDER:
            raise ValueError(f"{gid}: confidence must be high / medium / low, got {d.get('confidence')!r}")
        if ch == "none":
            pass
        elif ch == "custom":
            try:
                e, n = float(d["custom_east_m"]), float(d["custom_north_m"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"{gid}: custom needs custom_east_m and custom_north_m") from None
            why = check_custom(cases[gid], e, n)
            if why:
                raise ValueError(f"{gid}: custom spot rejected - {why}")
        else:
            try:
                ch = int(ch)
            except (TypeError, ValueError):
                raise ValueError(f"{gid}: choice must be a candidate number, 'none' or 'custom', got {d.get('choice')!r}") from None
            if ch not in {c["id"] for c in cases[gid]["candidates"]}:
                raise ValueError(f"{gid}: candidate {ch} does not exist")
            d = {**d, "choice": ch}
        out[gid] = d
    missing = set(cases) - set(out)
    if missing:
        raise ValueError(f"{len(missing)} cases not answered, e.g. {sorted(missing)[:3]}")
    return out


def decision_offset(case: dict, d: dict) -> np.ndarray | None:
    """Where a decision puts the point, as metres east / north of the raw point (None for ``none``)."""
    if d["choice"] == "none":
        return None
    if d["choice"] == "custom":
        return np.array([float(d["custom_east_m"]), float(d["custom_north_m"])])
    c = next(c for c in case["candidates"] if c["id"] == d["choice"])
    return np.array([c["east_m"], c["north_m"]])


def same_decision(case: dict, a: dict, b: dict, tol_m: float = 1.5) -> bool:
    """Agreement: the same candidate, spots within ``tol_m`` of each other, or both ``none``."""
    pa, pb = decision_offset(case, a), decision_offset(case, b)
    if pa is None or pb is None:
        return pa is None and pb is None
    return a["choice"] == b["choice"] != "custom" or float(np.linalg.norm(pa - pb)) <= tol_m


def keeps_auto(case: dict, d: dict, tol_m: float = 1.5) -> bool:
    auto = next((c for c in case["candidates"] if "automatic snap" in c["roles"]), None)
    p = decision_offset(case, d)
    return auto is not None and p is not None and float(np.linalg.norm(p - [auto["east_m"], auto["north_m"]])) <= tol_m


def calibration(cases: dict[str, dict], dec: dict[str, dict]) -> tuple[float, list[str]]:
    """Share of easy cases where the judge kept the automatic snap, and the easy cases it changed."""
    easy = [g for g, c in cases.items() if c["easy"] and g in dec]
    changed = [g for g in easy if not keeps_auto(cases[g], dec[g])]
    return (1.0 - len(changed) / len(easy)) if easy else 1.0, changed


def resolve(case: dict, a: dict, b: dict, c: dict | None) -> dict:
    """Final decision for one point (plan 4, section 7.4 step 5)."""
    def pack(d, result):
        p = decision_offset(case, d)
        status = "judge_none" if p is None else ("ok" if keeps_auto(case, d) else "judged")
        return {"offset": p, "snap_status": status, "judge_result": result}

    if same_decision(case, a, b):
        return pack(a, "agreed")
    if c is not None:
        for d in (a, b):
            if same_decision(case, c, d):
                return pack(d, "majority")
    auto = next((x for x in case["candidates"] if "automatic snap" in x["roles"]), None)
    if auto is None:  # no automatic snap to fall back on (the point failed the wall check): drop it
        return {"offset": None, "snap_status": "judge_none", "judge_result": "unresolved"}
    return {"offset": np.array([auto["east_m"], auto["north_m"]]), "snap_status": "ok", "judge_result": "unresolved"}


def order_reversed(final_xy: np.ndarray, raw_xy: np.ndarray, along: np.ndarray, times: np.ndarray, min_m: float = 1.0) -> np.ndarray:
    """Per point: did the order along the wall of this point and the next one (by time) reverse?"""
    flag = np.zeros(len(times), bool)
    order = np.argsort(times, kind="stable")
    for i, j in zip(order[:-1], order[1:]):
        if not (np.isfinite(final_xy[[i, j]]).all() and np.isfinite(along[i]).all()):
            continue
        r, f = (raw_xy[j] - raw_xy[i]) @ along[i], (final_xy[j] - final_xy[i]) @ along[i]
        if abs(r) > min_m and abs(f) > min_m and np.sign(r) != np.sign(f):
            flag[i] = flag[j] = True
    return flag


def custom_checker(grid_for_case, min_height: float = 6.0):
    """``check_custom`` for :func:`validate_decisions`, from a function giving (grid, raw_xy, pc, ground_z) per case."""
    def check(case, e, n):
        grid, raw_xy, pc, ground_z = grid_for_case(case)
        xy = np.asarray(raw_xy, float) + [e, n]
        if float(grid.distance(xy)[0]) > 2.0:
            return f"{float(grid.distance(xy)[0]):.1f} m from the wall mask (limit 2 m)"
        idx = pc.column(xy, 2.5)
        base = float(np.quantile(pc.xyz[idx, 2], 0.05)) if len(idx) else ground_z
        h = wall_height_at(grid, xy, base)[0]
        if h < min_height:
            return f"wall only {h:.1f} m tall there (minimum {min_height:g} m)"
        return None
    return check


def dumps(obj) -> str:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(obj, separators=(",", ":"), default=default)
