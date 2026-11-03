"""Match OSM/OpenClimbing route locations to documented Mountain Project routes.

Mountain Project provides *which* routes exist (name, grade, length, wall) but
only rough wall-level pins; OSM provides *where* a route starts. A route
location is kept for fine-tuning only if it matches a documented MP route:

1. ``mp_url``   - the OSM feature links the MP route (``climbing:url:mountainproject``);
2. ``name``     - normalised names equal, MP wall pin within ``max_pin_dist_m``;
3. ``fuzzy``    - name similarity >= ``fuzzy_min`` *and* compatible YDS grade,
                  MP pin within ``max_pin_dist_m``.

Each OSM route matches at most one MP route and vice versa (best score wins).
The MP pin distance is a sanity check only (pins are 10-500 m off), never a
location.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

import numpy as np
import pandas as pd

from crag.geo import haversine_m

_STOP = {"the", "a", "an"}


def normalize_name(s) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)", " ", s)  # "(aka ...)", "(variation)"
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    words = [w for w in s.split() if w not in _STOP]
    return " ".join(words)


def yds_core(grade) -> str | None:
    """'5.10b' / '5.10 PG13' / '5.10b/c' -> '5.10' (number only) for a loose compatibility check."""
    if not isinstance(grade, str):
        return None
    m = re.search(r"5\.(\d{1,2})", grade)
    return f"5.{m.group(1)}" if m else None


def grades_compatible(a, b) -> bool:
    ca, cb = yds_core(a), yds_core(b)
    if ca is None or cb is None:
        return True  # unknown grade never blocks a name match
    return abs(int(ca[2:]) - int(cb[2:])) <= 1


def match_routes(osm: pd.DataFrame, mpr: pd.DataFrame, max_pin_dist_m: float = 2000.0, fuzzy_min: float = 0.88) -> pd.DataFrame:
    """One row per OSM route with ``match_type`` in {mp_url, name, fuzzy, none} and MP columns."""
    osm = osm.reset_index(drop=True).copy()
    osm["norm"] = osm["name"].map(normalize_name)
    mpr = mpr.reset_index(drop=True).copy()
    mpr["norm"] = mpr["route"].map(normalize_name)
    by_id = {int(r): i for i, r in enumerate(mpr["route_id"])}
    by_name: dict[str, list[int]] = {}
    for i, n in enumerate(mpr["norm"]):
        if n:
            by_name.setdefault(n, []).append(i)

    cands = []  # (score, osm_i, mp_i, type, dist)
    for oi, o in osm.iterrows():
        d_all = None
        tag = o.get("mp_route_id_tag")
        if pd.notna(tag) and int(tag) in by_id:
            mi = by_id[int(tag)]
            d = float(haversine_m(o["lon"], o["lat"], mpr.at[mi, "lon"], mpr.at[mi, "lat"]))
            cands.append((3.0, oi, mi, "mp_url", d))
            continue
        if not o["norm"]:
            continue
        d_all = haversine_m(o["lon"], o["lat"], mpr["lon"].to_numpy(), mpr["lat"].to_numpy())
        for mi in by_name.get(o["norm"], []):
            if d_all[mi] <= max_pin_dist_m and grades_compatible(o.get("grade"), mpr.at[mi, "rating"]):
                cands.append((2.0 - d_all[mi] / 1e6, oi, mi, "name", float(d_all[mi])))
        near = np.flatnonzero(d_all <= max_pin_dist_m)
        for mi in near:
            n = mpr.at[mi, "norm"]
            if not n or n == o["norm"]:
                continue
            r = difflib.SequenceMatcher(None, o["norm"], n).ratio()
            if r >= fuzzy_min and grades_compatible(o.get("grade"), mpr.at[mi, "rating"]):
                cands.append((r - d_all[mi] / 1e6, oi, mi, "fuzzy", float(d_all[mi])))

    cands.sort(key=lambda c: -c[0])
    used_o, used_m, best = set(), set(), {}
    for score, oi, mi, typ, d in cands:
        if oi in used_o or mi in used_m:
            continue
        used_o.add(oi)
        used_m.add(mi)
        best[oi] = (mi, typ, score, d)
    cols = {"mp_route_id": [], "mp_name": [], "mp_wall": [], "mp_location": [], "mp_rating": [], "mp_length_m": [], "match_type": [], "match_score": [], "dist_to_mp_pin_m": []}
    for oi in range(len(osm)):
        if oi in best:
            mi, typ, score, d = best[oi]
            m = mpr.iloc[mi]
            vals = [int(m["route_id"]), m["route"], m["wall"], m["location"], m["rating"], m["length_m"], typ, round(min(score, 1.0) if typ == "fuzzy" else 1.0, 3), round(d, 1)]
        else:
            vals = [None, None, None, None, None, np.nan, "none", 0.0, np.nan]
        for k, v in zip(cols, vals):
            cols[k].append(v)
    out = pd.concat([osm.drop(columns=["norm"]), pd.DataFrame(cols)], axis=1)
    return out


def crag_coverage(matched: pd.DataFrame, mpr: pd.DataFrame) -> pd.DataFrame:
    """Per MP wall: documented routes vs routes with a verified OSM location.

    ``coverage`` near 1 means the wall is mapped completely, so cliff between
    mapped routes is unlikely to hide undocumented routes (used to trust
    "gap" negatives).
    """
    tot = mpr.groupby(["wall", "lat", "lon"]).size().rename("mp_routes")
    ok = matched[matched["match_type"] != "none"]
    got = ok.groupby("mp_wall").size().rename("located")
    cov = tot.reset_index().merge(got, left_on="wall", right_index=True, how="left").fillna({"located": 0})
    cov["coverage"] = (cov["located"] / cov["mp_routes"]).clip(upper=1.0)
    return cov.sort_values("located", ascending=False).reset_index(drop=True)
