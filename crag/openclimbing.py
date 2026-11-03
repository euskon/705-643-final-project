"""Route-level climbing locations from OpenStreetMap - the data openclimbing.org shows.

openclimbing.org (OsmAPP) stores climbing data directly in OSM using the
``climbing=*`` tagging scheme (https://wiki.openstreetmap.org/wiki/Key:climbing):

* ``climbing=route_bottom`` node - **the start of one route** (what we want),
* ``climbing=route`` way - a route line, bottom to top (start = lower end),
* ``climbing=route`` node - a route point (older / less precise mappings),
* ``climbing=crag`` / ``climbing=area`` - nodes or ``type=site`` relations
  grouping routes (relation members give each route its crag name).

Useful route tags: ``name``, ``climbing:grade:yds_class`` (/uiaa/french/
hueco), ``climbing:length``, ``climbing:sport|trad|boulder|toprope``,
``climbing:url:mountainproject`` (a direct link to the MP route - exact
cross-reference), ``climbing:url:openbeta``.

Queries go to the public Overpass API, one at a time, cached in
``data/raw/overpass/`` (keyed by query hash) so a re-run makes no requests.
Mountain Project coordinates are never used as route locations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time

import numpy as np
import pandas as pd
import requests

from crag.paths import RAW_DIR

log = logging.getLogger(__name__)

ENDPOINTS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
USER_AGENT = "crag-detection-research/0.2 (JHU EN.705.643 coursework; cached queries)"
CACHE_DIR = RAW_DIR / "overpass"
PAUSE_S = 5.0
_last = 0.0

GRADE_KEYS = ("climbing:grade:yds_class", "climbing:grade:uiaa", "climbing:grade:french", "climbing:grade:hueco", "climbing:grade:font")


def overpass(query: str, refresh: bool = False, timeout: int = 300) -> dict:
    global _last
    key = hashlib.sha1(query.encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{key}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    last_err = None
    for attempt in range(6):
        url = ENDPOINTS[attempt % len(ENDPOINTS)]
        wait = PAUSE_S - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=timeout + 30)
            _last = time.time()
            if resp.status_code in (429, 502, 503, 504):
                raise requests.HTTPError(f"{resp.status_code} from {url}")
            resp.raise_for_status()
            data = resp.json()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
            (CACHE_DIR / f"{key}.query").write_text(query)
            return data
        except (requests.RequestException, ValueError) as exc:
            last_err = exc
            log.warning("Overpass attempt %d failed (%s); backing off", attempt + 1, exc)
            time.sleep(min(30 * (attempt + 1), 120))
    raise RuntimeError(f"Overpass query failed: {last_err}")


def bbox_query(bbox_lonlat, timeout: int = 180) -> str:
    """Routes, route lines (with geometry) and crag/area groupings inside a bbox."""
    w, s, e, n = bbox_lonlat
    b = f"({s},{w},{n},{e})"
    return f"""[out:json][timeout:{timeout}];
(
  node["climbing"="route_bottom"]{b};
  node["climbing"="route"]{b};
  way["climbing"="route"]{b};
);
out tags geom;
(
  relation["climbing"~"^(crag|area)$"]{b};
  node["climbing"~"^(crag|area)$"]{b};
);
out tags center;
"""


def us_survey_query(timeout: int = 300) -> str:
    return f"""[out:json][timeout:{timeout}];
area["ISO3166-1"="US"][admin_level=2]->.us;
(
  node["climbing"="route_bottom"](area.us);
  node["climbing"="route"](area.us);
  way["climbing"="route"](area.us);
);
out tags center;
"""


def _mp_route_id(tags: dict) -> int | None:
    for k in ("climbing:url:mountainproject", "url", "website", "climbing:url"):
        m = re.search(r"mountainproject\.com/route/(\d+)", tags.get(k, ""))
        if m:
            return int(m.group(1))
    return None


def _length_m(tags: dict) -> float:
    v = tags.get("climbing:length", "")
    m = re.match(r"\s*([\d.]+)\s*(ft|')?", v)
    if not m:
        return np.nan
    x = float(m.group(1))
    return x * 0.3048 if m.group(2) else x


def _grade(tags: dict) -> tuple[str | None, str | None]:
    for k in GRADE_KEYS:
        if tags.get(k):
            return tags[k], k.rsplit(":", 1)[-1]
    return None, None


def parse_elements(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overpass JSON -> (routes, crags).

    Route location: the node for ``route_bottom``/``route`` nodes; for route
    *ways* both end points are kept (``lat, lon`` = first node, ``lat2, lon2``
    = last) - the lower one is chosen later from the LiDAR.
    """
    routes, crags, members = [], [], {}
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        kind = tags.get("climbing")
        if kind in ("crag", "area"):
            c = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
            crags.append({"osm_type": el["type"], "osm_id": el["id"], "kind": kind, "name": tags.get("name"), "lat": c.get("lat"), "lon": c.get("lon")})
            for m in el.get("members", []) or []:
                members[(m["type"], m["ref"])] = tags.get("name")
            continue
        if kind not in ("route_bottom", "route"):
            continue
        grade, system = _grade(tags)
        row = {
            "osm_type": el["type"],
            "osm_id": el["id"],
            "osm_kind": kind,
            "name": tags.get("name"),
            "grade": grade,
            "grade_system": system,
            "length_m": _length_m(tags),
            "is_boulder": tags.get("climbing:boulder") == "yes" or system in ("hueco", "font"),
            "is_sport": tags.get("climbing:sport") == "yes",
            "is_trad": tags.get("climbing:trad") == "yes",
            "mp_route_id_tag": _mp_route_id(tags),
            "openbeta_url": tags.get("climbing:url:openbeta"),
        }
        if el["type"] == "node":
            row.update(lat=el["lat"], lon=el["lon"], lat2=np.nan, lon2=np.nan)
        elif "geometry" in el and el["geometry"]:
            g = el["geometry"]
            row.update(lat=g[0]["lat"], lon=g[0]["lon"], lat2=g[-1]["lat"], lon2=g[-1]["lon"])
        elif "center" in el:
            row.update(lat=el["center"]["lat"], lon=el["center"]["lon"], lat2=np.nan, lon2=np.nan)
        else:
            continue
        routes.append(row)
    routes = pd.DataFrame(routes)
    if len(routes):
        routes["osm_crag"] = [members.get((t, i)) for t, i in zip(routes["osm_type"], routes["osm_id"])]
        routes = routes.drop_duplicates(["osm_type", "osm_id"]).reset_index(drop=True)
    return routes, pd.DataFrame(crags)


def bbox_cells(bbox_lonlat, max_cell_deg: float = 0.3, points: np.ndarray | None = None, margin_deg: float = 0.02) -> list[tuple]:
    """Split a bbox into a grid of cells no larger than ``max_cell_deg`` on a side.

    With ``points`` (lon, lat), cells with no point within ``margin_deg`` are
    dropped, since there is no crag there to query. A bbox that already fits is
    returned unchanged, so its cached query stays valid.
    """
    w, s, e, n = map(float, bbox_lonlat)
    nx, ny = max(1, int(np.ceil((e - w) / max_cell_deg))), max(1, int(np.ceil((n - s) / max_cell_deg)))
    if nx == 1 and ny == 1:
        return [(w, s, e, n)]
    xs, ys = np.linspace(w, e, nx + 1), np.linspace(s, n, ny + 1)
    cells = []
    for i in range(nx):
        for j in range(ny):
            c = (round(xs[i], 6), round(ys[j], 6), round(xs[i + 1], 6), round(ys[j + 1], 6))
            if points is not None and len(points):
                p = np.asarray(points, float)
                inside = (p[:, 0] >= c[0] - margin_deg) & (p[:, 0] <= c[2] + margin_deg) & (p[:, 1] >= c[1] - margin_deg) & (p[:, 1] <= c[3] + margin_deg)
                if not inside.any():
                    continue
            cells.append(c)
    return cells


def routes_in_bbox(bbox_lonlat, refresh: bool = False, points: np.ndarray | None = None, max_cell_deg: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OSM routes + crags inside a bbox.

    One query per cell (see :func:`bbox_cells`). A single query over a large area
    (Lake Tahoe is ~80 x 95 km) times out on the public Overpass servers (502/504).
    Elements are de-duplicated across cells before parsing.
    """
    cells = bbox_cells(bbox_lonlat, max_cell_deg, points)
    if len(cells) > 1:
        log.info("Overpass: bbox split into %d cells", len(cells))
    seen, elements = set(), []
    for c in cells:
        for el in overpass(bbox_query(c), refresh=refresh).get("elements", []):
            k = (el.get("type"), el.get("id"))
            if k not in seen:
                seen.add(k)
                elements.append(el)
    return parse_elements({"elements": elements})


def cluster_points(lonlat: np.ndarray, eps_m: float = 1500.0) -> np.ndarray:
    """Connected components of points closer than ``eps_m`` (single-linkage) -> cluster id."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    lat0 = np.radians(np.nanmean(lonlat[:, 1]))
    xy = np.stack([np.radians(lonlat[:, 0]) * np.cos(lat0), np.radians(lonlat[:, 1])], 1) * 6371008.8
    # equirectangular: a few % distortion of eps across the US, irrelevant for grouping crags
    tree = cKDTree(xy)
    pairs = tree.query_pairs(eps_m, output_type="ndarray")
    from scipy.sparse import coo_matrix

    n = len(xy)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    return connected_components(g, directed=False)[1]
