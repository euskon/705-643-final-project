"""Known climbing routes from Mountain Project, used as weak positive labels.

Rather than crawling hundreds of route pages, this uses the site's own
"Route Finder" CSV export: one HTTP request returns every rock route under an
area with its type, grade, length (ft) and the lat/lon of the wall it is on.
The response is cached so the site is hit once per area, ever. Mountain
Project's robots.txt asks for a 60 s crawl delay; :func:`fetch_route_csv`
enforces it between live requests, and its terms of use restrict automated
collection, so keep this to small, cached, non-commercial pulls (the CC-BY
OpenBeta dataset carries the same routes if a larger pull is ever needed).

Important limitation: the coordinates are per *wall*, not per route - all
142 North Forty routes share one pin - and pins are typically 10-30 m from
the actual rock. :mod:`labeling` therefore snaps each wall pin to the nearby
cliff line before placing route cylinders.

Example::

    from mountain_project import load_routes, walls_from_routes

    routes = load_routes()                 # one row per route
    walls = walls_from_routes(routes)      # one row per wall pin
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import DATA_DIR, FT_TO_M, MP_AREA_ID, MP_DEFAULT_PIN

EXPORT_URL = "https://www.mountainproject.com/route-finder-export"
USER_AGENT = "crag-detection-poc/0.1 (JHU EN.705.643 coursework; single cached request)"
CRAWL_DELAY_S = 60.0
_last_request = 0.0


def fetch_route_csv(area_id: int = MP_AREA_ID) -> str:
    """Download the route-finder CSV for every rock route under ``area_id``."""
    global _last_request
    wait = CRAWL_DELAY_S - (time.monotonic() - _last_request)
    if _last_request and wait > 0:
        time.sleep(wait)
    params = {
        "selectedIds": area_id,
        "type": "rock",
        # 800 / 12400 are the site's internal codes for 5.0 and 5.15d,
        # i.e. "any grade"
        "diffMinrock": 800,
        "diffMaxrock": 12400,
        "stars": 0,
        "pitches": 0,
        "sort1": "area",
        "sort2": "rating",
    }
    resp = requests.get(
        EXPORT_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=60
    )
    _last_request = time.monotonic()
    resp.raise_for_status()
    if not resp.text.startswith("Route,"):
        raise RuntimeError(f"unexpected route-finder response: {resp.text[:200]!r}")
    return resp.text


def parse_routes(csv_text: str) -> pd.DataFrame:
    """Raw export -> tidy one-row-per-route frame with metric lengths."""
    df = pd.read_csv(io.StringIO(csv_text))
    df = df.rename(
        columns={
            "Route": "route",
            "Location": "location",
            "URL": "url",
            "Avg Stars": "stars",
            "Route Type": "route_type",
            "Rating": "rating",
            "Pitches": "pitches",
            "Length": "length_ft",
            "Area Latitude": "lat",
            "Area Longitude": "lon",
        }
    ).drop(columns=["Your Stars"], errors="ignore")
    df["route_id"] = df["url"].str.extract(r"/route/(\d+)/")[0].astype("int64")
    # "North Forty Routes > North Side > Horseshoe Canyon Ranch > ..." -> wall
    df["wall"] = df["location"].str.split(" > ").str[0]
    df["length_m"] = df["length_ft"] * FT_TO_M
    df["default_pin"] = np.isclose(df["lat"], MP_DEFAULT_PIN[0]) & np.isclose(
        df["lon"], MP_DEFAULT_PIN[1]
    )
    return df


def load_routes(area_id: int = MP_AREA_ID, refresh: bool = False) -> pd.DataFrame:
    """Cached :func:`fetch_route_csv` + :func:`parse_routes`."""
    path = Path(DATA_DIR) / "mountain_project" / f"route_finder_{area_id}.csv"
    if path.exists() and not refresh:
        text = path.read_text()
    else:
        text = fetch_route_csv(area_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return parse_routes(text)


def filter_bbox(
    routes: pd.DataFrame, bbox_lonlat: tuple[float, float, float, float]
) -> pd.DataFrame:
    """Keep the routes whose wall pin lies inside (min_lon, min_lat, max_lon, max_lat)."""
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    inside = routes["lon"].between(min_lon, max_lon) & routes["lat"].between(
        min_lat, max_lat
    )
    return routes[inside].reset_index(drop=True)


def walls_from_routes(routes: pd.DataFrame) -> pd.DataFrame:
    """Collapse routes to one row per (wall, pin) with route count and lengths."""
    walls = (
        routes.groupby(["wall", "lat", "lon", "default_pin"], as_index=False)
        .agg(
            n_routes=("route_id", "size"),
            median_length_m=("length_m", "median"),
            max_length_m=("length_m", "max"),
        )
        .sort_values("n_routes", ascending=False)
        .reset_index(drop=True)
    )
    return walls
