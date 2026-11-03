"""Mountain Project: area pages (popularity, GPS, breadcrumb) and route-finder exports.

Politeness / reproducibility rules:

* every live request waits until ``CRAWL_DELAY_S`` (robots.txt: 60 s) has
  passed since the previous one - tracked in a file, so two scripts running at
  once still respect it;
* every response is cached under ``data/raw/mp`` and never re-fetched unless
  ``refresh=True``; the merged per-area route CSVs live in
  ``data/mountain_project`` (small, committed) so a fresh clone needs no
  requests at all.

The route-finder export silently truncates at 1000 routes (Red River Gorge has
several thousand). :func:`fetch_routes` therefore bisects the grade range
(``diffMinrock``/``diffMaxrock``) until every request returns < 1000 rows and
de-duplicates by route URL.
"""

from __future__ import annotations

import fcntl
import html as htmllib
import io
import logging
import re
import time

import numpy as np
import pandas as pd
import requests

from crag.paths import MP_ROUTES_DIR, RAW_DIR

log = logging.getLogger(__name__)

BASE = "https://www.mountainproject.com"
EXPORT_URL = f"{BASE}/route-finder-export"
USER_AGENT = "crag-detection-research/0.2 (JHU EN.705.643 coursework; cached, 60s crawl delay)"
CRAWL_DELAY_S = 60.0
EXPORT_ROW_CAP = 1000
# route-finder's internal YDS codes: 800 = 5.0 ... 12400 = 5.15d
GRADE_MIN, GRADE_MAX = 800, 12400
FT_TO_M = 0.3048

MP_RAW_DIR = RAW_DIR / "mp"
_LOCK_FILE = MP_RAW_DIR / ".last_request"


def _polite_get(url: str, params: dict | None = None, timeout: float = 120) -> str:
    """GET with the cross-process crawl delay and a few retries."""
    MP_RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(_LOCK_FILE, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)  # one MP request at a time, machine-wide
        fh.seek(0)
        txt = fh.read().strip()
        last = float(txt) if txt else 0.0
        wait = CRAWL_DELAY_S - (time.time() - last)
        if wait > 0:
            log.info("MP crawl delay: sleeping %.0fs", wait)
            time.sleep(wait)
        try:
            for attempt in range(3):
                try:
                    resp = requests.get(
                        url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout
                    )
                    resp.raise_for_status()
                    return resp.text
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(CRAWL_DELAY_S)
        finally:
            fh.seek(0)
            fh.truncate()
            fh.write(str(time.time()))
            fh.flush()
            fcntl.flock(fh, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# area pages
# ---------------------------------------------------------------------------


def area_url(mp_id: int) -> str:
    return f"{BASE}/area/{int(mp_id)}"


# MP pages embed their Mapbox access token (static map images). The cache is
# committed, so strip it before saving; nothing here needs it, and GitHub push
# protection rejects it.
_MAPBOX_TOKEN = re.compile(r"\b[ps]k\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")


def redact_tokens(text: str) -> str:
    return _MAPBOX_TOKEN.sub("REDACTED", text)


def fetch_area_page(mp_id: int, refresh: bool = False) -> str:
    path = MP_RAW_DIR / "areas" / f"{int(mp_id)}.html"
    if path.exists() and not refresh:
        return path.read_text()
    text = redact_tokens(_polite_get(area_url(mp_id)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse_area_page(page: str) -> dict:
    """Name, canonical URL, GPS, page views and breadcrumb of an MP area page."""
    out: dict = {}
    m = re.search(r"<h1>(.*?)<span", page, re.S) or re.search(r"<h1>(.*?)</h1>", page, re.S)
    out["name"] = _clean(m.group(1)) if m else None
    m = re.search(r"Page Views:</td>\s*<td>\s*([\d,]+)\s*total\s*&middot;\s*([\d,]+)/month", page)
    out["page_views_total"] = int(m.group(1).replace(",", "")) if m else None
    out["page_views_month"] = int(m.group(2).replace(",", "")) if m else None
    m = re.search(r"<td>GPS:</td>\s*<td>\s*(-?[\d.]+),\s*(-?[\d.]+)", page)
    out["lat"], out["lon"] = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = re.search(r'<div class="mb-half small text-warm">(.*?)</div>', page, re.S)
    crumbs = re.findall(r'/area/(\d+)/([^"]+)">([^<]+)</a>', m.group(1)) if m else []
    out["breadcrumb"] = [(int(i), htmllib.unescape(n).strip()) for i, _, n in crumbs]
    out["state"] = out["breadcrumb"][0][1] if out["breadcrumb"] else None
    m = re.search(r'<link rel="canonical" href="([^"]+)"', page)
    out["url"] = m.group(1) if m else None
    return out


# ---------------------------------------------------------------------------
# route-finder export
# ---------------------------------------------------------------------------


def _export_params(mp_id: int, lo: int, hi: int) -> dict:
    return {
        "selectedIds": int(mp_id),
        "type": "rock",
        "diffMinrock": lo,
        "diffMaxrock": hi,
        "stars": 0,
        "pitches": 0,
        "sort1": "area",
        "sort2": "rating",
    }


def _fetch_band(mp_id: int, lo: int, hi: int, refresh: bool) -> str:
    path = MP_RAW_DIR / "route_finder" / f"{int(mp_id)}_{lo}_{hi}.csv"
    if path.exists() and not refresh:
        return path.read_text()
    text = _polite_get(EXPORT_URL, _export_params(mp_id, lo, hi))
    if not text.startswith("Route,"):
        raise RuntimeError(f"unexpected route-finder response for {mp_id}: {text[:200]!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def fetch_routes_csv(mp_id: int, refresh: bool = False) -> str:
    """Every rock route under ``mp_id`` as one CSV text (bisecting past the 1000-row cap).

    Cached as ``data/mountain_project/route_finder_<id>.csv`` (same name the
    proof of concept used, so the Horseshoe Canyon Ranch file is reused).
    """
    merged = MP_ROUTES_DIR / f"route_finder_{int(mp_id)}.csv"
    if merged.exists() and not refresh:
        return merged.read_text()
    frames, truncated = [], []
    stack = [(GRADE_MIN, GRADE_MAX)]
    while stack:
        lo, hi = stack.pop()
        text = _fetch_band(mp_id, lo, hi, refresh)
        df = pd.read_csv(io.StringIO(text))
        if len(df) >= EXPORT_ROW_CAP:
            if hi - lo <= 1:
                truncated.append((lo, hi))  # can't split further; keep what we got
            else:
                mid = (lo + hi) // 2
                # overlapping at `mid` is harmless (dedup below) and safe whether
                # the bounds are inclusive or not
                stack += [(lo, mid), (mid, hi)]
                continue
        frames.append(df)
    if truncated:
        log.warning("area %s: grade bands %s still hit the %d-row cap", mp_id, truncated, EXPORT_ROW_CAP)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("URL")
    df = df.sort_values(["Location", "Route"]).reset_index(drop=True)
    merged.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(merged, index=False)
    return merged.read_text()


def parse_routes(csv_text: str, area_pin: tuple[float, float] | None = None) -> pd.DataFrame:
    """Export -> one row per route: ``route, route_id, wall, location, route_type, rating,
    length_m, lat, lon, default_pin`` (same columns as the proof of concept).

    ``default_pin`` flags routes whose pin was never placed on a real wall:
    it equals the area's own pin, or several different walls share it (an
    inherited parent-area pin).
    """
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
    df["route_id"] = df["url"].str.extract(r"/route/(\d+)")[0].astype("int64")
    df["wall"] = df["location"].str.split(" > ").str[0]
    df["length_m"] = pd.to_numeric(df["length_ft"], errors="coerce") * FT_TO_M
    walls_per_pin = df.groupby(["lat", "lon"])["wall"].transform("nunique")
    default = walls_per_pin > 1
    if area_pin is not None:
        default |= np.isclose(df["lat"], area_pin[0]) & np.isclose(df["lon"], area_pin[1])
    df["default_pin"] = default.to_numpy()
    return df


def load_routes(mp_id: int, area_pin=None, refresh: bool = False) -> pd.DataFrame:
    return parse_routes(fetch_routes_csv(mp_id, refresh), area_pin)


def crags_from_routes(routes: pd.DataFrame) -> pd.DataFrame:
    """One row per crag (MP leaf sub-area pin): ``name, location, lat, lon, n_routes, default_pin``."""
    crags = (
        routes.groupby(["wall", "lat", "lon"], as_index=False)
        .agg(
            location=("location", "first"),
            n_routes=("route_id", "size"),
            default_pin=("default_pin", "first"),
            median_length_m=("length_m", "median"),
        )
        .rename(columns={"wall": "name"})
        .sort_values("n_routes", ascending=False)
        .reset_index(drop=True)
    )
    return crags
