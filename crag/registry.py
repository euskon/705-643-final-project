"""The area registry: ``registry/areas.json``, one entry per climbing area.

Adding an area = adding one entry with at least ``slug``, ``name`` and
``mp_id``; ``scripts/s1_build_registry.py`` fills in every missing field
(MP GPS/popularity, crags, bounding box, UTM zone, USGS datasets, split) and
never overwrites a field that is already set, so hand edits stick.

Entry schema (see ``docs/registry-schema.md`` for the full description)::

    {
      "slug": "red-river-gorge",            # unique id, used for directories
      "name": "Red River Gorge",
      "state": "Kentucky",
      "mp_id": 105841134,
      "mp_url": "https://www.mountainproject.com/area/105841134/red-river-gorge",
      "lat": 37.67745, "lon": -83.68217,    # MP area pin
      "popularity": {"mp_page_views_total": 7036854, "mp_page_views_month": 28487,
                     "rank": 1, "retrieved": "2026-11-01"},
      "bbox": [min_lon, min_lat, max_lon, max_lat],   # encloses all crags + margin
      "utm_epsg": 32617,
      "usgs": {"datasets": [{"name": ..., "url": ..., "coverage": 0.98,
                             "points_per_m2": 11.2, "year": 2019}, ...]},
      "n_routes": 3342,
      "crags": [{"name": ..., "lat": ..., "lon": ..., "n_routes": ...,
                 "default_pin": false, "location": "A > B > Area > State"}, ...],
      "split": "train",                     # pretraining split: train|val|test|holdout
      "enabled": true,
      "notes": ""
    }
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from crag.geo import AreaCRS
from crag.paths import AREAS_JSON

SPLITS = ("train", "val", "test", "holdout")
SCHEMA_VERSION = 1


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower().replace("'", "")).strip("-")
    return s or "area"


def load(path: Path = AREAS_JSON) -> dict:
    if not Path(path).exists():
        return {"schema_version": SCHEMA_VERSION, "areas": []}
    reg = json.loads(Path(path).read_text())
    validate(reg)
    return reg


def save(reg: dict, path: Path = AREAS_JSON) -> None:
    """Pretty JSON with one crag per line, so diffs stay readable."""
    validate(reg)
    lines = ["{", f'  "schema_version": {SCHEMA_VERSION},']
    if "description" in reg:
        lines.append(f'  "description": {json.dumps(reg["description"])},')
    lines.append('  "areas": [')
    for i, a in enumerate(reg["areas"]):
        body = []
        for k, v in a.items():
            if k == "crags":
                crag_lines = ",\n".join("        " + json.dumps(c) for c in v)
                body.append(f'      "crags": [\n{crag_lines}\n      ]' if v else '      "crags": []')
            else:
                body.append(f"      {json.dumps(k)}: {json.dumps(v)}")
        lines.append("    {\n" + ",\n".join(body) + "\n    }" + ("," if i < len(reg["areas"]) - 1 else ""))
    lines += ["  ]", "}"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def validate(reg: dict) -> None:
    slugs = [a.get("slug") for a in reg["areas"]]
    dupes = {s for s in slugs if slugs.count(s) > 1}
    if dupes:
        raise ValueError(f"duplicate slugs in registry: {sorted(dupes)}")
    for a in reg["areas"]:
        for k in ("slug", "name"):
            if not a.get(k):
                raise ValueError(f"registry entry missing {k!r}: {a}")
        if a.get("split") is not None and a["split"] not in SPLITS:
            raise ValueError(f"{a['slug']}: split must be one of {SPLITS}, got {a['split']!r}")
        bb = a.get("bbox")
        if bb is not None and not (len(bb) == 4 and bb[0] < bb[2] and bb[1] < bb[3]):
            raise ValueError(f"{a['slug']}: bbox must be [min_lon, min_lat, max_lon, max_lat], got {bb}")


def areas(reg: dict, slugs: list[str] | None = None, enabled_only: bool = True) -> list[dict]:
    """Registry entries filtered by slug list (None = all) and ``enabled``."""
    out = [a for a in reg["areas"] if not enabled_only or a.get("enabled", True)]
    if slugs:
        unknown = set(slugs) - {a["slug"] for a in reg["areas"]}
        if unknown:
            raise KeyError(f"unknown area slug(s): {sorted(unknown)}")
        out = [a for a in out if a["slug"] in slugs]
    return out


def area_crs(area: dict) -> AreaCRS:
    return AreaCRS(area["utm_epsg"])


def stable_fraction(text: str) -> float:
    """Deterministic pseudo-random number in [0, 1) from a string (for splits)."""
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16) / 2**32


def assign_splits(reg: dict, val_frac: float = 0.1, test_frac: float = 0.1, seed: str = "crag") -> None:
    """Give every area without a ``split`` one, deterministically from its slug.

    Areas are ranked by a hash of ``seed + slug``; the first ``test_frac`` go
    to test, the next ``val_frac`` to val, the rest to train. Already-set
    splits (e.g. ``holdout``) are kept and not counted.
    """
    todo = [a for a in reg["areas"] if not a.get("split")]
    todo.sort(key=lambda a: stable_fraction(seed + a["slug"]))
    n = len(todo)
    n_test, n_val = round(n * test_frac), round(n * val_frac)
    for i, a in enumerate(todo):
        a["split"] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
