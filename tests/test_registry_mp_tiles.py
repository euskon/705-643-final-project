"""Step 1 building blocks: registry, Mountain Project parsing/bisection, tile planning."""

import io

import numpy as np
import pandas as pd
import pytest

from crag import mp, registry, tiles
from crag.geo import AreaCRS, utm_epsg


def test_slugify_and_validate(tmp_path):
    assert registry.slugify("Devil's Lake") == "devils-lake"
    assert registry.slugify("Tahquitz & Suicide Rocks") == "tahquitz-suicide-rocks"
    reg = {"areas": [{"slug": "a", "name": "A"}, {"slug": "a", "name": "B"}]}
    with pytest.raises(ValueError, match="duplicate"):
        registry.validate(reg)
    with pytest.raises(ValueError, match="bbox"):
        registry.validate({"areas": [{"slug": "a", "name": "A", "bbox": [1, 1, 0, 2]}]})


def test_registry_roundtrip_and_splits(tmp_path):
    reg = {"areas": [{"slug": f"area-{i}", "name": f"A{i}", "crags": [{"name": "c", "lat": 1.0, "lon": 2.0}]} for i in range(20)]}
    reg["areas"][0]["split"] = "holdout"
    registry.assign_splits(reg, 0.1, 0.1)
    splits = [a["split"] for a in reg["areas"]]
    assert splits[0] == "holdout"  # kept
    assert splits.count("val") == 2 and splits.count("test") == 2
    again = {"areas": [dict(a, split=None) if a["slug"] != "area-0" else dict(a) for a in reg["areas"]]}
    registry.assign_splits(again, 0.1, 0.1)
    assert [a["split"] for a in again["areas"]] == splits  # deterministic
    p = tmp_path / "areas.json"
    registry.save(reg, p)
    assert registry.load(p)["areas"] == reg["areas"]


AREA_HTML = """
<div class="mb-half small text-warm">
    <a href="https://www.mountainproject.com/route-guide">All Locations</a>  &gt;
            <a href="https://www.mountainproject.com/area/105868674/kentucky">Kentucky</a>
    </div>
<h1>
    Red River Gorge
            <span class="hidden-md-down">Climbing</span>
</h1>
<link rel="canonical" href="https://www.mountainproject.com/area/105841134/red-river-gorge" />
<tr><td>GPS:</td>
                    <td>
                        37.67745, -83.68217 <br />
<tr><td class="pr-half text-nowrap">Page Views:</td>
                <td>
                    7,036,854 total &middot;
                    28,487/month
"""


def test_parse_area_page():
    info = mp.parse_area_page(AREA_HTML)
    assert info["name"] == "Red River Gorge"
    assert info["page_views_total"] == 7036854 and info["page_views_month"] == 28487
    assert (info["lat"], info["lon"]) == (37.67745, -83.68217)
    assert info["breadcrumb"] == [(105868674, "Kentucky")] and info["state"] == "Kentucky"


def _csv(rows):
    head = 'Route,Location,URL,"Avg Stars","Your Stars","Route Type",Rating,Pitches,Length,"Area Latitude","Area Longitude"\n'
    return head + "".join(
        f'R{i},"{w} > Area > State",https://www.mountainproject.com/route/{i}/r,3,-1,Sport,{g},1,60,{lat},{lon}\n' for i, w, g, lat, lon in rows
    )


def test_parse_routes_default_pins():
    text = _csv([(1, "WallA", "5.10a", 1.0, 2.0), (2, "WallA", "5.11a", 1.0, 2.0), (3, "WallB", "5.9", 1.0, 2.0), (4, "WallC", "5.8", 5.0, 6.0), (5, "WallD", "5.7", 9.0, 9.0)])
    df = mp.parse_routes(text, area_pin=(9.0, 9.0))
    # pin shared by WallA and WallB -> inherited parent pin; WallD sits on the area pin
    assert df.set_index("route_id")["default_pin"].to_dict() == {1: True, 2: True, 3: True, 4: False, 5: True}
    crags = mp.crags_from_routes(df)
    assert set(crags["name"]) == {"WallA", "WallB", "WallC", "WallD"}
    assert np.isclose(df["length_m"].iloc[0], 60 * 0.3048)


def test_route_export_bisection(monkeypatch, tmp_path):
    """A band returning >= 1000 rows is split until every request is under the cap."""
    monkeypatch.setattr(mp, "MP_ROUTES_DIR", tmp_path)
    monkeypatch.setattr(mp, "EXPORT_ROW_CAP", 10)
    grades = np.linspace(mp.GRADE_MIN, mp.GRADE_MAX, 35).astype(int)  # 35 routes, cap 10
    calls = []

    def fake_band(mp_id, lo, hi, refresh):
        calls.append((lo, hi))
        sel = [(i, "W", "5.10a", 1.0, 2.0) for i, g in enumerate(grades) if lo <= g <= hi]
        return _csv(sel[: mp.EXPORT_ROW_CAP])  # the server truncates

    monkeypatch.setattr(mp, "_fetch_band", fake_band)
    df = pd.read_csv(io.StringIO(mp.fetch_routes_csv(123)))
    assert len(df) == 35 and df["URL"].is_unique
    assert len(calls) > 1


def test_utm_and_tile_plan():
    assert utm_epsg(-93.29, 36.0) == 32615 and utm_epsg(-83.7, 37.7) == 32617
    area = {"slug": "t", "utm_epsg": 32615, "lat": 36.0, "lon": -93.29, "crags": [{"name": "c", "lat": 36.0, "lon": -93.29, "n_routes": 5}]}
    plan = tiles.plan_tiles(area, tile_size_m=200, crag_radius_m=150)
    crs = AreaCRS(32615)
    x, y = crs.to_work(-93.29, 36.0)
    for t in plan:
        dx = max(t.x0 - x, 0, x - (t.x0 + 200))
        dy = max(t.y0 - y, 0, y - (t.y0 + 200))
        assert np.hypot(dx, dy) <= 150 + 1e-6
        assert t.x0 % 200 == 0 and t.y0 % 200 == 0
        assert tiles.Tile.from_id(t.id, 200) == t
    assert 4 <= len(plan) <= 9
    assert len(tiles.plan_tiles(area, 200, 150, max_tiles=2)) == 2
