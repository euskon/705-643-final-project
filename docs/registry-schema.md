# Registry schema (`registry/areas.json`)

One JSON file, one entry per climbing area. `crag/registry.py` loads and
validates it (unique slugs, valid bbox, known split).
`scripts/s1_build_registry.py` fills missing fields and never overwrites
existing ones (`--recompute usgs,bbox,crags` forces).

## Minimal entry (all you write to add an area)

```json
{"slug": "obed-clear-creek", "name": "Obed & Clear Creek", "mp_id": 105891970}
```

## Full entry

| field | type | source | meaning |
|---|---|---|---|
| `slug` | str | you / slugified MP name | unique id, used for directory names |
| `name` | str | MP | area name |
| `mp_id`, `mp_url` | int, str | MP | Mountain Project area |
| `mp_rank` | int | `mp_ranked.csv` | popularity rank among the candidates (by MP page views) |
| `state` | str | MP breadcrumb | |
| `lat`, `lon` | float | MP area page "GPS" | the area's own pin (rough) |
| `popularity` | obj | MP area page | `mp_page_views_total`, `mp_page_views_month`, `retrieved` date |
| `n_routes` | int | route-finder export | rock routes under the area |
| `crags` | list | route-finder export | every MP sub-area pin holding routes: `name`, `lat`, `lon`, `n_routes`, `default_pin` (pin shared by several walls or equal to the area pin, so not a real location), `location` (breadcrumb) |
| `bbox` | `[min_lon, min_lat, max_lon, max_lat]` | crags | encloses every non-default crag pin (pins > 60 km from the area pin are treated as mistakes) + 500 m margin; used for OSM route queries |
| `utm_epsg` | int | `lat`, `lon` | metric CRS for all geometry in this area |
| `usgs.datasets` | list | hobu `resources.geojson` | USGS 3DEP EPT datasets ranked by coverage of the crag footprint (150 m around crags), point density, and year: `name`, `url` (`ept.json`), `coverage`, `points_per_m2`, `year`. Tiles use the first dataset whose boundary contains them, so areas spanning two collections are covered |
| `split` | `train`/`val`/`test`/`holdout` | deterministic hash of the slug (10% / 10%) | pretraining split; `holdout` areas are never used in pretraining (Horseshoe Canyon Ranch: hand-gathered ground truth) |
| `enabled` | bool | default true | `false` skips the area everywhere |
| `notes` | str | free text | |

## Related files

* `registry/mp_candidates.csv`: the candidate pool for "most popular"
  (`name, state, mp_id, note`). Mountain Project publishes no ranking, so
  popularity = MP total page views, fetched by `s1_rank_areas.py`.
  Candidates nested inside another candidate are dropped (no double-counted crags).
  Caveat: this ranks a *curated* pool of 61 well-known US areas, not every MP
  area. Add rows to widen it.
* `registry/mp_ranked.csv`: the ranking, with retrieval date.
* `registry/heldout_regions.json`: lon/lat boxes of fine-tuning test crags,
  written by `s3_build_finetune_set.py`. Pretraining blocks inside them are dropped.

## Coverage gaps (registry built 2026-11-02)

The public USGS 3DEP EPT bucket does not cover every area. Of the 49
pretraining areas in the top 50 (Horseshoe Canyon is #35 and is `holdout`):

* **No coverage, skipped by the downloader:** smith-rock, index, shelf-road,
  custer-state-park, queen-creek-canyon, exit-38, mount-charleston.
  Oregon and Washington LiDAR is mostly state-hosted (DOGAMI, WA DNR), not in 3DEP EPT.
* **Partial coverage (best dataset < 60% of the crag footprint):** red-rocks,
  yosemite-national-park (~0%), lake-tahoe, little-cottonwood-canyon,
  zion-national-park, big-cottonwood-canyon, sedona-area. Tiles outside
  coverage are written as empty markers.
* **Mostly bouldering** (the route-finder export is `type=rock`, so few
  routes): joes-valley, hueco-tanks.

To replace weak areas, add rows to `mp_candidates.csv`, re-run
`s1_rank_areas.py`, then run `s1_build_registry.py --top N`. Set `"enabled": false` to drop one.
