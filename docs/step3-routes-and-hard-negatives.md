# Step 3: route-level GPS and hard negatives

## 1. The problem

Mountain Project tells us **which** routes exist (name, grade, length, wall),
but its coordinates are wall- or area-level pins, often 10-500 m from where a
climb starts. That's fine for pretraining (Step 1 only needs popular areas
and rough extents), but fine-tuning needs **where routes actually start**,
and negatives that are *hard*: rock that looks like a crag but isn't
climbable. We have almost none of those.

## 2. Data collection

### Sources, in order of trust

| source | what it gives | reliability | used as |
|---|---|---|---|
| hand-gathered waypoints (Horseshoe Canyon Ranch, 2026-10-24/25) | 126 `Climb` + 60 `Nc` points at the wall base, 40 named with MP links | on-site judgement; GPS 3-10 m | positives **and** verified negatives; held-out test area |
| OpenStreetMap `climbing=route_bottom` nodes (what openclimbing.org edits and renders) | start point of one route, often with grade, length, MP / OpenBeta links | mapped by climbers on aerial imagery / on site; a few metres | positives, after cross-referencing |
| OSM `climbing=route` ways / nodes | route line (bottom→top) or a point | as above; we take the **lower** end via the LiDAR ground height | positives, after cross-referencing |
| Mountain Project route-finder export | names, grades, lengths, wall pins | names excellent, locations rough | cross-reference only (never a location) |

openclimbing.org has no separate route database or API of its own. It is an
OSM editor (OsmAPP), so the reproducible way to "query openclimbing" is the
**Overpass API** with the climbing tagging scheme
(`crag/openclimbing.py`: one query per area bbox, cached in `data/raw/overpass/`).

### What the survey found (`s3_survey_openclimbing.py`, Nov 2026)

* The whole USA has **2,590** route-level OSM features (2,107 `route_bottom`
  nodes, 483 `route` nodes/ways); 144 carry a direct Mountain Project link.
* **Red River Gorge dominates**: about 1,600 route starts in several clusters
  (the largest, around Muir Valley / PMRP, has 900). None carry MP links,
  so they are cross-referenced by name.
* Then a long tail: Index WA ~51, Yosemite ~49, Vermont crags ~48, Red Rocks
  ~45 + 24, the Gunks ~43, Mt. Woodson CA 37 (35 with MP links), the Shelf Road
  area ~34, Adirondacks ~32, Tahoe ~25, Acadia ~20.
* Horseshoe Canyon Ranch has none (2 boulder problems), so our hand
  data is the only route-level data there.

Consequence: the fine-tuning set is RRG-heavy. Splits are therefore
**spatial by crag group** (150 m linkage), so the model is tested on crags it
never saw, and Horseshoe Canyon (different rock, the only real negatives)
is the test area. To grow the set, add the survey's top clusters to the
registry (`--resolve-mp` prints their MP area chain) and/or map more routes
on openclimbing.org at areas you know.

### Cross-referencing (`crag/crossref.py`)

An OSM route is kept only if it matches a documented MP route of the same
registry area, one-to-one:

1. `mp_url`: the OSM feature links the MP route id;
2. `name`: normalised names equal (accents, case, punctuation, "the",
   "&"/"and", parentheticals), compatible YDS grade (±1 number), MP pin
   within 2 km (a sanity check only);
3. `fuzzy`: difflib similarity ≥ 0.88 plus a compatible grade plus the 2 km check
   (confidence 0.9).

`coverage.csv` per area records, per MP wall, how many documented routes
have a verified location. Negatives use it (below).

## 3. Hard negatives

### Strategies compared

| strategy | what | hardness | label noise | volume | verdict |
|---|---|---|---|---|---|
| **hand Nc** | rock judged unclimbable on site | hardest | none | 60 (one area) | keep; also the test set's negatives |
| **band / gap** (your proposal) | cliff columns 10-40 m from every known route on the same cliff line | hard (same rock, same wall) | **depends on documentation coverage** (measured below) | ~1 per positive | keep, only where coverage ≥ 0.6; `bracketed` (routes on both sides) gets higher confidence |
| **stepback** (PoC) | talus / plateau 12-20 m out from / behind a route | easy | low | 2 per positive | keep, so the model learns crag context |
| **far cliff** | cliff > 300 m from any route or MP crag | medium | medium-high: could be unclimbed good rock | ~40 per area | keep at low confidence (0.4) or drop via `dataset.exclude_sources` |
| **augmentation** (`--aug squash,layback`) | a positive block with its wall shrunk to 1/5 height, or laid back to ~45° | medium | none by construction | 1 per positive per kind | optional ablation: certain labels, but synthetic geometry the model may learn as a shortcut; whole-block transforms avoid an obvious "notch" artifact |
| model-mined (OHEM, later) | highest-scoring cliff far from routes, after a first fine-tune | hardest | confirmation bias | any | next iteration: review the top false positives in `samples.geojson` / `review_negatives.csv` before adding them |
| PU learning (alternative) | treat unlabelled rock as *unlabelled*, not negative (nnPU loss) | n/a | avoids false negatives | n/a | worth trying if band precision stays low; needs a class-prior estimate |

Training uses a class-balanced sampler weighted by each sample's
`confidence`, and per-sample labels are capped at 48 points, so big talus
columns don't outweigh thin route cylinders.

### How we verify that negatives are negative

1. **Exclusion radius.** No negative cylinder within 7.5 m (2.5 m route
   cylinder + 5 m input column) of *any* known route location: all OSM routes,
   matched or not, boulders included, plus all hand `Climb` points.
2. **Documentation gate (band).** At least 3 verified routes within 75 m, and
   MP-wall coverage ≥ `min_coverage` (default 0.6). Where a crag is
   completely mapped, unclimbed cliff between routes is most likely bad rock.
   Where it isn't, it is most likely undocumented routes.
3. **Measured precision at Horseshoe Canyon** (`validate_on_handgathered`,
   in every `report.json`). Pretend only a random fraction *f* of the 126
   Climb waypoints were documented, generate band negatives from them, and
   check each against the labels the generator did **not** see. Within 8 m
   of an Nc point counts as confirmed; within 8 m of a held-out Climb point
   counts as contradicted.

   | simulated coverage *f* | band precision | negatives per run |
   |---|---|---|
   | 0.30 | 0.45 | 38 |
   | 0.50 | 0.79 | 66 |
   | 0.75 | 0.82 | 94 |
   | 0.90 | 0.93 | 115 |

   *(5 repeats per f. Only ~5-7 negatives per run land within 8 m of an
   unseen label, so this is directional evidence, not a precise estimate.)*
   An earlier run that used only the 40 **named** waypoints as documentation
   (≈ 30% coverage, coverage gate bypassed) gave precision 0.25, worse than
   random cliff (≈ the 32% Nc base rate). **The coverage gate is essential.**
4. **Audit features** for every negative in `samples.csv`: cliff height
   (q95-q5 of steep points), steep-point count, multi-return (vegetation)
   fraction on the face, distance to the nearest route. `report.json` averages
   them per source.
5. **Human review.** `review_negatives.csv` holds a stratified random sample with
   satellite links and an empty `review_label` column. `samples.geojson`
   shows every sample by source/split.

## 4. Running it

```bash
python scripts/s3_survey_openclimbing.py --resolve-mp
python scripts/s3_collect_routes.py
python scripts/s3_build_finetune_set.py --name v1                       # default strategies
python scripts/s3_build_finetune_set.py --name v1-aug --aug squash,layback
python scripts/s3_build_finetune_set.py --name v1-strict --min-coverage 0.8 --strategies handgathered_nc,band,stepback
```
