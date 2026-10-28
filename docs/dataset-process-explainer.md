# How the dataset gets built and labeled

## The very simple version

1. **Get the climbing routes.** Download Mountain Project's list of routes at Horseshoe Canyon Ranch. Each route comes with a GPS pin, but the pin marks the wall, not the route.
2. **Get the 3D terrain.** Download USGS airborne LiDAR for the ranch. It's millions of 3D points covering the ground, trees and rock.
3. **Find the cliffs.** Mark every LiDAR point that sits on a steep surface and isn't a tree as "rock".
4. **Make "yes" examples.** Move each wall pin onto the nearest cliff, and drop one small cylinder on the cliff for each route on that wall. Each cylinder becomes a "climbable" sample.
5. **Make "no" examples.** Pick spots at the base of the cliff, on top of it, and at random far away. These are "not climbable" samples. Any spot that looks like a lot of cliff is skipped instead of being called "no".
6. **Cut out a column of points.** For each sample, take every LiDAR point within 5 m sideways of its center, at any height. That column is what the model sees.
7. **Split by wall.** Each wall goes entirely into train, val or test, so nearly identical neighboring samples never end up on both sides.
8. **Train and map.** Fine-tune a pretrained PointNet++, then slide it across the whole ranch to get a P(climbable) heat map.

The labels are "self-supervised" (really weak supervision) because nobody looked at any sample. Every label comes from rules applied to Mountain Project pins and LiDAR geometry.

---

## The detailed version, in order

### Step 1: Mountain Project routes (`mountain_project.py`)

- **Source:** one request to Mountain Project's route-finder CSV export for area 105903004, cached at `data/mountain_project/route_finder_105903004.csv` (530 routes).
- **Raw row:** `Route, Location, URL, Avg Stars, Your Stars, Route Type, Rating, Pitches, Length (ft), Area Latitude, Area Longitude`
- **`parse_routes`** produces one row per route with these columns: `route, route_id, wall, route_type, rating, pitches, length_ft, length_m, lat, lon, default_pin`
  - `wall` is the first part of the `Location` breadcrumb.
  - `default_pin` is True when the route sits on the ranch's own placeholder pin (36.0118, −93.2922). Those routes were never placed on a real wall.
- **`filter_bbox`** keeps routes inside the study box in `config.RegionConfig.bbox_lonlat`.
- **`walls_from_routes`** collapses them to one row per wall: `wall, lat, lon, default_pin, n_routes, median_length_m, max_length_m` (33 walls).
- **The key limitation:** coordinates are per wall, and usually 10–30 m off the rock. Every route on a wall shares one pin (all 142 North Forty routes share one).

### Step 2: USGS LiDAR (`usgs_ept.py`)

- **Source:** USGS 3DEP `AR_North_Corridor_2017` Entwine Point Tiles on public S3, read with PDAL. `AR_BuffaloRiver_2010` is the fallback.
- **Process:**
  1. Cut the bounding box into 250 m tiles.
  2. Run one PDAL pipeline per tile: read, drop noise classes 7 and 18, reproject from Web Mercator to **UTM 15N (EPSG:32615)** so distances are real meters.
  3. Merge the tiles and remove duplicate points on tile edges.
- **Output:** a dict of arrays, cached at `data/horseshoe_canyon_ranch/AR_North_Corridor_2017.npz` (gitignored, about 19 M points):
  - `xyz` (N,3) float64, meters
  - `classification`, `return_number`, `number_of_returns` (uint8)
  - `intensity` (uint16)
- **Format of one data point:** one laser return, `(easting, northing, elevation)` plus its return info.

### Step 3: Normals, then "rock" points (`labeling.compute_normals`, `PointCloud.build`)

1. **Thin the cloud** to one point per 0.5 m voxel. Without this, flight-line overlap strips (twice the point density) show up as fake steep stripes.
2. **Compute a normal for each point** by PCA over its 12 nearest neighbors. The normal is the eigenvector with the smallest eigenvalue, flipped so it points upward. Each original point takes the normal of its nearest thinned point.
3. **Slope** is the angle between the normal and vertical: 0° is flat, 90° is a vertical wall.
4. **A point is "steep" (rock)** when its slope is at least 60° **and** `number_of_returns == 1`. The 2017 data has no vegetation class, and tree canopy returns several echoes per pulse, while bare rock returns one.

- **Output:** a `PointCloud` holding `xyz`, `normals` (N,3 float32), `steep` (N bool), a 2D KD-tree over all points, and a 2D KD-tree over steep points only. Normals are cached at `normals_k12_vox0.5.npy`.

### Step 4: Positives, one route cylinder per route (`build_positives`, `place_route_cylinders`)

For each wall group `(wall, lat, lon)`:

1. **Skip** the wall if it's on the default pin (`status=default_pin`).
2. **Snap to the cliff:**
   - Gather every steep point within **30 m** of the pin, sorted nearest-first and thinned to a 0.5 m grid.
   - Greedily accept a candidate center when it is at least **3 m** from every center already placed (on any wall), and a **2.5 m**-radius cylinder around it holds at least **8** steep points.
   - Stop after `n_routes` centers. If none qualify, the wall gets `status=no_cliff_found`.
3. **Pair centers with routes,** longest route to nearest center (routes are sorted by length). This pairing is arbitrary: we don't know which route is actually where.
4. **Set the vertical extent:**
   - `base_z` is the 5th-percentile elevation in the cylinder.
   - `top_z` is `base_z` plus the MP route length. If the length is missing, it falls back to the wall's median length, or 15 m.
5. **Find the wall direction:**
   - `wall_direction` averages the horizontal normals of the steep points, which gives a unit vector `(out_dx, out_dy)`.
   - It's oriented to point from the cliff top toward the base, i.e. out of the wall.
6. **Run the feasibility diagnostics** (`route_profile`, on steep points between `base_z` and `top_z`):
   - `n_steep`: number of those points.
   - `coverage`: fraction of 1 m height bins that contain at least one return.
   - `max_gap_m`: tallest run of empty bins.
   - `lean_deg`: a line fit of horizontal offset against height. Positive means the wall overhangs.
7. **Set the status:** `ok` if `n_steep >= 8`, otherwise `unseen_by_lidar`. Only `ok` rows become positives.

- **Positive row (`positives_all`):** `wall, route, route_id, route_type, rating, pin_x, pin_y, x, y, snap_dist_m, base_z, top_z, length_m, out_dx, out_dy, n_steep, coverage, max_gap_m, lean_deg, status, lon, lat`

### Step 5: Negatives (`build_negatives`)

There are two kinds, and both go through the same filter, `try_add`, using the 5 m input column:

**Contextual negatives.** From each positive, step 12 m and 20 m along its wall direction:

- `+out` gives `stepback_base` (talus below the cliff).
- `−out` gives `stepback_top` (plateau above it).
- A candidate is skipped if it's within 7.5 m (5 m patch radius + 2.5 m route radius) of any positive.

**Random negatives.** Up to 300 uniform-random points in the region, each at least **40 m** from every route pin and every positive.

**Filter for both:**

- Reject the candidate if its column has fewer than 64 points.
- If more than **25%** of its column is steep, it's **ignored** (probably rock nobody has logged) and counted in `ignored_mostly_cliff`. It is never labeled 0.

- **Negative row:** `x, y, source, wall` (the source positive's wall, or None for random), `steep_frac, n_points`

### Step 6: Samples and patches (`build_samples`, `extract_patch`)

- Positives and negatives are concatenated into **`samples`**: `x, y, label (1/0), source (mp_route | stepback_base | stepback_top | random), wall, route, n_points`.
- **`extract_patch`** builds each sample's input:
  - Take **all points within 5 m horizontally**, at every height. The 2.5 m label cylinder is never used for cropping.
  - Center X/Y on the sample.
  - Shift Z so the column's lowest point is 0.
  - Result: a ragged `(n, 3)` float32 array.
- Columns with fewer than 64 points are dropped.
- **Current counts** in `data/horseshoe_canyon_ranch/samples.csv` (1,447 rows):

  | Source | Label | Count |
  |---|---|---|
  | `mp_route` | 1 | 354 |
  | `stepback_base` | 0 | 403 |
  | `stepback_top` | 0 | 390 |
  | `random` | 0 | 300 |

### Step 7: Split, normalize, Dataset (`dataset.py`)

- **`split_walls`**:
  - Walls are dealt to train/val/test (60/20/20 of the positives), biggest walls first, each going to whichever split is furthest below its target.
  - Every negative takes the split of its **nearest positive's** wall. That's the `split_wall` column.
- **`save_samples`** writes `samples.csv` (adds `split_wall, split`) and `samples.npz`. The npz holds all patch points concatenated, plus offsets.
- **`PatchDataset.__getitem__`** produces each training example:
  1. Resample to exactly **1024 points**.
  2. Normalize to the unit sphere: center on the mean, divide by the max radius, the same as ModelNet40.
  3. On train only, augment: random rotation about Z, scale 0.9–1.1, jitter.
- **The tensor the model sees:** `(points [3, 1024] float32, label int64)`.

### Step 8: Train and infer (`model.py`, `trainer.py`, `inference.py`)

- **Model:** PointNet++ SSG with ModelNet40 weights and a new 2-way head.
- **Three modes compared:** `scratch`, `head` (frozen backbone), `finetune`.
- **Training:** class-weighted cross-entropy. The best checkpoint is chosen by validation balanced accuracy and then scored on the test walls.
- **`predict_grid`:** slides the same 5 m column across a grid. Each grid node becomes a row of `x, y, n_points, prob`, saved as `probability_grid_finetune.csv` (about 11.3k nodes at 4 m spacing).

---

## Where the hand-gathered data fits

The GPS files have a different format from everything above:

- **Format:** `Waypoints_2026-10-2{4,5}.csv` / `.gpx` from a Garmin Montana 700i, with columns `name, lat, lon, ele_m, time, comment, description, symbol` in WGS84.
- **What each point is:** a ground-truth label at a **specific route location**, not a wall pin.

  | File | Waypoints | `Climb` | `Nc` | Named route | Blank |
  |---|---|---|---|---|---|
  | 10-24 | 120 | 41 | 40 | 35 | 4 |
  | 10-25 | 69 | 44 | 19 | 5 | 1 |

- **Coverage:** all 189 points fall inside the study box.

**How they line up with the pipeline:**

- **Positives (Step 4):** named waypoints can be matched to `route` names, so you can measure the error of the pin→cliff snap and of the arbitrary route-to-cylinder pairing.
- **Negatives (Step 5):** `Nc` points are real negatives right at the cliff. The automatic labeling never makes these; it either ignores mostly-cliff spots or uses talus and plateau spots 12–20 m away.
- **Model check (Step 8):** each waypoint can go through `lonlat_to_work` → `extract_patch` → the model, or be looked up against the nearest node in the probability grid.

**Things to watch before using them:**

- **GPS error:** handheld GPS under a cliff is often off by 3–10 m. That's comparable to the 2.5 m cylinder and the 3 m spacing, so you'll want a distance tolerance.
- **Elevations:** `ele_m` is GPS elevation, not LiDAR Z, so don't trust it as an absolute height.
- **Names:** waypoint names like `0-?` aren't unique.

The `Climbable/` and `NotClimbable/` photos aren't used by any code yet.
