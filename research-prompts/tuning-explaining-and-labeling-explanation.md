# Labeling Algorithm, Its Name, and What to Tune

This document covers three things:

1. **What kind of labeling this is**, and its name in the literature.
2. **The labeling algorithm in depth**, step by step, tied to the code in
   `labeling.py` and the knobs in `config.py`.
3. **Tuning options.** It covers whether cylinders are the right shape,
   what `steep_slope_deg` actually measures, and what changing the patch
   radius, `route_radius_m`, `stepback_m` and `steep_slope_deg` should do.
   It ends with how to run a small sweep whose results you can trust.

Numbers quoted below come from the current cached Horseshoe Canyon Ranch data
(`data/horseshoe_canyon_ranch/`, 18.66 M points) and the notebook outputs.

---

## 1. What this type of labeling is called

The short name is **weak supervision**. More precisely it is **distant
supervision with programmatic (heuristic) labeling functions**.

- **Distant supervision** (Mintz et al., 2009) means taking labels from an
  external knowledge base that was never built for your task and projecting
  them onto your raw data. Here the knowledge base is Mountain Project and the
  raw data is the LiDAR cloud. No person ever looked at a point cloud and drew
  a box around a climb.
- **Programmatic labeling** (the "labeling functions" idea behind Snorkel,
  Ratner et al., 2017) means the labels come from code-level rules
  ("steep + near a pin + LiDAR saw it → 1") rather than hand annotation.

Zhou (2018, *A brief introduction to weakly supervised learning*) splits weak
supervision into three kinds. This pipeline has all three:

| Kind of weakness | Where it shows up here | How the pipeline copes |
|---|---|---|
| **Inaccurate** (labels are in the wrong place) | MP wall pins sit 10–30 m off the rock | *Snapping* the pin to the nearest real cliff (step 2) |
| **Inexact** (labels are coarser than what you predict) | MP gives a wall pin plus a list of routes, not each route's exact line | One cylinder per route, placed greedily along the cliff |
| **Incomplete** (many true positives are unlabeled) | Plenty of good rock has no MP entry | *Ignore* label for anything that looks like cliff but isn't listed |

Several standard techniques sit inside this framework, and you can name them
in the report:

- **Hard negative mining.** The step-back negatives are deliberately chosen
  to sit right beside a positive.
- **Ignore / "don't-care" regions.** These play the same role as KITTI's
  `DontCare` boxes or the ignore index in segmentation losses. The
  `ignore_buffer_m` band and the "mostly cliff" rejection are both examples.
- **Positive–unlabeled (PU) learning flavor.** Strictly, this data is
  positives plus *unlabeled* terrain, not positives plus true negatives. The
  ignore rules are a cheap way to keep the "negatives" mostly clean.
- **Label registration / snapping.** This is the step-2 correction of
  positional error in the labels.

A one-line description for the report: *"Weakly supervised (distantly
supervised) point-cloud labeling. Mountain Project route metadata is
registered onto airborne LiDAR with geometric labeling functions, using
contextual hard negatives and ignore regions for unlabeled rock."*

---

## 2. The labeling algorithm in depth

### Overview

```
LiDAR cloud (UTM 15N, metres)                 Mountain Project routes
        │                                      (wall, lat/lon pin, length)
        ▼                                                │
[A] per-point normals → slope → steep mask               │
        │                                                │
        └──────────────┬─────────────────────────────────┘
                       ▼
[B] for each wall pin: snap → place ≤ n_routes route cylinders
                       ▼
[C] for each cylinder: base_z, top_z, wall direction, profile
                       → status ok / unseen_by_lidar          (positives, label 1)
                       ▼
[D] negatives: step-back (base, top) + random far-field
                       → keep (label 0) or IGNORE if mostly cliff
                       ▼
[E] every sample → full vertical input column, radius PatchConfig.radius_m
                       ▼
[F] split by wall (dataset.split_walls)
```

Two different radii appear throughout. Mixing them up is the easiest way to
misread the pipeline:

| Name | Default | What it is | Does the model ever see it? |
|---|---|---|---|
| `LabelConfig.route_radius_m` | 2.5 m | The **label cylinder**. It decides whether a spot counts as a positive, and where its base is. | **No** |
| `PatchConfig.radius_m` | 5.0 m | The **input column**. These are the points actually fed to PointNet++. | **Yes** |

### [A] Normals, slope, and the steep mask
(`compute_normals`, `slope_deg`, `PointCloud.build`)

1. **Voxel thinning.** The cloud is bucketed into 0.5 m cubes
   (`normal_voxel_m`) and one point per cube is kept. Without this step, the
   flight-line overlap strips (about 2× density) give a fixed-`k`
   neighborhood a smaller physical size. Normals there then pick up
   centimetre roughness, which shows up as fake N–S stripes of "steep" points.
2. **Local PCA.** For each thinned point, take its `normal_k = 12` nearest
   neighbors, subtract their mean and form the 3×3 covariance matrix. The
   eigenvector with the **smallest** eigenvalue is the direction the
   neighborhood is "thinnest" in, which is the surface normal **n**.
   At 0.5 m voxels, 12 neighbors span roughly 1.5–2 m of surface.
3. **Sign fix.** Airborne LiDAR stores no per-point sensor position, so the
   normal's sign is arbitrary. The code flips every normal so that
   `n_z ≥ 0`.
4. **Copy back.** Every original point takes the normal of its nearest
   thinned point.
5. **Slope.** `slope = arccos(|n_z|)` in degrees.
6. **Steep mask.** A point is "steep" (cliff) if `slope ≥ steep_slope_deg`
   (60°). With `steep_single_return_only = True` it must also be a
   **single-return** pulse. The 2017 collection has no vegetation class, and
   about 58% of steep-normal points are multi-return, which means canopy. Bare
   rock returns a pulse once.

On this data, 61.5% of all points are single-return, and **7.6% of all points
are "steep" under the defaults**.

#### What `steep_slope_deg` actually measures

You asked whether it is the angle between the surface's normal and the normal
coming out of the ground. It is. Precisely:

> **slope = the angle between the point's local surface normal and the
> vertical axis (0, 0, 1).**

The vertical axis is the normal of perfectly flat ground, so this is the
same as **the angle between the local surface and the horizontal plane**,
which geologists call the *dip*:

```
   slope 0°            slope 45°            slope 60°           slope 90°
   n ↑                    n ↗                  n →(ish)             n →
 ───────────            ╱                     ╱│                    │
 flat ground           ╱  45° hillside       ╱ │ steep face          │ vertical wall
```

Things worth knowing about this definition:

- **It uses `|n_z|`, so overhangs fold back onto the 0–90° range.** A roof
  angled 30° past vertical has a downward-tilted normal, but `|n_z|` makes it
  read as 60°. The slope therefore cannot tell a 60° slab from a 30°
  overhang. That job falls to the separate `lean_deg` diagnostic.
- **It is not "is this a climb".** It answers "is this *point* part of a
  rock face". Whether a *location* is a climb is decided later from how many
  steep points are near it (`min_steep_points`), whether MP lists a route
  there, and so on.
- **PCA smooths edges.** A 1.5–2 m neighborhood that straddles a cliff lip
  mixes plateau points with face points, so edges read as 30–60° instead of
  90°. True vertical faces are also sparsely sampled from the air. Inside the
  2.5 m positive cylinders, the slope percentiles are 25%: 22°, 50%: 41°,
  75%: 69° and 90%: 82°. Most points inside a "route" cylinder are not on
  the face at all.
- **It never reaches the model.** The network receives only xyz coordinates.
  `steep_slope_deg` changes the *labels* (which samples exist and which
  label they get), not the *features*.

### [B] Snapping the wall pin onto rock (`place_route_cylinders`)

For each wall `(wall, lat, lon)` group of MP routes:

1. If the pin equals the ranch's default pin (`MP_DEFAULT_PIN`), the routes
   were never really placed, so the wall is dropped (`status = default_pin`,
   3 cases).
2. Routes are sorted longest-first, so the longest routes get the
   best-supported cylinders.
3. Gather every **steep** point within `snap_search_radius_m = 30 m` of the
   pin (2-D distance). Sort them by distance to the pin. Thin them to one per
   0.5 m grid cell so the loop stays short.
4. **Greedy placement**, walking candidates from nearest to farthest:
   - skip a candidate if it is closer than `route_spacing_m = 3 m` to any
     centre already taken, including centres from *other* walls (`taken` is
     shared, so two walls with near-identical pins can't claim the same rock
     and leak across splits);
   - accept it if the 2-D circle of radius `route_radius_m` around it holds
     at least `min_steep_points = 8` steep points (counted over **all
     heights**);
   - stop after `n_routes` acceptances.
5. If nothing is accepted: `status = no_cliff_found`.

Because candidates are themselves steep points, every centre sits *on* the
cliff line, wherever that line is closest to the pin. Median snap distance is
6.8 m, and the maximum is 29.8 m, right at the search limit.

### [C] Building and accepting each route cylinder (`build_positives`, `route_profile`, `wall_direction`)

For each placed centre `c` paired with a route:

1. **Column**: all points within `route_radius_m` of `c` horizontally, at any
   height.
2. **Base**: `base_z` is the 5th-percentile Z of that column. Using the 5th
   percentile rather than the minimum stops one low noise point from setting
   the base.
3. **Top**: `top_z = base_z + length_m`, using the MP route length. When that
   is missing, the wall's median length is used, and failing that 15 m.
4. **Wall direction** (`wall_direction`): average the horizontal components
   of the steep points' normals, after aligning their arbitrary signs to one
   reference normal. That gives the line perpendicular to the face. Then pick
   the sign that points from the column's top 20% of points (high side) to
   its bottom 20% (low side). The result is a horizontal unit vector pointing
   **out of the wall, into the air**.
5. **Profile** (steep points only, `base_z ≤ z ≤ top_z`):
   - `coverage`: the fraction of 1 m height bins with at least one steep
     return;
   - `max_gap_m`: the tallest run of empty bins;
   - `lean_deg`: a line fit of each point's outward offset against height.
     Below 0 the wall lays back, about 0 is vertical, above 0 it overhangs.
6. **Accept** as a positive (`status = ok`, label 1) if `n_steep ≥
   min_steep_points` inside the height band. Otherwise mark it
   `unseen_by_lidar`: we are confident a route is there, but the scan has
   nothing to learn from.

Result: **354 ok, 75 unseen_by_lidar, 3 default_pin**. For accepted routes,
median coverage is 0.43 and median lean is −6° (laid back), and only 8.5% have
≥ 80% coverage.

### [D] Negatives (`build_negatives`)

Both kinds go through the same gate, `try_add(c)`:

1. Take the **input column** at `c` (radius `PatchConfig.radius_m`). If it
   holds fewer than `min_points = 64` points, skip it.
2. Compute `steep_frac`, the share of steep points in that column.
3. If `steep_frac > max_steep_fraction_random_neg = 0.25`, **ignore** it.
   It is probably unlisted rock, so it is neither labeled 0 nor used
   (12 candidates were ignored).
4. Otherwise keep it as label 0.

**Contextual (step-back) negatives.** For each accepted positive, with out
vector `o` and each distance `d` in `stepback_m = (12, 20)`:

- `c + d·o` → `stepback_base` (out into the air, landing on the talus below);
- `c − d·o` → `stepback_top` (back over the lip, onto the plateau);
- **overlap guard:** skip the candidate if it lies within
  `PatchConfig.radius_m + route_radius_m = 7.5 m` of *any* positive, so a
  negative's input never overlaps a route's label cylinder.

**Random negatives.** Sample uniformly inside the region, inset by the patch
radius. Reject points within `ignore_buffer_m = 40 m` of any wall pin or
positive (that band is "unknown, don't label"). Continue until 300 are
accepted or 15,000 attempts have been made.

Result: **403 stepback_base + 390 stepback_top + 300 random = 1,093
negatives** against 354 positives, about 3:1. Training uses class-weighted
cross-entropy to compensate.

### [E] The input patch (`extract_patch`)

For **every** sample, positive or negative, the input is the full vertical
column of points within `PatchConfig.radius_m = 5 m`. X and Y are
re-centred on `c`, and Z is shifted so the lowest point is at 0. Later,
`dataset.normalize_patch` resamples it to 1,024 points and scales it into
the unit sphere.

The label cylinder (radius, base, top) is **never** used to crop the input.
If it were, every positive would be cut to exactly "cliff base → base + route
length" while negatives were cut some other way, and the model could learn
the *shape of the crop* instead of the terrain.

### [F] Split by wall (`dataset.split_walls`)

Whole walls are assigned to train, val or test, with the biggest walls dealt
first and each split filled toward its target share of positives. Each
negative follows the wall of its **nearest positive**. This matters because
positives on one wall are only 3 m apart while each input is 10 m across, so
neighboring patches overlap heavily. A random split would put
near-duplicates in train and test.

---

## 3. Are cylinders the right shape?

Two different cylinders are in play, and the answer differs for each.

### The input column: a vertical cylinder is the right choice

- **No orientation needed at inference.** `inference.predict_grid` slides
  over a grid and doesn't know where any wall is or which way it faces. A
  circle is the only footprint that needs no orientation. Any oriented box
  would require estimating wall direction at every grid node first.
- **It matches the augmentation.** Training rotates patches only about the
  vertical axis. A circular footprint is invariant to exactly that rotation,
  so rotating never moves points in or out of the crop.
- **It matches the sensor.** Airborne LiDAR is close to 2.5-D (it looks
  down), so "everything above this patch of ground" is a natural unit.

**Recommendation:** keep the vertical cylinder for the input and tune its
radius (§4.1).

### The label cylinder: it works, but it is the weakest part of the geometry

What's wrong with a 2.5 m vertical cylinder as the definition of "the route":

- **A route is a strip on a surface, not a column of air.** The centre is a
  steep point on the face, so roughly half the cylinder is air in front of
  the wall and the other half is the rock or plateau behind it. That's why
  the median slope inside a positive cylinder is only 41°.
- **Laid-back walls leave the cylinder.** The median lean is −6° and the 25th
  percentile is −12°. At −12°, a 15 m route's face moves about
  `15·tan 12° ≈ 3.2 m` horizontally from bottom to top. That is a large
  fraction of the 5 m diameter, so part of the face falls outside the
  cylinder. Some of the low `coverage` (median 0.43) is caused by the
  cylinder's geometry, not by the LiDAR missing the rock.
- **Base estimation mixes in the ground.** The 5th percentile of a column
  that includes talus in front of the wall gives the ground level, which is
  usually what you want. On a wall with a ledge or terrace it may pick the
  wrong level.

Alternatives, roughly in order of effort:

| Option | Idea | Pros | Cons |
|---|---|---|---|
| **Keep cylinder, tune radius** | §4.2 | Zero code | Doesn't fix the geometry |
| **Wall-aligned box** (oriented prism) | Box `w` m along the wall × `d` m across × route height, oriented with `wall_direction` | Matches a route's real footprint; along-wall spacing becomes meaningful | Depends on a good wall direction; ~30 lines of code |
| **Leaning prism** | Shift the box's centre along `o` by `tan(lean)·(z − base_z)` | Follows slabs and overhangs | Needs `lean_deg` first (chicken and egg; solvable by iterating) |
| **Face strip** | Fit a local surface to the steep points; the route is the points within ±1 m of that surface between base and top | Most faithful | Most code; fragile when coverage is poor |
| **2-D raster alternative** | Rasterize to DEM, slope and hillshade tiles, then a CNN or segmentation model | Simple, fast, strong baselines exist | Throws away 3-D overhang structure |

**Recommendation:** for this project, try the **wall-aligned box** as the
label region (acceptance + base/top + profile) while keeping the circular
input column. It fixes most of the cylinder's problems for little code, and
because the label region never crops the input, it doesn't affect inference.
If time is short, a radius sweep (§4.2) is the zero-code version of the same
question.

---

## 4. Tuning: what each knob should do

The key idea before any sweep: **most of these knobs change the dataset,
not the model.** Changing `steep_slope_deg`, `route_radius_m` or
`stepback_m` changes which positives exist, which negatives exist and how
hard they are. If each variant is scored on its *own* test set, you are
partly measuring how easy that variant made its own exam. §5 shows how to
compare variants fairly.

### 4.1 `PatchConfig.radius_m` (input column radius, default 5 m)

What it controls: how much context the model sees around a location.

| Change | Expected effect |
|---|---|
| **Smaller (3 m)** | Less context: the talus and plateau around the face mostly disappear, so the model has to recognize the face itself. Fewer raw points, so more patches may fall below `min_points = 64`, especially on steep faces that are sparse from the air. The probability map gets sharper. |
| **Larger (7.5–10 m)** | More context (base talus + face + lip + plateau in one patch), which may help, since cliffs are recognizable by that profile. But with a fixed 1,024 points the face is a smaller share of the input, neighboring samples overlap more, and the map gets blurrier. |

Coupled settings to watch:

- **The overlap guard grows with it.** Step-back negatives are skipped
  within `radius_m + route_radius_m` of any positive. At `radius_m = 10`
  that's 12.5 m, so **the 12 m step-back negatives would silently all
  disappear**. Raise `stepback_m` to match, e.g. `(radius_m + route_radius_m
  + 4, …)`.
- **The ignore rule uses this radius too.** `steep_frac` is measured over the
  input column, so a larger radius dilutes the steep share and fewer
  candidates get ignored.
- **`unit_sphere` normalization removes absolute scale.** Every patch is
  rescaled into a unit sphere, so the model doesn't know whether a column is
  10 m or 40 m tall. If radius matters, also try `normalize = "fixed"`. It's
  one config flag.

Suggested values: **3, 5 (baseline), 7.5, 10**, with stepback adjusted at 10.

### 4.2 `LabelConfig.route_radius_m` (label cylinder radius, default 2.5 m)

What it controls: how strict the "is there rock here?" test is, plus the
base/top estimate and the profile. It does **not** change the model input.

| Change | Expected effect |
|---|---|
| **Smaller (1.5 m)** | Stricter: fewer steep points per circle, so fewer acceptances and more `no_cliff_found` / `unseen_by_lidar`. Surviving positives sit on denser, more obvious rock. You get fewer, cleaner positives, plus more spare spots for the greedy placer, which may then pick spots farther from the pin. |
| **Larger (4 m)** | More lenient: more positives accepted, including marginal ones on steep vegetated slopes. Laid-back faces stay inside the cylinder, so coverage goes up. The base estimate drifts lower. The overlap guard grows, removing some step-back negatives. |

Coupled setting: **`min_steep_points` should scale with area (∝ r²)**.
Otherwise you change two things at once. From 8 at 2.5 m, use about **3 at
1.5 m** and about **20 at 4 m**.

Suggested values: **1.5, 2.5 (baseline), 4**, each with scaled
`min_steep_points`. Track the number of `ok` positives per variant as well as
model metrics.

### 4.3 `LabelConfig.stepback_m` (contextual negative distances, default (12, 20))

What it controls: **how hard the hard negatives are.**

| Change | Expected effect |
|---|---|
| **Closer, e.g. (8, 12)** | Harder negatives. The model has to locate the face precisely instead of just recognizing "this area is canyon-ish." That should give a sharper probability map and lower headline accuracy. It also adds label noise: an 8 m step-back with a 5 m input column still contains the lip or face edge (its column reaches to 3 m from the face). |
| **Farther, e.g. (20, 30)** | Easier negatives, so higher headline accuracy and a blurrier boundary. At Horseshoe Canyon's stacked sandstone tiers, 20–30 m back from the top may land on the *next* tier's face, which is either ignored or a mislabeled 0. |
| **More distances, e.g. (10, 15, 20, 30)** | More negatives (each distance adds up to 2 per positive). The class ratio shifts further toward negatives; class weighting compensates, but precision on positives can drop. |
| **Asymmetric** | Talus (base) and plateau (top) aren't equally hard. Making the base side closer and the top side farther is a reasonable variant, but it needs a small code change: separate `stepback_base_m` and `stepback_top_m`. |

Hard constraints:

- **Every distance must be > `PatchConfig.radius_m + route_radius_m` (7.5 m
  by default)**, or the overlap guard drops all of them. With the defaults,
  (8, 12) is the closest setting that does anything, and the 8 m ones only
  just pass.
- Today the step-back sources are already almost as steep as the positives.
  Median `steep_frac` in the 5 m column at 60° is 0.116 for positives, 0.092
  for `stepback_top`, 0.077 for `stepback_base` and 0.064 for random. These
  negatives are already hard. Making them closer is a test of how far the
  boundary can be pushed, not a free gain.

Suggested values: **(8, 12), (12, 20) baseline, (20, 30)**.

How to judge the result: for this knob, "accuracy on its own test set" is
mostly a measure of how hard that variant's negatives were. Use the fixed
reference test set (§5) and, ideally, look at the probability map: does the
high-probability band hug the cliff line?

### 4.4 `LabelConfig.steep_slope_deg` (cliff threshold, default 60°)

What it controls: the steep mask, which feeds **snapping, positive
acceptance, coverage and the negative ignore rule**. It never changes the
model input.

Share of steep (single-return) points, and the median steep fraction per
5 m input column, on the actual data:

| Threshold | All points steep | Positives | Step-back base | Step-back top | Random | Pos ÷ step-back-base |
|---|---|---|---|---|---|---|
| 45° | 12.1% | 0.199 | 0.137 | 0.156 | 0.111 | 1.45 |
| 50° | 10.5% | 0.169 | 0.116 | 0.134 | 0.096 | 1.46 |
| 55° | 9.1% | 0.139 | 0.096 | 0.112 | 0.079 | 1.45 |
| **60°** | **7.6%** | **0.116** | **0.077** | **0.092** | **0.064** | **1.51** |
| 65° | 6.3% | 0.089 | 0.062 | 0.071 | 0.049 | 1.44 |
| 70° | 5.0% | 0.066 | 0.046 | 0.054 | 0.038 | 1.43 |
| 75° | 3.8% | 0.047 | 0.033 | 0.039 | 0.027 | 1.42 |
| 80° | 2.6% | 0.027 | 0.022 | 0.025 | 0.017 | 1.23 |

(The positive and negative locations in this table were chosen at 60°, so it
is only a rough guide. It still shows the trend.)

How to read it:

- **Separation between positive and negative columns is roughly flat from
  45° to 75° and peaks near 60°.** It falls off at 80°, where airborne LiDAR
  simply has very few returns that steep. **60° is a sensible default**, and
  this data argues against going above about 70°.
- **Lower (45–50°)** makes about 60% more points steep. Snapping and
  acceptance become more lenient, so more positives survive and fewer are
  `unseen_by_lidar`. But steep wooded hillsides and dirt banks start to count
  as "cliff", so positives can snap onto non-rock. Many more negative
  candidates also exceed the 0.25 ignore threshold, so the negatives lose
  exactly the steep-but-not-climbable examples that matter most.
- **Higher (70–75°)** makes the mask cleaner but sparser. More routes become
  `unseen_by_lidar`, which already happens to 75 of them. Snapping drifts to
  wherever the few very steep returns are, and coverage drops.

Coupled setting: `max_steep_fraction_random_neg = 0.25` only means something
relative to the threshold. At 45° the typical random column is already at
0.11, and at 75° it is 0.03. To isolate the threshold's effect, scale the
ignore cutoff with the overall steep share, e.g. keep it at about 3× the
random-column median. That gives roughly 0.33 at 45°, 0.25 at 60° and 0.10
at 75°.

Suggested values: **50, 60 (baseline), 70**. Also record how many routes are
`ok` versus `unseen_by_lidar` and how many negatives are ignored.

### 4.5 Other knobs worth knowing about (not in your list)

| Knob | Why it might matter |
|---|---|
| `steep_single_return_only` | Turning it off brings tree canopy back into "steep". This is a good ablation showing the filter is worth having. |
| `normal_k`, `normal_voxel_m` | Set the physical scale of a "surface". Larger gives smoother, more stable normals, but blurs edges more. Changing these means recomputing the normals cache (`normals_k12_vox0.5.npy`). |
| `route_spacing_m` (3 m) | Smaller gives more, nearly duplicate positives per wall; larger gives fewer, more independent ones. |
| `snap_search_radius_m` (30 m) | Several snaps sit near the 30 m limit. Shrinking it trades positives for confidence that they're the right rock. |
| `PatchConfig.normalize` | `"fixed"` keeps metric scale (see §4.1). |

---

## 5. How to run a small sweep you can trust

The current runs in `runs/` are `FAST_DEV_RUN` runs: 2 epochs, with 71
positives in each of val and test drawn from only a few walls. Test AUROC is
0.65–0.70 across the three modes. With so few positives and walls, differences
of a few points between variants are within noise. Some guardrails:

1. **Fix the evaluation set across variants.** Build the baseline config once
   and save its wall-to-split mapping and its test samples (centres + labels)
   as a **reference test set**. For each variant:
   - reuse the **same wall → split mapping**, so that train, val and test
     contain the same walls every time;
   - score the model on (a) its own test split *and* (b) the reference test
     centres, re-extracted with the variant's `PatchConfig`.

   Score (b) is the comparable one. Score (a) mostly tells you how hard the
   variant's own labels are.
2. **Change one knob at a time** from the baseline, and adjust its coupled
   settings as described in §4.1–4.4.
3. **Use one fine-tuning mode**, e.g. `head` or `finetune` (not all three),
   with **3 seeds** per variant, and report mean ± std.
4. **Train long enough to rank variants.** Two epochs can't separate them.
   Use something like 15–20 epochs with the existing early stopping, and the
   same budget for every variant.
5. **Report label statistics next to metrics.** For each variant record
   number of `ok` positives, `unseen_by_lidar`, negatives by source, ignored
   count, median coverage, and the positive:negative ratio. Often the label
   statistics explain the metric change.
6. **Primary metrics:** AUROC and balanced accuracy, since both handle the
   3:1 imbalance. Also look at the probability map for the step-back and
   steep-threshold variants.

### Suggested minimal sweep (13 variants × 3 seeds)

| # | Variant | Change from baseline |
|---|---|---|
| 0 | baseline | (as configured) |
| 1–3 | input radius | `patch.radius_m` = 3, 7.5, 10 (at 10, `stepback_m = (16, 24)`) |
| 4–5 | label radius | `route_radius_m` = 1.5 (`min_steep_points = 3`), 4 (`min_steep_points = 20`) |
| 6–7 | step-back | `stepback_m` = (8, 12), (20, 30) |
| 8–9 | steep threshold | `steep_slope_deg` = 50 (ignore cutoff 0.30), 70 (ignore cutoff 0.15) |
| 10 | canopy filter ablation | `steep_single_return_only = False` |
| 11 | scale-aware input | `patch.normalize = "fixed"` |
| 12 | label shape (code change) | wall-aligned box instead of label cylinder (§3) |

### Sketch of the sweep loop (uses only existing functions)

```python
from dataclasses import replace
import numpy as np
from config import ExperimentConfig
from labeling import PointCloud, build_samples, extract_patch
from dataset import split_walls, make_loaders, PatchDataset
from trainer import Trainer, TrainConfig
from torch.utils.data import DataLoader

base = ExperimentConfig()
# xyz, normals, cloud, routes loaded as in crag_poc.ipynb

def build(cfg):
    pc = PointCloud.build(xyz, cfg.label, normals, cloud["number_of_returns"])
    samples, patches, pos_all = build_samples(pc, routes, cfg)
    return pc, samples, patches, pos_all

# 1. baseline: fix the wall → split mapping and the reference test centres
pc0, s0, p0, _ = build(base)
s0 = split_walls(s0, base.split)
wall_split = s0.drop_duplicates("split_wall").set_index("split_wall")["split"].to_dict()
ref_test = s0[s0["split"] == "test"][["x", "y", "label"]].reset_index(drop=True)

def apply_fixed_split(samples):
    s = split_walls(samples, base.split)          # computes split_wall per sample
    s["split"] = s["split_wall"].map(wall_split).fillna("train")
    return s

variants = {
    "baseline": base,
    "patch_r3": replace(base, patch=replace(base.patch, radius_m=3.0)),
    "stepback_8_12": replace(base, label=replace(base.label, stepback_m=(8.0, 12.0))),
    "steep_70": replace(base, label=replace(base.label, steep_slope_deg=70.0,
                                            max_steep_fraction_random_neg=0.15)),
    # ...
}

for name, cfg in variants.items():
    pc, s, p, pos_all = build(cfg)
    s = apply_fixed_split(s)
    loaders = make_loaders(s, p, cfg.patch)
    # reference test: same centres + labels, inputs re-cut with this variant's patch radius
    ref_patches = [extract_patch(pc, np.array([x, y]), cfg.patch) for x, y in ref_test[["x", "y"]].to_numpy()]
    ref_loader = DataLoader(PatchDataset(ref_test, ref_patches, None, cfg.patch), batch_size=16)
    for seed in (0, 1, 2):
        t = Trainer(TrainConfig(run_name=f"sweep-{name}-s{seed}", mode="head", epochs=20, seed=seed))
        t.fit(loaders["train"], loaders["val"])
        t.load_best()
        own = t.evaluate(loaders["test"])
        ref = t.evaluate(ref_loader)
        # log own/ref metrics + label stats (pos_all.status.value_counts(), s.groupby(['label','source']).size(), ...)
```

(`apply_fixed_split` sends walls the baseline never saw to train. That only
happens if a variant accepts positives on a wall the baseline rejected.)

---

## 6. TL;DR

- **Name:** weak supervision, specifically distant supervision with
  programmatic labeling functions. It includes label snapping, hard-negative
  mining and ignore regions, and has a positive–unlabeled flavor.
- **`steep_slope_deg`** is the angle between a point's surface normal and
  straight up (the normal of flat ground), which equals the surface's dip
  from horizontal. Because it uses `|n_z|`, overhangs fold back into 0–90°.
  It labels *points* as cliff, never *locations* as climbs, and it never
  enters the model input. On this data, separation peaks around 60°, so stay
  in the 50–70° range.
- **Cylinders:** keep the vertical circular **input** column, because it is
  needed for orientation-free sliding-window inference. The **label**
  cylinder is the weak point. A wall-aligned box is the best upgrade for the
  effort.
- **Radius:** the input radius trades context against focus. Scale the
  step-back distances with it, or the 12 m negatives disappear at 10 m. The
  label radius trades the number of positives against how clean they are.
  Scale `min_steep_points` ∝ r².
- **`stepback_m`:** controls how hard the negatives are. Every distance must
  be > 7.5 m with the defaults. Judge it on a fixed reference test set and
  the probability map, not on its own test accuracy.
- **Before trusting any comparison:** fixed wall split, fixed reference test
  set, 3 seeds, more than 2 epochs.
