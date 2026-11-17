# Horseshoe Canyon points and the wall height check (2026-11-15)

Follow-up to `docs/fable-investigation-project-issues.md`, section 4. The
plan that turns this into code is
`research-prompts/fable-investigation-project-issues-resolution-plan4.md`.

## 1. What was asked

* Snap a route GPS point only to a wall that is **at least about 10 m tall**,
  for all data, as part of preprocessing. Very short rock is bouldering, not
  sport climbing.
* Apply the same rule to the "not climbable" (Nc) points: they are mostly on
  faces next to climbable routes, or on faces of climbable height.
* For Horseshoe Canyon Ranch: back up the original GPS points, snap them
  **once**, save the result, and use that file from then on.
* **No hand review** (decided 2026-11-15). From a map the user can't see the
  rock any better than a model can. Subagent judges check each Horseshoe
  point once, and their decision is final.

## 2. Done so far

* **Backup.** The original files are copied, read-only, to
  `data/handgathered/original/`: both waypoint CSVs and both GPX files. They
  are also in git. Nothing else has changed yet.

## 3. How to measure wall height: today's measure doesn't work

The wall check (`snap_one` in `crag/groundtruth.py`) measures height as the
spread of cliff points in an **upright 2.5 m-wide column** at the snap spot.
On the points that pass today, I compared three measures:

| measure | Horseshoe median | Red River Gorge median | Horseshoe Climb under 10 m | Red River Gorge under 10 m |
|---|---|---|---|---|
| upright column (today) | 9.9 m | 15.9 m | 52% | 23% |
| along the face, stopping at the first 2 m gap with no cliff points | 11 m | 8 m | 46% | 56% |
| highest non-tree point within 12 m, minus the foot | 22 m | 36 m | 2% | 0% |

* **The column undercounts.** Walls lean back, so the face leaves the column
  as it rises. Aircraft LiDAR also puts few points on vertical rock.
* **The face measure undercounts even more on tall walls.** Gaps of more
  than 2 m with no cliff points are common (roofs, sparse returns). That
  also affects plan 1's "wall top" rule; it is noted in plan 1's risks table.
* **The 12 m measure overcounts.** It picks up the hillside behind the wall.

### The new measure: from the bare-earth ground grid

The 1 m ground grid (`crag/walls.py`, built from the provider's ground
points) covers the whole wall: where the face has no points, the grid fills
in from the nearest ground points at the foot and the top. So:

1. At the candidate snap spot, find the uphill direction from the smoothed
   ground grid.
2. Walk uphill in 0.5 m steps across the wall-mask strip. It must start
   within 3 m; a break of up to 1.5 m in the strip is allowed.
3. **Wall height = the highest ground in the strip (plus 1 m past its top
   edge) minus the foot height.** The foot height is the same `base_z` the
   wall check already uses.

**Checked against real route lengths** (405 OpenStreetMap routes with a
length from Mountain Project):

| | upright column (today) | ground grid (new) |
|---|---|---|
| agreement with route length (rank correlation) | -0.06 (none) | **0.43** |
| share of walls at least 0.8 × the route length | 37% | **79%** |
| median wall height ÷ route length | 0.62 | 1.22 |

A route can't be longer than its wall, so a good measure should give wall
heights at or a bit above route lengths. The ground grid does; the column
doesn't. This was measured with a prototype in a scratch folder; plan 4
builds it properly into `crag/walls.py`.

## 4. Dry run: what each minimum height costs

All route GPS points: all 186 labelled Horseshoe points, 250 random Red
River Gorge routes, and every route in the other areas with LiDAR (175).
Each point is snapped to the nearest wall within 20 m that passes the
existing checks (enough cliff points; foot within 6 m of the standing
height) **and** is at least the minimum height on the ground grid. The first
row is today's check, for reference.

**Share of points that pass:**

| rule | Horseshoe Climb (126) | Horseshoe Nc (60) | Red River Gorge (250) | other areas (175) |
|---|---|---|---|---|
| today (column ≥ 4 m) | 0.98 | 0.97 | 0.99 | 0.98 |
| ground grid ≥ 4 m | 0.95 | 0.95 | 0.99 | 0.99 |
| ground grid ≥ 6 m | 0.94 | 0.95 | 0.98 | 0.98 |
| ground grid ≥ 8 m | 0.82 | 0.80 | 0.98 | 0.98 |
| **ground grid ≥ 10 m** | **0.78** | **0.73** | **0.96** | **0.97** |
| ground grid ≥ 12 m | 0.69 | 0.50 | 0.96 | 0.95 |

**Median distance moved** (passing points): Horseshoe Climb 4.8 m today vs
6.2 m at 10 m; Horseshoe Nc 3.0 vs 5.7 m; Red River Gorge 1.3 m at every
threshold; other areas 0.4-0.5 m.

**Share of passing points that skipped a nearer, shorter wall** (moved more
than 3 m further than the nearest wall, because the nearest was too short):

| rule | Horseshoe Climb | Horseshoe Nc | Red River Gorge | other areas |
|---|---|---|---|---|
| ≥ 4 m | 0.04 | 0.00 | 0.01 | 0.01 |
| ≥ 8 m | 0.17 | 0.15 | 0.01 | 0.03 |
| ≥ 10 m | 0.15 | 0.16 | 0.00 | 0.02 |

### What the table says

* **Outside Horseshoe, 10 m costs little:** 3-4% of routes. Their walls are
  mostly tall (ground-grid median 29 m at Red River Gorge).
* **At Horseshoe, 10 m drops 28 Climb and 16 Nc points** (of 126 and 60).
  For the points that fail, the tallest wall within 20 m is a median 6.4 m
  (middle half 5.8-7.5 m). So these are mostly genuinely short rock, or rock
  the ground grid can't see under trees. Named Climb routes among them:
  Maximus, Mexican Mija, Mijo, Newton County Mentality, Playing Hooky, Season
  of the Storm, Spam. (Season of the Storm and Spam were already 20-31 m from
  any wall in the earlier audit.) The judges see these too.
* **At Horseshoe, 10 m moves many points a long way:** 80 of 142 passing
  points move more than 5 m, and 37 more than 10 m. About 1 in 6 skips past
  a nearer, shorter wall. These are exactly the points the judges should
  look at hardest.
* **Nc points sit on shorter rock than Climb points.** At today's snap spots
  the ground-grid height is a median 9.5 m for Nc and 15.6 m for Climb. So
  height is a real part of why rock was judged not climbable, and the height
  rule removes many of those cases from the task. That is intended: the
  model's question becomes "which **sport-height** walls are climbable", and
  walls under the minimum are "too short" by rule, never asked about.
* **It also reduces Climb/Nc conflicts.** Nc points within 2.5 m of a Climb
  point after snapping: 21 of 57 at 4 m, 13 of 44 at 10 m.

**Decision (2026-11-15): 6 m.** The user picked 6 m as the lowest height
reasonable for a climbing route, from their own outdoor climbing; 10 m cut
more routes than expected. At 6 m:

* 94% of Horseshoe Climb points pass (8 of 126 dropped), 95% of Nc (3 of 60
  dropped), and 98% elsewhere;
* at Horseshoe, 91 of the 175 passing points move more than 5 m, 35 more
  than 10 m, and 17 skip a nearer, shorter wall;
* the Climb/Nc conflicts mostly remain: 28 of 57 Nc points end up within 5 m
  of a Climb point (23 within 2.5 m). Most of that is the snap squashing
  points together (section 4b), not the height rule.

Named Climb routes dropped at 6 m: Mexican Mija, Mijo, Newton County
Mentality, Season of the Storm, Spam.

## 4b. The snap squashes neighbouring points together

A Climb point within a few metres of an Nc point is normal (a route beside
a blank or loose stretch). But the current snap makes such pairs look closer
than they are. Each point independently jumps to the **nearest spot that
passes the checks**. On sparse faces only a few spots pass, so points
recorded several metres apart often land on the same one.

At Horseshoe, 6 m minimum, current snap:

* **15 Nc points land exactly on a Climb point (0.0 m apart)**;
* 28 Nc points end up within 5 m of a Climb point. Their raw GPS points were
  a median 5.1 m apart, and half were more than 5 m apart;
* 90 pairs of Horseshoe points (any labels) end up within 1 m.

What follows from it (details in plan 4 and plan 1):

* **The snap moves points straight toward their wall**, keeping their
  spacing along it, instead of to the nearest passing spot. It applies to all
  data.
* **Close Climb/Nc pairs are kept.** Their labelled strips are split so no
  rock gets both labels. A pair is dropped only when the two feet are under
  1 m apart and the judges didn't confirm both positions.
* **For Horseshoe only, once:** subagent judges check each point: two per
  wall section, plus a tie-break judge where they disagree. Their decision
  is final.
* **The evaluation gets a pair check** (does the Climb point score above its
  Nc neighbour?), and part of Horseshoe moves into training by wall section.

## 5. How the Horseshoe files are used today

Every reader goes through `load_waypoints` in `crag/groundtruth.py`, which
reads two fixed paths:

* `data/handgathered/Waypoints_2026-10-24.csv`
* `data/handgathered/Waypoints_2026-10-25.csv`

| reader | what for |
|---|---|
| `scripts/s3_collect_routes.py` | copies the points into `data/routes/verified_routes.csv`, which the fine-tuning build reads |
| `scripts/s3_build_finetune_set.py` (`validate_on_handgathered`) | checks the band negatives against the hand labels |
| `crag/gt_eval.py` | Horseshoe scoring of pretraining checkpoints |
| `scripts/plot_hcr_waypoints.py` | the audit maps |
| `proof-of-concept/` | its own copy of the same paths (old code; it will see the snapped points too) |

A point's id is `<file name>#<row number>`, so the files must keep their
names and row order.

**So the snapped points can go in place of the originals at the same paths.
No reader needs a new path**, as you suggested. Two things still need
changing:

1. `data/routes/verified_routes.csv` holds its **own copy** of the
   coordinates. It must be rebuilt after the snap, or the fine-tuning build
   keeps using the old points.
2. The pipeline must **not snap the points again**. Snapping an already
   snapped point usually moves it very little, but it can move it to
   another wall, and it would undo the judges' decisions. The saved
   file marks each point as already snapped, and the wall check then only
   confirms a wall is still there (within 3 m) instead of searching 20 m.

## 6. No hand review

Earlier drafts planned a QGIS review by the user. It was dropped
(2026-11-15): a map shows the user the same LiDAR the judges see, not the
rock itself, so a hand guess wouldn't beat the models'. Instead, plan 4
relies on:

* two independent judges per wall section, plus a tie-break judge;
* a calibration check: the judges must keep the automatic snap on easy
  points, or nothing is applied;
* a walking-order check;
* strict limits on where a judge may put a point (on the wall mask, on a
  wall at least 6 m tall);
* Horseshoe results also reported without the unresolved and low-confidence
  points.
