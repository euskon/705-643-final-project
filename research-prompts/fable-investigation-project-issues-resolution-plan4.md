# Plan: wall height check and a better snap for all route points, and a one-time clean-up of the Horseshoe points

Resolves **problem 4** of `docs/fable-investigation-project-issues.md`. The
investigation and the dry-run numbers this plan relies on are in
`docs/fable-investigation-hcr-data-fix-and-snapping-logic-updates.md`.

There are two parts, kept strictly apart:

* **For all data (every area, every build):** the wall check gets a minimum
  wall height and a snap that keeps points' spacing along the wall. It runs
  automatically every time the fine-tuning set is built.
* **For Horseshoe Canyon Ranch only, once:** the points are snapped by the
  new wall check, checked by subagent judges, and saved. From then on they
  are used as saved and never snapped again.

**No hand review.** The user decided (2026-11-15) not to review the
Horseshoe points by hand in QGIS. From a map they can't see the rock any
better than the judges can, so their guess wouldn't beat the models'. The
judges are the final check, so the plan builds in its own safeguards: two
independent judges, a tie-break judge, a calibration check, a walking-order
check, and strict limits on where a judge may put a point.

## 0. Before you start

* **Order.** Do this before building fine-tuning set V4 (plan 1). V4's
  positives, negatives and Horseshoe points all come out of the wall check
  changed here. It can be coded alongside plan 1. Plan 1's `wall_foot` runs
  **after** `snap_one`, and plan 1's close-pair rules (section 5.6) use the
  saved Horseshoe points.
* **The minimum wall height (`MIN_WALL_HEIGHT_M`) is 6 m**, the user's choice
  (2026-11-15) after the dry run (investigation doc, section 4): "the lowest
  threshold reasonable for a climbing route", from their outdoor climbing.
  Below 6 m is bouldering height. At 6 m the dry run keeps 94% of Horseshoe
  Climb points, 95% of Nc and 98% elsewhere.
* **Backup already done:** `data/handgathered/original/` holds read-only
  copies of `Waypoints_2026-10-24.csv`, `Waypoints_2026-10-25.csv` and both
  GPX files. Never write to that folder. If it is missing or incomplete,
  stop and ask.
* **Environment:** `conda activate crag-pointnext`. No new packages.
* **Run `pytest` first**, so any test that already fails is known.

## 1. The goal

1. Every route GPS point, in every area, is snapped only to a wall at least
   6 m tall, measured on the bare-earth ground grid. Climb and Nc points are
   treated alike.
2. The snap moves a point **straight toward its wall**, so points keep their
   order and spacing along the wall. Today each point jumps to the nearest
   spot that passes the checks, which squashes neighbours together. With the
   6 m rule, **15 Horseshoe Nc points land exactly on a Climb point (0.0 m
   apart)**, and 90 pairs of Horseshoe points end up within 1 m. For the
   Climb/Nc pairs within 5 m after snapping, the raw GPS points were a median
   5.1 m apart.
3. Every not-climbable sample made by the code (band, far cliff) stands on a
   wall that passes the same height rule. Otherwise short walls would appear
   only as "not climbable", which is a giveaway (plan 1's group B leak check).
4. **Horseshoe only, once:** automatic snap -> subagent judges -> saved at
   the same file paths. Nothing in this step runs for any other area.
5. A Climb point and an Nc point **may** be close together. That is real (a
   route beside a blank or loose stretch). They are kept, not dropped; plan 1
   splits their labelled strips.

---

## 2. Step 1: wall height from the ground grid (`crag/walls.py`)

Build the investigation's prototype into the shared wall module.

```python
def wall_height_at(grid: WallGrid, xy, base_z, *, max_walk_m=30.0, step_m=0.5,
                   start_m=3.0, gap_m=1.5, past_top_m=1.0) -> tuple[float, float]:
    """(wall height in m, distance walked across the wall strip in m).

    From ``xy`` (a candidate foot), walk uphill (ground-grid gradient after
    the usual Gaussian smooth, averaged over 5 x 5 cells) across the wall
    mask. The strip must be reached within ``start_m``; breaks up to
    ``gap_m`` are allowed. Height = highest ground value in the strip and
    ``past_top_m`` beyond it, minus ``base_z``. (0, nan) when no strip is
    reached."""

def wall_frame_at(grid: WallGrid, xy) -> tuple[np.ndarray, np.ndarray]:
    """(out, along) unit vectors: out = downhill (from the wall into the air), along = out turned 90 degrees."""
```

* Keep the gradient in `WallGrid` (add `gx`, `gy` fields filled by
  `wall_grid`), so it's computed once per grid.
* `max_walk_m` limits how far back the walk goes, not the wall height. If
  plan 1's ground-grid wall top is used for tall walls, raise it to 60 m
  there.
* **Grid building for the wall check.** `snap_one` today gets a
  `PointCloud` from the cliff-point crop and has no access to the provider's
  ground classes. Add `las` (the provider's class), `return_number` and
  `number_of_returns` to `AreaCloud.crop` in `crag/crop.py` (plan 3 adds the
  last two as well; whichever lands first adds them). Then add
  `crag.walls.local_grid(cloud, cx, cy, half=45.0)`, which builds the grid
  from a raw crop with the same inputs as the pseudolabels (`surface_mask`,
  ground = class 2).
* **Speed:** the prototype took about 6 minutes for 611 points (one grid per
  point, 90 m wide). If the full build gets slow, cache grids per 200 m tile
  with a 45 m border in `AreaCloud`. Don't do it pre-emptively.

## 3. Step 2: the new wall check (`crag/groundtruth.py`, all data)

`GroundTruthConfig`:

```python
min_wall_height_m: float = 6.0         # user's choice; measured on the ground grid
height_measure: str = "grid"           # "grid" (new) | "column" (old, for comparison runs only)
snap_method: str = "toward_wall"       # "toward_wall" (new) | "nearest" (old, for comparison runs only)
target_min_steep_points: int = 3       # lighter check for the spot straight in front of the person
max_along_shift_m: float = 3.0         # flag a point slid further than this along the wall
presnapped_radius_m: float = 3.0       # confirmation radius for points already snapped (step 7)
```

### 3.1 Which wall: the anchor

The candidate loop stays as today: candidate cliff spots within 20 m,
nearest first. A candidate passes when it has at least 8 cliff points within
2.5 m, its foot is within 6 m of the standing height, and **its ground-grid
height is at least `min_wall_height_m`**. The first candidate that passes is
the **anchor**: it identifies which wall the point belongs to.

### 3.2 Where on that wall: straight in front of the person

1. Get the wall's directions at the anchor with `wall_frame_at` (`out`,
   `along`).
2. Project the GPS point onto the wall line through the anchor:
   `s = (gps - anchor) · along`, and the target is `anchor + s · along`. That
   is the foot of the wall straight in front of where the person stood,
   measured at right angles to the wall.
3. **Accept the target** if:
   * it lies within 2 m of the wall mask;
   * at least `target_min_steep_points` (3) cliff points are within 2.5 m.
     Fewer than the anchor's 8, because sparse faces often have few points
     at any single spot. The anchor already proved the wall is real;
   * its foot is within 6 m of the standing height; and
   * its ground-grid height is at least `min_wall_height_m`.
4. Otherwise **slide along the wall toward the anchor** in 0.5 m steps until a
   spot passes. The anchor always passes, so this ends. Record the slide as
   `along_shift_m`.
5. The snapped position is the nearest cliff column (within 1 m) to the
   accepted spot, so it sits on real points. Plan 1's `wall_foot` refines the
   foot afterwards.

`snap_method: nearest` keeps the old behaviour (the anchor is the snapped
spot) for comparison runs.

### 3.3 Fields returned

| field | meaning |
|---|---|
| `wall_height_m` | ground-grid height at the snapped spot |
| `wall_height_column_m` | the old column measure, kept for comparison |
| `anchor_x`, `anchor_y` | the anchor spot |
| `along_shift_m` | how far the point slid along the wall from straight in front |
| `nearest_wall_height_m`, `nearest_wall_dist_m` | the nearest candidate that passed every check except height |
| `skipped_shorter_wall` | true if the anchor is more than 3 m further than that nearest candidate, which was under 6 m |
| `status` | `ok`, `too_short` (walls within reach, but all under 6 m), `off_wall` (no wall at all), `no_lidar` |

* `too_short` is kept apart from `off_wall` in every report (`wall_check.csv`,
  `rejected_points.csv`).
* Callers that only check `status == "ok"` need no change. Check that each
  one doesn't treat `off_wall` as the only failure status.

### 3.4 Spacing check after snapping (all areas)

New function `spacing_report(snapped: pd.DataFrame) -> pd.DataFrame` in
`crag/negatives.py`, run in every fine-tuning build after `snap_routes`.
For each pair of `ok` points in the same area whose raw GPS points are 2-20 m
apart:

* raw spacing **along the wall** = `|(raw_a - raw_b) · along|`, using the
  `along` direction at their anchor;
* snapped spacing = the distance between the snapped positions.

Flag the pair `squashed_pair` if the snapped spacing is under half the raw
along-wall spacing. Write the pairs to `squashed_pairs.csv` and the count per
area to `report.json`. Nothing is dropped here: plan 1's close-pair rules
decide.

## 4. Step 3: the same height rule for code-made negatives

In `crag/negatives.py` (and plan 1's `wall_foot`, once it exists):

* **band** and **far cliff** candidates: after centring on the wall foot,
  compute `wall_height_at` at the foot; drop the candidate if it is under
  6 m. Count the drops per source in `report.json`.
* **step-back** samples are off the wall (and removed from the task by plan
  1), so no height rule applies to them.
* Store `wall_height_m` for every sample in `samples.csv`. The plan 1 leak
  test reads it as a group B number. It should now look alike for climbable
  and not climbable, since both are at least 6 m.

## 5. Step 4: a permanent dry-run report (all data)

New script **`scripts/s3_wall_height_dryrun.py`**, built on steps 1-2:

* Input: `data/routes/verified_routes.csv` (all areas). For Horseshoe it
  reads the **originals** in `data/handgathered/original/`, so it always
  measures the automatic method. Plus `--sample-per-area 250`,
  `--thresholds 4,6,8,10,12` and `--methods nearest,toward_wall`.
* Per threshold and method, per area and label:
  * share passing; median distance moved; share that skipped a shorter wall;
  * share slid more than 3 m along the wall (`toward_wall` only);
  * `squashed_pair` count; Climb/Nc pairs within 1, 2.5 and 5 m at
    Horseshoe; Horseshoe Nc points landing on a Climb point (under 0.5 m);
  * the route-length check (rank correlation, share of walls at least 0.8 ×
    route length) for both height measures.
* Output: printed tables plus `data/routes/wall_height_dryrun.json`.
* **Run it and compare with the investigation.** For `nearest` at 6 m the
  prototype gave:

  | | Horseshoe Climb | Horseshoe Nc | Red River Gorge | other areas |
  |---|---|---|---|---|
  | share passing | 0.94 (8 of 126 dropped) | 0.95 (3 of 60 dropped) | 0.98 | 0.98 |
  | median move | 5.9 m | 3.4 m | 1.3 m | 0.4 m |
  | skipped a shorter wall | 0.11 | 0.07 | 0.01 | 0.01 |

  Horseshoe, `nearest`, 6 m: 15 Nc points on a Climb point (0.0 m), 23 Nc
  points within 2.5 m of a Climb point, 90 pairs of points within 1 m.
  Route-length correlation about 0.43.

  Big differences for `nearest` mean the code differs from the prototype, so
  find out why first. **Expected for `toward_wall`:**
  * about the same pass rates (the anchor uses the same checks);
  * Nc points on a Climb point near 0;
  * pairs within 1 m close to the number already within 1 m in the raw GPS;
  * few `squashed_pair` flags.

  If `toward_wall` doesn't clearly beat `nearest` on spacing, stop and look
  at the cases before going on.

---

## 6. Step 5: the one-time automatic snap of the Horseshoe points

New script **`scripts/snap_hcr_waypoints.py`**. It refuses any area other
than `horseshoe-canyon-ranch`.

**Input: always the originals** in `data/handgathered/original/`, never the
working files. So the script can be rerun without snapping already-snapped
points.

For each row of each original CSV, in the original order:

* `Climb` and `Nc` rows: run the step 2 wall check (`toward_wall`, 6 m, same
  settings as the rest of the pipeline).
* `?` rows (unlabelled): copy unchanged.

**Output: the working files at the original paths**
(`data/handgathered/Waypoints_2026-10-24.csv`, `..._25.csv`). Same rows,
same order, same original columns, so every `gt_id` (`<file>#<row>`) and
every reader stays the same:

| column | value |
|---|---|
| `lat`, `lon` | **final position**: after this step the automatic snap; replaced by step 6 (judges) |
| `ele_m` | unchanged (GPS elevation, for reference) |
| `raw_lat`, `raw_lon` | the original position |
| `auto_lat`, `auto_lon` | the automatic snap (kept even after the judges change `lat/lon`) |
| `snap_status` | `ok` / `too_short` / `off_wall` / `no_lidar` / `unlabelled`; later `judged`, `judge_none` |
| `snap_dist_m`, `along_shift_m`, `wall_height_m`, `wall_height_column_m`, `nearest_wall_height_m`, `skipped_shorter_wall` | from step 2 |
| `presnapped` | `True` for rows with a usable position |
| `snap_flags` | semicolon-separated notes: `moved_gt_5m`, `moved_gt_10m`, `skipped_shorter_wall`, `slid_along_wall`, `squashed_pair`, `near_opposite_label` (a Climb and an Nc within 5 m; information only, not an error), `too_short`, `off_wall`; the judges add more (step 6) |
| `snap_min_wall_height_m`, `snap_method` | the rule used, so the file records it |
| judge columns | empty here; filled in by step 6 |

**Safety:**

* If a working file already has anything in the judge columns, **stop**
  unless `--force` is given, and say how many judged rows would be lost.
* Write to a temporary file, then rename.
* Print a summary: counts per status and label, and per flag.

---

## 7. Step 6: subagent judges for the Horseshoe points (once, Horseshoe only)

The automatic snap uses fixed rules. Some Horseshoe cases need judgement:
the East Side points sit 7-12 m out on the downhill side, some points have
two walls in reach, and neighbouring points were recorded in walking order.
Claude subagents judge each point from a picture and the numbers. **Their
decision is final**: there is no hand review afterwards.

**This step runs once, for Horseshoe Canyon Ranch only.** Its scripts refuse
any other area. Its outputs are saved and committed. Rerunning needs
`--force`.

### 7.1 Build the cases: `scripts/hcr_judge_cases.py`

For every labelled Horseshoe point (all 186, so the judges see each point's
neighbours), write to `data/handgathered/review/judge/cases/`:

**`<gt_id>.png`**, one image per point:

* a 60 m × 60 m map centred on the raw GPS point, at 1 m resolution:
  hillshade, 2 m contour lines, the wall mask, and cliff points coloured by
  height above the anchor's foot;
* the raw GPS point (×), the automatic snap (○), and **numbered candidate
  spots** (up to 8):
  1. the automatic snap,
  2. the anchor,
  3. the nearest spot that passes every check,
  4. the spot straight in front on each other wall within 20 m,
  5. the tallest wall within 20 m;
* the neighbouring waypoints: label, name, raw position and their own
  automatic snap;
* scale bar and north arrow;
* a second panel: an elevation profile through the raw point along the
  anchor's `out` direction, marking the standing height, each wall's foot and
  top.

**`<gt_id>.json`:**

* the point: label, name, `mp_url`, `area_name`, `route_index`, time, raw
  position;
* each candidate: id, position, distance from the raw point,
  `along_shift_m`, ground-grid height, column height, foot height minus
  standing height, cliff points within 2.5 m, and which wall (connected
  wall-mask piece);
* the previous and next three waypoints in walking order (by `time`), with
  label, name and automatic snap.

**`batches/<section>.json`:** the case ids per wall section (`area_name`:
West Side, North Forty, East Side), in walking order.

### 7.2 The judge instructions: `data/handgathered/review/judge/PROMPT.md`

Written by the script, the same for every judge:

* **Facts:**
  * the waypoints are where a person stood at the wall base;
  * handheld GPS under a cliff is off by 3-10 m;
  * East Side points sit 7-12 m out on the downhill side (a known offset);
  * walls under 6 m don't count (bouldering height);
  * a Climb and an Nc point may be close together: that's real, not an error;
  * **nobody will check your answer by hand**, so mark real uncertainty as
    low confidence.
* **What to decide for each point:** which candidate is the foot of the rock
  the label is about, or `none` (no candidate fits, so the point should be
  dropped), or `custom` with coordinates. `custom` must lie within 2 m of the
  wall mask, on a wall at least 6 m tall.
* **How to decide:**
  * prefer the wall the person faced, on the downhill side;
  * the foot should be near the standing height;
  * keep the walking order: consecutive waypoints should progress along the
    wall unless the times show the person turned back;
  * a route name and its neighbours should end up on the same stretch of
    wall;
  * never pick a wall just because it is taller;
  * if unsure, say so with low confidence rather than guessing.
* **Output:** `decisions_<section>_<judge>.json` (one entry per case:
  `gt_id`, `choice`, `custom_lat`/`custom_lon` if any, `confidence`
  high/medium/low, `reason` in one or two sentences).
* **Read the case files; write only the one decisions file; change nothing
  else.**

### 7.3 Run the judges

The implementing Claude Code session starts the judges with the Agent tool:

* **Round 1:** two independent judges (`a` and `b`) per wall section: 6
  subagents. Each prompt names its batch file and `PROMPT.md`. Neither judge
  sees the other's decisions. Each reads the case images with the Read tool
  and writes its decisions file.
* **Round 2, tie-break:** after 7.4 steps 1-3, one more judge (`c`) per
  section that has disagreements: at most 3 subagents. It sees **only the
  disagreed cases**, written to `batches/<section>_tiebreak.json`. It gets the
  same `PROMPT.md`, plus judges `a` and `b`'s choices and reasons as two
  opinions to weigh, not to copy. It writes `decisions_<section>_c.json`.
* At most 9 subagents in total. The user asked for this; don't start more.
* If a judge fails or writes an invalid file, rerun only that judge, once.
  If it fails again, treat its cases as unresolved.

### 7.4 Check and apply: `scripts/hcr_judge_apply.py`

1. **Validate** every decisions file: every case answered once; candidate
   ids exist; `custom` positions within 2 m of the wall mask and on a wall at
   least 6 m tall. Reject the file with a clear message otherwise.
2. **Agreement** between `a` and `b`: the same candidate, or `custom`
   positions within 1.5 m of each other, or both `none`.
3. **Calibration check** on easy points (the automatic snap moved under 1 m
   and only one wall is within 20 m). Each judge should keep the automatic
   snap on at least 90% of them. If either judge is below that, **stop**,
   report the disagreeing easy cases to the user, and apply nothing. The
   instructions or images need fixing first. This is the main safeguard
   now that there's no hand review.
4. Write the tie-break batches; run round 2 (7.3); validate its files.
5. **Final position per point:**

   | case | position | `snap_status` | `judge_result` |
   |---|---|---|---|
   | `a` and `b` agree on the automatic snap | unchanged | `ok` | `agreed` |
   | `a` and `b` agree on another candidate or a custom spot | the agreed spot | `judged` | `agreed` |
   | `a` and `b` agree on `none` | unchanged | `judge_none` (dropped downstream) | `agreed` |
   | they disagree; `c` sides with one of them | that judge's choice | `judged` / `ok` / `judge_none` as above | `majority` |
   | they disagree; `c` picks a third answer | the automatic snap | `ok` | `unresolved` |

6. **Walking-order check:** within each section, list consecutive waypoints
   (by time) whose order along the wall reversed after judging, and flag them
   `order_reversed`. If any reversal involves an `agreed` or `majority`
   point, keep it, but report it in the summary.
7. **Write** to the working CSVs (same safety rules as step 5):
   * `judge_a_choice`, `judge_b_choice`, `judge_c_choice`;
   * `judge_confidence` (the lowest of the judges that decided);
   * `judge_reason_a`, `judge_reason_b`, `judge_reason_c`;
   * `judge_result`, the new `lat/lon` and `snap_status`;
   * flags `judge_low_confidence`, `judge_moved_from_auto`, `judge_none`,
     `judge_unresolved` and `order_reversed`.
8. **Print a summary:** agreement rate, tie-break outcomes, calibration rate,
   how many points moved from the automatic snap and by how much, and counts
   per flag.

Then rebuild the routes copy:

```bash
python scripts/s3_collect_routes.py    # refresh data/routes/verified_routes.csv
```

### 7.5 How the judge results are used downstream

* **`agreed` and `majority`** points count as **confirmed** positions
  (plan 1 section 4.3: confirmed close pairs are never dropped).
* **`unresolved`** points keep the automatic snap and are **not confirmed**.
  They are used, but every Horseshoe evaluation also reports its numbers
  **without them**, so we can see whether they change the result.
* **`judge_none`** points are dropped and counted in the reports.
* Points with `judge_low_confidence` are used. The evaluation reports
  Horseshoe numbers on **high + medium confidence only** as well.

---

## 8. Step 7: the pipeline uses the saved Horseshoe points without snapping again

* `crag/groundtruth.load_waypoints`: read the new columns when present. Old
  files without them still load, with `presnapped` = False. Rows whose
  `snap_status` is not usable (`too_short`, `off_wall`, `no_lidar`,
  `judge_none`) keep their label but carry their status, so reports still
  count them.
* `scripts/s3_collect_routes.handgathered_rows`: pass `presnapped`,
  `snap_status`, `judge_result` and `judge_confidence` through to
  `verified_routes.csv`.
* `crag/negatives.snap_routes`: for `presnapped` rows, call `snap_one` with
  `snap_radius_m = presnapped_radius_m` (3 m). That confirms a tall-enough
  wall is still there without moving the point. If the check fails, keep the
  point, mark it `presnapped_lost` and log a warning naming it. The judges'
  placements are never overridden. Rows with an unusable status are not
  re-checked.
* `crag/gt_eval.score_waypoints` and `crag/negatives.validate_on_handgathered`
  build their own frame for `snap_routes`; pass the same columns in it.
* `scripts/plot_hcr_waypoints.py`: draw raw -> automatic -> final, coloured
  by `judge_result`. A record of what was done, for reports. No editing.
* `proof-of-concept/` reads the same paths and will see the final points.
  Leave it; add one line to its config saying so.
* **Check after every rebuild of `verified_routes.csv`:** its Horseshoe rows
  match the working CSVs' `lat/lon` exactly (a test, and an assert in
  `s3_collect_routes.py`).

## 9. Step 8: close Climb and Nc points

A Climb point and an Nc point may be close together; that is real.

* `near_opposite_label` is **information, not an error**. Plan 1 section 5.6
  uses it to split the two labelled strips, so no rock gets both labels.
* Points are dropped only when the two feet are **under 1 m apart and the
  judges didn't confirm both positions** (plan 1 section 4.3).
* The new snap (step 2) should make those under-1 m cases rare. Under the
  old snap, 15 Nc points sat exactly on a Climb point. The judges handle the
  rest.
* Plan 1 section 8.1 adds a **pair check** to the evaluation (does the Climb
  point score above its Nc neighbour?).

## 10. Step 9: rerun downstream and report

1. `python scripts/s3_wall_height_dryrun.py`: record the final tables in
   `docs/results/v4/README.md`.
2. Rebuild the fine-tuning set (plan 1's V4 build). `wall_check.csv` shows
   `too_short` separately; `squashed_pairs.csv` lists any remaining squashed
   pairs.
3. In the V4 results, report the Horseshoe points at each stage (automatic,
   judged): Climb and Nc counts, positions changed, and counts per
   `judge_result` and flag.
4. Add a short note to `docs/hcr-waypoint-audit.md`: the points are now
   pre-snapped and judged, how, and where the decisions are kept.

---

## 11. Tests to add (`tests/test_labels_negatives_crossref.py`)

| test | checks |
|---|---|
| `test_wall_height_at_synthetic` | a synthetic ground grid with a 12 m step: about 12 m (±1); a 5 m step: about 5 m; flat ground: 0 |
| `test_wall_height_ignores_hillside` | a 10 m wall with a gentle slope rising another 20 m behind it: about 10 m, not 30 |
| `test_snap_skips_short_wall` | a 5 m wall 4 m away and a 12 m wall 12 m away: with a 6 m minimum, snaps to the 12 m wall and sets `skipped_shorter_wall`; with a 4 m minimum, the 5 m wall |
| `test_too_short_status` | only a 5 m wall within reach: status `too_short`, not `off_wall` |
| `test_toward_wall_keeps_spacing` | two GPS points 4 m apart along a straight wall, both 8 m out: snapped 4 m apart (±0.5); with `nearest` on a wall whose cliff points are sparse they may collapse, and the test documents that |
| `test_toward_wall_slides_when_needed` | the spot straight in front has no cliff points: the point slides to the nearest passing spot and records `along_shift_m` |
| `test_spacing_report_flags_squash` | a pair snapped 0.5 m apart from raw points 5 m apart along the wall is flagged |
| `test_presnapped_not_moved` | a presnapped point beside two walls stays put (moves under 3 m) |
| `test_snap_script_keeps_rows` | on a small copy of a waypoint CSV: same row count, same order, same `gt_id`s, originals untouched, `?` rows unchanged |
| `test_hcr_only` | the snap, judge-case and judge-apply scripts refuse another area |
| `test_scripts_protect_judging` | filled judge columns make the scripts stop without `--force` |
| `test_judge_apply_rules` | made-up decisions: agreement; disagreement settled by `c` (majority); three different answers (automatic snap, `unresolved`); `none`; an invalid candidate id (rejected); a `custom` point off the wall (rejected); a calibration failure (stops without writing) |
| `test_routes_copy_matches` | after `s3_collect_routes`, the Horseshoe rows equal the working CSVs' `lat/lon` |
| `test_negatives_height_rule` | a band candidate on a 5 m wall is dropped with a 6 m minimum |

Run the whole suite after each step.

## 12. Files that change

| file | change |
|---|---|
| `crag/walls.py` | `wall_height_at`, `wall_frame_at`; gradient in `WallGrid`; `local_grid` |
| `crag/crop.py` | `crop` also returns `las`, `return_number`, `number_of_returns` |
| `crag/groundtruth.py` | config fields; `snap_one` anchor + straight-in-front snap, height rule, new fields and `too_short`; `load_waypoints` reads the new columns |
| `crag/negatives.py` | `snap_routes` presnapped handling; `spacing_report`; height rule for band and far-cliff candidates |
| `crag/gt_eval.py` | pass the new columns; report Horseshoe also without `unresolved` and on high + medium confidence |
| `scripts/s3_collect_routes.py` | pass the new columns through; assert the Horseshoe rows match |
| `scripts/s3_build_finetune_set.py` | report `too_short`, negative drops, `squashed_pairs.csv` |
| `scripts/s3_wall_height_dryrun.py` | new |
| `scripts/snap_hcr_waypoints.py` | new (Horseshoe only) |
| `scripts/hcr_judge_cases.py`, `scripts/hcr_judge_apply.py` | new (Horseshoe only, once) |
| `scripts/plot_hcr_waypoints.py` | draw raw -> automatic -> final |
| `data/handgathered/Waypoints_2026-10-2{4,5}.csv` | rewritten by steps 5 and 6 (originals in `original/`) |
| `data/handgathered/review/judge/` | cases, `PROMPT.md`, decisions (committed) |
| `data/routes/verified_routes.csv` | rebuilt |
| `docs/pipeline.md` | runbook: dry run; Horseshoe one-time snap and judges; rebuild routes |
| `docs/hcr-waypoint-audit.md` | note on the pre-snapped, judged points |

## 13. Risks

| risk | what we do |
|---|---|
| The ground grid misses a wall under dense trees and calls it short | `too_short` counted separately; those points are shown to the judges |
| The straight-in-front spot is on the wrong wall where walls curve or stack | it must pass the same checks as the anchor (within 2 m of the wall mask, foot at standing height, 6 m tall) or the point slides toward the anchor; long slides are flagged |
| Moving past a shorter wall puts a label on the wrong rock (17 Horseshoe points at 6 m) | flagged `skipped_shorter_wall`; judged |
| The judges are confidently wrong, with no human to catch it | two independent judges plus a tie-break; the calibration check stops everything below 90%; the walking-order check; Horseshoe results also reported without `unresolved` and without low-confidence points |
| The judges invent positions | `custom` must be within 2 m of the wall mask on a 6 m wall; otherwise rejected |
| A script overwrites the judging | always starts from `original/`; refuses without `--force` when judge columns are filled |
| `verified_routes.csv` keeps old coordinates | rebuilt after every step that changes points; asserted to match |
| The 6 m rule drops real routes | few: 2% outside Horseshoe; 8 Climb points at Horseshoe (named: Mexican Mija, Mijo, Newton County Mentality, Season of the Storm, Spam; the last two were already 20-31 m from any wall). Listed with `too_short`; shown to the judges |
| The new snap changes the Red River Gorge points used in V4 | expected; V4 is a new set. The dry run compares the two methods for every area |

## 14. Decisions for the user

| decision | default | other options |
|---|---|---|
| minimum wall height | **6 m: decided by the user** (2026-11-15) | - |
| Nc points | same rule as Climb | decided |
| close Climb / Nc points | kept; strips split (plan 1 section 5.6); dropped only under 1 m and not confirmed by the judges | decided |
| hand review of the Horseshoe points | **none: decided by the user** (2026-11-15); the judges are final | - |
| judges | Horseshoe only, once; 2 per wall section plus a tie-break judge for disagreements (at most 9) | decided |
| calibration bar for the judges | 90% agreement with the automatic snap on easy points | 80% |
| confirmation radius for saved points | 3 m | 2 m |
| lighter cliff-point check for the spot straight in front | 3 points | 5 |
