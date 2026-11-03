# Evaluating the dataset and the pseudolabeling heuristic with hand-gathered ground truth

## The short version

- **What we have:** 189 GPS waypoints, taken by hand on 2026-10-24/25 at the base of the walls at Horseshoe Canyon Ranch. Each one is labeled `Climb`, `Nc` (not climbable rock) or `?`.
- **What we want to know:**
  1. Does the surface-normal "rock" heuristic find the walls we actually stood under?
  2. Can that heuristic tell climbable rock from non-climbable rock?
  3. How wrong are the weak labels in `samples.csv`?
  4. How well does the trained model do on real ground truth?
- **The main point:** the heuristic in `labeling.py` is a **rock detector, not a climbability detector**. `Nc` points are rock too, so we expect good rock recall but poor Climb-vs-Nc separation. This evaluation measures both. The `Nc` points are also the only true hard negatives at the wall that we have.
- **Rule:** the waypoints are a **held-out test set**. Nothing in this plan trains on them. Training with them (Plan 1, below) comes after this evaluation.

---

## 1. The heuristic under test

This is how the current labels are made (from `labeling.py` and `config.LabelConfig`):

| Step | Rule | Knob (default) |
|---|---|---|
| Normals | PCA over k nearest neighbors on a voxel-thinned copy | `normal_k=12`, `normal_voxel_m=0.5` |
| Steep / "rock" point | slope ≥ threshold **and** `number_of_returns == 1` | `steep_slope_deg=60`, `steep_single_return_only=True` |
| Cliff column | at least N steep points within 2.5 m horizontally | `min_steep_points=8`, `route_radius_m=2.5` |
| Pin snapping | search for cliff columns within R of the MP wall pin | `snap_search_radius_m=30` |
| Negative ignored | a negative's 5 m column is more than 25% steep | `max_steep_fraction_random_neg=0.25` |

Each one is a threshold we can sweep against the ground truth.

## 2. Ground-truth data (`data/handgathered/`)

- **Files:** `Waypoints_2026-10-24.csv` and `Waypoints_2026-10-25.csv`.
- **Columns:** `name, area, area_name, route_index, lat, lon, ele_m, time, comment, description, symbol, mp_url`

  | Column | Meaning |
  |---|---|
  | `name` | Mountain Project route name, or `?` |
  | `area` | 0 = North Forty, 1 = West Side, 2 = East Side |
  | `route_index` | Order of the route in the Mountain Project listing, or `?` |
  | `comment` | Label: `Climb` / `Nc` / `?` |
  | `mp_url` | Mountain Project link for named routes |

- **Counts:** 126 `Climb`, 60 `Nc`, 3 `?` (the `?` points are excluded).
- **Caveats:**
  - Phone/handheld GPS under a cliff is off by about **3–10 m**, and the waypoint is where the person stood, a few meters out from the face. Every metric therefore uses a distance tolerance.
  - `ele_m` is GPS elevation. We only use it to estimate a median offset against the LiDAR, never as a height.
  - Points 0-48/0-49 (both Trapeze Artist) and the two 0-56 points have suspect `route_index` values. Names come from the notes and are fine.

## 3. Loading and snapping (`groundtruth.py`)

1. **`load_waypoints()`** reads both CSVs. It adds `gt_id`, `label` (1/0/−1) and `x, y` in UTM 15N (`usgs_ept.lonlat_to_work`).
2. **`snap_to_cliff(pc, gt, cfg)`** moves each point onto the nearest cliff column:
   - It searches steep points within `GroundTruthConfig.snap_radius_m` (**12 m**, not the 30 m used for MP pins).
   - It keeps a candidate only if:
     - its 2.5 m cylinder has ≥ `min_steep_points` steep points;
     - its base height (5th-percentile Z) is within `max_dz_m` (**6 m**) of the ground height where the person stood (10th-percentile Z of a 3 m column at the waypoint). This stops a snap onto a higher or lower ledge of a tiered cliff.
   - It records `snap_x, snap_y, snap_dist_m, ground_z, base_z, out_dx, out_dy, n_steep_snap, status` (`ok` / `no_cliff`).

## 4. Questions and metrics (`eval_pseudolabels.py`)

The tolerances are **r ∈ {5, 10, 15} m**, applied around the raw waypoint. The results table reports how each metric changes with r.

### Q1. Does the heuristic find the rock? (rock recall)

- **Rock recall** is the share of `Climb ∪ Nc` points with a cliff column within r. It's reported separately for `Climb` and `Nc`.
- **False-cliff rate:** the same test at random points more than 40 m from any wall pin or waypoint. This gives the other side of the trade-off.
- **Low recall** means the walls are hidden by canopy, too sparse, or the threshold is too strict.

### Q2. Can steepness separate Climb from Nc?

- **Scores at each point** (the snapped point if there is one, otherwise the raw point):
  - the max steep-point count in a 2.5 m cylinder within r;
  - `steep_fraction` of the 5 m column;
  - `coverage`, `max_gap_m` and `lean_deg` from `route_profile` (over 15 m of height);
  - the wall height, i.e. the Z range of the steep points in the column.
- **Metrics:** AUROC per score, plus the precision–recall curve and best F1 for the strongest one.
- **AUROC near 0.5** confirms that the heuristic only finds rock, and that we need `Nc`-style hard negatives.

### Q3. How do the weak labels agree with the ground truth?

This uses the labels actually produced (`samples.csv`).

| Ground truth | Nearest weak sample within r | What it means |
|---|---|---|
| `Climb` | `mp_route` (1) | agreement |
| `Climb` | stepback / random (0) | **wrong weak negative** |
| `Climb` | nothing | missed: a route with no positive nearby |
| `Nc` | `mp_route` (1) | **wrong weak positive** |
| `Nc` | stepback / random (0) | agreement |
| `Nc` | nothing | the pipeline has no label there; these were most likely ignored as "mostly cliff" |

- **MP pin error:** for named `Climb` points, the distance from the waypoint to its MP wall pin. This tests the "pins are 10–30 m off" assumption in `config.LabelConfig`.

### Q4. How well does the model do on the waypoints?

- For each waypoint, run the trained checkpoint (`runs/hcr-<mode>/best.pt`) on the 5 m column at the snapped point if there is one, otherwise the raw point. Take the max probability over a small ring of centers within r, the same tolerance idea as above.
- **Metrics:** AUROC, the PR curve, and F1 at 0.5.
- **Caveat:** the checkpoints in `runs/` are from a 2-epoch `FAST_DEV_RUN`, so these numbers set up the pipeline rather than make a claim. Re-run after a full GPU training run.

### Reporting rules

- **Per area** (North Forty / West Side / East Side) as well as overall.
- **Bootstrap 95% CIs** (1,000 resamples). With n ≈ 186, differences under about 0.05 are noise.
- **Leakage:** North Forty's MP positives are in the weak training set. Report Q4 split by distance from the waypoint to the nearest *train* sample (< 7.5 m vs ≥ 7.5 m).

## 5. Ablations (`threshold_sweep`)

To keep each run fast, normals are recomputed only on the LiDAR within 60 m of a waypoint, plus a random far-field set for the false-cliff rate.

| Knob | Values |
|---|---|
| `steep_slope_deg` | 45, 50, 55, 60, 65, 70, 75 |
| `steep_single_return_only` | on, off |
| `normal_k` | 8, 12, 24 |
| `normal_voxel_m` | 0.25, 0.5, 1.0 |
| `min_steep_points` | 4, 8, 16 |

Each run reports rock recall, the false-cliff rate and the Climb-vs-Nc AUROC. Plotting rock recall against the false-cliff rate gives a trade-off curve for picking defaults.

## 6. Visual sanity checks

- **Per-area maps:** steep points in grey, MP pins, `mp_route` and negative samples, waypoints (`Climb` green, `Nc` red) and raw → snapped arrows.
- **Probability map with waypoints:** `inference.plot_probability_map` with the waypoints overlaid.
- **Side profiles** (the notebook's `show_wall` style) for a few `Climb` and `Nc` points.
- **Score histograms** per class, for the Q2 scores.

## 7. What the results would tell us

| Result | Action |
|---|---|
| Low rock recall | Lower `steep_slope_deg`, relax single-return, use a smaller voxel |
| High false-cliff rate | Raise the threshold or `min_steep_points` |
| High rock recall, Climb/Nc AUROC ≈ 0.5 | Steepness ≠ climbability. Add `Nc` hard negatives (Plan 1) and reconsider the "ignore mostly-cliff" rule near ground truth |
| Many `Nc` near `mp_route` | Tighten MP snapping (smaller radius) or replace MP positives with ground-truth positives where available |
| Many `Climb` near stepback negatives | Increase `stepback_m` or check the sign of `wall_direction` |
| Big differences between areas | Report per area; don't tune one global threshold |

## 8. Next: training with the waypoints (Plan 1, after this)

- **`build_gt_samples`:** emit `gt_climb` / `gt_nc` rows with a `weight` (about 2.0 for ground truth vs 1.0 for weak labels).
- **Conflicts, where ground truth wins:**
  - drop `mp_route` positives within 5 m of an `Nc`;
  - drop weak negatives within 7.5 m of a `Climb`;
  - drop opposite-label waypoints closer than 3 m.
- **Code changes:**
  - `dataset.PatchDataset` returns the weight;
  - `trainer.fit` uses per-sample weighted cross-entropy;
  - waypoint rows get split `gt_test` so `make_loaders` never trains on them by accident.
- **Experiment:** leave one area out. Train on the weak labels plus two areas of waypoints, and test on the third. Remove weak samples within 15 m of the held-out waypoints. Never split nearby waypoints randomly.

## Open questions and risks

- Is the `route_index` order correct? It only affects bookkeeping, not labels.
- A GPS standoff of 3–10 m is close to the 5 m patch radius, so tiered cliffs can still snap to the wrong ledge. The 18 photos in `Climbable/` and `NotClimbable/` could be used to spot-check snaps.
- `Nc` counts per area are small (East Side has 19), so per-area CIs will be wide.
