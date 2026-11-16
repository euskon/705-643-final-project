# Plan: fix how fine-tuning samples are cut and labelled (fine-tuning set V4)

Resolves **problem 1** of `docs/fable-investigation-project-issues.md`:
*the model is learning how samples were cut, not the rock.* Background and
the exact current labelling rules: `docs/fable-investigation-v3-labeling-problems.md`.

Problems 2-4 of that doc (neighbourhood scale, input channels, hand-cleaning
the Horseshoe points) are **not** part of this plan. Section 9 lists where
they connect, so the work here doesn't block them.

---

## 1. The goal in one sentence

Every sample on a wall, climbable or not, is centred, cut, labelled, counted
and scored **by the same code with the same settings**, so the only thing that
differs between a climbable and a not-climbable sample is the rock itself.

## 2. How we will know it worked

We measure before and after with a permanent **leak test** (Phase 0). It
fits simple classifiers on a few summary numbers per sample, with no shape
information, and reports their scores.

The summary numbers come in two groups, and they mean different things:

| group | examples | why it exists | target after the fix |
|---|---|---|---|
| **A. how the sample was built** | number of labelled points; height of the labelled points above the wall foot; distance from block centre to the wall foot; labelled height span | these only reflect our rules, never the rock | **AUROC 0.45-0.55** (95% range includes 0.5) on validation, Red River Gorge test and Horseshoe |
| **B. plain wall geometry** | wall height; block height; how steep the face is on average; share of cliff points in the block | real properties of a wall that may honestly relate to climbing (a 4 m wall is rarely a route) | allowed above 0.5; becomes the **simple baseline the network must beat** |

Pass conditions for V4:

1. Group A scores 0.45-0.55 on every split. If not, there is still a leak:
   find it and fix it before any training.
2. Horseshoe Climb vs Nc uses exactly the same labelling for both labels
   (checked by a unit test, section 7).
3. After training (Phase 6), the network beats the group B baseline on
   validation and on Horseshoe Climb vs Nc. If it doesn't, the network adds
   nothing over wall height and steepness. That is a real finding, and it
   points us to problems 2 and 3.

We expect the **validation score to drop** from 0.78, because the shortcut is
gone. That is the intended result, not a failure. The numbers that matter are
Horseshoe Climb vs Nc and the gap above the group B baseline.

---

## 3. Phase 0: the leak test (build first)

Turn the one-off checks from the investigation into a permanent script, so
every future dataset can be checked in a minute.

**New script: `scripts/s3_leak_test.py`**

* Input: `--name v3` (any dataset under `data/finetune/`).
* For every real sample (augmented samples skipped), read the block arrays
  and compute both groups of numbers from section 2.
* Fit one logistic regression per group on the **train** split. Score
  validation, Red River Gorge test, Horseshoe (all samples) and Horseshoe
  Climb vs Nc, with 95% ranges from 1,000 resamples.
* Also report each single number's AUROC. A single number far from 0.5 is
  the quickest way to see *which* rule leaks.
* Output: `data/finetune/<name>/leak_test.json` and a short table printed to
  the terminal. Optionally logged to W&B (Weights & Biases).

**First run: on V3, as a reference.** It should reproduce the investigation:
group A about 0.90 on validation, and the "block centre" numbers about 0.73.
Record these numbers in `docs/results/v4/README.md` later as the "before".

---

## 4. Phase 1: one rule for where a sample is centred

**Today:** climbable samples and Horseshoe Nc are snapped to the wall foot;
band samples can be anywhere on the face; far-cliff samples sit on the
densest cliff spot; step-back samples are off the wall.

**Change: every wall sample is centred on the foot of its wall, using one
function.**

### 4.1 New function `wall_foot(pc, xy)` in `crag/groundtruth.py`

Input: any spot on or near a wall. Output: the foot of that wall, plus the
wall's frame (section 5). Steps:

1. Find the cliff column at `xy`: the steep points within 2.5 m. If there are
   fewer than 8, use the nearest column that has 8 or more, within 5 m.
2. Get the outward direction `out` (from the wall into the air) with the
   existing `wall_direction`.
3. Walk outward along `out` in 0.5 m steps until a 1 m-wide column holds no
   steep points. The last column that still had steep points is the foot
   column.
4. Foot height `base_z` = 5th percentile height of the points in the foot
   column.
5. Return `foot_x, foot_y, base_z, out_dx, out_dy` and a status (`ok`, or
   `no_wall` if step 1 fails).

### 4.2 Who uses it

| sample | how it is centred in V4 |
|---|---|
| climbable (route) | the wall check from plan 4 (`snap_one`: moves the point straight toward the nearest wall at least 6 m tall, keeping its position along the wall), **then** `wall_foot` from the snapped spot |
| Horseshoe Climb and Nc | already snapped once and saved (plan 4: automatic snap, then checked by subagent judges; there is no hand review); only confirmed, not moved (`presnapped`), **then** `wall_foot` |
| band | candidate cliff spot -> `wall_foot` |
| far cliff | candidate cliff spot -> `wall_foot` |
| step-back | removed from this task (Phase 4) |

Climbable and Nc points keep `snap_one` first because they start as GPS
points off the wall. Afterwards every label goes through the same last step,
so the final centre is defined the same way for all.

### 4.3 Checks after centring (all labels)

* Run the **exclusion check after centring**: a not-climbable centre must be
  at least 7.5 m from every known route foot. Moving a band point to the
  foot can bring it closer to a route.
* **Same-label duplicates.** If two samples with the **same** label end up
  within 2 m of each other, keep one, preferring the higher-confidence one.
* **Climbable and not climbable close together is allowed.** A route next to
  a blank or loose stretch is normal, and these pairs are the most useful
  examples for telling rock apart. Don't drop them; the labelled strips are
  split between them instead (section 5.6). Drop a pair only when **both**
  hold:
  * the two feet are **under 1 m apart**, too close to tell which rock each
    label means; and
  * the judges did not confirm both positions. At Horseshoe a position is
    confirmed when the judges agreed on it (plan 4: `judge_agree` or a
    tie-break majority). Confirmed points are always kept; if they are still
    under 1 m apart, log a warning. Points outside Horseshoe have no judges,
    so the rule applies to them as plain "under 1 m apart".

  Log every dropped pair with both ids.
* Store `foot_x, foot_y, base_z, out_dx, out_dy` for every sample in
  `samples.csv`.

---

## 5. Phase 2: one labelled shape that follows the face

**Today:** an upright 2.5 m tube. Climbable samples are cut at 15 m or the
route length; not-climbable ones are not cut at all. On a leaning wall the
tube loses the face partway up: climbable labels reach a median 13 m above
the foot, and only 29% reach the top of their window.

**Change: a strip of the face, measured in the wall's own directions.**

### 5.1 The wall frame

At the foot, use three directions:

* **along**: sideways along the wall (horizontal, at right angles to `out`);
* **out**: horizontal, from the wall into the air;
* **up**: height above `base_z`.

Every point in the block gets `(along, out, up)` coordinates.

### 5.2 The face strip (which points can be labelled)

A point is in the sample's strip when **all** of these hold:

| rule | value | why |
|---|---|---|
| steep (cliff pseudolabel) | yes | only rock is labelled, never ground or trees |
| sideways from the centre | \|along\| <= 2.5 m | same 5 m width as today's tube |
| in front of / behind the foot | -15 m <= out <= +5 m | follows a face that leans back up to 15 m, or overhangs up to 5 m |
| height | -1 m <= up <= wall top | from just below the foot to the top of *this* wall |

### 5.3 Wall top (the same for every sample)

1. Take the steep points that pass the first three rules.
2. Put them into 1 m height bins, starting at the foot.
3. Go up from the foot. The wall top is the last bin before the first gap of
   **more than 2 m with no steep points**, capped at **300 m** (the user's
   choice, 2026-11-15; in practice no cap).

**The block limits the height anyway.** Only points inside the 20 m block
can be labelled, so a face that leans back more than about 10 m leaves the
block. The highest strip points measured on V3 positives are 64 m, and the
cap never binds. A 40 m cap would have cut a third of Red River Gorge strips
(their highest strip point is a median 35 m, and over 40 m for 34%).

**The gap rule, not the cap, decides the top.** As noted in the risks table,
the 2 m gap rule stops early on tall faces. Compare it with the ground-grid
height from plan 4 (`wall_height_at`) before building V4, and use whichever
matches route lengths better.

This stops at a ledge, so a second cliff tier above it isn't counted as the
same wall. **Route length is not used** for anything: not-climbable samples
have no route length, and using it for one label only would bring back a
difference between labels. Record `wall_top_m` per sample. It is a group B
number, so it is allowed to differ between labels.

### 5.4 Everything outside the strip

* Points in the strip of a **neighbouring** sample get codes 3 or 4, using
  the same strip shape (not the old tube), so the "ignored because it
  belongs to someone else" area has the same shape too.
* "Unknown rock" (code -1): steep points within 7.5 m of any known route
  stay ignored, unchanged.
* Everything else is background (code 2), ignored in training.

### 5.5 Code changes

* `crag/finetune_data.py`: new `face_strip_mask(xyz, foot, out_dir, cfg)`
  and `wall_top(...)`; rewrite step d of `block_codes` to use them for every
  label. Delete the "climbable only" height window and the
  label-specific branches.

### 5.6 Splitting strips between close climbable and not-climbable samples

Two strips 5 m wide overlap when their feet are less than 5 m apart along
the wall. The same rock would then be labelled climbable in one block and
not climbable in the other, which trains the model to answer "maybe" there.

**Rule:** for every sample, find the nearest sample of the **opposite**
label on the same wall (feet within 5 m, and its foot inside this sample's
`out` range from section 5.2).

* Let `d` be the distance between the two feet, measured along the wall.
* This sample's strip ends at `d/2 - 0.25 m` on that side; the other side
  keeps its normal 2.5 m. So a 0.5 m unlabelled gap separates the two
  labels, and no point ever gets both.
* The other sample of the pair gets the mirror rule, so both are treated the
  same way (no difference between labels).
* If a side shrinks below **1 m**, or the sample ends up with fewer than the
  minimum labelled points (section 6.1), drop the sample and log it as
  `strip_too_narrow` with its pair.

**Only opposite-label pairs are split.** Same-label neighbours (routes 2-3 m
apart at Red River Gorge) can share rock: the label is the same, so there's
no contradiction, and splitting would leave dense crags with thin strips.

**Leak check.** Split strips are narrower, and they occur only next to an
opposite-label sample. Add `strip_width_m` and `has_opposite_neighbour` to
the group A numbers of the leak test. Both members of a pair are narrowed
the same way, so they should not separate the labels; check that they
don't.

Code: `split_strip_widths(samples) -> (left_m, right_m)` per sample in
`crag/finetune_data.py`, computed once after centring (section 4.3) and
stored in `samples.csv` (`strip_left_m`, `strip_right_m`, `pair_id`).
`face_strip_mask` takes the two widths instead of one half-width.
* New `StripConfig` dataclass (half-width 2.5, out range -15..+5, gap 2 m,
  cap 300 m, below-foot 1 m), stored in `report.json` so each dataset records
  its rules.

---

## 6. Phase 3: same number of labelled points, same weight in the loss

**Today:** climbable median 10 labelled points, not-climbable 24-48 (many
hit the 48 cap). The loss counts points, so one not-climbable sample pushes
training 2-5 times harder than a climbable one. The counts also differ
because of the different height windows.

### 6.1 Pick the same number of points, spread over the height

In `block_codes`, after the strip is found:

1. Split the strip's height range into **4 equal bands**. The range runs
   from the foot to the **highest strip point actually in the block**, or the
   wall top if that is lower. Otherwise a tall, leaning wall whose top lies
   outside the block would get empty top bands, and fewer points just
   because it's tall.
2. From each band pick up to **12** points at random (seeded per sample), so
   **48** at most in total (the user's choice, 2026-11-15).
3. If a band has fewer than 12 points, take what is there; don't top it up
   from the other bands. (Otherwise the height spread would differ again.)
4. All other strip points become -1 (ignored).
5. A sample with **fewer than 12** labelled points in total is dropped, for
   every label, and logged with its source.

**Measured on V3 positives with the face strip** (2026-11-15, full-width
strips):

| | Horseshoe Climb | Horseshoe Nc | Red River Gorge | other areas |
|---|---|---|---|---|
| cliff points in the strip, median | 33 | 30 | 48 | 290 |
| picked with 12 per band, median | 30 | 29 | 32 | 48 |
| dropped (under 12) | 4% | 0% | 0% | 1% |
| at the full 48 | 8% | 3% | 12% | 72% |

So 48 / 12 costs almost nothing in drops, and gives each sample about 30
points instead of about 20 at 24 / 8, so scores are less noisy.

Things to watch:

* **The count now depends on the area.** The western areas' LiDAR is much
  denser, so 72% reach the cap, against about 10% at Horseshoe and Red River
  Gorge. That's not a label difference (both labels come from the same
  areas), and the per-sample loss weight (section 6.3) keeps big samples from
  counting more. The leak test's "number of labelled points" (group A)
  checks it stays label-neutral.
* **Split strips (section 5.6) are narrower**, so close pairs will be
  dropped more often at a minimum of 12. Report those drops separately
  (`strip_too_narrow`). If they remove more than a quarter of the close
  Horseshoe pairs, lower the minimum for split strips only to 8.
* **Tall walls put some climbable labels above the route.** Red River Gorge
  routes are a median 20 m long (28 m at the 90th percentile), but the strip
  reaches a median 35 m. So on tall walls the top band (about a quarter of
  the points) is often rock the route never reaches. That's mild label noise,
  not a leak, because the same rule applies to both labels.
  * Optional fix: bands at **fixed heights** above the foot (0-8, 8-16,
    16-24 and 24 m+) instead of equal shares. The route-start rock then
    gets three quarters of the points on every wall, for both labels.

Result: both labels get the same maximum count and the same spread over the
wall. Short walls get fewer points, but they do so for both labels in the
same way.

### 6.2 Keep every labelled point through resampling

`BlockDataset.__getitem__` draws 8,192 points at random from each block. With
more than 8,192 points some labelled points can be dropped; with fewer they
are copied a random number of times. Change it to **always keep each
labelled point exactly once**, then fill the rest at random from the other
points.

### 6.3 Each sample counts once in the loss

* `BlockDataset` returns a per-point weight `w`: 1 ÷ (number of labelled
  points in that sample) for labelled points, 0 for the rest.
* `Trainer.loss_fn` uses that weight when `loss_weighting: per_sample` is set
  in the config: cross-entropy per point × `w`, summed, divided by the number
  of samples in the batch. Pretraining keeps the current per-point loss.
* The validation loss used to pick the best checkpoint uses the same weights,
  so "best" means best per sample.
* The sample score (`SampleScoreHook`) is already a per-sample mean; no
  change needed.

---

## 7. Phase 4: narrow the task to wall points only

**Step-back samples leave this task.** They are labelled on ground and
talus, so they can never be built the same way as wall samples. They also
inflate the Horseshoe score (always rejected correctly). The question the
model now answers is: *given a point on a wall, is this wall climbable?*

* Default `--strategies handgathered_nc,band,far_cliff` for V4. Step-back
  code stays available but off.
* Augmented negatives (`aug_squash`, `aug_layback`) stay off. They change
  wall height and lean, both group B numbers.
* When the model is later used on a whole area, it is only asked about
  points the pretraining model (or the wall mask) calls cliff. Ground and
  trees are never scored as "climbable or not".

**Far-cliff samples are kept, but watched.** They come from different walls
than the routes, so their group B numbers (wall height, block height) may
differ a lot. The leak test reports a group B score for "climbable vs far
cliff" separately. If it is very high (above about 0.85), far cliffs are
teaching "short wall = not climbable" and little else. We then either lower
their weight or keep only far-cliff walls whose height matches the route
walls.

---

## 8. Phase 5: the Horseshoe test uses the same rules

* Horseshoe Climb and Nc samples are built by the same Phases 1-3 code as
  everything else, so this follows automatically. Add a unit test anyway:
  build one Climb and one Nc sample on the *same* synthetic wall and check
  the labelled points have the same count and the same heights.
* `crag/gt_eval.py` (the pretraining waypoint check) scores the 90th
  percentile within 5 m, a different rule. Switch it to the face strip and
  the same 48-point pick, so pretraining and fine-tuning numbers on Horseshoe
  measure the same thing.
* `scripts/plot_hcr_waypoints.py`: draw each sample's foot and strip (as a
  short line along the wall) so you can see by eye where every label lands.

### 8.1 A pair check for close climbable / not-climbable samples

The usual AUROC mixes easy cases (an Nc on very different rock) with hard
ones (an Nc a few metres from a route). Add to `crag/finetune.sample_metrics`:

* **`pair_acc`**: for every opposite-label pair from section 5.6 (feet within
  5 m), the share where the climbable sample scores higher than its
  not-climbable neighbour. 0.5 is guessing. Report the number of pairs, and a
  95% range from 1,000 resamples (pairs are few).
* **AUROC split two ways**: `auroc_isolated` (samples with no
  opposite-label sample within 5 m) and `auroc_near` (the rest).

Report all three for Horseshoe in every evaluation (`s4_eval_finetune.py`,
`s4_compare.py`). They show whether the model can make the fine call, or
only the easy ones.

### 8.2 Part of Horseshoe goes into training (by wall section)

Today every Horseshoe sample is in the test set, so the model never trains
on hand-checked "not climbable" rock, or on close pairs.

* Split Horseshoe by **wall section**, using the waypoints' `area_name`
  column: West Side, North Forty, East Side. Code-made samples at Horseshoe
  (band, far cliff) take the section of the nearest waypoint.
* Two folds, each a full fine-tuning run:

  | fold | trains on | tests on |
  |---|---|---|
  | `hcr-a` | East Side | West Side + North Forty |
  | `hcr-b` | West Side + North Forty | East Side |

  Report the Horseshoe test numbers of both folds, and their mean.
* Keep a third setting, `hcr-none` (all of Horseshoe in test, as today), as
  the reference, since it doesn't depend on the section split.
* A section in training never also appears in validation. Model selection
  (best checkpoint) still uses the normal validation split, which holds no
  Horseshoe samples.
* Pretraining is unchanged: Horseshoe stays a pretraining hold-out in every
  fold.
* Code: `scripts/s3_build_finetune_set.py --hcr-fold {none,hcr-a,hcr-b}`;
  `assign_splits` gets the section rule. The dataset name records the fold
  (for example `v4-hcr-a`); the block arrays are shared, only `index.csv`
  differs.

---

## 9. Phase 6: build V4, check it, train, compare

### 9.1 Build and check (no training yet)

```bash
conda activate crag-pointnext
python scripts/s3_build_finetune_set.py --name v4 --strategies handgathered_nc,band,far_cliff --no-download   # hcr-none
python scripts/s3_build_finetune_set.py --name v4-hcr-a --hcr-fold hcr-a --strategies handgathered_nc,band,far_cliff --no-download
python scripts/s3_build_finetune_set.py --name v4-hcr-b --hcr-fold hcr-b --strategies handgathered_nc,band,far_cliff --no-download
python scripts/s3_leak_test.py --name v4
python scripts/plot_hcr_waypoints.py        # look at the strips on the map
```

Stop here if group A is outside 0.45-0.55. Use the single-number scores to
find which rule still differs, fix it, rebuild.

Also check in `report.json`:

* samples kept / dropped per source and the reason (fewer than 12 points,
  duplicate, conflict, no wall);
* labelled points per sample, per source (should look alike);
* labelled heights above the foot, per source (should look alike);
* wall top per source (group B; may differ, but write it down).

### 9.2 Train (smaller sweep first)

Same backbones as V3 (`runs/pretrain/pointnext-s`, `runs/pretrain/pointnet2`).
Start with three strategies per backbone: `decoder_head`,
`full_discriminative`, `scratch`. Add `loss_weighting: per_sample` and
`dataset.name: v4` to `cfgs/finetune/default.yaml`.

```bash
STRATEGIES="decoder_head full_discriminative scratch" bash scripts/s4_sweep.sh
BACKBONE=runs/pretrain/pointnet2 STRATEGIES="decoder_head full_discriminative scratch" bash scripts/s4_sweep.sh
```

Fix the sweep bug at the same time: the `s3dis_init` strategy ignores the
backbone setting and always builds PointNeXt-S, so the
"pointnet2-s3dis_init" run is really PointNeXt-S. Either pass the right
architecture or skip `s3dis_init` for PointNet++.

### 9.3 Compare and write up

`docs/results/v4/README.md`, with for every run:

* validation AUROC, Red River Gorge test AUROC, Horseshoe Climb vs Nc AUROC,
  each with 95% ranges;
* the same three for the group A and group B baselines;
* V3 numbers next to them, marked "not directly comparable" (different
  samples and labels).

Move V3 runs to `runs/archive-ft-v3/` before training, as was done for
earlier versions.

---

## 10. Tests to add (`tests/test_labels_negatives_crossref.py`)

| test | checks |
|---|---|
| `test_wall_foot_from_any_height` | on a synthetic 20 m wall, starting from the top, the middle and 3 m in front all give the same foot (within 0.5 m) |
| `test_face_strip_follows_leaning_wall` | on a 70 degree wall the strip reaches the top; the old upright tube would not |
| `test_wall_top_stops_at_ledge` | two tiers with a 4 m ledge between: the top is the lower tier's top |
| `test_same_labels_for_both_classes` | a Climb and an Nc sample on the same wall give the same labelled-point count and height spread |
| `test_point_pick_is_balanced` | at most 12 points per height band, at most 48 total, under 12 -> dropped; bands span only the strip points present in the block |
| `test_resampling_keeps_labelled_points` | every labelled point appears exactly once after resampling to 8,192 |
| `test_per_sample_loss_weight` | a 12-point and a 48-point sample contribute equally to the loss |
| `test_exclusion_after_centring` | a band point that moves to within 7.5 m of a route foot is dropped |
| `test_close_pair_strips_split` | a Climb and an Nc 3 m apart on one wall: no point labelled twice, a 0.5 m gap between the labels, both strips trimmed the same way |
| `test_close_pair_too_narrow` | 1.5 m apart: both dropped as `strip_too_narrow` |
| `test_pair_under_1m_unconfirmed_dropped` | 0.6 m apart and not confirmed by the judges: both dropped; the same pair with both positions confirmed: kept, with a warning |
| `test_same_label_not_split` | two Climb samples 2.5 m apart keep full strips |
| `test_pair_acc` | made-up scores: `pair_acc` and the isolated / near AUROC match hand-computed values |
| `test_hcr_fold_split` | `hcr-a` puts only East Side in training, never in validation; code-made Horseshoe samples follow their nearest waypoint's section |

Run the whole suite (`pytest`) after each phase.

---

## 11. Files that change

| file | change |
|---|---|
| `crag/groundtruth.py` | new `wall_foot` |
| `crag/negatives.py` | band and far-cliff candidates go through `wall_foot`; exclusion and duplicate check after centring; step-back off by default |
| `crag/finetune_data.py` | `StripConfig`, `face_strip_mask` (two widths), `wall_top`, `split_strip_widths`, new step d / e in `block_codes`, balanced point pick, drop under the minimum points; Horseshoe section folds in `assign_splits` |
| `crag/finetune.py` | `pair_acc`, `auroc_isolated`, `auroc_near` in `sample_metrics` |
| `crag/datasets.py` | keep labelled points through resampling; return per-point weight `w` |
| `crag/train.py` | `loss_weighting: per_sample` in training and validation loss |
| `crag/gt_eval.py` | score with the face strip and point pick |
| `scripts/s3_build_finetune_set.py` | new defaults, new report fields, write foot and frame per sample |
| `scripts/s3_leak_test.py` | new |
| `scripts/plot_hcr_waypoints.py` | draw foot and strip |
| `scripts/s4_sweep.sh` | fix the `s3dis_init` architecture bug |
| `cfgs/finetune/default.yaml` | `dataset.name: v4`, `loss_weighting: per_sample` |
| `docs/pipeline.md`, `docs/results/v4/README.md` | document the new rules and results |

---

## 12. Risks and what we do about them

| risk | what we do |
|---|---|
| Fewer samples: drops for under the minimum points, duplicates, pairs under 1 m and strips too narrow | log every drop by source; if more than 15% of climbable samples are lost, lower the minimum or widen the strip to 3 m and rebuild |
| Split strips give close-pair samples fewer labelled points, which could separate them as a group | `strip_width_m` and `has_opposite_neighbour` in the group A leak check (section 5.6) |
| A Horseshoe fold has too few test points to compare runs | report both folds and their mean, with 95% ranges; keep `hcr-none` as the stable reference |
| The 2 m gap rule for the wall top stops too early. Measured 2026-11-15 on V3 points: it gives Red River Gorge a median wall height of 8 m, against 16 m from the plain column measure, because tall faces often have gaps of more than 2 m with no cliff points (roofs, sparse LiDAR on vertical rock) | before building V4, compare the gap rule against a wall height measured from the bare-earth grid (problem 4 plan), on Red River Gorge and Horseshoe; try a 4-5 m gap; pick the rule whose heights match the known route lengths best |
| `wall_foot` walks off the wrong side on a thin fin or a boulder | the unit tests; the strip plots at Horseshoe and two Red River Gorge crags; samples whose foot moves more than 10 m are logged for a look |
| Climbable labels get noisier (the strip up the full wall can include rock off the route line) | accepted: a route starting there almost always climbs that strip. The 4-band pick keeps the upper wall from outweighing the start |
| Band negatives are still only 42-83% right | not fixed by this plan. Once the shortcut is gone, the quality of the negatives becomes the main limit, which raises the value of hand-cleaned Horseshoe Nc points (problem 4) |
| Group B is high for far cliff (short walls) | the separate "climbable vs far cliff" check in section 7 |
| The network does not beat group B | a real result: the network is not seeing more than wall height and steepness yet. Next step: problems 2 and 3 |

---

## 13. Decisions for you (defaults used unless you say otherwise)

| decision | default | other options |
|---|---|---|
| strip half-width | 2.5 m | 2 m (tighter to the route) or 3 m (more points) |
| wall top: gap and cap | 2 m gap, 300 m cap | 3 m gap for broken walls |
| points per sample | 4 bands × 12 = 48 | 4 × 4 = 16 (fewer drops on short walls) |
| minimum points to keep a sample | 12 | 6 |
| step-back samples | removed | keep, but only with a separate "not a wall" class |
| far-cliff samples | kept, watched by the section 7 check | removed until the check passes |
| box size (from the earlier question 1) | stays 20 m for V4 | after V4: try 10 m and 15 m, keep whichever scores best on Horseshoe Climb vs Nc |

## 14. Order of work

1. Phase 0: leak test, and run it on V3 (reference numbers).
2. Phase 1: `wall_foot`, plus its tests.
3. Phase 2: face strip and wall top, plus tests.
4. Phase 3: balanced point pick, resampling fix, per-sample loss, plus tests.
5. Phase 4 and 5: strategies, `gt_eval`, plots.
6. **Plan 4 must be done first:** the wall height check, the new snap, and
   the Horseshoe points snapped and judged.
7. Close pairs: split strips (section 5.6), the pair check (8.1), the
   Horseshoe section folds (8.2), plus tests.
8. Build V4 (all three folds), run the leak test, look at the strip plots.
   **Stop and review together before training.**
9. Phase 6: train, compare, write `docs/results/v4/README.md`.
