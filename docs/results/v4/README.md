# V4 results: one labelling rule, filled neighbourhoods, surface inputs, judged Horseshoe points (2026-11-15)

What was wrong with V3 and why: `docs/fable-investigation-project-issues.md` (and the four
detailed docs it links). The plans carried out here:
`research-prompts/fable-investigation-project-issues-resolution-plan{1,2,3,4}.md`.
The runbook is in `docs/pipeline.md` ("V4 changes").

**Clean start.** Every old checkpoint under `runs/` was deleted before training; the V3 run
folders (metrics and evaluations only) are in `runs/archive-v3/`. Nothing in V4 resumes or
starts from an old run. The downloaded S3DIS weights in `pretrained/` were kept.

Files in this folder:

| file | contents |
|---|---|
| `leak_test_v3.json`, `leak_test_v4.json` | the leak test (below) on V3 and V4 |
| `finetune_set_report.json` | fine-tuning set V4: samples, drops by reason, labelled points and heights per source |
| `dropped_samples.csv` | every sample dropped while building V4, with the reason |
| `confirmation_runs.csv` | the scale and input-channel test runs (mean of 2 seeds) |
| `horseshoe_judges_summary.json` | the one-time check of the Horseshoe points |

AUROC: 0.5 = guessing, 1 = perfect. With about 70 Horseshoe hand points, differences under
about 0.05 are noise.

## 1. Route points: wall height and snap (plan 4)

* Every route GPS point now snaps only to a wall **at least 6 m tall**, measured on the
  bare-earth ground grid. The old measure (spread of cliff points in an upright column) did not
  agree with real route lengths (rank correlation -0.04); the ground grid does (0.46, and 79%
  of walls are at least 0.8 x the route length, against 34%).
* The snap moves a point **straight toward its wall**, so neighbours keep their spacing. At
  Horseshoe, pairs of points squashed together fell from 155 to 52; Horseshoe Nc points landing
  on a Climb point fell from 14 to 9.
* Cost at 6 m: 93% of Horseshoe Climb, 95% of Nc and 98-99% of other routes pass.

### Horseshoe points: snapped once, then judged

The originals stay read-only in `data/handgathered/original/`. The automatic snap was checked
by Claude subagent judges, two per wall section plus a tie-break judge (9 in total); there was
no hand review (your decision).

| check | result |
|---|---|
| calibration: judges keep the automatic snap on easy points (needed: 90%) | 100% and 100% (31 points) |
| the two judges agree | 166 of 186 (89%) |
| disagreements settled by the tie-break judge | 20 of 20 (none unresolved) |
| moved away from the automatic snap | 15 points, median 3.1 m |
| dropped (no fitting wall, or under 6 m) | 13 (8 Climb, 5 Nc), e.g. Mijo, Mexican Mija, Season of the Storm, Spam |
| low confidence | 70 of 186; evaluations also report Horseshoe without them |

Points the judges found doubtful: West Side routes #95-#100 (they may be on a lower rock step
the 6 m rule excludes), North Forty #34-#37 (raw points on top of the wall), and the East Side
boulder field #42-#61. Maps: `docs/figures/hcr-waypoints/judged_*.png`.

## 2. One labelling rule for every sample (plan 1)

Every sample, climbable or not, is centred on the foot of its wall (the outer edge of the
ground grid's wall strip, the same whichever side a sample starts from). Its labels sit on a
strip of the face measured along / out from / up the wall, up to the ground-grid wall height,
and every sample gets the same pick: 4 height bands x up to 12 points. Close climbable and
not-climbable samples split their strips so no rock gets both labels. Step-back samples are
off (they are not on a wall).

**Wall top.** The plan's 2 m gap rule stopped too early on tall walls. Against 279 real
route lengths: ground-grid height rank correlation 0.44 (median wall/route 1.10); gap rules
2-8 m 0.24-0.30. The ground grid is used.

### The leak test

Simple classifiers on a few numbers per sample, with no shape information. Group A = how the
sample was built (should say nothing about the rock); group B = plain wall geometry (allowed;
the baseline a network must beat).

| | V3 group A | V4 group A | V4 group B (baseline) |
|---|---|---|---|
| validation | **0.89** | 0.66 | 0.76 |
| Red River Gorge test | 0.91 | 0.68 | 0.76 |
| Horseshoe Climb vs Nc | 0.73 | 0.60 (range 0.46-0.74) | 0.69 |

**V4 group A is still above the plan's 0.55 bar outside Horseshoe, but only through wall
height.** Climbable walls are taller (median 29.5 m against 20.7 m for band negatives), so a
taller strip follows. On validation, wall height alone scores 0.708 and adding every group A
number *lowers* it to 0.68; within walls of similar height the group A numbers sit at
0.38-0.60. One small remainder: climbable faces lean back about 1 m more (labelled points
further behind the foot), which looks like real face shape. An earlier V4 build did have a
rule difference (route feet placed further out than negative feet, because routes arrive from
outside the wall); that was found by this test and fixed.

### What V4 holds

2,522 samples: 1,251 climbable (1,201 OpenStreetMap routes + 50 Horseshoe) and 1,271 not
climbable (1,049 band, 202 far cliff, 20 Horseshoe Nc); details in `finetune_set_report.json`. Horseshoe keeps 50 Climb and 20 Nc of the 118 / 55 judged points.
The losses follow plan 1's rules: same-label points under 2 m apart (30 + 8), opposite-label
points too close to split their strips (17 + 15), and sparse faces with too few labelled points
(21 + 12, already using the plan's lower minimum of 8 for split strips).

Horseshoe folds (each trains on one part, tests on the rest): `v4-hcr-a` trains on East Side
(25 Climb, 5 Nc) and tests on North Forty + West Side (25 / 19); `v4-hcr-b` the reverse.

## 3. Neighbourhood scale (plan 2)

Real (distinct) points per neighbourhood ball, median, fine-tuning blocks:

| setting | stage 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| V3: 0.625 m first radius, 8,192 points | 10 | 8 | 8 | 7 |
| 2 m, 4,096 points | 32 | 32 | 32 | 30 |
| **1.5 m, 4,096 points (chosen)** | 27 | 26 | 24 | 18 |

Scale test (training from scratch, PointNeXt-S, mean of 2 seeds):

| setting | validation | Red River Gorge | Horseshoe Climb vs Nc |
|---|---|---|---|
| V3 setting | 0.702 | 0.689 | 0.631 |
| 2 m, 8,192 | 0.748 | 0.699 | 0.655 |
| 2 m, 4,096 (proposed) | 0.766 | 0.669 | 0.612 |
| **1.5 m, 4,096** | **0.757** | **0.739** | **0.698** |

Every wider setting beats V3 on validation. 2 m missed the plan's "Horseshoe not lower"
condition; 1.5 m passed all three conditions, so 1.5 m (`coord_scale 0.067`) is used.
Training blocks are not thinned (the plan's optional step 6c was not run).

## 4. Input channels (plan 3)

Same test design, at 1.5 m:

| inputs | validation | Red River Gorge | Horseshoe |
|---|---|---|---|
| a: x, y, z + height (V3) | 0.734 | 0.671 | 0.720 |
| b: height above ground | 0.752 | 0.715 | 0.643 |
| c: + steepness, flatness | 0.766 | 0.699 | 0.648 |
| d: + return number flags | 0.733 | 0.697 | 0.633 |
| **e: c + intensity rank** | **0.778** | 0.678 | **0.730** |
| f: c + distance to wall | 0.762 | 0.734 | 0.575 |
| baseline: wall geometry | 0.760 | 0.764 | 0.690 |

Kept: height above ground, steepness, flatness, intensity rank (the only set above the
baseline on validation and Horseshoe). Dropped: return flags, wall distance. Pretraining uses
height above ground + flatness (the other channels define its cliff label).

Height above ground is measured from the lowest ground within 10 m, so on steep slopes next to
cliffs even ground points read several metres (Red River Gorge ground median 9 m on a
cliff-heavy tile). That is the plan's definition; it is noted here, not changed.

## 5. Full training (2026-11-15/16)

`bash scripts/v4_full.sh`: everything from a clean start, at 1.5 m / 4,096 points.

### Pretraining (`pretrain_eval_<model>.json`)

| model | best epoch | cliff IoU validation | cliff IoU test | V3 (validation / test) |
|---|---|---|---|---|
| PointNeXt-S | 23 | 0.497 | 0.558 | 0.47-0.48 / 0.56-0.57 |
| PointNet++ | 70 | 0.495 | 0.556 | |

Cliff detection is as good as V3 or slightly better, with height above ground + flatness
instead of x, y, z. On the Horseshoe points, P(cliff) does not separate Climb from Nc (AUROC
0.55), as expected: pretraining knows nothing about climbability.

### Fine-tuning (`finetune_comparison.csv`)

Sample AUROC. Horseshoe = Climb vs Nc hand points (50 / 20); "high + medium" leaves out the
points the judges were unsure about.

| run | validation | Red River Gorge test | Horseshoe | Horseshoe high + medium |
|---|---|---|---|---|
| PointNet++ scratch | **0.752** | **0.724** | **0.754** | 0.761 |
| PointNet++ full_discriminative | 0.721 | 0.669 | 0.720 | 0.765 |
| PointNet++ gradual_unfreeze | 0.729 | 0.679 | 0.676 | 0.714 |
| PointNet++ decoder_head | 0.704 | 0.710 | 0.683 | 0.693 |
| PointNeXt-S scratch | 0.751 | 0.688 | 0.670 | 0.706 |
| PointNeXt-S decoder_head | 0.723 | 0.710 | 0.616 | 0.657 |
| PointNeXt-S full_discriminative | 0.708 | 0.705 | 0.603 | 0.626 |
| PointNeXt-S gradual_unfreeze | 0.675 | 0.717 | 0.683 | 0.722 |
| **baseline: wall geometry (group B)** | 0.760 | 0.764 | 0.690 | |

Horseshoe folds (best strategy, scratch, trained with one part of Horseshoe):

| fold (tests on) | PointNeXt-S | PointNet++ | wall-geometry baseline on the same test |
|---|---|---|---|
| `hcr-a` (North Forty + West Side, 25 Climb / 15 Nc scored) | 0.555 | 0.611 | 0.757 |
| `hcr-b` (East Side, 25 Climb / 5 Nc) | 0.448 | 0.456 | 0.528 (range 0.23-0.90) |

### What it means

1. **The V3 shortcut is gone, and with it the inflated validation score.** V3's best
   validation AUROC was 0.785, but a six-number summary of how the samples were cut scored
   0.905. In V4 that summary carries nothing beyond wall height. V3 and V4 numbers are not
   directly comparable (different samples and labels).
2. **The networks do not beat plain wall geometry.** On validation and Red River Gorge every
   run is at or below the group B baseline (0.76). On Horseshoe the best run (PointNet++
   from scratch, 0.754) is above it (0.69), but with 70 points the 95% range is about ±0.12,
   and in the folds both models fall below the baseline. Per plan 1 this is a real finding:
   the network sees little beyond wall height and steepness yet.
3. **Pretraining does not help.** No pretrained run beats `scratch` on Horseshoe by 0.05 or
   more (plan 3, 10c: keep fine-tuning from scratch; "change the pretraining task" is future
   work). The pretraining task (cliff / not cliff) is learned well, but it is not what
   separates climbable rock.
4. **The close-pair check is too small to read.** Only 4 Climb / Nc pairs within 5 m
   survived V4's rules at Horseshoe (most close pairs were dropped as too close to split).
5. **Runs peak early** (best checkpoint at epoch 3-17): the training set is small for these
   models.

### Where to go next

* **More and better "not climbable" examples.** The band negatives are right only 42-68% of
  the time at 50-75% documentation coverage (`finetune_set_report.json`,
  `band_validation_horseshoe`), and Horseshoe's 20 Nc points are the only checked ones. This
  is now the main limit (plan 1 risks table; plan 3, 10b).
* **Horseshoe is too small to rank runs.** Differences under about 0.1 there are noise. More
  hand-labelled sites would help more than further model changes.
* **A pretraining task closer to climbing** (plan 3 doc, section 7, option 3).
