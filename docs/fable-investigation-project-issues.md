# Investigation: what is going wrong with model training (2026-11-15)

I read the whole pipeline (`crag/`, `scripts/`, `cfgs/`) and ran checks on the
fine-tuning set V3 and the saved predictions in `runs/finetune/`. Nothing in
the code was changed.

## Short answer

Your two suspicions are both real, but neither is the biggest problem.

1. **Biggest problem (new finding): the climbable and not-climbable examples
   are cut and labelled in different ways, and the model learns that
   difference instead of anything about the rock.** A six-number summary with
   no shape information beats every network we trained.
2. **The model sees almost no local shape.** The neighbourhood size was set
   from the wrong number, so each point has about 3 real neighbours where the
   model expects 32.
3. **The colour (RGB) input mismatch is real** but small in effect. The
   bigger input problem is that the 4 input channels carry only 3 pieces of
   information and no surface information.
4. **The Horseshoe Canyon points do need cleaning.** A third of the
   "not climbable" points end up on the same piece of rock as a "climbable"
   point.

Details below, most important first.

---

## 1. The model is learning how samples were cut, not the rock

**What the code does.** Each fine-tuning sample is a 20 m block with a 2.5 m
circle in the middle. Only points inside that circle are labelled and scored.
But the rules differ by label:

| | climbable | not climbable ("band", "far cliff") | not climbable ("step-back") |
|---|---|---|---|
| Where the block centre is put | foot of the wall, on the side nearest the GPS point (`snap_one`) | any cliff point on a 2 m grid, often mid-wall or the top (`band_candidates`) | 12-20 m out on the talus or plateau |
| Which points get the label | steep points from the wall base up to 15 m only (`block_codes`, the `zok` rule) | every steep point in the column, any height | every point in the column |
| Typical labelled points per block | 10 | 24-48 | 48 |

**What I measured.**

* Labelled points sit at 28% of the block's height for climbable samples and
  about 50% for not-climbable ones.
* A plain logistic regression given only six numbers about *where the
  labelled points are* (their heights and how many there are; no shape at
  all) scores **0.905** on validation. Our best network scores 0.785.
* A second one given only five numbers about the block centre (using no label
  information) scores 0.73 on validation, and 0.55 on Horseshoe Canyon
  climbable versus not-climbable. The networks score 0.72-0.785 and
  0.46-0.59. That is the same pattern.
* In every run, the network's score falls as the labelled points get higher
  in the block (rank correlation -0.54 to -0.66).

**What it means.** The model has learned "low in the block = climbable, high
in the block = not climbable". At Horseshoe Canyon both the Climb and Nc
points go through the same snapping, so that trick stops working and the
score drops to guessing. This explains the gap between validation (0.78) and
the Horseshoe hand points (about 0.5).

**The test itself is also uneven.** Horseshoe Climb points are scored on
"base to base + 15 m" and Nc points on the whole column. The six-number
summary scores 0.74 on Climb versus Nc from that difference alone, so the
Horseshoe number is not a clean measurement yet.

**Fix.** Use one rule for every sample, whatever its label: same way of
placing the centre (wall foot), same height range, same cap on labelled
points. Then rerun the simple baselines. They should fall to about 0.5; if
they do not, there is still a leak.

## 2. The model can barely see local shape

**What the code does.** `coord_scale: 0.16` was chosen so that our 0.25 m
voxels match the 4 cm voxels of the indoor dataset (S3DIS) the backbone came
from. That assumes our points are 0.25 m apart.

**What I measured.**

* Real spacing between neighbouring points is 0.31 m (0.45 m at Horseshoe
  Canyon). On vertical rock it is worse, because the aircraft looks down.
* The first layer looks 0.625 m around each point and expects up to 32
  neighbours. It finds a median of **3** (and **1** at Horseshoe Canyon).
* A follow-up simulation of all four grouping layers (radius 0.625, 1.25,
  2.5 and 5 m) found a median of only **2-5 real points out of 32 slots in
  every layer**, not just the first.
* Blocks hold about 3,100 points (2,100 at Horseshoe), but the loader always
  draws 8,192, so every point is copied 2 to 4 times. The copies add no
  information and waste memory. (They do little harm to the neighbourhoods
  themselves, because each layer keeps the largest value over its
  neighbours, and a copy never changes that.)

**What it means.** No layer of the model sees a well-filled neighbourhood.
It works mostly from each point's own position plus 1-4 neighbours, which is
close to looking at points one at a time.

**Fix.** Set the scale from the measured spacing, not the voxel size: a
first radius of about 2 m (`coord_scale` 0.05) fills the first layer with
about 15-18 real points and every later layer with 25-32. Set `num_points`
near the real block size (4,096) so points are not copied. Full explanation:
`docs/fable-investigation-neighbourhood-scale.md`.

## 3. The input channels (your RGB concern)

**What is true.** The downloaded backbones were trained with 4 feature
channels: red, green, blue, height. We put x, y, z, height into those same
slots (`feature_keys: pos,heights`). So the first layer's starting weights
were tuned for colours and receive positions. This is confirmed in
`crag/datasets.py` and the S3DIS configs.

**How much it matters.** Less than expected:

* Only the first small layer (128 weights in PointNeXt-S) is affected. The
  position of each point also reaches the model through a separate path
  (`pos`) that is used correctly.
* That layer is retrained during our pretraining.
* Starting from scratch, from S3DIS, or from our pretraining all score about
  the same. So the starting weights are not what holds the score down.

**The bigger input problems.**

* **Channel 4 is a copy of channel 3.** The transform `PointCloudXYZAlign`
  already sets the lowest point to z = 0, so "height" equals z. The model
  gets 3 pieces of information in 4 channels.
* **x and y are positions inside the block**, which mean nothing about the
  rock. Their main use to the model is to tell where the block centre is,
  which feeds the shortcut in section 1.
* **Nothing describes the surface.** We already compute surface direction
  (normals), steepness, distance to the wall, laser intensity and return
  number per point, and none of it is given to the model.

**Fix.** Replace the feature channels with things that describe the rock, for
example: height above the local ground, surface normal, intensity, return
number. The first layer then no longer matches S3DIS and starts fresh, which
costs nothing since S3DIS weights are not helping.

## 4. Horseshoe Canyon points

* 43% of the hand points are moved more than 5 m by the automatic snap, and
  14% more than 10 m (Red River Gorge: median 1.3 m). Large moves are where
  the snap can pick the wrong formation.
* The snap takes the nearest wall at least 4 m tall. One in ten Horseshoe
  points lands on a wall under 4.8 m, which is probably a boulder or ledge
  and not the route's wall.
* **18 of 58 Nc points end up within 2.5 m of a Climb point** (27 within
  5 m). The labelled circle has a 2.5 m radius, so those pairs cover the same
  rock with opposite labels. No model can separate them. 13 of these were
  already that close in the raw GPS data, so this is partly how the points
  were recorded, not only the snap.
* After cleaning there are only 181 points, in clusters. Differences smaller
  than about 0.05 between runs are noise.

So yes: placing these by hand is worth doing. It will not fix sections 1-2,
but without it the Horseshoe score cannot be trusted.

### How to move the points by hand

Recommended: **QGIS** (free desktop map program).

1. I add a small script that writes a hillshade image and the wall mask for
   Horseshoe Canyon as GeoTIFF files, and the waypoints (raw position, snapped
   position, label, name) as one GeoPackage file.
2. You open them in QGIS, turn on editing for the point layer, and drag each
   point onto the wall with the "Move Feature" tool. Delete points you do not
   trust. Save.
3. I add a `manual_x, manual_y` override to the loader: a hand-placed point
   is used as is and skips the automatic snap.

A lighter option is geojson.io in a browser (drag points over satellite
photos), but tree cover hides the walls there, so the LiDAR hillshade in QGIS
is much better. While placing points, keep Climb and Nc points at least 5 m
apart, or drop one of the pair.

## 5. Smaller setup problems found

* **Two runs are mislabelled.** `runs/finetune/pointnet2-s3dis_init` is
  actually a PointNeXt-S run, identical to `pointnext-s-s3dis_init`. The sweep
  script passes no architecture for that strategy, so it falls back to
  PointNeXt-S. There is no PointNet++ result from S3DIS weights.
* **66 climbable samples have no labelled points** (44 in training, 18 in
  test, 4 of them Horseshoe). They add nothing in training and get no score in
  testing.
* **Climbable samples have very few labelled points** (median 10 of about
  3,100 in the block), so each one's score is an average over a handful of
  points and is noisy.
* **Pretraining teaches something the input already contains.** The cliff
  label is computed from steepness and distance to a wall, both worked out
  from the same x, y, z the model receives. This is why pretraining does not
  help.
* **No "not climbable" hand points in training**, and 81% of climbable
  samples come from Red River Gorge (both already listed in
  `docs/results/v3/README.md`).
* Stale comments in `crag/models.py` still say 5 classes (now 4). Harmless.

## 6. What I checked and found correct

* Training and evaluation build blocks the same way (same crop, scale,
  transforms, feature channels). No train/inference mismatch.
* Horseshoe Canyon never enters pretraining; test regions are removed from
  pretraining data.
* Train, validation and test groups are split by location (no crag is in two
  splits).
* Label codes are mapped correctly, ignored points do not enter the loss, and
  frozen parts stay frozen including their normalisation statistics.
* Checkpoints load fully; only the head is new, as intended.

## 7. Suggested order of work

1. Make sample cutting and labelling identical for both labels (section 1),
   then rerun the simple baselines as a leak test.
2. Fix the scale and point count (section 2).
3. Change the input channels to surface information (section 3).
4. Hand-place the Horseshoe points in QGIS (section 4), and move part of them
   into training so the model sees real "not climbable" rock.
5. Retrain and compare against the simple baselines, not only against 0.5.
