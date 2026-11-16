# V3 fine-tuning labels: what is wrong (2026-11-15)

Follow-up to `docs/fable-investigation-project-issues.md`, section 1. This
explains exactly how a fine-tuning sample is cut and labelled today
(`crag/crop.py`, `crag/finetune_data.py`, `crag/finetune.py`), and why the
labelling lets the model score well without learning anything about the rock.

## 1. What a block is

A block is 3D: a **20 m x 20 m square seen from above, with no height
limit**. Every LiDAR point whose x and y are within 10 m of the sample's spot
is kept, from the valley floor to the top of the wall and the trees. A 40 m
cliff gives a block about 40-50 m tall.

Then:

1. the points are thinned to one per 0.25 m cube;
2. they are turned into local coordinates: x and y from the sample spot, z
   from the lowest point in the block;
3. at training time the block is resampled to 8,192 points.

The model receives **every point in the block as input**: ground, trees, the
whole wall.

## 2. What the cylinder is

A point is inside the sample's cylinder when its distance from the sample
spot, **measured flat on the map**, is 2.5 m or less. Height is not checked at
this step, so the cylinder runs from the bottom of the block to the top.

## 3. How each point gets its label

A point is **steep** when it was marked as cliff in pretraining labelling: a
last return (the laser reached a solid surface), not vegetation, on a surface
of 60 degrees or steeper, within 2 m of the bare-earth wall mask.

Every point in the block goes through these steps in order; a later step
overwrites an earlier one (`block_codes`).

| step | which points | code |
|---|---|---|
| a | every point | **2** = background |
| b | steep points within 7.5 m (flat distance) of *any* known route location | **-1** = unknown rock |
| c | steep points inside a *neighbouring* sample's cylinder | **3** (neighbour climbable) or **4** (neighbour not climbable) |
| d | points inside *this* sample's cylinder: see below | **1**, **0** or **-1** |
| e | if this sample has more than 48 points coded 0 or 1, random extras go back to **-1** | |

Step d depends on the kind of sample:

* **Climbable** (OpenStreetMap route or Horseshoe Climb point):
  non-steep points -> -1; steep points **from 1 m below the wall foot up to
  15 m above it** (or the route length, if known) -> **1**; steep points above
  that -> left as they were (usually -1).
* **Not climbable, on the cliff** (band, far cliff, Horseshoe Nc):
  non-steep points -> -1; steep points **at any height** -> **0**.
* **Not climbable, step-back** (talus or plateau): **every point** in the
  cylinder, steep or not -> **0**.

During training everything except 1 and 0 is set to "ignore". The model
predicts all ~8,000 points, but **the loss only counts the points coded 1 or
0**. A sample's score at evaluation is the average predicted chance of
"climbable" over those same points.

Where the centre of the block is put also differs:

* climbable: the GPS point is snapped to the **foot of the wall**
  (`snap_one`);
* band: any steep point on a 2 m grid, so often **mid-wall or near the top**;
* far cliff: the densest cliff spot in a pretraining block;
* step-back: 12-20 m out from or behind the wall;
* Horseshoe Nc: snapped to the wall foot, the same as Climb.

## 4. Why this gives the model a shortcut

One 30 m wall, with a climbable sample at its foot and a band sample 15 m
along the same wall:

```
height above wall foot
 30 m |  ignored     |  0 0 0      <- band: labelled up to the top
 20 m |  ignored     |  0 0 0
 15 m |--------------|-------
 10 m |  1 1 1       |  0 0 0
  0 m |  1 1 1       |  0 0
        climbable       band
```

* A point more than 15 m up a wall can **only ever be labelled 0**.
* Band centres are often mid-wall; climbable centres are always at the foot.
* Labelled points per sample: median 10 for climbable, 24-48 for not
  climbable (many hit the 48 cap). The loss counts points, so one
  not-climbable sample pushes training 2-5 times harder than one climbable
  sample, even though the sampler draws both labels equally often.

Each point's height is part of the input, so the easiest rule to learn is
**"high in the block = not climbable, low = climbable"**. That is a fact
about our labelling, not about the rock. Every trained run shows it: the
score falls as the labelled points get higher in the block (rank correlation
-0.54 to -0.66 on validation).

At Horseshoe Canyon, Climb and Nc points are both snapped to a wall foot, so
the Nc samples look "low" too and the shortcut stops working. That is why
validation is about 0.78 and Horseshoe Climb vs Nc is about 0.5. One
difference remains there: Nc is scored over the whole wall height and Climb
stops at 15 m, so the Horseshoe test is not clean yet either.

## 5. The two baseline tests

Both are logistic regressions (the simplest classifier), trained on the
training samples and given a few summary numbers per sample, with no shape
information.

**Six numbers about the labelled points:**

1. median height of the labelled points above the block's lowest point
2. lowest labelled point
3. highest labelled point
4. height span of the labelled points
5. median labelled height divided by the block's height
6. number of labelled points

Scores: **0.905** validation, 0.91 Red River Gorge test, 0.74 Horseshoe Climb
vs Nc. The best network gets 0.785 on validation. The labelling rules alone
give away most of the answer.

**Five numbers that use no labels**, only the points within 2.5 m of the
block centre:

1. the block's height
2. number of points within 2.5 m of the centre
3. lowest of those points divided by the block's height
4. median of those points divided by the block's height
5. share of the block's points below that median

Scores: 0.73 validation, **0.55 Horseshoe Climb vs Nc**. The networks score
about the same (0.72-0.785 and 0.46-0.59), which suggests they mostly pick up
where the sample sits, not the rock.

## 6. The fix, and how to check it

Make steps d and e identical for climbable and not-climbable samples: same
centre placement (wall foot), same height window, same number of labelled
points. Then rerun both baselines. They should drop to about 0.5. If they do
not, something is still leaking.
