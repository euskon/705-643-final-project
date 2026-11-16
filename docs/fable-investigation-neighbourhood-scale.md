# Problem 2: the model barely sees local shape (2026-11-15)

Follow-up to `docs/fable-investigation-project-issues.md`, section 2.

## How the model looks at points

PointNeXt-S and PointNet++ build their understanding in stages. Each stage:

1. picks a smaller set of **centre points**, spread out as evenly as
   possible (farthest point sampling);
2. for each centre, collects up to **32 neighbours** within a fixed
   **radius**: a ball query (if fewer are found, the slots are padded with
   repeats);
3. combines what those neighbours say into one description of the centre,
   by keeping the largest value of each feature over the neighbours.

For our settings (8,192 points in, a radius that doubles each stage):

| stage | centre points | radius (model units) | radius on the ground |
|---|---|---|---|
| 1 | 2,048 | 0.1 | 0.625 m |
| 2 | 512 | 0.2 | 1.25 m |
| 3 | 128 | 0.4 | 2.5 m |
| 4 | 32 | 0.8 | 5 m |

Before stage 1 there is a step that looks at each point alone. After stage
4, the decoder passes the stage results back down to every point.

## Where the radius comes from

Model units are metres × `coord_scale`. The radius 0.1 is fixed by the
downloaded S3DIS weights, so `coord_scale` sets how many metres it covers.

S3DIS is indoor scans thinned to one point per 4 cm cube. Those scans are
dense: almost every 4 cm cube on a wall or table holds a point, so a 0.1
radius ball finds about 20-30 neighbours. We thin ours to one point per
0.25 m cube, and chose `coord_scale = 0.04 / 0.25 = 0.16` so that 0.1 covers
"the same number of cubes" (0.625 m).

**The mistake: this assumes our cubes are full too.** They are not.
Aircraft LiDAR gives only a few points per square metre, so most 0.25 m
cubes are empty, and the thinning hardly changes anything. The voxel size is
the *densest* our data can ever be, not how dense it actually is. Measured
spacing between neighbouring points: 0.31 m overall, 0.45 m at Horseshoe
Canyon. Steep rock is sparser still, because the aircraft looks down at it.

## What the model actually gets

I simulated each stage on 100 real fine-tuning blocks: the same even picking
of centres, the same radius, counting **real** points (copies removed) in
each ball. Median out of 32 slots, centre point included:

| first radius | stage 1 | stage 2 | stage 3 | stage 4 |
|---|---|---|---|---|
| **0.625 m (today)** | **2** | **3** | **3** | **4** |
| 1.5 m | 9 | 26 | 30 | 29 |
| 2.0 m | 18 | 32 | 32 | 32 |

Horseshoe Canyon on its own: 2 / 4 / 5 / 5 today, and 18-32 at 2.0 m.

Every stage, not only the first, is nearly empty today. The later stages are
empty too, because each stage's centres are spread out evenly, so the next
stage again finds only a few of them inside its doubled radius.

**What that means in practice:**

* The model combines each point with only 1-3 others. A wall, a ledge or an
  overhang is a pattern over tens of points, and the model never sees enough
  of them together to recognise it.
* The rest is filled with repeats, so "32 neighbours" is really "the same 2
  points over and over".
* Its best remaining clue is each point's own position, especially its height
  in the block. That is exactly the shortcut from problem 1, so the two
  problems make each other worse.

## The second part: 8,192 points per block

Blocks hold a median 3,100 real points (5% under 1,800, 5% over 7,700;
Horseshoe about 2,100). The loader always draws 8,192, picking with
replacement, so a typical point appears 2-4 times.

Correction to the first write-up: the copies don't make the neighbourhoods
worse. Each stage keeps the largest value over its neighbours, and a copy
never changes that. The real costs are:

* half the memory and time is spent on copies;
* labelled points are copied a random number of times, so some count more
  in the loss than others (the problem 1 plan already fixes this by keeping
  each labelled point exactly once);
* a block of 8,192 mostly-copied points looks different from what the model
  will see on a new area.