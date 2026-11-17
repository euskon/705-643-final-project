# Problem 3: what the model is given about each point (2026-11-15)

Follow-up to `docs/fable-investigation-project-issues.md`, section 3.

## 1. The two inputs the model gets

PointNeXt-S and PointNet++ take two things per point:

| input | what it is | how the model uses it |
|---|---|---|
| `pos` | the point's x, y, z | only to find neighbours and to measure each neighbour's **offset** from a centre point. Offsets are relative, so moving the whole block changes nothing. |
| `x` (the "feature channels") | a short list of numbers describing the point | the very first step reads **only these**, one point at a time. Every later stage combines each neighbour's features with its offset. |

So `pos` tells the model the **shape** (where points are relative to each
other), and the feature channels tell it **what each point is**.

## 2. What the downloaded weights were trained with

The S3DIS weights (indoor rooms) use 4 feature channels per point:

| channel | S3DIS meaning |
|---|---|
| 1-3 | red, green, blue colour from the scanner's camera |
| 4 | height above the floor |

S3DIS never puts x and y into the feature channels. Position only enters
through `pos`, as relative offsets.

## 3. What we give it

`feature_keys: pos,heights` (`crag/datasets.py`, `make_item`). After the
augmentation steps (`PointCloudXYZAlign` centres x and y on the block's mean
and sets the lowest z to 0; then a random spin around the vertical axis):

| channel | what it actually holds |
|---|---|
| 1 | x: distance from the block's mean point, along a random direction |
| 2 | y: the same, at right angles |
| 3 | z: height above the **lowest point in this block** |
| 4 | "height": z minus the lowest z, which is **channel 3 again** (the lowest z is already 0) |

Laser intensity is stored in every block but not given to the model.
Nothing else is stored in the blocks.

## 4. Why this is a problem

### 4a. The model is handed its absolute position

The neighbour offsets are built so that only *relative* shape matters. By
also putting x, y, z into the feature channels, we hand the model the
point's position inside the block directly.

* **x and y mean nothing about rock.** Where the block starts is our choice,
  and the random spin scrambles the direction. Mostly they tell the model
  how far a point is from the block's middle, which is where the labelled
  points are.
* **z is "height above the lowest point in the block".** That number depends
  on how the block was cut, not on the wall. It is exactly the number the
  problem 1 shortcut uses ("low in the block = climbable"). The feature
  channels make that shortcut available from the very first step.

### 4b. Nothing describes the surface

Apart from position, the model gets no information about a point. It has to
work out "this is a steep, flat rock face" purely from its neighbours'
offsets. Problem 2 showed each neighbourhood holds only 2-5 real points. So
today the model knows little more than **where each point is**.

That is how the three problems lock together:

* problem 1 makes position predict the label;
* problem 2 starves the model of shape;
* problem 3 hands it position directly and nothing else.

### 4c. Channel 4 is a copy of channel 3

A wasted channel. Harmless on its own, but it shows the "height" channel
was meant to carry something and doesn't.

### 4d. The colour mismatch you spotted

The first step's starting weights were tuned to read colour and receive x,
y, z. This is real, but it costs little:

* it touches only the first step (128 weights in PointNeXt-S; the rest of
  the model, about 0.8 million weights, is unaffected);
* our pretraining retrains that step anyway;
* random start, S3DIS start and our pretraining all score about the same on
  V3.

So the starting weights are not what holds the model back. The **content**
of the channels is.

## 5. What we already know about each point

| information | where it is today | what it says | computed the same in every area? |
|---|---|---|---|
| surface direction (normal) | stage files (`normals`) | which way the surface faces; how steep it is | yes, from x, y, z (fitted on ~12 neighbours, about 1.5 m) |
| flatness (planarity) | stage files (`planarity`) | flat patch (rock face) vs line (trunk) vs blob (leaves) | yes, from x, y, z |
| return number / number of returns | raw tiles | whether the laser passed through something first (vegetation in front of the rock) | mostly; the maximum number of returns depends on the sensor |
| laser intensity | raw tiles, blocks | how strongly the surface reflected | **no**: depends on the sensor, flight height, year |
| provider's class (ground, vegetation, ...) | raw tiles | the provider's own labelling | **no**: quality differs by provider |
| distance to the wall mask | stage files (`wall_dist`) | how near a point is to a bare-earth wall | yes, but see section 7 |
| height above the local ground | not stored; easy from the bare-earth grid (`crag/walls.py`) | how high up a wall or tree a point is | yes |

### Intensity is an area label, not a rock label

I measured intensity in V3 (already scaled per tile to its 99th percentile):

* median per area ranges from **0.14 (Horseshoe Canyon)** and 0.20 (Red River
  Gorge) to 0.82 (Leavenworth);
* for telling climbable from not climbable it scores **0.49** on validation
  and **0.49** on Horseshoe Climb vs Nc: no information at all.

So intensity mostly tells the model *which survey the block came from*. With
81% of climbable samples from one area, that invites a new shortcut, and
Horseshoe (the darkest area) would look unlike anything in training.

## 6. Options, and which to choose

### Position: drop x, y, z; use height above the local ground instead

* **Remove x and y** from the feature channels. The offsets in `pos` already
  give the model all the shape it needs.
* **Replace z** with **height above the local ground**: the point's z minus
  the lowest bare-earth ground height within 10 m (from the 1 m grid in
  `crag/walls.py`). Worked out from the whole area *before* the block is
  cut, it doesn't depend on where the block edges fall. On a wall it means
  "how far up the wall"; in a tree, "how high in the tree".
* The model then can't see the point's position inside the block at all.
  That cuts off the problem 1 shortcut at its source, as a second safety net
  behind the plan 1 fix.

### Surface shape: verticality and flatness first

* **Verticality**: the vertical part of the normal (1 = flat ground, 0 = a
  vertical wall).
* **Planarity**: flat face vs trunk vs leaves.
* Both stay the same when the block is spun around the vertical axis, so
  they work with the rotation augmentation as-is.
* **The full normal** (its two horizontal parts as well) would show which
  way a face points. That needs two code changes:
  * the rotation augmentation must spin the normals together with the points
    (today `PointCloudRotationZ` only spins `pos`);
  * our normals are flipped to always point up, so an overhang and a slab
    look the same.

  So add the full normal later, as a separate test.

These work at about 1.5 m, just under the new 2 m first radius (problem 2):
the features describe the surface right at the point, and the stages
describe the shape around it.

### Laser returns: two simple flags

* **Return position**: return number ÷ number of returns (1 = the last thing
  the laser hit).
* **Single return**: 1 if the pulse came back only once (bare surface), 0
  otherwise.

They show vegetation in front of or on the rock, which plausibly matters for
climbing. They're mildly sensor-dependent, so check them with the area test
in section 8.

### Leave out for now

| channel | why not |
|---|---|
| intensity | measured: area identity, no label information (section 5) |
| provider's class | quality differs by provider; it already shapes the wall mask indirectly |
| x, y, z | section 4a |

### Context channels: distance to the wall, slope of the ground grid

Strong and computed the same everywhere, but they define the pretraining
labels (section 7). Test them only in fine-tuning, as a separate step.

### Later: colour from aerial photos

US aerial photos (the National Agriculture Imagery Program, NAIP, about
0.6 m pixels) could give each point a red/green/blue colour, which is what
the downloaded first-layer weights were built for. But:

* the photos look straight down, so a vertical face barely shows (a wall
  point would take the colour of its top edge or the ground);
* photo dates and seasons differ between areas;
* it means a new download.

Worth trying only after the geometry channels are working.

## 7. The catch: pretraining

The pretraining "cliff" label is: last return, surface 60 degrees or steeper,
within 2 m of the wall mask, not vegetation. If the input includes
verticality, return position and wall distance, the label is a simple
lookup of the input. Pretraining would learn a rule, not a skill.

It already teaches little: on V3, random start and pretrained start score
about the same. Choices:

1. **Pretrain on a small set** (height above ground and planarity only), then
   fine-tune with the full set. The first step is rebuilt for the extra
   channels; the rest of the backbone still loads.
2. **Same channels for both**, accepting pretraining becomes easy and teaches
   even less.
3. **Change the pretraining task** to something the input doesn't give
   away. A bigger project.

**Recommended:** option 1, plus a cheap test first. Train `scratch` with the
new channels. If pretrained still scores no better than scratch, take
pretraining out of the main path until option 3 is worth doing.

## 8. The recommended channel set, and why

**Start with 5 channels:** height above local ground, verticality,
planarity, return position, single-return flag.

Why this fits our project:

* **Small, lopsided dataset** (about 1,566 climbable samples, 81% from Red
  River Gorge): a few meaningful channels are less likely to be used as
  shortcuts than many loosely related ones.
* **Several LiDAR surveys, and the test area is a different one:** everything
  here comes from x, y, z or the return count, and is worked out the same
  way in every area. Intensity and the provider's class are not.
* **The problem 1 shortcut:** no channel carries the point's position inside
  the block.
* **Pretraining:** works with option 1 above.

### How to test it (after plans 1 and 2)

On V4 with the new scale, `scratch` strategy, two seeds each, add one group
at a time:

| step | channels | question it answers |
|---|---|---|
| a | `pos,heights` (today) | reference |
| b | height above local ground only | does removing position alone change anything? |
| c | b + verticality + planarity | does surface shape help? |
| d | c + return position + single return | do vegetation flags help? |
| e | d + intensity (ranked within each block) | does intensity add anything once it can't identify the area? |
| f | d + distance to wall | fine-tuning only: does the context channel help? |

Keep a step only if Horseshoe Climb vs Nc improves (beyond about 0.05, the
noise level) and the gap above the plain-geometry baseline grows.

### Two checks on every channel, added to the leak test

1. **Label check:** how well the channel alone, averaged over the labelled
   points, separates climbable from not climbable. Should look like the
   model's real job, not a giveaway.
2. **Area check:** how well it tells Horseshoe from Red River Gorge. A
   channel that strongly identifies the area but not the label is a risk:
   it lets the model learn "which survey" instead of "which rock".

## 9. What changes in the code (for a later plan)

* **Blocks must store more per-point values.** Today `feat` holds intensity
  only. Fine-tuning blocks (`crag/finetune_data.build_blocks`) and
  pretraining blocks (`crag/pseudolabels.make_blocks`) need the new
  channels, so both are rebuilt. V4 is being rebuilt anyway; pretraining
  blocks took about 30 minutes last time.
* **`crag/datasets.py`:** new names in `FEATURE_DIMS` (`hag`, `vert`,
  `planarity`, `ret_pos`, `single`) and `make_item` reads them from the
  stored array.
* **`in_channels`** is already checked against the feature list in
  `crag/pretrain.py` and `crag/finetune.py`; the model configs follow it.
* **Loading weights:** the first step no longer matches the S3DIS or old
  backbones. `load_pretrained` already skips shape mismatches and reports
  them. Expect exactly the first layer to be reported, nothing else.
* **Full normals (later):** `PointCloudRotationZ` must rotate them with the
  points.
