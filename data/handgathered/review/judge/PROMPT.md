# Judging the Horseshoe Canyon Ranch waypoints

You are checking where hand-recorded GPS waypoints belong on the rock. For each case you
get a picture (`cases/<id>.png`) and numbers (`cases/<id>.json`). Decide which candidate
spot is the foot of the rock the label is about.

## Facts

* Each waypoint is where a person **stood at the base of a wall** and recorded a label:
  `Climb` (a climbing route starts here; often with the route name) or `Nc` (this rock is
  not climbable).
* Handheld GPS under a cliff is off by **3-10 m**.
* **East Side** points sit **7-12 m out on the downhill side** of the wall (a known offset).
* Walls **under 6 m** tall don't count (bouldering height). Heights are measured on the
  bare-earth ground grid.
* A `Climb` and an `Nc` point may be close together. That is real (a route beside a blank
  or loose stretch), not an error.
* **Nobody will check your answer by hand.** Mark real uncertainty as low confidence.

## The picture

* Left, the map (60 m x 60 m, north up, axes in metres east / north of the raw GPS point):
  grey hillshade of the ground, brown 2 m contour lines, **red shade = wall mask** (steep
  ground with relief), coloured dots = cliff LiDAR points (colour = height above the wall
  foot), **yellow x = the raw GPS point**, **yellow o = the automatic snap**, **numbered
  black circles = candidate spots**, blue (Climb) / red (Nc) x = neighbouring waypoints'
  raw points, joined by a line to their own automatic snap (o).
* Right, the profile: a cut across the wall through the raw point. Left is into the hill,
  right is out into the air. Brown line = ground, green dots = cliff points, black bars =
  candidates (foot to wall top). The person stood at the black x.

## The numbers (`<id>.json`)

* `point`: label, name, wall section (`area_name`), time, the automatic snap's status and flags.
* `candidates`: for each numbered spot: what it is (`roles`), distance from the raw point,
  how far it is along the wall from straight in front (`along_shift_m`), wall height on the
  ground grid (`wall_height_grid_m`) and in the upright column, foot height minus the
  standing height, cliff points within 2.5 m, and which wall piece it is on (`wall_piece`).
* `walking_order`: the three waypoints recorded before and after this one (by time), with
  their labels, names and where their own automatic snaps are relative to this raw point.

## What to decide, per case

* a candidate number: that spot is the foot of the rock this label is about; or
* `"none"`: no candidate fits and no better spot exists - the point should be dropped; or
* `"custom"` with `custom_east_m` and `custom_north_m` (metres east / north of the raw GPS
  point, read from the map axes). A custom spot **must** lie within 2 m of the wall mask
  (red shade) on a wall at least 6 m tall, or it is rejected. Use it rarely.

## How to decide

* Prefer the wall the person faced, on the downhill side of it.
* The foot should be near the standing height (the black x in the profile).
* Keep the walking order: consecutive waypoints should progress along the wall, unless the
  times show the person turned back.
* A route and its neighbours should end up on the same stretch of wall.
* Keep neighbouring points apart about as far as their raw points are apart along the wall;
  do not put two points on the same spot unless their raw points are that close.
* Never pick a wall just because it is taller.
* If the automatic snap (candidate 1) is reasonable, keep it.
* If unsure, say so with low confidence rather than guessing.

## Output

Read your batch file (a JSON list of case ids in walking order). For every id read
`cases/<id>.png` (with the Read tool) and `cases/<id>.json`. Then write **one** file, the
decisions file named in your instructions, as JSON:

```json
[
  {"gt_id": "Waypoints_2026-10-24#0", "choice": 1, "confidence": "high", "reason": "one or two sentences"},
  {"gt_id": "Waypoints_2026-10-24#1", "choice": "custom", "custom_east_m": -3.5, "custom_north_m": 6.0, "confidence": "low", "reason": "..."},
  {"gt_id": "Waypoints_2026-10-24#2", "choice": "none", "confidence": "medium", "reason": "..."}
]
```

`confidence` is `high`, `medium` or `low`. One entry per case, every case answered once.
**Read the case files; write only that one decisions file; change nothing else.**
