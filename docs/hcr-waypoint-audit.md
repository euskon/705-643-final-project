# Horseshoe Canyon waypoint audit

Do the hand-gathered GPS waypoints (`data/handgathered/`) sit on the rock they
label, and where does fine-tuning's snapping (`crag.groundtruth.snap_one`)
move them? Regenerate everything here with:

    python scripts/plot_hcr_waypoints.py      # -> docs/figures/hcr-waypoints/

## Figures

| file | what it shows |
|---|---|
| `overview.png` | whole area: hillshade + 5 m contours, black = bare-earth wall, red / pink = single- / multi-return `cliff` pseudolabel, dots = waypoints |
| `zoom_<area>_<k>.png` | ~160 m windows; each waypoint labelled with its route name and distance to the bare-earth wall |
| `wall_mask_steps.png` | how the wall mask is built (below), one East Side window |
| `profiles.png` | elevation cross-sections through 15 waypoints, toward the nearest wall |
| `distances.png` | Climb vs Nc: distance to wall, snap distance, cliff points within 5 m, relief |
| `waypoints_metrics.csv` | per-waypoint numbers |

## The bare-earth wall mask

A deliberately simple definition of "wall" built only from ground height
(`crag/walls.py`). Pseudolabels, the wall check on route points and these
figures all use this same code. Everything is on a 1 m grid:

1. **Bare-earth DEM.** Per 1 m cell, the lowest point the LiDAR provider
   classified as ground (LAS class 2). Cells with no ground point (often right
   under a tree) take the nearest cell that has one. An earlier version used
   the lowest last return of any kind, and under trees that was canopy, which
   drew false "walls" as rings around trees. Tiles with almost no ground
   classification fall back to that method.
2. **Slope.** Gradient of the DEM after a light Gaussian smooth (sigma = 1 m,
   so single noisy cells don't make fake slopes), converted to degrees.
3. **Relief.** Max minus min elevation in a 5 x 5 m window.
4. **Steep:** slope >= 50 deg. Gridding a vertical wall on 1 m cells smears it
   into a strip that reads as ~60-80 deg, so 50 is a safe threshold for it.
5. **Tall:** relief >= 4 m. Removes short steps: road cuts, banks, a single
   boulder edge.
6. **Clean-up:** a 2 x 2 m morphological opening removes 1-cell specks.

The wall comes out as a 3-6 m wide strip from the foot of the wall to its top.
"Distance to wall" is the horizontal distance to the nearest strip cell, which
is in practice **the distance to the foot of the wall** from the side the
person stood on.

Limitations:
- Overhangs show only as their base.
- A wall hidden under dense canopy with no ground returns gets interpolated
  and can be missed.
- Boulders and small ledges at least 4 m tall count as walls too (the small
  closed shapes on the valley floor in `overview.png`).
- Where the provider's ground classification cuts off a cliff top or face,
  the DEM interpolates across it; a wall is still found as long as its top
  and foot have ground points within a few metres.

## What changed because of this (2026-11-10)

* `cliff` pseudolabels now require a point within 2 m of the wall mask.
* Every route GPS point (OpenStreetMap route starts and hand-gathered
  waypoints) goes through the wall check: it is moved to the foot of the
  nearest wall within 20 m that is at least 4 m tall and starts at the
  person's ground height, or rejected as `off_wall`. Fine-tune builds write
  `wall_check.csv` and `rejected_points.csv`. In the zoom figures an arrow shows
  where each point moved to and a black x marks a rejected point.

## Findings before the change (fine-tune set v2)

| area | Climb median / 90th pct distance to wall | Nc median |
|---|---|---|
| West Side | 0.5 / 2.2 m | 0.4 m |
| North Forty | 3.5 / 13.8 m | 2.6 m |
| East Side | 7.3 / 10.3 m | 7.4 m |

- 44% of Climb and 33% of Nc waypoints are more than 5 m from the wall.
- **East Side** waypoints run parallel to the wall about 7-12 m out on the
  downhill side. This looks like a systematic offset (GPS bias, or standing
  on the approach trail), not random error.
- **North Forty:** points along the bluff edge are within 1-5 m. The Big Top /
  Trapeze Artist / Fat Hand cluster is 10-19 m out. Spam is 31 m out and
  Season of the Storm 20 m out, unless those two routes are on boulders.
- **The snap does not correct this.** It moves points a median 0.6 m (max
  5 m). The `cliff` pseudolabel (7.6% single-return + 3.0% multi-return of all
  Horseshoe points) is speckled across the forest floor densely enough that a
  "cliff column" (>= 8 cliff points within 2.5 m) exists almost everywhere, so
  the snap stops where the person stood. Fine-tuning cylinders at East Side
  and North Forty are therefore centred on forest floor, not the wall.

## Which points the ground grid is built from

`docs/figures/wall-dem-methods.png` compares three ways of building the
ground grid at Tower Rock and Dip Wall (Red River Gorge) and on the Horseshoe
Canyon plateau and North Forty wall (white = wall mask):

* **lowest surface return** (first version): rings around trees everywhere,
  because under a tree the lowest return is often canopy;
* **provider ground points only** (current): clean walls, no rings;
* **ground, plus "bare-looking" cells where nearly every pulse returned once**:
  the rings come back, since some tree crowns also return single pulses.

With the current method, 24 of 1,720 route GPS points fail the wall check
(fine-tune set v3). 16 are at Red River Gorge (Dip Wall, Tower Rock): those
OpenStreetMap points sit 24-97 m from the nearest wall, and with the old
method they had only "passed" because of tree rings next to them. The full
list is in `data/finetune/v3/rejected_points.csv`.
