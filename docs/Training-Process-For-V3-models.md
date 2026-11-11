The process, in plain terms

1. Find the walls (new).
For every LiDAR tile, I build a 1 m grid of ground height. Each cell uses the lowest laser return that reached the surface, which ignores trees. A cell is a wall if the ground there is steep (50° or more) and rises at least 4 m within 5 m. Each point then gets stored with its distance to the nearest wall.

2. Label the points for pretraining.
The four classes stay the same: ground, vegetation, steep terrain, cliff.
- New rule: a point only counts as cliff if it is within 2 m of a wall.
- Steep-looking points on the forest floor (tree bases, logs, rough ground) become steep terrain instead.
- This removes the red speckle you saw all over the maps.

3. Check every route GPS point (new and mandatory). There are two sources of route points:
- route locations people have tagged on OpenStreetMap, a free public map anyone can edit;
- your hand-gathered Horseshoe Canyon waypoints.

Every point from either source goes through the same check before it's used anywhere:
- Look for a real wall within 20 m. "Real" means a column of cliff points at least 4 m tall whose base is at about the height where the person stood (within 6 m).
- Pass: the point moves to the foot of that wall, so the training sample sits on the rock.
- Fail: the point is marked "off wall" and dropped, for both "climbable" and "not climbable" labels.
- Every build writes wall_check.csv with counts per area and rejected_points.csv listing exactly which points were dropped and why.

The check runs in three places, so no route GPS point can skip it:
- building the fine-tuning data (training, validation and test);
- the Horseshoe Canyon validation of the negatives;
- the pretraining evaluation on your waypoints.

4. Train.
- Pretrain both models on the new labels.
- Build fine-tuning set v3 from the checked points.
- Fine-tune with every strategy and compare.

5. Audit.
scripts/plot_hcr_waypoints.py redraws the maps. Each arrow runs from where the GPS point was to where it moved, and an ✗ marks a rejected point. That lets us confirm by eye that points now land on the rock.

Code changes so far (all tests pass):
- New: crag/walls.py, the one shared definition of a wall.
- Changed:
  - the cliff rule, in crag/pseudolabels.py;
  - the GPS point check, in crag/groundtruth.py;
  - the code that applies the check and drops off-wall points, in crag/negatives.py, crag/gt_eval.py and scripts/s3_build_finetune_set.py;
  - the audit script, so it uses the shared wall definition;
  - the fine-tuning config, which now defaults to v3;
  - docs/pipeline.md.
Update (after this was written): step 1 now builds the 1 m ground grid from the points the LiDAR provider classified as ground (LAS class 2), filling gaps from the nearest ground cell. Using the lowest surface return drew false "walls" as rings around trees. See docs/hcr-waypoint-audit.md and docs/figures/wall-dem-methods.png. Results: docs/results/v3/README.md.
