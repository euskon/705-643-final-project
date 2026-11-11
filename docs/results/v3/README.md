# V3 results: wall-gated cliff labels + wall-checked route points (2026-11-11)

What changed in V3 and why: `docs/Training-Process-For-V3-models.md` and
`docs/hcr-waypoint-audit.md`. The previous runs (V2: same 4 classes, no wall
rule) are archived in `runs/archive-4class-v2/`.

Files in this folder:

| file | contents |
|---|---|
| `pretrain_eval_<model>.json` | pretraining evaluation: per-class accuracy on validation and test areas, Horseshoe waypoint check |
| `finetune_comparison.csv` | every fine-tuning run, validation and test scores |
| `horseshoe_confidence_and_cutoffs.txt` | Horseshoe Canyon scores with 95% confidence ranges, and accuracy when the cutoff is chosen on validation data |
| `wall_check.csv`, `rejected_points.csv` | how many route GPS points passed the wall check, and the ones that failed |
| `finetune_set_report.json` | fine-tuning set V3: samples per area and source, check of the "not climbable" sampling rule |
| `cliff_rule_check.csv` | cliff label rules compared on known climbing spots |

How to read the scores:

* **IoU** (0-1): overlap between the predicted and true points of a class; 1 is perfect.
* **AUROC** (0-1): how well a score ranks climbable above not-climbable; 0.5 is
  guessing, 1 is perfect.
* **Balanced accuracy** (0-1): the average of "share of climbable called climbable"
  and "share of not-climbable called not climbable"; 0.5 is guessing.

## Data

* Route GPS points: 1,720 in total; 1,624 passed the wall check, 24 were more than 20 m from
  any wall (rejected), 72 have no LiDAR coverage.
* Horseshoe Canyon hand-gathered points: 123 of 126 climbable and 58 of 60
  not-climbable passed.
* Cliff labels: 2.8% of all points (7.3% in V2). Cliff points within 20 m of 98% of known climbing
  spots are kept; isolated cliff specks fell from 4.7% to 2.1%.
* Fine-tuning samples: 6,254 blocks; Red River Gorge has 1,271 of the 1,566
  climbable samples (81%).

## Pretraining (ground / vegetation / steep terrain / cliff)

| | V2 | V3 |
|---|---|---|
| cliff IoU, validation areas | 0.33-0.36 | 0.47-0.48 |
| cliff IoU, test areas | 0.40-0.45 | 0.56-0.57 |
| mean IoU of all 4 classes, test areas | 0.46-0.48 | 0.52 |

## Fine-tuning (climbable or not)

| | V2 | V3 |
|---|---|---|
| best validation AUROC | 0.868 | 0.785 |
| Red River Gorge test AUROC | 0.71-0.80 | 0.69-0.72 |
| Horseshoe Canyon AUROC, all test samples | 0.62-0.79 | 0.76-0.83 |
| Horseshoe Canyon climbable vs not-climbable hand points | 0.36-0.56 | 0.46-0.59 |
| Horseshoe climbable hand points predicted climbable (cutoff 0.5) | 0-3% | 4-22% |

Best Horseshoe Canyon runs with 95% confidence ranges (1,000 resamples):

| run | Horseshoe AUROC | climbable vs not-climbable AUROC | test balanced accuracy, cutoff 0.5 | same, cutoff chosen on validation |
|---|---|---|---|---|
| PointNet++ decoder_head | 0.83 (0.79-0.88) | 0.57 (0.47-0.66) | 0.62 | 0.71 |
| PointNet++ scratch | 0.82 (0.77-0.86) | 0.57 (0.49-0.67) | 0.58 | 0.70 |
| S3DIS initialisation | 0.80 (0.76-0.85) | 0.59 (0.50-0.67) | 0.54 | 0.70 |
| PointNeXt-S gradual_unfreeze | 0.80 (0.75-0.84) | 0.50 (0.42-0.59) | 0.63 | 0.69 |

## What is still wrong (most important first)

1. **The model cannot yet tell good rock from bad rock.** Climbable versus
   not-climbable hand points at Horseshoe Canyon scores 0.46-0.59, and every
   95% range includes 0.5 (guessing). The high overall Horseshoe score
   (0.83) mostly comes from easy negatives: "step-back" samples on the talus
   below a wall are rejected 100% of the time. So today the model is a good
   *wall finder*, not yet a *climbable-rock finder*.
2. **Our "not climbable" training examples are unreliable.** Most of them come
   from the "band" rule: wall between documented routes is assumed
   unclimbable. Checked against the Horseshoe hand labels with the wall check
   in place, it is right 42% of the time at half documentation and 83% at
   90%, which is no better than picking random spots on the wall (71% and
   82%). The model is being taught "not climbable" largely on rock that is
   climbable or unknown.
3. **No on-site "not climbable" examples in training.** The only reliable ones
   (the 58 Horseshoe Nc points) are all in the test set.
4. **One area dominates.** 81% of climbable samples are from Red River Gorge;
   the best checkpoint is chosen on a validation set that is mostly Red
   River Gorge too, so selection favours what works there.
5. **The decision cutoff is wrong.** The model's scores are not calibrated, so
   0.5 calls too little climbable (balanced accuracy 0.53-0.64). A cutoff
   chosen on validation data raises this to 0.65-0.71 on the test set. The
   evaluation script does not do this yet.
6. **Pretraining on our cliff labels does not help yet.** Training from
   scratch scores about the same as starting from our pretrained models. The
   pretraining task (find steep, wall-adjacent surfaces) is easy to learn
   from local shape and probably teaches little that climbability needs
   beyond "this is a wall".
7. **Airborne LiDAR may not show what makes rock climbable.** Vertical faces
   get few points (a few per square metre), so cracks, holds and rock
   quality are invisible. The model can see wall height, steepness,
   overhangs, continuity, vegetation and the approach. Some of the
   climbable / not-climbable difference may not be learnable from this data
   alone (aerial photos or geology maps could add it).
8. **Red River Gorge scores dropped (0.80 to 0.70) and this is not explained
   yet.** The validation and test samples changed with V3, so the numbers
   are not exactly comparable, and some runs now call more distant cliffs
   climbable (PointNet++ decoder_head rejects 66-78% of them, down from
   95-98%).
9. Smaller issues:
   * the wall check sometimes moves a point to a smaller formation nearby
     rather than the route's main wall (a few Horseshoe East Side points);
   * 27 test/validation samples have no points in the scored cylinder and get no score;
   * single laser returns high in tree crowns are labelled "steep terrain"
     instead of vegetation;
   * Horseshoe has only 181 checked hand points, mostly in clusters, so
     differences under about ±0.05 between runs are noise.
