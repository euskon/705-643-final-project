Okay I have some ideas I want to refine and eventually implement

We should take some time to refine these ideas and outline a clear implementation plan.

**Step 1: Pseudolabeled pretraining dataset of climbing areas**

Goal: a broad, noisy pretraining set for a convolutional backbone (I'll later drop the prediction head and fine-tune), built before any formal annotation. Pretraining may run for days, so the data pipeline must scale to many locations.

Build a simple, modular, reproducible dataset-management tool:

1. **Area list:** Use the web to find the 50 most popular climbing areas on Mountain Project. For each, record the name, Mountain Project link, and a bounding box that accurately encompasses the area.
2. **Registry:** Store all areas in a single CSV or JSON file the script loads. Adding an area means adding one entry.
3. **Per-area data:** For each area, include the USGS file to download and a list of crags (name + coordinates) from its subareas.
4. **Script:** Download datasets incrementally (skip what's already fetched) and generate pseudolabels with our existing pseudolabeling approach.
5. **Design constraints:** Keep it simple and runnable by hand with simple scripts. Results must be reproducible. Structure the pipeline so the backbone's conv layers train separately from a swappable prediction head.

Start with the plan and the registry schema before writing the full script.


**Step 2: Initial model training**
We need to train a the model on the pseudolabeled pretraining dataset to learn general features of climbing areas. This should take a fairly long time and we should consider training two different already pretrained models we have already been using one but there is another 

When fine-tuning on massive USGS public LiDAR point cloud datasets (typically distributed in Entwine Point Tile (EPT) or LAS/LAZ formats) using an RTX 3080 (10GB VRAM), memory optimization is your main hurdle. PointNet++’s custom CUDA operators (specifically Farthest Point Sampling and Ball Querying) scale poorly in memory usage when raw block point densities get too high.
For this specific setup, OpenPoints (PointNeXt) or sshaoshuai’s Pointnet2.PyTorch are your best practical baselines, but you will need to structure your data pipeline aggressively to avoid Out-Of-Memory (OOM) errors.

Hardware Strategy for an RTX 3080 (10GB)

The RTX 3080 has incredible compute capability (Ampere architecture), but 10GB of VRAM fills up instantly if you feed large 3D point cloud chunks. Implement these configurations immediately:
• Point Count Budget: Do not exceed 8,192 to 16,384 points per sample block during training. PointNet++ handles local hierarchies by keeping dense index tensors in memory.
• Batch Size: Set your batch size low (4 to 8 samples per GPU step). To compensate for the low batch size and stabilize your gradients, use Gradient Accumulation (e.g., accumulate over 4 or 8 steps to simulate a true batch size of 32 or 64).
• Mixed Precision (AMP): Use PyTorch’s torch.cuda.amp.autocast(). This drops memory consumption significantly by utilizing FP16 Tensor Cores while preserving FP32 for unstable operations like coordinate normalization.

Designing Your USGS Dataset Pipeline

USGS point clouds cover large geographic coordinates with varying absolute heights. PointNet++ cannot process raw global coordinates.
1. Grid-Based Chunking: Subdivide your USGS tiles into localized 2D bounding boxes (e.g., blocks of 10m × 10m or 20m × 20m in the XY plane).
2. Coordinate Normalization: For every sampled block, translate the coordinates so that the centroid of the block sits at (0,0,0). Scale the points so they fit entirely within a unit sphere (coordinates bounded between \([-1, 1]\)). PointNet++ will fail to converge if you do not do this.
3. Point Downsampling: USGS data can be dense. Use Voxel Grid Downsampling (via open3d or laspy) to enforce a uniform point density across your blocks before feeding them into your PointNet++ Farthest Point Sampling (FPS) layer.

Repository Recommendation for Your Task

Framework Choice	Strategy for USGS Data
OpenPoints (PointNeXt)	Best for Accuracy. Use their PointNet++ MSG configuration. They include modern data augmentations (random rotation, scaling, jitter) built-in, which prevents your model from overfitting to specific USGS flight-line flight directions.

Lets use this OpenPoints (PointNeXt) framework for our USGS data processing and model training. Consider pulling this repository and following their documentation for setting up the environment and data pipeline. We can utilize this as a submodule in our repo:`https://github.com/guochengqian/PointNeXt`

I think this is an example of using a pretrained PointNeXt model for segmentation on the S3DIS dataset. You can adapt this command for your USGS dataset by changing the configuration file, pretrained path, and number of classes accordingly.
```bash
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py \
  --cfg cfgs/s3dis/pointnext-s.yaml \
  --pretrained_path pretrained/s3dis/pointnext-s/checkpoint/pointnext-s_ckpt_best.pth \
  batch_size=8 \
  num_points=8192 \
  num_classes=<OUR_CLIMB_CLASS_COUNT>
```

Here is an example yaml configuration for the PointNeXt model:
```yaml
# ==============================================================================
# Model Architecture & Segmentation Head
# ==============================================================================
model:
  NAME: BaseSeg
  encoder_args:
    NAME: PointNextEncoder
    blocks: [1, 1, 1, 1, 1, 1]
    strides: [1, 2, 2, 2, 2, 1]
    width: 32
    in_channels: 4          # 4 channels: XYZ coordinates + Intensity (or 3 if XYZ only)
    sa_layers: 2
    sa_use_res: True
    radius: 0.1
    radius_scaling: 2.5
    nsample: 32
    expansion: 4
    aggr_args:
      feature_type: 'dp_df'
      reduction: 'max'
    group_args:
      NAME: 'ballquery'
      normalize_dp: True
    conv_args:
      order: conv-norm-act
    act_args:
      act: 'relu'
    norm_args:
      norm: 'bn'
  decoder_args:
    NAME: PointNextDecoder
  head_args:
    NAME: SegHead
    num_classes: 5          # SET TO YOUR CLIMBING CLASS COUNT (e.g., PossiblyClimbableRock, NotPossiblyClimbableRock)
    in_channels: null

# Path to pretrained weights (e.g., S3DIS pretrained checkpoint)
pretrained_path: "pretrained/pointnext-s_s3dis.pth"

# ==============================================================================
# Dataset & Data Augmentation Setup
# ==============================================================================
dataset:
  common:
    NAME: USGSPointCloudDataset   # Replace with your registered Dataset class name
    data_dir: "data/usgs_processed"
    num_points: 8192              # Keep between 8192 and 16384 for 10GB VRAM
  train:
    split: train
  val:
    split: val
  test:
    split: test

# Data augmentations suitable for aerial LiDAR (XY rot, scale, jitter)
datatransforms:
  train:
    - NAME: PointsToTensor
    - NAME: PointCloudScaling
      scale: [0.9, 1.1]
    - NAME: PointCloudRotation
      angle: [0, 0, 1]           # Aerial LiDAR should strictly rotate around the Z (upright) axis
    - NAME: PointCloudJitter
      jitter_sigma: 0.005
      jitter_clip: 0.02
  val:
    - NAME: PointsToTensor

# ==============================================================================
# Hardware, Batching & Training Hyperparameters (RTX 3080 10GB)
# ==============================================================================
batch_size: 8                     # Keep at 4–8 to prevent CUDA OOM during ball queries
val_batch_size: 8
dataloader:
  num_workers: 6
  pin_memory: True

epochs: 100
use_amp: True                     # Automatic Mixed Precision (FP16) saves ~35-40% VRAM

optimizer:
  NAME: 'adamw'
  lr: 0.001                       # Lower learning rate for fine-tuning
  weight_decay: 1.0e-4

sched: cosine
warmup_epochs: 5
min_lr: 1.0e-5

# Loss function
criterion_args:
  NAME: CrossEntropy
  label_smoothing: 0.1            # Helps with noisy boundary points in dense canopy/structures
```

Once you have revamped the repository with the new configuration, you can proceed to train the model using the specified dataset and hyperparameters. This should again be highly reproducible and easy for me to invoke. Please create easily runnable scripts for running the pretraining similar to what you did for the dataset in the previous step. This step should also have a script that we can easily run for evaluation of the pretrained backbone at its best checkpointed weights. The weights from the pretraining should be saved after each improvement of the validation loss over a threshold such as 0.001 reduction in loss. We want to set this up to use weights and balances to track our pretraining runs. I will provide a key in a separate file for tracking these runs.

**Step 3: Determining Fine Tuning Strategy ** 
We need to find out how to fine-tune the pretrained model effectively for our specific task, including determining which layers to freeze, the appropriate learning rate, and the number of epochs for fine-tuning. This step should also include a script for running the fine-tuning process and evaluating the model's performance on the validation set to ensure that the fine-tuning is improving the model's performance. Additionally, we should save the best checkpointed weights during fine-tuning, similar to the pretraining step, and track the fine-tuning runs using the key provided in the separate file.

**Step 3: Route-level GPS data and hard negatives**

Problem: we have very few hard negatives, so the dataset is severely imbalanced. We'll train mostly on true positives and need to generate negatives that are actually hard for the model to classify.

1. **Route-level GPS collection:** For each of the 50 areas from Step 1, collect GPS coordinates for individual climbing routes only, not crag-level points (those are only rough positions). Use API's for OpenClimbing.org (which uses open street maps in some ways) to query each area's bounding box for user-uploaded routes. Write a reproducible, easily runnable script for this.
2. **Cross-referencing:** Match collected routes against the route CSVs we scraped from Mountain Project, keeping only points that correspond to documented routes names. Remember mountain project does NOT have specific climbing route locations, it only has general area information. We  want to find a way to get the most well documented areas on openclimbing.org in the USA so we can then use the USGS survey data from that to fine-tune on since we have the route-level GPS data from OpenClimbing.org. Pretraining we don't need specific route-level GPS data. 
3. **Hard negative generation:** Research how to best generate and use hard negatives given how few exist. Start with this idea: sample synthetic points near verified route locations but not on them (e.g., within a defined distance band). Also evaluate other strategies, such as data augmentation, and compare their tradeoffs.
4. **Constraints:** Keep it simple, modular, and reproducible.

Before writing code, propose the data-collection approach and the negative-sampling strategy, including how you'd verify that the negatives are actually negative.

Again once we are ready to implement the data-collection and negative-sampling strategies, create easily runnable scripts for each step, ensuring that the process is reproducible and can be easily invoked by others. Use the key provided in the separate file to track the progress and results of these steps in weights and balances.


# FINAL OUTPUT:
All four steps are implemented and their scripts are tested. Per your instruction, I built the datasets but didn't run any training or the full LiDAR download. All 18 tests pass, including GPU checks of both backbones, and each training and eval script ran end to end on Horseshoe Canyon data in a few-step smoke mode. Nothing is committed yet.

Step 1: area registry and pseudolabelled pretraining data
- Registry: registry/areas.json is the single registry file. Adding an area means adding one line with its slug, name and Mountain Project ID; the build script fills in the rest and only processes what's missing.
- What it holds: the 50 most-viewed areas from a list of 61 well-known US areas I picked by hand. Mountain Project publishes no popularity ranking, so I ranked by page views.
  - Horseshoe Canyon Ranch ranks #35, so it fills one of the 50 slots. It's marked as held out and never used in pretraining, which leaves 49 pretraining areas: 39 train / 5 val / 5 test.
  - Each area has its crags (name and coordinates), a bounding box and the USGS LiDAR dataset(s) to download.
- LiDAR coverage gaps:
  - No USGS coverage, so they contribute nothing: Smith Rock, Index, Shelf Road, Custer State Park, Queen Creek, Exit 38 and Mount Charleston. Oregon and Washington LiDAR is mostly hosted by the states, not USGS.
  - Partial coverage: Red Rocks, Yosemite (about 0%), Lake Tahoe, Big and Little Cottonwood, Zion and Sedona.
  - Mostly bouldering: Joe's Valley and Hueco have very few roped routes.
  - That leaves roughly 40 areas with useful data. To replace weak ones, add candidates and re-run the build; the steps are in docs/registry-schema.md.
- Download size: the estimate is 4,836 tiles, about 29 GB. Downloads resume where they stopped. Pseudolabels use the existing approach, and the backbone is saved separately from the head (backbone.pt).

Step 2: pretraining compares PointNeXt-S and PointNet++, both starting from public S3DIS weights, with PointNeXt as a git submodule.
- It uses your RTX 3080 settings: 8 × 8192 points per batch, gradient accumulation of 4, mixed precision.
- The best checkpoint is saved whenever validation loss improves by more than 0.001, runs resume after interruption, and everything is tracked in W&B.
- GPU memory is much lower than expected (0.45–0.82 GB per step), so batch 32 or 16384 points also fit.

Step 3: real route locations and hard negatives
- Route coordinates: these come from OpenStreetMap, which is what openclimbing.org edits, plus your Horseshoe Canyon waypoints: 126 climbs and 60 non-climbable points. Mountain Project coordinates are used only to match route names, never as locations.
- Data is thin: only about 2,600 route-level points exist across the US, and about 1,600 are in Red River Gorge, so the fine-tuning set is Red River Gorge-heavy. Horseshoe Canyon is the test area.
- Negatives are only reliable where routes are well documented: I checked hard negatives (non-climbable points 10–40 m from known routes) against your Horseshoe data. About 45% were truly non-climbable when 30% of a wall's routes were mapped, and about 93% at 90%. So negatives are only generated on walls with at least 60% of routes mapped.
- Rationale: the full write-up is in docs/step3-routes-and-hard-negatives.md.

Step 4: fine-tuning has six strategies, compared on validation data: linear probe, decoder + head, full fine-tune with lower learning rates for earlier layers, gradual
unfreezing, from scratch, and S3DIS weights only. Each saves its best he reasoning on frozen layers, learning rates and epochs is indocs/step4-finetuning-strategy.md.

Order to run (full runbook in docs/pipeline.md):
1. make lidar pseudolabels (Step 1 data).
2. make survey routes finetune-data (Step 3 data). This has to come before pretraining because it marks the fine-tuning test crags, which pretraining then skips.
3. make pretrain, then make eval-pretrain.
4. make sweep compare.