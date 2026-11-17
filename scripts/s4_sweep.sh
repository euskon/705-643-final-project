#!/usr/bin/env bash
# Step 4 - run every fine-tuning strategy on the same dataset, evaluate, compare.
#
#   bash scripts/s4_sweep.sh                                   # PointNeXt-S backbone
#   BACKBONE=runs/pretrain/pointnet2 bash scripts/s4_sweep.sh  # the PointNet++ backbone
#   STRATEGIES="linear_probe full_discriminative" LRS="1e-3 3e-4" bash scripts/s4_sweep.sh
#   WANDB=offline bash scripts/s4_sweep.sh
#
# Extra args are passed to every s4_finetune.py call as key=value overrides.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKBONE=${BACKBONE:-runs/pretrain/pointnext-s}
STRATEGIES=${STRATEGIES:-"decoder_head full_discriminative scratch"}
LRS=${LRS:-""}            # empty = each strategy's own LR
WANDB=${WANDB:-online}
TAG=$(basename "$BACKBONE")

for s in $STRATEGIES; do
  init=("init.arch=$BACKBONE" "init.weights=$BACKBONE/backbone.pt")
  [[ $s == scratch ]] && init=("init.arch=$BACKBONE")       # scratch.yaml sets weights: null
  if [[ $s == s3dis_init ]]; then                            # its own S3DIS checkpoint, for the matching architecture
    case "$TAG" in
      pointnext-s*) init=() ;;                               # s3dis_init.yaml is PointNeXt-S
      pointnet2*) init=("init.arch=cfgs/pretrain/pointnet2.yaml" "init.weights=pretrained/openpoints/s3dis-pointnet2/checkpoint/s3dis-train-pointnet++-ngpus1-seed5583-20220506-155113-AobHJJtq8wdeMfRbcHpZGD_ckpt_best.pth") ;;
      *) echo "s3dis_init: no S3DIS checkpoint for $TAG - skipped"; continue ;;
    esac
  fi
  for lr in ${LRS:-default}; do
    name="$TAG-$s"; lr_arg=()
    if [[ $lr != default ]]; then name="$name-lr$lr"; lr_arg=("optimizer.lr=$lr"); fi
    echo "=== $name"
    python scripts/s4_finetune.py --strategy "$s" --run-name "$name" --wandb "$WANDB" "${init[@]}" "${lr_arg[@]}" "$@"
    python scripts/s4_eval_finetune.py --run "runs/finetune/$name" --wandb "$WANDB"
  done
done
python scripts/s4_compare.py --wandb "$WANDB"
