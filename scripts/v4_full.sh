#!/usr/bin/env bash
# The full V4 training run, from a clean state (runs/pretrain and runs/finetune must hold no old runs):
#   1. pretrain PointNeXt-S and PointNet++ (feature_keys hag,planarity; coord_scale 0.067; 4,096 points)
#   2. evaluate both (cliff IoU on val / test areas, Horseshoe waypoint check)
#   3. fine-tune each backbone on V4: decoder_head, full_discriminative, gradual_unfreeze, scratch
#   4. both Horseshoe folds (v4-hcr-a, v4-hcr-b) for each backbone's best strategy by validation AUROC
#   5. comparison table
#
#   bash scripts/v4_full.sh            # WANDB=offline to log locally
set -euo pipefail
cd "$(dirname "$0")/.."
WANDB=${WANDB:-online}
STRATEGIES=${STRATEGIES:-"decoder_head full_discriminative gradual_unfreeze scratch"}

for m in pointnext-s pointnet2; do
  if [[ -f runs/pretrain/$m/last.pt && ! -f runs/pretrain/$m/summary.json ]]; then
    echo "runs/pretrain/$m holds an unfinished run - resuming it (it was started by this script)"
  fi
  if [[ ! -f runs/pretrain/$m/summary.json ]]; then
    python scripts/s2_pretrain.py --cfg cfgs/pretrain/$m.yaml --wandb "$WANDB" wandb.group=v4-pretrain
  fi
  [[ -f runs/pretrain/$m/eval_best.json ]] || python scripts/s2_eval_pretrain.py --run runs/pretrain/$m --wandb "$WANDB"
done

for m in pointnext-s pointnet2; do
  BACKBONE=runs/pretrain/$m STRATEGIES="$STRATEGIES" WANDB=$WANDB bash scripts/s4_sweep.sh wandb.group=v4-finetune dataset.name=v4
done

for m in pointnext-s pointnet2; do
  best=$(python - "$m" <<'EOF'
import json, sys, glob
m = sys.argv[1]
runs = []
for f in glob.glob(f"runs/finetune/{m}-*/eval_best.json"):
    name = f.split("/")[-2]
    if "-hcr-" in name:
        continue
    v = json.load(open(f)).get("val", {}).get("sample_auroc")
    if v is not None:
        runs.append((v, name[len(m) + 1:]))
print(max(runs)[1])
EOF
)
  echo "=== $m: best strategy by validation AUROC: $best"
  for fold in hcr-a hcr-b; do
    name="$m-$best-$fold"
    [[ -f runs/finetune/$name/eval_best.json ]] && continue
    init=("init.arch=runs/pretrain/$m" "init.weights=runs/pretrain/$m/backbone.pt")
    [[ $best == scratch ]] && init=("init.arch=runs/pretrain/$m")
    python scripts/s4_finetune.py --strategy "$best" --run-name "$name" --wandb "$WANDB" "${init[@]}" dataset.name=v4-$fold wandb.group=v4-finetune
    python scripts/s4_eval_finetune.py --run "runs/finetune/$name" --wandb "$WANDB"
  done
done
python scripts/s4_compare.py --wandb "$WANDB"
