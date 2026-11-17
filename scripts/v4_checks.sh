#!/usr/bin/env bash
# V4 confirmation runs before the full training (plans 2 and 3): cheap "scratch" fine-tuning runs
# (no pretraining needed) on fine-tuning set V4, PointNeXt-S, two seeds each.
#
#   bash scripts/v4_checks.sh scale      # plan 2, 6b: neighbourhood scale A/B
#   bash scripts/v4_checks.sh channels   # plan 3, 10a: input channel groups a-d
#   bash scripts/v4_checks.sh channels2  # plan 3, 10a: e (intensity rank) and f (wall distance)
#   python scripts/v4_summary.py         # table of all check runs
#
# Run names start with "chk-"; each is evaluated right after training.
set -euo pipefail
cd "$(dirname "$0")/.."
WANDB=${WANDB:-online}
SEEDS=${SEEDS:-"42 7"}
ARCH=init.arch=cfgs/pretrain/pointnext-s.yaml

run() {  # name, overrides...
  local name=$1; shift
  if [[ -f runs/finetune/$name/eval_best.json ]]; then echo "=== $name (done)"; return; fi
  echo "=== $name"
  python scripts/s4_finetune.py --strategy scratch --run-name "$name" --wandb "$WANDB" "$ARCH" wandb.group=v4-checks "$@"
  python scripts/s4_eval_finetune.py --run "runs/finetune/$name" --wandb "$WANDB"
}

case "${1:-}" in
  scale)
    # jitter kept in metres for every scale: sigma = 0.03 m x scale, clip = 0.12 m x scale
    for s in $SEEDS; do
      run chk-scale-cs016-n8192-s$s  dataset.coord_scale=0.16  dataset.num_points=8192 datatransforms.kwargs.jitter_sigma=0.0048 datatransforms.kwargs.jitter_clip=0.0192 batch_size=8 grad_accum_steps=2 seed=$s
      run chk-scale-cs005-n8192-s$s  dataset.coord_scale=0.05  dataset.num_points=8192 batch_size=8 grad_accum_steps=2 seed=$s
      run chk-scale-cs005-n4096-s$s  dataset.coord_scale=0.05  dataset.num_points=4096 seed=$s
      run chk-scale-cs0067-n4096-s$s dataset.coord_scale=0.067 dataset.num_points=4096 datatransforms.kwargs.jitter_sigma=0.002 datatransforms.kwargs.jitter_clip=0.008 seed=$s
    done ;;
  channels)
    for s in $SEEDS; do
      run chk-chan-a-posheights-s$s dataset.feature_keys=pos,heights seed=$s
      run chk-chan-b-hag-s$s        dataset.feature_keys=hag seed=$s
      run chk-chan-c-shape-s$s      dataset.feature_keys=hag,vert,planarity seed=$s
      run chk-chan-d-returns-s$s    dataset.feature_keys=hag,vert,planarity,ret_pos,single seed=$s
    done ;;
  channels2)
    for s in $SEEDS; do
      run chk-chan-e-intensity-s$s dataset.feature_keys=hag,vert,planarity,intensity_rank seed=$s   # on top of the chosen set (c)
      run chk-chan-f-walldist-s$s  dataset.feature_keys=hag,vert,planarity,wall_dist seed=$s
    done ;;
  *) echo "usage: $0 scale|channels|channels2"; exit 2 ;;
esac
