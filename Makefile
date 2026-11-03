# One-word entry points for each step (each target just calls a script; see docs/pipeline.md).
# Override variables on the command line, e.g.  make pretrain MODEL=pointnet2 WANDB=offline
PY      ?= python
MODEL   ?= pointnext-s
WANDB   ?= online
AREAS   ?=
STRATEGY?= full_discriminative
DATASET ?= v1
area_arg = $(if $(AREAS),--areas $(AREAS),)

.PHONY: env test rank registry lidar-plan lidar pseudolabels pretrain pretrain-smoke eval-pretrain \
        survey routes finetune-data finetune eval-finetune sweep compare

env:            ; bash scripts/setup_env.sh
test:           ; $(PY) -m pytest tests -q
# --- step 1: pretraining dataset
rank:           ; $(PY) scripts/s1_rank_areas.py
registry:       ; $(PY) scripts/s1_build_registry.py $(area_arg)
lidar-plan:     ; $(PY) scripts/s1_download_lidar.py --dry-run $(area_arg)
lidar:          ; $(PY) scripts/s1_download_lidar.py $(area_arg)
pseudolabels:   ; $(PY) scripts/s1_pseudolabel.py $(area_arg) --wandb $(WANDB)
# --- step 2: pretraining
pretrain:       ; $(PY) scripts/s2_pretrain.py --cfg cfgs/pretrain/$(MODEL).yaml --wandb $(WANDB)
pretrain-smoke: ; $(PY) scripts/s2_pretrain.py --cfg cfgs/pretrain/$(MODEL).yaml --smoke
eval-pretrain:  ; $(PY) scripts/s2_eval_pretrain.py --run runs/pretrain/$(MODEL) --wandb $(WANDB)
# --- step 3: route-level GPS + hard negatives
survey:         ; $(PY) scripts/s3_survey_openclimbing.py --wandb $(WANDB)
routes:         ; $(PY) scripts/s3_collect_routes.py $(area_arg) --wandb $(WANDB)
finetune-data:  ; $(PY) scripts/s3_build_finetune_set.py --name $(DATASET) --wandb $(WANDB)
# --- step 4: fine-tuning
finetune:       ; $(PY) scripts/s4_finetune.py --strategy $(STRATEGY) init.arch=runs/pretrain/$(MODEL) init.weights=runs/pretrain/$(MODEL)/backbone.pt dataset.name=$(DATASET) --wandb $(WANDB)
eval-finetune:  ; $(PY) scripts/s4_eval_finetune.py --run runs/finetune/$(STRATEGY) --wandb $(WANDB)
sweep:          ; BACKBONE=runs/pretrain/$(MODEL) WANDB=$(WANDB) bash scripts/s4_sweep.sh dataset.name=$(DATASET)
compare:        ; $(PY) scripts/s4_compare.py
