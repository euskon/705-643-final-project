#!/usr/bin/env bash
# Create the `crag-pointnext` conda env and compile OpenPoints' CUDA operators.
#
#   bash scripts/setup_env.sh
#   conda activate crag-pointnext
#
# Needs: conda, an NVIDIA driver, and a CUDA toolkit whose major version matches
# the torch build in environment.yml (cu124 -> CUDA 12.x; default /usr/local/cuda-12.4).
set -euo pipefail
cd "$(dirname "$0")/.."

ENV=crag-pointnext
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
export PATH="$CUDA_HOME/bin:$PATH"
# RTX 30xx = 8.6; add others if you train elsewhere (e.g. "8.6;8.9")
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export MAX_JOBS=${MAX_JOBS:-8}

git submodule update --init --recursive || {
  # PointNeXt's nested `openpoints` submodule uses an SSH URL; fall back to HTTPS
  git -C third_party/PointNeXt config submodule.openpoints.url https://github.com/guochengqian/openpoints.git
  git submodule update --init --recursive
}

if ! conda env list | grep -q "^$ENV "; then
  conda env create -f environment.yml
fi
PY="$(conda run -n $ENV python -c 'import sys; print(sys.executable)')"

# pointnet2_batch: ball query / FPS / grouping / interpolation (the only ops our models use)
( cd third_party/PointNeXt/openpoints/cpp/pointnet2_batch && "$PY" -m pip install --no-build-isolation -e . )

"$PY" - <<'EOF'
import torch, pdal, pointnet2_batch_cuda  # noqa: F401
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("pdal", pdal.__version__, "| pointnet2_batch_cuda OK")
EOF

# pretrained S3DIS checkpoints for the two backbones (Google Drive, via gdown)
"$PY" scripts/download_pretrained.py
echo "done: conda activate $ENV"
