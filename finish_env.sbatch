#!/bin/bash -l
#SBATCH --job-name=causal-env-finish
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=finish-env-%j.out
#SBATCH --error=finish-env-%j.err

set -eo pipefail

WORK="/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter"
REPO="$WORK/Causal-Adapter"
ENV="/sc-projects/sc-proj-ukb-cvd/envs/causal-adapter"

export PIP_CACHE_DIR="$WORK/.cache/pip"
export TMPDIR="$WORK/.cache/tmp"

mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# Initialize Conda without strict undefined-variable checking
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
set -u

cd "$REPO"

echo "Python:"
which python
python --version

echo "Installing custom Diffusers..."
python -m pip install --no-deps -e ./diffusers

echo "Verifying installation..."

python - <<'PY'
import torch
import diffusers

from diffusers import (
    Causal_ControlNetModel,
    StableDiffusionCausalControlNetPipeline,
)

print("PyTorch:", torch.__version__)
print("Diffusers:", diffusers.__version__)
print("Diffusers location:", diffusers.__file__)
print("CUDA build:", torch.version.cuda)
print("Causal-Adapter imports: OK")
PY

echo "Environment setup completed successfully."
