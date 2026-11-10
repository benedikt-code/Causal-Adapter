#!/bin/bash -l

#SBATCH --job-name=pendulum-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail

# ============================================================
# 1. Paths
# ============================================================

WORK=/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter
REPO="$WORK/Causal-Adapter"
ENV=/sc-projects/sc-proj-ukb-cvd/envs/causal-adapter

# ============================================================
# 2. Environment configuration
# ============================================================

export CA_WORK="$WORK"

export PYTHONPATH="$REPO/notebook_benchmarks:$REPO${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="$WORK/.cache/huggingface"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$WORK/.cache/matplotlib"
export TMPDIR="$WORK/.cache/tmp"
export PYTHONUNBUFFERED=1

# Experiment parameters
export ABLATION_N_INPUTS="${ABLATION_N_INPUTS:-5}"
export ABLATION_STEPS="${ABLATION_STEPS:-50}"

mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$TMPDIR"

# ============================================================
# 3. Activate Conda
# ============================================================

# Avoid MKL_INTERFACE_LAYER unbound-variable error
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
set -u

# ============================================================
# 4. Job information
# ============================================================

cd "$WORK"

echo "=========================================="
echo "Pendulum SCM Ablation Experiment"
echo "=========================================="

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $WORK"
echo "Input images: $ABLATION_N_INPUTS"
echo "Diffusion steps: $ABLATION_STEPS"

echo ""
echo "Python:"
which python
python --version

echo ""
echo "GPU:"
nvidia-smi

# ============================================================
# 5. Verify GPU
# ============================================================

python - <<'PY'
import torch

assert torch.cuda.is_available(), "No CUDA device available"

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

# Test GPU computation
x = torch.ones(10, device="cuda")
assert (x + x).sum().item() == 20

print("GPU computation successful!")
PY

# ============================================================
# 6. Run ablation experiment
# ============================================================

echo ""
echo "Starting SCM ablation..."

python "$WORK/ablation_pendulum.py" run

# ============================================================
# 7. Finish
# ============================================================

echo ""
echo "=========================================="
echo "Experiment completed: $(date)"
echo "=========================================="
