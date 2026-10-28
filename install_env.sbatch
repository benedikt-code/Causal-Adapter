#!/bin/bash -l
#SBATCH --job-name=causal-adapter-env
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT="/sc-projects/sc-proj-ukb-cvd"
ENV_PREFIX="$PROJECT/envs/causal-adapter"
CACHE="$PROJECT/.cache/causal-adapter"

cd "$SLURM_SUBMIT_DIR"

echo "Starting: $(date)"
echo "Node: $(hostname)"
echo "Repository: $PWD"

# Configure package caches
mkdir -p "$CACHE/conda-pkgs" "$CACHE/pip" "$CACHE/tmp"

export CONDA_PKGS_DIRS="$CACHE/conda-pkgs"
export PIP_CACHE_DIR="$CACHE/pip"
export TMPDIR="$CACHE/tmp"

# Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check repository files
test -f environment.yml
test -d diffusers

# Remove target directory only if empty
if [ -d "$ENV_PREFIX" ]; then
    rmdir "$ENV_PREFIX" || {
        echo "ERROR: Target directory is not empty."
        exit 1
    }
fi

# Create environment
echo "Creating Conda environment..."

conda env create \
    --prefix "$ENV_PREFIX" \
    --file environment.yml

# Activate environment
conda activate "$ENV_PREFIX"

# Install custom Diffusers fork
echo "Installing modified Diffusers..."
python -m pip install -e ./diffusers

# Set repository import path
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Verify
echo "Checking installation..."
python --version
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import diffusers; print('Diffusers:', diffusers.__version__)"
python -c "from diffusers import Causal_ControlNetModel; print('Causal ControlNet OK')"

echo "Installation finished: $(date)"