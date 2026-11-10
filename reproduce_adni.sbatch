#!/bin/bash -l
#SBATCH --job-name=adni-reproduce
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=adni-%j.out
#SBATCH --error=adni-%j.err
set -euo pipefail
: "${REPO:?Set REPO to the checked-out repository}"
: "${ENV:?Set ENV to the existing conda environment}"
: "${ADNI_CONFIG:?Set ADNI_CONFIG to the completed JSON configuration}"
: "${ADNI_OUTPUT:?Set ADNI_OUTPUT to a new, unique output directory}"
MODE="${MODE:-smoke}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
: "${TORCH_HOME:?Set TORCH_HOME to the prepared public metric cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# This job consumes prepared local checkpoints; it never installs or trains.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
set -u
cd "$REPO"
python scripts/adni/run.py --config "$ADNI_CONFIG" --mode "$MODE" --output-root "$ADNI_OUTPUT" "$@"
