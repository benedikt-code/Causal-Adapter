#!/bin/bash -l
#SBATCH --job-name=adni-qualitative
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=adni-qualitative-%j.out
#SBATCH --error=adni-qualitative-%j.err
set -euo pipefail
# Select the site's A100 40GB partition/GRES/constraint at submission time.
# Python checks the actual allocation before loading diffusion weights.
WORK="${WORK:-/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter}"
REPO="${REPO:-$WORK/Causal-Adapter}"
ENV="${ENV:-/sc-projects/sc-proj-ukb-cvd/envs/causal-adapter}"
: "${ADNI_DATA_ROOT:?Set ADNI_DATA_ROOT to the preprocessing root containing preprocessed_data/}"
: "${ADNI_BASE_MODEL:?Set ADNI_BASE_MODEL to the downloaded miniSD-diffusers directory}"
: "${ADNI_WEIGHTS:?Set ADNI_WEIGHTS to the released causal-adapter-weights/ADNI directory}"
ADNI_OUTPUT="${ADNI_OUTPUT:-$WORK/results/adni/qualitative-${SLURM_JOB_ID:?Submit this script with sbatch}}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED="${ADNI_SEED:-40}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
set -u
cd "$REPO"
args=(--data-root "$ADNI_DATA_ROOT" --base-model "$ADNI_BASE_MODEL" --weights-root "$ADNI_WEIGHTS"
      --output-root "$ADNI_OUTPUT" --subjects "${ADNI_SUBJECTS:-4}" --levels "${ADNI_LEVELS:-5}"
      --seed "${ADNI_SEED:-40}" --split "${ADNI_SPLIT:-test}")
if [[ -n "${ADNIMERGE_CSV:-}" ]]; then
    args+=(--adnimerge-csv "$ADNIMERGE_CSV")
fi
python -u scripts/adni/qualitative.py "${args[@]}" "$@"
