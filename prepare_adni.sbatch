#!/bin/bash -l
#SBATCH --job-name=adni-preprocess
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=adni-preprocess-%j.out
#SBATCH --error=adni-preprocess-%j.err
set -euo pipefail
: "${REPO:?Set REPO}"
: "${ENV:?Set ENV}"
: "${ADNI_INPUT:?Set ADNI_INPUT to the raw collection root}"
: "${ADNI_SELECTED_IDS:?Set ADNI_SELECTED_IDS to final_image_ids.csv}"
: "${ADNI_UPSTREAM_PREPROCESSING:?Set ADNI_UPSTREAM_PREPROCESSING to pinned upstream source}"
: "${ADNI_PREPROCESSING_OUTPUT:?Set ADNI_PREPROCESSING_OUTPUT to a new directory}"
: "${FSLDIR:?Initialize the existing FSL installation}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
source "$FSLDIR/etc/fslconf/fsl.sh"
set -u
cd "$REPO"
python scripts/adni/preprocess.py --input-root "$ADNI_INPUT" --selected-ids "$ADNI_SELECTED_IDS" \
    --upstream-preprocessing "$ADNI_UPSTREAM_PREPROCESSING" --output-root "$ADNI_PREPROCESSING_OUTPUT" "$@"
