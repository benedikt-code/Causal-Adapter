#!/bin/bash -l
#SBATCH --job-name=pendulum-prepare
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00

set -euo pipefail

WORK="/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter"
REPO="$WORK/Causal-Adapter"
ENV="/sc-projects/sc-proj-ukb-cvd/envs/causal-adapter"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Cache directories
export HF_HOME="$WORK/.cache/huggingface"
export PIP_CACHE_DIR="$WORK/.cache/pip"
export TMPDIR="$WORK/.cache/tmp"
export MPLCONFIGDIR="$WORK/.cache/matplotlib"
export MPLBACKEND=Agg

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$MPLCONFIGDIR"

# Activate Conda
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
set -u

# Point the editable installation to the new repository
python -m pip install --no-deps -e "$REPO/diffusers"

# Verify dependencies
python -c "from diffusers import Causal_ControlNetModel; print('Diffusers OK')"
python -c "import nbclient, nbformat, ipykernel; print('Notebook dependencies OK')"

echo "Downloading pretrained models..."

python - <<'PY'
import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

work = Path("/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter")
repo = work / "Causal-Adapter"
data = work / "data" / "pendulum_quick"

# Download mini Stable Diffusion backbone
snapshot_download(
    repo_id="lambda/miniSD-diffusers",
    local_dir=str(work / "models" / "miniSD-diffusers"),
)

# Download only Pendulum checkpoints
snapshot_download(
    repo_id="LeiTong/Causal-Adapter",
    allow_patterns=["pendulum/**"],
    local_dir=str(work / "models" / "causal-adapter-weights"),
)

print("Model downloads completed.", flush=True)

# Generate exactly one synthetic image
data.mkdir(parents=True, exist_ok=True)
(data / "pendulum" / "train").mkdir(parents=True, exist_ok=True)
(data / "pendulum" / "test").mkdir(parents=True, exist_ok=True)

source = (repo / "pendulum.py").read_text()

assert "range(-40,44)" in source
assert "range(60,148)" in source

source = source.replace("range(-40,44)", "[-20]")
source = source.replace("range(60,148)", "[80]")

(data / "generate_one.py").write_text(source)

subprocess.run(
    [sys.executable, "generate_one.py"],
    cwd=str(data),
    check=True,
)

images = list((data / "pendulum" / "train").glob("*.png"))
assert len(images) == 1, f"Expected one image, found {len(images)}"

print("Generated image:", images[0])
print("Preparation completed.")
PY
