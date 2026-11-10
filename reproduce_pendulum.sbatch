#!/bin/bash -l

#SBATCH --job-name=pendulum-reproduce
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00

#SBATCH --output=/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter/inference-%j.out
#SBATCH --error=/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter/inference-%j.err

set -euo pipefail

# ============================================================
# 1. Paths
# ============================================================

export WORK="/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter"

REPO="$WORK/Causal-Adapter"
ENV="/sc-projects/sc-proj-ukb-cvd/envs/causal-adapter"

export MODEL_ROOT="$WORK/models"
export DATA_ROOT="$WORK/data/pendulum_quick/pendulum/train"
export OUTPUT_DIR="$WORK/results/pendulum"

# ============================================================
# 2. Environment configuration
# ============================================================

export HF_HOME="$WORK/.cache/huggingface"
export MPLCONFIGDIR="$WORK/.cache/matplotlib"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Required for causal_modules imports
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$OUTPUT_DIR"

# Initialize Conda
# Disable undefined-variable checking during activation
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
set -u

echo "=========================================="
echo "Pendulum Counterfactual Reproduction"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Repository: $REPO"
echo "Environment: $ENV"
echo "Output: $OUTPUT_DIR"

echo ""
echo "GPU information:"
nvidia-smi

echo ""
echo "Python:"
which python
python --version

# ============================================================
# 3. Run inference
# ============================================================

cd "$REPO/notebook_benchmarks"

python - <<'PY'

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from causal_modules.ddim_modules import (
    sample,
    ddim_editing,
)

from inference_utils import (
    build_transforms,
    load_causal_adapter,
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

WORK = Path(os.environ["WORK"])
MODEL_ROOT = Path(os.environ["MODEL_ROOT"])
DATA_ROOT = Path(os.environ["DATA_ROOT"])
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])

BASE_MODEL_PATH = (
    MODEL_ROOT / "miniSD-diffusers"
)

WEIGHTS = (
    MODEL_ROOT
    / "causal-adapter-weights"
    / "pendulum"
)

CONTROLNET_PATH = (
    WEIGHTS
    / "controlnet"
    / "controlnet-steps-20000.safetensors"
)

TEXT_EMBEDDING_PATH = (
    WEIGHTS
    / "controlnet"
    / "learned_embeds-steps-20000.safetensors"
)

SCM_PATH = (
    WEIGHTS
    / "scm"
    / "best_model.pt"
)

DATASET = "pendulum"
SIZE = 256
NUM_STEPS = 50

# ------------------------------------------------------------
# Check required files
# ------------------------------------------------------------

print("\nChecking required files...", flush=True)

assert (BASE_MODEL_PATH / "model_index.json").is_file(), (
    f"Missing base model: {BASE_MODEL_PATH}"
)

assert CONTROLNET_PATH.is_dir(), (
    f"Missing ControlNet: {CONTROLNET_PATH}"
)

assert TEXT_EMBEDDING_PATH.is_file(), (
    f"Missing embeddings: {TEXT_EMBEDDING_PATH}"
)

assert SCM_PATH.is_file(), (
    f"Missing SCM: {SCM_PATH}"
)

images = sorted(DATA_ROOT.glob("*.png"))

assert len(images) > 0, (
    f"No Pendulum images found in {DATA_ROOT}"
)

print("All required files found.", flush=True)

# ------------------------------------------------------------
# GPU verification
# ------------------------------------------------------------

print("\nChecking GPU...", flush=True)

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), (
    "CUDA is not available. Check GPU allocation."
)

device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))

# Test actual GPU computation
x = torch.randn(1000, 1000, device=device)
y = x @ x
torch.cuda.synchronize()

print("GPU computation successful.", flush=True)

del x, y

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading Causal-Adapter...", flush=True)

image_transforms, original_transforms, _ = (
    build_transforms(DATASET, size=SIZE)
)

assets = load_causal_adapter(
    DATASET,
    base_model_path=str(BASE_MODEL_PATH),
    controlnet_path=str(CONTROLNET_PATH),
    text_embedding_path=str(TEXT_EMBEDDING_PATH),
    scm_path=str(SCM_PATH),
)

pipe = assets.pipe
device = assets.device
prompt = assets.prompt
presudo_token_ids = assets.presudo_token_ids

print("Model loaded successfully.", flush=True)

# ------------------------------------------------------------
# Load one Pendulum image
# ------------------------------------------------------------

img_path = images[0]

print("\nInput image:", img_path, flush=True)

# Extract attributes from filename
label_values = list(
    map(float, img_path.stem.split("_")[1:])
)

assert len(label_values) == 4

# Same normalization as authors' notebook
scale = np.array([
    [2, 42],
    [104, 44],
    [7.5, 4.5],
    [11, 8],
])

norm_label = (
    (np.array(label_values) - scale[:, 0])
    / scale[:, 1]
).astype(np.float32)

label = (
    torch.tensor(norm_label)
    .unsqueeze(0)
    .to(device)
    .float()
)

img = Image.open(img_path).convert("RGB")

original_img = original_transforms(img.copy())
image_t = image_transforms(img)

original_img.save(OUTPUT_DIR / "original.png")

print("Raw attributes:", label_values)
print("Normalized attributes:", norm_label)

# ------------------------------------------------------------
# DDIM inversion + reconstruction
# ------------------------------------------------------------

print("\nStarting DDIM inversion...", flush=True)

final_im, inverted_latents, _, uncond_embeddings = (
    ddim_editing(
        pipe,
        image_t.unsqueeze(0),
        label.clone(),
        presudo_token_ids,
        prompt,
        num_steps=NUM_STEPS,
        invert_guidance_scale=1.0,
        set_guidance_scale=1.0,
        intervention_indx=None,
        intervention_values=None,
        return_PIL=True,
    )
)

final_im[0].save(
    OUTPUT_DIR / "reconstruction.png"
)

print("DDIM reconstruction completed.", flush=True)

# ------------------------------------------------------------
# Counterfactual interventions
# ------------------------------------------------------------

print("\nStarting counterfactual generation...", flush=True)

RANGE_LEN = 8

# Same intervention ranges as the paper's notebook
inter_value = [
    np.linspace(-1.0, 1.0, num=RANGE_LEN),
    np.linspace(-0.5, 0.5, num=RANGE_LEN),
    np.linspace(-0.5, 0.5, num=RANGE_LEN),
    np.linspace(-0.5, 0.5, num=RANGE_LEN),
]

attribute_names = [
    "pendulum",
    "light",
    "shadow_length",
    "shadow_position",
]

image_lists = []

individual_dir = OUTPUT_DIR / "individual"
individual_dir.mkdir(parents=True, exist_ok=True)

for inter_id in range(4):

    print(
        f"\nIntervention: {attribute_names[inter_id]}",
        flush=True,
    )

    row_images = []

    for i in range(RANGE_LEN):

        value = float(inter_value[inter_id][i])

        print(
            f"  Sample {i+1}/{RANGE_LEN}, value={value:.3f}",
            flush=True,
        )

        interved_image, _ = sample(
            pipe,
            prompt,
            presudo_token_ids,
            start_step=0,
            start_latents=inverted_latents[-1].clone(),
            guidance_scale=1.0,
            num_inference_steps=NUM_STEPS,
            num_images_per_prompt=1,
            negative_prompt=None,
            device=device,
            controlnet_image=None,
            intervention_indx=inter_id,
            intervention_values=value,
            label=label.clone(),
            return_PIL=True,
            disentangle=False,
            uncond_embeddings=uncond_embeddings,
        )

        result = interved_image[0].convert("RGB")

        # Save each counterfactual separately
        filename = (
            f"{attribute_names[inter_id]}_{i:02d}.png"
        )

        result.save(individual_dir / filename)

        row_images.append(result)

    image_lists.append(row_images)

# ------------------------------------------------------------
# Create 4 x 8 intervention grid
# ------------------------------------------------------------

print("\nCreating intervention grid...", flush=True)

rows = 4
cols = RANGE_LEN
border = 1

width, height = image_lists[0][0].size

grid = Image.new(
    "RGB",
    (
        cols * (width + border) + border,
        rows * (height + border) + border,
    ),
    color="black",
)

for r, row_images in enumerate(image_lists):
    for c, image in enumerate(row_images):

        x = border + c * (width + border)
        y = border + r * (height + border)

        grid.paste(image, (x, y))

grid.save(
    OUTPUT_DIR / "intervention_pendulum.png"
)

print("\n==========================================")
print("Inference completed successfully!")
print("==========================================")

print("Original image:")
print(OUTPUT_DIR / "original.png")

print("Reconstruction:")
print(OUTPUT_DIR / "reconstruction.png")

print("Counterfactual grid:")
print(OUTPUT_DIR / "intervention_pendulum.png")

print("Individual counterfactuals:")
print(individual_dir)

PY

echo ""
echo "=========================================="
echo "SLURM job finished: $(date)"
echo "=========================================="

ls -lh "$OUTPUT_DIR"