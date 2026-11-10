#!/bin/bash -l

#SBATCH --job-name=pendulum-notebook
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00

#SBATCH --output=/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter/notebook-%j.out
#SBATCH --error=/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter/notebook-%j.err

set -euo pipefail

# ============================================================
# 1. Paths
# ============================================================

export WORK="/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter"

REPO="$WORK/Causal-Adapter"
ENV="/sc-projects/sc-proj-ukb-cvd/envs/causal-adapter"
OUT="$WORK/results/notebooks/pendulum"

export REPO OUT

mkdir -p "$OUT"

# ============================================================
# 2. Environment
# ============================================================

export PYTHONPATH="$REPO:$REPO/notebook_benchmarks${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="$WORK/.cache/huggingface"
export MPLCONFIGDIR="$WORK/.cache/matplotlib"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$MPLCONFIGDIR"

# Fix MKL activation issue
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
set -u

# ============================================================
# 3. Job information
# ============================================================

echo "=========================================="
echo "Causal-Adapter Pendulum Notebook"
echo "=========================================="

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Repository: $REPO"
echo "Output: $OUT"

echo ""
echo "Python:"
which python
python --version

echo ""
echo "GPU:"
nvidia-smi

python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA unavailable"

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

# ============================================================
# 4. Register Jupyter kernel
# ============================================================

python -m ipykernel install \
    --prefix "$ENV" \
    --name causal-adapter-hpc \
    --display-name "Causal Adapter HPC"

# ============================================================
# 5. Execute notebook
# ============================================================

python - <<'PY'

import os
import re
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

work = Path(os.environ["WORK"])
repo = Path(os.environ["REPO"])
out = Path(os.environ["OUT"])

notebook_path = (
    repo
    / "notebook_benchmarks"
    / "counterfactuals_pendulum.ipynb"
)

output_notebook = (
    out / "counterfactuals_pendulum_executed.ipynb"
)

# ------------------------------------------------------------
# Load notebook
# ------------------------------------------------------------

print("\nLoading notebook:", notebook_path, flush=True)

nb = nbformat.read(notebook_path, as_version=4)

print("Original cells:", len(nb.cells), flush=True)

# ------------------------------------------------------------
# Configure model and data paths
# ------------------------------------------------------------

weights = (
    work
    / "models"
    / "causal-adapter-weights"
    / "pendulum"
)

paths = {
    "BASE_MODEL_PATH":
        work / "models" / "miniSD-diffusers",

    "CONTROLNET_PATH":
        weights / "controlnet" /
        "controlnet-steps-20000.safetensors",

    "TEXT_EMBEDDING_PATH":
        weights / "controlnet" /
        "learned_embeds-steps-20000.safetensors",

    "SCM_PATH":
        weights / "scm" / "best_model.pt",

    "DATA_ROOT":
        work / "data" / "pendulum_quick" /
        "pendulum" / "train",
}

print("\nChecking required paths...", flush=True)

for name, path in paths.items():
    assert path.exists(), f"Missing {name}: {path}"
    print(f"OK: {name} = {path}", flush=True)

assert list(paths["DATA_ROOT"].glob("*.png")), (
    "No Pendulum PNG images found."
)

# ------------------------------------------------------------
# Configure notebook
# ------------------------------------------------------------

cells = []
configuration_found = False
grid_found = False

for cell in nb.cells:

    # Skip optional attention-map section
    if (
        cell.cell_type == "markdown"
        and "## 6. Cross-attention maps" in cell.source
    ):
        print(
            "\nSkipping optional attention-map section.",
            flush=True,
        )
        break

    if cell.cell_type == "code":

        src = cell.source

        # IMPORTANT:
        # Match an actual variable assignment, not merely
        # a reference to BASE_MODEL_PATH elsewhere.
        if re.search(
            r"(?m)^[ \t]*BASE_MODEL_PATH[ \t]*=",
            src,
        ):

            if configuration_found:
                raise RuntimeError(
                    "Multiple configuration cells found."
                )

            print(
                "\nConfiguring notebook paths...",
                flush=True,
            )

            for name, path in paths.items():

                pattern = (
                    rf"(?m)^[ \t]*{re.escape(name)}"
                    rf"[ \t]*=[^\n]*"
                )

                replacement = (
                    f"{name} = {str(path)!r}"
                )

                src, count = re.subn(
                    pattern,
                    lambda match, value=replacement: value,
                    src,
                    count=1,
                )

                if count != 1:
                    raise RuntimeError(
                        f"Cannot configure {name}"
                    )

                print(
                    f"Configured: {name}",
                    flush=True,
                )

            configuration_found = True

        # Redirect image grid to results directory
        if "save_images_grid_box(image_lists" in src:

            old = "'./intervention_pendulum.png'"

            new = repr(
                str(out / "intervention_pendulum.png")
            )

            if old not in src:
                raise RuntimeError(
                    "Cannot find intervention grid output path."
                )

            src = src.replace(old, new)
            grid_found = True

        cell.source = src

    cells.append(cell)

if not configuration_found:
    raise RuntimeError(
        "Notebook configuration cell not found."
    )

if not grid_found:
    raise RuntimeError(
        "Intervention grid cell not found."
    )

nb.cells = cells

print(
    "\nNotebook configured successfully.",
    flush=True,
)
print(
    "Cells to execute:",
    len(nb.cells),
    flush=True,
)

# ------------------------------------------------------------
# Save progress incrementally
# ------------------------------------------------------------

def save_notebook():

    # Write to a temporary file before replacing the
    # previous checkpoint to reduce corruption risk.
    temporary = out / "pendulum_checkpoint.tmp.ipynb"

    nbformat.write(nb, temporary)

    os.replace(temporary, output_notebook)


def on_cell_start(cell, cell_index, **kwargs):

    if cell.cell_type != "code":
        return

    source_lines = cell.source.strip().splitlines()
    first_line = source_lines[0] if source_lines else ""

    print(
        f"\n[START] Cell {cell_index}: {first_line[:100]}",
        flush=True,
    )


def on_cell_executed(cell, cell_index, **kwargs):

    print(
        f"[DONE] Cell {cell_index}",
        flush=True,
    )

    # Save results after every executed cell
    save_notebook()

    print(
        f"[SAVED] {output_notebook}",
        flush=True,
    )


# Save the configured notebook before execution
save_notebook()

print(
    "\nInitial notebook saved:",
    output_notebook,
    flush=True,
)

# ------------------------------------------------------------
# Execute notebook
# ------------------------------------------------------------

client = NotebookClient(
    nb,
    timeout=7200,
    kernel_name="causal-adapter-hpc",
    resources={
        "metadata": {
            "path": str(
                repo / "notebook_benchmarks"
            )
        }
    },
    on_cell_start=on_cell_start,
    on_cell_executed=on_cell_executed,
)

print(
    "\nStarting notebook execution...",
    flush=True,
)

try:

    client.execute()

    print(
        "\nNotebook completed successfully!",
        flush=True,
    )

except Exception:

    print(
        "\nERROR: Notebook execution failed!",
        flush=True,
    )

    traceback.print_exc()

    raise

finally:

    # Preserve whatever outputs were produced
    save_notebook()

    print(
        "\nLatest notebook saved:",
        output_notebook,
        flush=True,
    )

PY

# ============================================================
# 6. Final output verification
# ============================================================

echo ""
echo "=========================================="
echo "Notebook job completed"
echo "=========================================="

echo "Finished: $(date)"

echo ""
echo "Output files:"
ls -lh "$OUT"