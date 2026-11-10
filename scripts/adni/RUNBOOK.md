ADNI reproduction status and commands (2026-11-10, paper **arXiv 2509.24798v2**)

No GPU experiments or preprocessing jobs have been submitted. No quantitative reproduction has completed. Existing handoff/Pendulum work is preserved. Preparation outputs below contain local collection metadata; keep them on approved storage.

Small qualitative demonstration (current image-first workflow):

Submit `qualitative_adni.sbatch`; it calls `scripts/adni/qualitative.py`. Defaults: four distinct subjects, five ventricular-volume targets (.2–.8 normalized), two brain-volume targets (.25/.75) and two age targets (.25/.75). It uses the released notebook SCM head and reuses one factual inversion for all nine edits per subject. No benchmark flows, predictor checkpoints, unconditional evaluation VAE, metric cache, or training are needed. Clinical attributes and preprocessed float TIFFs are still required. The extra SCM output labels are saved rather than assumed to equal the targets for downstream variables.

```bash
# Set these to the ACTUAL prepared paths; <release-tag> is a placeholder.
export WORK=/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter
export REPO="$WORK/Causal-Adapter"
export ENV=/sc-projects/sc-proj-ukb-cvd/envs/causal-adapter
export ADNI_DATA_ROOT="<root-containing-preprocessed_data>"
export ADNIMERGE_CSV="<matching-clinical-ADNIMERGE.csv>"
export ADNI_BASE_MODEL="$WORK/models/adni-release-<release-tag>/miniSD-diffusers"
export ADNI_WEIGHTS="$WORK/models/adni-release-<release-tag>/causal-adapter-weights/ADNI"
cd "$REPO"
# Check the site's GPU type/features and add its actual A100 40GB selector:
sinfo -p gpu -N -o '%N %G %f'
sbatch qualitative_adni.sbatch
# Example ONLY if sinfo reports the GRES type exactly as a100:
# sbatch --gres=gpu:a100:1 qualitative_adni.sbatch
```

The allocation requests one GPU and one hour; the Python runner verifies one visible A100 with approximately 40 GB before loading diffusion weights. GPU type/constraint names have not been verified on the site. Set `ADNI_SUBJECTS=8` for eight subjects, `ADNI_LEVELS=7` for seven ventricular points, or `ADNI_SEED=...` for a different selection. Default split is `test`, using the existing sorted-subject boundaries. If only a small preprocessing demonstration root exists, explicitly set `ADNI_SPLIT=all` and treat outputs as qualitative examples, not held-out reproduction. The runner never silently reduces the requested subject count or alters clinical pairing. It records scan/clinical date mismatches and excludes out-of-range factual labels only when selecting these qualitative examples.

Optional CPU input validation after activating the existing environment:

```bash
python scripts/adni/qualitative.py --data-root "$ADNI_DATA_ROOT" --adnimerge-csv "$ADNIMERGE_CSV" --base-model "$ADNI_BASE_MODEL" --weights-root "$ADNI_WEIGHTS" --output-root "$WORK/results/adni/qualitative-preflight-$(date -u +%Y%m%dT%H%M%SZ)" --subjects 4 --split "${ADNI_SPLIT:-test}" --preflight-only
```

Outputs default to `$WORK/results/adni/qualitative-<job-id>/` (override with `ADNI_OUTPUT` pointing at a new directory). Open `index.html` locally after copying results from the HPC. Each `case_XX/` contains original/reconstruction/edited 256px PNGs, three intervention grids, grayscale bicubic 192px versions, signed difference PNGs against both original and reconstruction, full numeric differences/labels in `labels_and_differences.npz`, and source/target/propagated labels in `case.json`. Blue indicates decreased intensity, red increased intensity, white no change; display saturates at +/-64 of 255 but the NPZ retains full signed values. The manifest records checkpoint/image hashes, revisions, actual GPU/environment, seed, selected source cases and partial/completed status. Existing results are refused. SLURM logs are `adni-qualitative-<job-id>.out/.err` in the submission directory. No quantitative Table 3 claims are made.

Optional focused quantitative plan for one A100 40 GB:

Use the example config's `metrics: ["composition", "effectiveness"]`, `focused_samples: 512`, batch size 2 and seed 40 for smoke, pilot and focused runs. This samples slices across held-out subjects with a fixed seed and saves the exact original indices. It retains 100 DDIM steps and guidance 1.0. Outputs cover the 3×3 volume/slice effectiveness matrix, null-intervention MAE/LPIPS and the smoke's qualitative grids/traversals. FID, CLD, attention, training and baseline runs are omitted; no unconditional evaluation VAE is needed. This is a focused quantitative reproduction, not a complete Table 3 reproduction.

Budget approximately 10–20 minutes for smoke/pilot plus 3–5 GPU-hours for the 512-slice run; request a 6-hour allocation. The latter extrapolates 512 × (three interventions + one reconstruction) × the paper's 5–7 seconds per image, plus overhead. This is not an HPC measurement; batching and storage can change it. Queue time, downloads and CPU FSL preprocessing are excluded. Reserve exactly one GPU with `--gres=gpu:1`; select an A100 40 GB using the site's actual partition/constraint syntax (not yet verified).

After preparing the inputs, complete smoke and pilot with the same config, then use:

```bash
# ADNI_SMOKE and ADNI_PILOT point to successful manifests from those stages.
export ADNI_OUTPUT="$WORK/results/adni/focused-$RUN_TAG"
MODE=focused sbatch --time=06:00:00 --gres=gpu:1 --export=ALL reproduce_adni.sbatch --validated-smoke "$ADNI_SMOKE" --validated-pilot "$ADNI_PILOT"
# Use the same aggregate.py command below, pointing at this output directory.
```

The full-metric workflow below remains available only if needed: set config `metrics` to `["composition", "effectiveness", "fid", "minimality"]` and supply the unconditional VAE checkpoint. Do not use that profile for the focused run. Full preprocessing still refuses missing selected scans, and the clinical export plus benchmark flows/image-parent predictors are still required.

Confirmed inventory:

- Checkout: `a9752e9a6ba57456eb3edbebc695cec7d94f1342`, plus this uncommitted patch.
- Local input: `/home/benedikt/Desktop/Charite/Datasets/ADNI`. Five extensionless ZIP containers hold 1,724 NIfTI members. The newly supplied metadata contains 4,588 XML files representing 2,294 processed image IDs; 570 metadata image IDs have no corresponding NIfTI in the inspected downloads. No extracted `.nii` files or preprocessed TIFFs were visible in the last scan. ZIP input is supported directly by the prepared preprocessor.
- XML conversion succeeded for every available image ID and preserves all 2,294 metadata records. Exact upstream preselection over the full metadata retained **1,916 scans from 639 subjects**; **1,424 selected scans are available and 492 selected scans are missing**. Primary outputs: `outputs/adni/preparation-local-20261110/collection_metadata-full.csv`, `preselection-full/final_image_ids.csv`, `preselection-full/summary.json`, and the upstream log. Earlier available-only exploratory outputs remain in `preselection/`; do not use those for full reproduction. These are preparation results, not model smoke tests.
- For the complete metadata population, the unchanged sorted-subject 70/15/15 split predicts train 447 subjects / 1,340 scans, valid 95 / 285, test 95 / 285 before FSL/clinical exclusions; two subjects fall outside the floored split sizes. Only 1,005 / 209 / 205 of those scans are currently available. The earlier available-only subset yields 631 subjects and upper bounds of 10,020 train / 2,130 valid / 2,040 test slices, which cannot establish the paper's 10,780 / 0 / 2,240. Missing scans also remove eight subjects entirely and therefore shift the sorted-subject boundaries. Neither the population nor split was changed to obtain expected counts. Final counts require the complete sources, preprocessing and the clinical join.
- `ADNIMERGE2.tar.gz` and the data dictionary are present, but no `ADNIMERGE*.csv`. The extracted `ADSL` object has only `AGE` among the eight required legacy clinical columns. XML collection metadata does not replace that clinical table. The original downloader `.csv` is a URL list.
- Base Python 3.13 has no ML packages. The existing `causal_scm` Python 3.11 has PyTorch 2.12.1 **CPU**, NumPy/pandas/Pillow/torchvision; it lacks diffusers, transformers, accelerate, normflows, torchmetrics and Lightning. It was used only for synthetic CPU checks and metadata selection. Nothing was reinstalled.
- No local GPU or SLURM commands; recorded `/sc-projects` paths are not mounted. Existing `charite-hpc` SSH reaches the frontend when network access is available, but authentication fails. No cluster inventory or submission is claimed.

Verified public sources:

- [Paper v2, Table 3 and Appendix C](https://arxiv.org/html/2509.24798v2): 100 DDIM steps, inversion/forward guidance 1.0, 256px generation, bicubic 192px grayscale evaluation. Quantitative interventions are brain volume, ventricular volume and slice. Composition is null intervention/reconstruction; reverse editing is a separate diagnostic.
- [Released ADNI files](https://huggingface.co/LeiTong/Causal-Adapter/tree/e868925fbcb0bf7d0979c8bf2a4d1b8be400f43c/ADNI): `controlnet/controlnet-steps-100000.safetensors/{config.json,diffusion_pytorch_model.safetensors}`, matching `learned_embeds-steps-100000.safetensors`, optional `scm/best_model.pt`. The first `.safetensors` path is a **directory**. Its inspected config declares `Causal_ControlNetModel`, `dataset=ADNI`, six causal concepts and `task_cond=generation_text_global_after`. The release does not contain the benchmark flow/predictor/unconditional-VAE checkpoints.
- Backbone: `lambda/miniSD-diffusers`, revision `26ed8a9bfbf76f46a6cf60517dde321f900c44ce` (the historical `lambdalabs` ID redirects).
- [Benchmark revision](https://github.com/gulnazaki/counterfactual-benchmark/tree/581dbb495421ca863c4a6e62ad759ac79de076e9) and [preprocessing README](https://github.com/gulnazaki/counterfactual-benchmark/blob/581dbb495421ca863c4a6e62ad759ac79de076e9/counterfactual_benchmark/datasets/adni/preprocessing/README.md). The actual upstream ADNI checkpoint tree currently contains only age/apoE/sex predictors. Required brain_vol/vent_vol/slice predictors and both volume flows are absent. Obtain the authors' matching evaluation artifacts; do not substitute the adapter SCM head. The benchmark README links a [separate unconditional-VAE folder](https://drive.google.com/drive/folders/1SAak8-S3HOCmPyk8JFojAjV-xv2spSlY); no specific downloaded VAE identity is yet confirmed.

Preparation commands (proposed cluster destinations, not confirmed existing files):

```bash
export WORK=/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter
export REPO="$WORK/Causal-Adapter"
export ENV=/sc-projects/sc-proj-ukb-cvd/envs/causal-adapter
export ADNI_INPUT="$WORK/data/ADNI"  # user-designated future input location
export RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export PREP="$WORK/preparation/adni-$RUN_TAG"
export TORCH_HOME="$WORK/.cache/torch-adni"
cd "$REPO"
mkdir -p "$PREP"
# Activate the existing ENV and inspect versions before installing anything.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
python -c 'import torch,diffusers; print(torch.__version__, torch.version.cuda, diffusers.__file__)'
# Only if needed, retain the project fork and add the missing benchmark packages:
python -m pip install -e diffusers
python -m pip install normflows==1.7.3 torch-fidelity==0.3.0
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python scripts/adni/prepare_public.py --model-root "$WORK/models/adni-release-$RUN_TAG"
# Or --upstream-only with a NEW model-root to fetch just the preprocessing source.
python scripts/adni/prepare_metric_cache.py
python scripts/adni/inventory_raw.py --input-root "$ADNI_INPUT" --output "$PREP/raw_inventory.json"
python scripts/adni/metadata_from_xml.py --input-root "$ADNI_INPUT" --output "$PREP/collection_metadata.csv"
export ADNI_UPSTREAM_PREPROCESSING="$WORK/models/adni-release-$RUN_TAG/benchmark_preprocessing"
mkdir "$PREP/preselection"
(cd "$PREP/preselection" && python "$ADNI_UPSTREAM_PREPROCESSING/image_preselection.py" --csv "$PREP/collection_metadata.csv" > selection.log)
export ADNI_SELECTED_IDS="$PREP/preselection/final_image_ids.csv"
```

Preprocessing requires an existing FSL installation (`FSLDIR`, fslreorient2std/robustfov/bet/flirt and MNI152 1mm brain atlas), SimpleITK, pandas, NumPy and Pillow. Inspect/install these in the compute environment separately; the GPU wrapper does not install software. Upstream's move script relocates originals; the prepared wrapper instead reads original NIfTI files/ZIP members into temporary storage and preserves them. It reuses pinned upstream normalization/cropping/TIFF writers and uses checked subprocess calls for the same FSL sequence. **FAST is commented out in upstream code**, despite the README listing it. The wrapper follows code and records that discrepancy. No raw MRI data is downloaded or transferred by these scripts. Full preprocessing refuses missing selected scans; a capped smoke run may process an available selected scan while clearly recording its partial scope.

```bash
# Initialize the site's EXISTING FSL installation before submission.
export ADNI_PREPROCESSING_OUTPUT="$ADNI_INPUT/preprocessing-smoke-$RUN_TAG"
sbatch prepare_adni.sbatch --max-scans 1
# After inspecting its 30 finite 180x180 TIFF slices, run into a NEW root:
export ADNI_PREPROCESSING_OUTPUT="$ADNI_INPUT/preprocessing-$RUN_TAG"
sbatch prepare_adni.sbatch
# Specify the site's CPU partition with sbatch --partition=... if required.
```

A compatible clinical CSV is still required: `PTID,VISCODE,EXAMDATE,APOE4,AGE,PTGENDER,WholeBrain,Ventricles`. Obtain the matching ADNIMERGE export through approved ADNI access, or supply a documented derivation from ADNIMERGE2. Do not rename ADSL or the data dictionary to ADNIMERGE. Set `adnimerge_csv` explicitly when multiple exports exist. The loader retains positional visit joins; the audit reports mismatching dates, missing/null visits, duplicate exam dates and dropped slice counts without repairing pairings. Raw and processed slice IDs remain subject to the supplied collection population.

Configure a personal JSON copied from `scripts/adni/config.example.json`. Use the completed preprocessing root containing `preprocessed_data/`, the explicit clinical CSV, actual base/ControlNet/embedding paths, a curated directory containing exactly one checkpoint for each `brain_vol_flow-epoch*` and `vent_vol_flow-epoch*`, predictor checkpoints starting `brain_vol`, `vent_vol`, `slice`, and the unconditional VAE file. Set seed 40, batch size 2 (reduce for memory if necessary), workers 0. The configuration file must contain real paths, not the example placeholders. Changing batch size requires rerunning validation stages.

```bash
export ADNI_CONFIG="$WORK/adni-config.json"
python scripts/adni/run.py --config "$ADNI_CONFIG" --mode preflight --output-root "$WORK/results/adni/preflight-$RUN_TAG"
export ADNI_OUTPUT="$WORK/results/adni/smoke-$RUN_TAG"
MODE=smoke sbatch --export=ALL reproduce_adni.sbatch
# Wait for smoke manifest status=smoke_completed and inspect grids/traversals/log.
export ADNI_SMOKE="$ADNI_OUTPUT/manifest.json"
export ADNI_OUTPUT="$WORK/results/adni/pilot-$RUN_TAG"
MODE=pilot sbatch --export=ALL reproduce_adni.sbatch --validated-smoke "$ADNI_SMOKE"
# Pilot explicitly adds --max-batches 1; inspect its metrics before full evaluation.
export ADNI_PILOT="$ADNI_OUTPUT/manifest.json"
export ADNI_OUTPUT="$WORK/results/adni/full-$RUN_TAG"
MODE=full sbatch --export=ALL reproduce_adni.sbatch --validated-smoke "$ADNI_SMOKE" --validated-pilot "$ADNI_PILOT"
squeue -u "$USER"
sacct -j <actual-job-id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
# SLURM logs: submission directory/adni-<actual-job-id>.out and .err
# Evaluator log: $ADNI_OUTPUT/evaluation.log
python scripts/adni/aggregate.py --run-dir "$ADNI_OUTPUT/ADNI/DSCM_effectiveness_step100.0_scale1.0_NTIFalse_seed40" --output-dir "$WORK/results/adni/table3-$RUN_TAG"
```

The runner prints the full evaluator command before execution: `evaluate_SD_DSCM.py --dataset ADNI --editing standard --NTI False --num_steps 100 --guidance_scale 1.0 --resolution 256 --seed 40 --bs 2 --metrics composition effectiveness fid minimality --embeddings lpips --cycles 1`, plus explicit inputs/outputs and intervention subset. The CLI key is uppercase `ADNI`; configs/dataset name are lowercase `adni`. In this implementation `NTI=True` routes standard editing through PnP, not null-text optimization. Therefore it is false. The shared helper's padded `assets.prompt` is used exactly once; the actual pseudo-token multiplicities must be `[1,1,10]`. Encoded labels retain `apoE:2,age:1,sex:1,brain_vol:1,vent_vol:1,slice:10`; slice classes are **0–9 thermometer** codes, not one-hot.

Every run refuses existing output directories and records arguments, revision/patch/source hashes, checkpoint/image hashes, clinical CSV hash, split counts, seed, package versions, GPU/import locations, SLURM job ID when present, and completion/failure status. Pilot/full require successful validation manifests with matching inputs and code. The smoke stage reconstructs factuals, edits all three image parents with benchmark flows, loads every metric network, saves original/reconstruction/CF/absolute-difference grids, labels and min/middle/max traversals. Even smoke/pilot use the full test reference for corrected CLD and the full train reference for FID; they remain explicitly labeled partial experiments.

Metric interpretation and unresolved equivalence:

- Volumes: sigmoid of regressor logits, normalized MAE (not physical mm³). Slice: argmax logits vs thermometer sum, micro multiclass F1, matching the benchmark's default TorchMetrics reduction. Aggregation reports global values and descriptive mean/SD over fixed 256-image chunks including the last partial chunk. The paper's exact SD grouping and F1 averaging are unspecified; equivalence is unconfirmed. Inference batch size does not define uncertainty chunks.
- Composition: one null-intervention reconstruction, per-image absolute pixel MAE on a 0–255 scale and per-image AlexNet LPIPS. No unsigned subtraction, signed-int8 wraparound or dropped final batch. Reverse/identity distances are not substituted for composition.
- FID: TorchMetrics Inception-2048, full training images vs one uniformly selected image-parent CF per held-out sample, matching `compute_FID.py`'s real split. Generated grayscale tensors are already `[0,1]`; repeat to RGB without postprocessing them again. Real images follow the checked-out 256→192 bicubic path. FID population choice is recorded; paper v2 does not specify that split. One generated image yields a missing FID, never zero.
- CLD: separate unconditional VAE with both posterior mean and log variance, W1=.8/W2=.2 reduction, normalized continuous labels and full held-out reference, using train-derived bins. The legacy script reuses mean as log scale, truncates continuous labels to integers and mixes labels belonging to different intervened variables. Prepared results are explicitly **cld_corrected**; a match to published .346 cannot be asserted until the authors' precise metric implementation is resolved. Skipped classes and sample counts are saved.
- Intervention draws use independent per-index/parent seeded streams, uniformly choosing distinct held-out values outside the factual train-derived volume bin (distinct thermometer vectors for slice). This corrects a boundary inconsistency in the old digitize/searchsorted comparison. Minimality/FID parent choice is per sample, rather than the old random-per-batch choice. These changes are recorded deviations; DDIM itself resets Torch's seed internally. Numerical batching differences may remain on CUDA, and deterministic unsupported operations fail rather than silently varying.

Attention is kept separate and **disabled**. `find_update_token_index` correctly addresses ADNI's last 12 encoded values and maps `@`, `*`, and all ten `&` positions; compatibility with the actual checkpoint tokenizer still needs a GPU smoke run. The evaluator's P2P fallback schedule has zero cross/self replacement with unverified blend thresholds; its `NTI` flag controls local blend in this branch. LocalBlend divides by attention maxima without a zero guard and assumes particular 16×16 layer maps. These need runtime validation and an author-confirmed ADNI schedule before enabling. `--mode attention` records a blocked manifest instead of silently using CelebA settings; no attention run was submitted. Table 3's attention row omits composition, and it must not be filled with zero.

Training remains a separate stage. Do not run either existing ADNI training wrapper unchanged. `train_scm_adni.sh` has duplicated `--data-root` and an unquoted placeholder. Its `SCM_modeling/train_scm.py` trains the adapter's CausalNet, **not** the benchmark normalizing flows. `train_adni.sh` uses effective batch 2 rather than 40. The checked-out flow config says `layers=3` but the implementation reads `n_layers`, defaulting to 10; released flow topology must match the checkpoint. No causal discovery is required.

If the authors cannot provide the two flows, the explicit fallback command is prepared (not launched, not claimed paper-matched):

```bash
# Run only within a separately scoped SLURM job after agreeing to retraining.
python scripts/adni/train_flows.py --data-root "$ADNI_PREPROCESSING_OUTPUT" --adnimerge-csv <clinical-export.csv> --output-dir "$WORK/models/adni-flows-$RUN_TAG" --seed 42
```

For an independently agreed adapter retraining stage, the reviewed candidate settings are single GPU, batch 2 × accumulation 20 = effective batch 40, AdamW 1e-5/weight decay .01, MSE+CTC temperature .2/scale .0005, known six-variable adjacency, seed 40, 100k steps, explicit pretrained SCM head, embeddings+ControlNet saves and full-state checkpoints. Accumulation changes CTC's per-microbatch negatives, so it is not equivalent to a true batch of 40. `train.py` creates a fresh timestamped output directory before resolving resume checkpoints; robust resume into an existing run needs a separate fix before claiming resumable training. No training was launched.

```bash
python train.py --dataset ADNI --pretrained_model_name_or_path <local-miniSD> --train_data_dir <preprocessing-root> --scm_path <released-ADNI-scm/best_model.pt> --output_dir <new-training-root> --output_name adni-paper-candidate --resolution 256 --train_batch_size 2 --gradient_accumulation_steps 20 --max_train_steps 100000 --learning_rate 1e-5 --adam_weight_decay .01 --infonce_temperature .2 --infonce_scale .0005 --seed 40 --save_steps 1000 --checkpointing_steps 1000 --placeholder_string 'a mri image of @ and * and &' --presudo_words '@,*,&' --presudo_words_infonce '@,*,&' --mcpl_training True --causal_training False --task_cond generation_text_global_after
```

Validation performed locally: 17 synthetic CPU regression tests (loader/joins/codecs, split counts, XML/archive mapping, DDIM call boundary, deterministic sampling, Gaussian KL, partial-batch aggregation, coverage rejection, saved grids), shell syntax and Python compilation. Real model/predictor loading, FSL output, CUDA kernels, complete metrics and attention remain untested because their prerequisites are unavailable.
