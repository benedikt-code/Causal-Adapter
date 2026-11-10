# Prompt for a new chat

Help me reproduce the ADNI experiments from **Causal-Adapter: Taming Text-to-Image Diffusion for Faithful Counterfactual Generation** in this repository. Inspect the implementation, prepare reproducible scripts, fix necessary ADNI execution issues, and run the experiments where the required data and compute are available. Do not stop at giving general instructions.

## Scope and working rules

- Start with the released pretrained ADNI adapter and evaluation checkpoints to establish a working reproduction. Training from scratch is a separate stage; do not launch a long training run merely because a path is unset. If retraining is necessary, explain the missing artifact and prepare the training command before asking about the scope change.
- Reproduce the quantitative ADNI evaluation and representative qualitative interventions. A successful notebook or a few images alone do not constitute a quantitative reproduction.
- Follow the repository's AGENTS.md instructions. Keep communication concise, modify only relevant files, preserve existing work, and run the smallest relevant checks. Do not commit, push, or overwrite results unless requested.
- Inspect available files and configuration before asking me for paths. If necessary data, access, or compute is unavailable, finish the independent preparation and ask for the specific missing prerequisite. Never invent paths, checkpoints, results, or completed runs.

## Workspace and cluster context

Local repository:

```text
/home/benedikt/Desktop/Charite/Sturopx/group2/causal-adapter-code
```

Existing Pendulum SLURM scripts record these cluster paths. Treat them as leads to verify, not proof of current access or successful prior runs:

```text
WORK=/sc-projects/sc-proj-ukb-cvd/projects/bepe10/sturopx/causal-adapter
REPO=$WORK/Causal-Adapter
ENV=/sc-projects/sc-proj-ukb-cvd/envs/causal-adapter
MODEL_ROOT=$WORK/models
```

Read `reproduce_pendulum.sbatch`, `run_pendulum_notebook.sbatch`, and, only as needed, `install_env.sbatch` / `finish_env.sbatch` for cluster conventions. Add dedicated ADNI scripts and output directories. Do not assume this chat has a cluster connection or GPU. Use SLURM for cluster GPU work; do not run heavy workloads on a login node. If execution is unavailable, supply exact commands, required inputs, and expected output locations without claiming submission or completion.

## Sources and reproduction target

Read the relevant paper sections and reconcile them with the checked-out code:

- Paper inspected for this handoff: https://arxiv.org/html/2509.24798v2 — Table 3, Appendix C, and ADNI qualitative results. Record the paper version used.
- Released weights: https://huggingface.co/LeiTong/Causal-Adapter — inspect its actual ADNI files; do not guess filenames or checkpoint steps.
- Upstream benchmark and data preparation: https://github.com/gulnazaki/counterfactual-benchmark

The paper's primary ADNI evaluation intervenes on `brain_vol`, `vent_vol`, and `slice`, measuring the two volumes with MAE and slice with F1 under each intervention. It also reports composition MAE/LPIPS, realism FID, and minimality CLD, with separate results using attention guidance. Appendix C specifies 100 DDIM steps, inversion and forward guidance 1.0, generation at 256×256, and bicubic downsampling to the benchmark's 192×192 grayscale images. It reports 10,780 training and 2,240 test images. Verify the evaluation population and explain discrepancies.

Keep the main standard-DDIM reproduction distinct from the attention-guidance variant. Audit ADNI-specific attention/token handling before enabling that variant. Treat reverse evaluation as an additional diagnostic, not a substitute for the paper's composition metric.

## Relevant implementation

Start with these files; follow imports only when needed:

```text
README.md
environment.yml
causal_datasets/adni_dataset.py
causal_datasets/_adapters/adni.py
causal_datasets/_normalization.py
causal_datasets/_constants.py
notebook_benchmarks/counterfactuals_ADNI.ipynb
notebook_benchmarks/inference_utils.py
counterfactual-benchmark/README.md
counterfactual-benchmark/counterfactual_benchmark/ctf_datasets/adni/dataset_SD.py
counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/evaluate_SD_DSCM.py
counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/adni/vae.json
counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/adni/classifier.json
counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/compute_FID.py
counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/compute_final_metrics/README.md
counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/compute_final_metrics/metrics_utils.py
counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/compute_final_metrics/compute_final_metrics.ipynb
```

Use `evaluate_SD_DSCM.py` for quantitative ADNI evaluation. Its Deep-SCM normalizing-flow mechanisms differ from the notebook's adapter SCM head: do not silently replace one with the other. The benchmark's lowercase directory/config name `adni` and evaluator CLI key `ADNI` are distinct.

## Known details and issues to verify

1. **Data layout:** the preprocessing root must contain `preprocessed_data/` with subject/scan/date TIFF subdirectories matching the loader's glob patterns, plus an `ADNIMERGE*.csv`. Raw MRI volumes alone will not satisfy this loader. Identify whether data are already preprocessed; otherwise establish the upstream preprocessing requirements. Do not download or upload controlled ADNI participant data without the required access and my instruction.
2. **Splits and clinical joins:** the local loader splits sorted subjects 70/15/15 and uses all visits by default. Audit subject separation, dropped records, visit/date alignment, and actual counts against the paper and benchmark. Do not change the split or CSV pairing just to obtain expected counts. If multiple ADNIMERGE files exist, resolve the intended file explicitly.
3. **Encoded attributes:** preserve this order and width: `apoE:2, age:1, sex:1, brain_vol:1, vent_vol:1, slice:10`, totaling 16 encoded values for six causal variables. APOE uses binary encoding; slice uses thermometer/ordinal encoding, despite some comments calling it one-hot. The code selects the central 10 of 30 stored slices. Verify normalization and image ranges across training, inference, and predictors.
4. **Prompt compatibility:** current code uses `a mri image of @ and * and &`, mapping the pseudo-tokens to brain volume, ventricular volume, and slice. The inference helper adds nine trailing repetitions of `&` to match training. Use its padded `assets.prompt`; do not pad twice or replace it with the paper's prose template without checking checkpoint compatibility.
5. **Checkpoints:** discover the actual miniSD base snapshot, ADNI ControlNet, matching learned embeddings from the same training step, adapter SCM if needed, benchmark flow mechanisms, and anti-causal predictor checkpoints. A name ending in `.safetensors` may refer to a model directory; check the loading API and filesystem. The benchmark README claims bundled checkpoints, but its expected `methods/deepscm/checkpoints/adni` directory was absent in this local checkout when this handoff was written.
6. **Environment:** retain the project diffusers fork installed with `pip install -e diffusers`, and put the repository root on `PYTHONPATH`. Verify the runtime import location and custom model classes. The pinned environment records PyTorch 2.4.1/CUDA 12.1 and transformers 4.46; inspect the actual environment before reinstalling anything. The evaluator accepts `CAUSAL_ADAPTER_SD15_BASE_MODEL` for a local backbone.
7. **Paths and evaluator defaults:** ADNI notebook paths are empty. Classifier `ckpt_path` and flow `checkpoint_dir` are relative to the working directory. The evaluator currently constructs ADNI datasets without an explicit root, and its default intervention loop includes all attribute keys. Add minimal explicit configuration where necessary so the intended dataset, paper intervention subset, and output paths are unambiguous. Verify metric dispatch, FID postprocessing, aggregation, and sample counts from code rather than assuming README examples run unchanged.
8. **Training only if needed:** inspect `train.py`, `scripts/adapter_training/train_adni.sh`, `SCM_modeling/train_scm.py`, and `SCM_modeling/common.py`. The adapter wrapper uses batch size 2 while the paper reports 40. The SCM wrapper contains a malformed duplicated `--data-root` and an unquoted placeholder. Do not execute either unchanged. Resolve effective batch size, optimizer, loss settings, graph, seeds, checkpointing, and resume support before proposing a paper-matched training run. Causal discovery is not automatically required.

## Execution sequence and deliverables

1. Inventory the available data, models, environment, GPU/cluster access, and required changes. Record confirmed paths separately from missing prerequisites.
2. Create a small configurable ADNI runner and SLURM wrapper, reusing existing loading/editing code. Keep model/data/output roots configurable, validate inputs before expensive initialization, and avoid hard-coded personal paths in reusable Python code.
3. Run a deterministic smoke test on a few held-out slices: load checkpoints, reconstruct factuals, intervene on each image parent, save original/reconstruction/counterfactual/difference grids, and check shape, dtype, ranges, finite values, label codecs, and nonempty outputs. Exercise predictor and metric loading too. Label these outputs as smoke-test results.
4. Establish the standard-DDIM evaluation command with `--dataset ADNI`, 100 steps, guidance 1.0, explicit checkpoints, seed, batch size, metrics, and a unique output root. Begin with `--max-batches 1`; then run the complete held-out evaluation once dependencies and smoke checks pass. Confirm the appropriate `--NTI` setting from implementation/protocol. Keep paper settings fixed when reducing batch size for memory and report any metric dependence on batching.
5. Aggregate the ADNI effectiveness matrix, composition, FID, and CLD using existing metric utilities where valid. Verify units, reductions, uncertainty definitions, intervention sampling, and final partial batches. Save machine-readable results and a readable comparison with Table 3. Never report a missing metric as zero, or tune on the test set to match published numbers.
6. Run the attention-guidance variant after the primary pipeline works and its ADNI configuration is verified. Preserve separate manifests and output roots. Save representative qualitative traversals; upstream-variable interventions can be additional qualitative examples.
7. Record the git revision, model/checkpoint identities, data split/counts, environment, all arguments, seeds, job IDs if submitted, output paths, and deviations in each run manifest. Provide a short runbook with exact preparation, smoke, full-evaluation, aggregation, and monitoring commands.

Start by inspecting the repository and available prerequisites. Continue through implementation and available validation. At completion, report only what changed, checks/run status, and unresolved problems in at most five concise bullets. Clearly distinguish scripts prepared, jobs submitted, and experiments completed.
