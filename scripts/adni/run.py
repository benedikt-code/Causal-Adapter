#!/usr/bin/env python3
"""Validate and run released-weight ADNI evaluation; no automatic training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.adni.protocol import ATTRIBUTES, PARENTS, PAPER, ALL_METRICS, CORE_METRICS, identify, validate_paths


def command(config, mode, output):
    args = [sys.executable, str(REPO / 'counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/evaluate_SD_DSCM.py'),
            '--dataset', 'ADNI', '--editing', 'standard', '--NTI', 'False', '--num_steps', '100',
            '--guidance_scale', '1.0', '--resolution', '256', '--seed', str(config.get('seed', 40)),
            '--bs', str(config.get('batch_size', 2)), '--dataloader_num_workers', str(config.get('workers', 0)),
            '--metrics', *config.get('metrics', ALL_METRICS), '--embeddings', 'lpips',
            '--cycles', '1', '--interventions', *PARENTS, '--output-root', str(output)]
    for key, flag in [('data_root', '--data-root'), ('adnimerge_csv', '--adnimerge-csv'),
                      ('controlnet', '--controlnet_path'), ('embeddings', '--mcpl_embedding_path'),
                      ('flow_dir', '--flow-checkpoint-dir'), ('predictor_dir', '--predictor-checkpoint-dir')]:
        args.extend([flag, config[key]])
    if 'minimality' in config.get('metrics', ALL_METRICS):
        args.extend(['--unconditional-vae', config['unconditional_vae']])
    if mode == 'focused':
        args.extend(['--max-samples', str(config.get('focused_samples', 512))])
    if mode == 'smoke':
        args.extend(['--max-batches', str(max(1, (2 + config.get('batch_size', 2) - 1) // config.get('batch_size', 2))), '--smoke-samples', '2'])
    if mode == 'pilot':
        args.extend(['--max-batches', '1'])
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['preflight', 'smoke', 'pilot', 'focused', 'full', 'attention'], default='preflight')
    parser.add_argument('--validated-smoke', type=Path, help='Successful smoke manifest with matching inputs')
    parser.add_argument('--validated-pilot', type=Path, help='Successful one-batch quantitative pilot manifest')
    parser.add_argument('--output-root', type=Path, required=True, help='New directory; existing outputs are refused')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'preflight', 'paper': PAPER, 'config': config, 'mode': args.mode,
                'output_root': str(output), 'started_utc': datetime.now(timezone.utc).isoformat(),
                'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'git_diff': subprocess.check_output(['git', 'diff'], cwd=REPO, text=True),
                'python': sys.version, 'executable': sys.executable, 'host': socket.gethostname(),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'attributes': ATTRIBUTES,
                'imports': {name: bool(importlib.util.find_spec(name)) for name in
                            ['torch', 'torchvision', 'transformers', 'diffusers', 'accelerate', 'normflows', 'torchmetrics', 'torch_fidelity', 'clip']}}
    path = output / 'manifest.json'
    try:
        audit = validate_paths(config)
        manifest['data_audit'] = audit
        path.write_text(json.dumps(manifest, indent=2))
        if any(audit['splits'][s]['kept_slices'] == 0 for s in ['train', 'test']):
            raise ValueError('Train and test must both have usable slices')
        if args.mode == 'focused' and config.get('metrics', ALL_METRICS) != CORE_METRICS:
            raise ValueError(f'Focused mode requires config metrics={CORE_METRICS}; use it for smoke and pilot too')
        for stage, required in [('smoke', args.mode in ['pilot', 'focused', 'full']), ('pilot', args.mode in ['focused', 'full'])]:
            if required:
                prior_path = getattr(args, 'validated_' + stage)
                if prior_path is None:
                    raise ValueError(f'{args.mode} requires --validated-{stage} manifest')
                prior = json.loads(prior_path.read_text())
                if prior.get('status') != stage + '_completed' or prior.get('config') != config:
                    raise ValueError(f'{stage} validation must have completed with the same configuration')
                if prior.get('data_audit') != audit:
                    raise ValueError(f'Data changed since the validated {stage} run')
                if prior.get('git_diff') != manifest['git_diff']:
                    raise ValueError(f'Implementation changed since the validated {stage} run')
                manifest['validated_' + stage] = str(prior_path.resolve())
        if args.mode == 'attention':
            raise ValueError('ADNI attention variant is blocked: local blend thresholds/schedules are unverified; see RUNBOOK.md')
        cmd = command(config, args.mode, output)
        manifest['command'] = cmd
        print(shlex.join(cmd), flush=True)
        if args.mode == 'preflight':
            manifest['status'] = 'inputs_validated_no_experiment_run'
            return
        if not os.environ.get('SLURM_JOB_ID'):
            raise ValueError('GPU execution requires a SLURM allocation; use reproduce_adni.sbatch')
        missing = [k for k, available in manifest['imports'].items() if not available]
        if missing:
            raise ValueError(f'Missing runtime packages: {missing}')
        import torch
        import diffusers
        from diffusers import Causal_ControlNetModel, StableDiffusionCausalControlNetPipeline
        if REPO / 'diffusers' not in Path(diffusers.__file__).resolve().parents:
            raise ValueError('Install this repository fork with pip install -e diffusers')
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable in allocated job')
        manifest['runtime'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                               'gpu': torch.cuda.get_device_name(), 'diffusers': diffusers.__file__}
        if 'TORCH_HOME' not in os.environ:
            raise ValueError('Set TORCH_HOME to the prepared metric cache')
        manifest['torch_home'] = os.environ['TORCH_HOME']
        manifest['metric_cache_identity'] = identify(Path(os.environ['TORCH_HOME']) / 'hub/checkpoints')
        manifest['packages'] = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()
        artifact_keys = ['base_model', 'controlnet', 'embeddings', 'flow_dir', 'predictor_dir']
        if 'minimality' in config.get('metrics', ALL_METRICS):
            artifact_keys.append('unconditional_vae')
        manifest['artifacts'] = {k: identify(Path(config[k])) for k in artifact_keys}
        manifest['data_identity'] = identify(Path(config['data_root']) / 'preprocessed_data')
        source_files = [p for folder in ['scripts/adni'] for p in (REPO / folder).glob('*.py')]
        source_files += [REPO / 'counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/adni_support.py']
        manifest['source_hashes'] = {str(p.relative_to(REPO)): identify(p) for p in source_files}
        for stage in ['smoke', 'pilot']:
            if 'validated_' + stage in manifest:
                prior = json.loads(Path(manifest['validated_' + stage]).read_text())
                if prior.get('artifacts') != manifest['artifacts']:
                    raise ValueError(f'Checkpoint identities changed since {stage}')
                if prior.get('metric_cache_identity') != manifest['metric_cache_identity']:
                    raise ValueError(f'Metric cache changed since {stage}')
                if prior.get('data_identity') != manifest['data_identity']:
                    raise ValueError(f'Preprocessed images changed since {stage}')
                if prior.get('source_hashes') != manifest['source_hashes']:
                    raise ValueError(f'Runner sources changed since {stage}')
        manifest['status'] = 'running'
        path.write_text(json.dumps(manifest, indent=2))
        env = dict(os.environ, CAUSAL_ADAPTER_SD15_BASE_MODEL=config['base_model'],
                   PYTHONPATH=str(REPO) + os.pathsep + os.environ.get('PYTHONPATH', ''),
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED=str(config.get('seed', 40)))
        with (output / 'evaluation.log').open('x') as log:
            subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        metric_paths = list(output.glob('ADNI/*/metrics.json'))
        if len(metric_paths) != 1:
            raise ValueError('Evaluator did not produce exactly one metrics.json')
        metric_report = json.loads(metric_paths[0].read_text())
        manifest['metrics_path'] = str(metric_paths[0])
        manifest['evaluated_samples'] = metric_report['evaluated_samples']
        if args.mode == 'focused' and metric_report['evaluated_samples'] != config.get('focused_samples', 512):
            raise ValueError('Focused run did not evaluate the requested sample count')
        if args.mode == 'full' and metric_report['evaluated_samples'] != audit['splits']['test']['kept_slices']:
            raise ValueError('Full evaluation missed held-out samples')
        if args.mode == 'smoke' and not list(metric_paths[0].parent.glob('smoke/*.png')):
            raise ValueError('Smoke run produced no qualitative grids')
        manifest['status'] = 'smoke_completed' if args.mode == 'smoke' else ('pilot_completed' if args.mode == 'pilot' else ('focused_completed' if args.mode == 'focused' else 'evaluation_completed'))
    except Exception as exc:
        manifest['status'] = 'blocked' if manifest['status'] == 'preflight' else 'failed'
        manifest['error'] = str(exc)
        print(str(exc), file=sys.stderr)
        raise
    finally:
        manifest['ended_utc'] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
