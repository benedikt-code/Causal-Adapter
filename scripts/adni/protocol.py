"""Dependency-free ADNI input checks and split/clinical-join inventory."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ATTRIBUTES = {"apoE": 2, "age": 1, "sex": 1, "brain_vol": 1, "vent_vol": 1, "slice": 10}
PARENTS = ["brain_vol", "vent_vol", "slice"]
ALL_METRICS = ['composition', 'effectiveness', 'fid', 'minimality']
CORE_METRICS = ['composition', 'effectiveness']
ADAPTER_REVISION = "e868925fbcb0bf7d0979c8bf2a4d1b8be400f43c"
BASE_REVISION = "26ed8a9bfbf76f46a6cf60517dde321f900c44ce"
PAPER = "https://arxiv.org/html/2509.24798v2"


def unique_checkpoint(directory: Path, prefix: str) -> Path:
    matches = sorted(p for p in directory.glob(prefix + '*') if p.is_file() and p.suffix in {'.ckpt', '.pt', '.pth'})
    if len(matches) != 1:
        raise ValueError(f"Select exactly one {prefix} checkpoint in {directory}; found {len(matches)}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identify(path: Path) -> dict:
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    return {str(p.relative_to(path) if path.is_dir() else p.name):
            {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files}


def audit_data(root: Path, clinical: Path) -> dict:
    """Mirror the existing positional visit join; report differences, never repair it."""
    required = ['PTID', 'VISCODE', 'EXAMDATE', 'APOE4', 'AGE', 'PTGENDER', 'WholeBrain', 'Ventricles']
    rows = {}
    with clinical.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"Clinical CSV must contain {required}")
        for row in reader:
            rows.setdefault(row['PTID'], []).append(row)
    for visits in rows.values():
        visits.sort(key=lambda row: row['EXAMDATE'])
    subjects = sorted((root / 'preprocessed_data').glob('*'))
    if any(not p.is_dir() for p in subjects):
        raise ValueError('preprocessed_data must contain subject directories only')
    sizes = [int(len(subjects) * x) for x in (.7, .15, .15)]
    report = {'subjects_total': len(subjects), 'unassigned_subjects_due_to_floor': len(subjects) - sum(sizes),
              'pairing': 'scan dates and EXAMDATE sorted independently, paired by visit position; all visits',
              'csv_sha256': sha256(clinical), 'splits': {}, 'subject_overlap': False}
    offset = 0
    for split, size in zip(['train', 'valid', 'test'], sizes):
        selected = subjects[offset:offset + size]
        offset += size
        counts = dict(subjects=size, visits=0, selected_slices=0, kept_slices=0, dropped_slices=0,
                      missing_clinical_visits=0, null_clinical_visits=0, date_mismatch_visits=0,
                      duplicate_clinical_dates=0, incomplete_slice_visits=0)
        counts['subject_order_sha256'] = hashlib.sha256('\n'.join(p.name for p in selected).encode()).hexdigest()
        for subject in selected:
            visits = sorted(subject.glob('*/*'), key=lambda p: p.name)
            clinical_visits = rows.get(subject.name, [])
            counts['duplicate_clinical_dates'] += len(clinical_visits) - len({r['EXAMDATE'] for r in clinical_visits})
            for index, visit in enumerate(visits):
                slices = [p for p in sorted(visit.glob('*/*.tiff')) if 10 <= int(p.stem.split('slice')[1]) < 20]
                counts['visits'] += 1
                counts['selected_slices'] += len(slices)
                counts['incomplete_slice_visits'] += len(slices) != 10
                row = clinical_visits[index] if index < len(clinical_visits) else None
                missing = row is None
                null = row is not None and any(row[k].strip().lower() in {'', 'na', 'nan', 'n/a', 'null'} for k in required)
                counts['missing_clinical_visits'] += missing
                counts['null_clinical_visits'] += null
                if row and visit.name[:10] != row['EXAMDATE']:
                    counts['date_mismatch_visits'] += 1
                counts['dropped_slices' if missing or null else 'kept_slices'] += len(slices)
        report['splits'][split] = counts
    report['paper_counts'] = {'train': 10780, 'valid': 0, 'test': 2240}
    report['matches_paper_counts'] = all(report['splits'][s]['kept_slices'] == n for s, n in report['paper_counts'].items())
    return report


def validate_paths(config: dict) -> dict:
    metrics = config.get('metrics', ALL_METRICS)
    if not isinstance(metrics, list) or not metrics or any(m not in ALL_METRICS for m in metrics):
        raise ValueError(f'metrics must be a nonempty list drawn from {ALL_METRICS}')
    vae_keys = ['unconditional_vae'] if 'minimality' in metrics else []
    needed = ['data_root', 'base_model', 'controlnet', 'embeddings', 'flow_dir', 'predictor_dir'] + vae_keys
    errors = [f"Missing {key}: {config.get(key)}" for key in needed if not config.get(key) or not Path(config[key]).exists()]
    if errors:
        raise ValueError('\n'.join(errors))
    for key, default, minimum in [('seed', 40, 0), ('batch_size', 2, 1), ('workers', 0, 0), ('focused_samples', 512, 1)]:
        value = config.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    root = Path(config['data_root'])
    if not (root / 'preprocessed_data').is_dir():
        errors.append('data_root must contain preprocessed_data; raw MRI volumes are insufficient')
    if not (Path(config['base_model']) / 'model_index.json').is_file():
        errors.append('base_model must be a local Diffusers snapshot directory')
    for key in ['tokenizer', 'text_encoder', 'unet', 'vae', 'scheduler']:
        if not (Path(config['base_model']) / key).is_dir():
            errors.append(f'base_model is missing {key}/')
    control = Path(config['controlnet'])
    if not control.is_dir() or not (control / 'config.json').is_file():
        errors.append('ControlNet .safetensors path must be a Diffusers model DIRECTORY with config.json')
    elif not any(p.is_file() for pattern in ['*.safetensors', '*.bin'] for p in control.glob(pattern)):
        errors.append('ControlNet directory has no model weights')
    for key in ['embeddings'] + vae_keys:
        if not Path(config[key]).is_file():
            errors.append(f'{key} must be a file')
    # Release identity is explicit; never infer a different training step.
    if control.name != 'controlnet-steps-100000.safetensors' or Path(config['embeddings']).name != 'learned_embeds-steps-100000.safetensors':
        errors.append('Expected released ADNI ControlNet and embeddings from matching step 100000')
    csv_path = config.get('adnimerge_csv')
    candidates = [Path(csv_path)] if csv_path else sorted(root.glob('ADNIMERGE*.csv'))
    if len(candidates) != 1 or not candidates[0].is_file():
        errors.append('Set adnimerge_csv explicitly if ADNIMERGE*.csv is missing or ambiguous')
    for key, prefixes in [('flow_dir', ['brain_vol_flow-epoch', 'vent_vol_flow-epoch']), ('predictor_dir', PARENTS)]:
        for prefix in prefixes:
            try:
                unique_checkpoint(Path(config[key]), prefix)
            except ValueError as exc:
                errors.append(str(exc))
    if errors:
        raise ValueError('\n'.join(errors))
    config['adnimerge_csv'] = str(candidates[0].resolve())
    return audit_data(root, candidates[0])
