#!/usr/bin/env python3
"""Small ADNI qualitative demo using the released notebook SCM, not benchmark flows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.adni.protocol import ATTRIBUTES, ADAPTER_REVISION, BASE_REVISION, PAPER, identify, sha256


def validate_inputs(args):
    paths = {'base_model': args.base_model, 'data_root': args.data_root,
             'controlnet': args.weights_root / 'controlnet/controlnet-steps-100000.safetensors',
             'embeddings': args.weights_root / 'controlnet/learned_embeds-steps-100000.safetensors',
             'scm': args.weights_root / 'scm/best_model.pt'}
    for name, path in paths.items():
        if not path.exists():
            raise ValueError(f'Missing {name}: {path}')
    for name in ['embeddings', 'scm']:
        if not paths[name].is_file():
            raise ValueError(f'{name} must be a checkpoint file: {paths[name]}')
    for component in ['tokenizer', 'text_encoder', 'unet', 'vae', 'scheduler']:
        if not (args.base_model / component).is_dir():
            raise ValueError(f'Base snapshot lacks {component}/')
    if not (args.base_model / 'model_index.json').is_file():
        raise ValueError('base-model must be a local miniSD Diffusers snapshot')
    if not paths['controlnet'].is_dir():
        raise ValueError('ControlNet .safetensors path must be a Diffusers DIRECTORY')
    config = json.loads((paths['controlnet'] / 'config.json').read_text())
    if (config.get('_class_name'), config.get('dataset'), config.get('task_cond')) != (
            'Causal_ControlNetModel', 'ADNI', 'generation_text_global_after'):
        raise ValueError('ControlNet config does not match the inspected ADNI text-conditioned release')
    if not (paths['controlnet'] / 'diffusion_pytorch_model.safetensors').is_file():
        raise ValueError('Missing released ControlNet model weights')
    if not (args.data_root / 'preprocessed_data').is_dir():
        raise ValueError('data-root must contain preprocessed_data/ with float TIFF slices; raw NIfTI/ZIP input is not supported here')
    from causal_datasets.adni_dataset import resolve_adnimerge
    paths['clinical_csv'] = resolve_adnimerge(args.data_root, args.adnimerge_csv)
    return paths


def select_cases(root, clinical_csv, split, count, seed):
    """Use the loader's split and positional clinical join; load only selected images."""
    import numpy as np
    import pandas as pd
    import torch
    from causal_datasets.adni_dataset import split_subjects, load_extra_attributes, encode_attribute
    from causal_datasets._normalization import normalize

    subjects = sorted((root / 'preprocessed_data').glob('*'))
    if any(not p.is_dir() for p in subjects):
        raise ValueError('preprocessed_data must contain only subject directories')
    selected_subjects = subjects if split == 'all' else split_subjects(subjects, split)
    metadata = {'subject': [], 'date': [], 'slice': []}
    dates_by_subject, candidates = {}, []
    for subject in selected_subjects:
        dates = sorted(subject.glob('*/*'), key=lambda path: path.name)
        dates_by_subject[subject.name] = [path.name for path in dates]
        for date in dates:
            images = sorted(date.glob('*/*.tiff'))
            central = [(int(p.stem.split('slice')[1]), p) for p in images]
            central = [(index, path) for index, path in central if 10 <= index < 20]
            if not central:
                continue
            index, path = min(central, key=lambda entry: (abs(entry[0] - 15), str(entry[1])))
            metadata['subject'].append(subject.name)
            metadata['date'].append(date.name)
            metadata['slice'].append(encode_attribute(index - 10, 'slice', 10))
            candidates.append({'subject': subject.name, 'scan_date': date.name, 'image_path': str(path), 'stored_slice': index})
    if not candidates:
        raise ValueError(f'No central TIFF slices in split={split}. For a small demonstration root, explicitly use --split all (not held-out).')
    attributes, removed = load_extra_attributes(clinical_csv, ATTRIBUTES, metadata, dates_by_subject)
    attributes['slice'] = np.delete(attributes['slice'], removed, axis=0)
    attributes = {name: normalize(torch.tensor(np.asarray(values), dtype=torch.float32), name) for name, values in attributes.items()}
    encoded = torch.cat([attributes[name].reshape(-1, width) for name, width in ATTRIBUTES.items()], dim=1)
    removed_set = set(removed)
    candidates = [case for i, case in enumerate(candidates) if i not in removed_set]
    clinical = pd.read_csv(clinical_csv, usecols=['PTID', 'EXAMDATE']).set_index('PTID')
    groups, excluded = {}, 0
    for case, label in zip(candidates, encoded):
        # Exclude out-of-range factual examples for this small demonstration;
        # never clip labels or modify the dataset population/split.
        if not torch.isfinite(label).all() or ((label[2:6] < 0) | (label[2:6] > 1)).any():
            excluded += 1
            continue
        if not torch.equal(label[:2], label[:2].round()) or (label[:2] @ torch.tensor([2., 1.])).item() not in (0., 1., 2.):
            raise ValueError('Invalid factual APOE binary encoding')
        if label[3].item() not in (0., 1.):
            raise ValueError('Invalid factual sex encoding')
        case['label'] = label.tolist()
        visit = dates_by_subject[case['subject']].index(case['scan_date'])
        exam_dates = clinical.loc[[case['subject']]].sort_values('EXAMDATE')
        case['paired_exam_date'] = str(exam_dates.iloc[visit]['EXAMDATE'])
        case['scan_and_exam_dates_match'] = case['scan_date'][:10] == case['paired_exam_date']
        groups.setdefault(case['subject'], []).append(case)
    if len(groups) < count:
        raise ValueError(f'Requested {count} distinct subjects; only {len(groups)} have usable clinical records in split={split}. Reduce --subjects or explicitly choose --split all for a qualitative-only subset.')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(sorted(groups), size=count, replace=False)
    cases = [groups[subject][int(rng.integers(len(groups[subject])))] for subject in chosen]
    return cases, {'split': split, 'held_out_claim': split == 'test', 'split_defined_on': 'available preprocessed subject directories; not proof of the released checkpoint training split',
                   'subjects_total': len(subjects), 'subjects_in_split': len(selected_subjects), 'eligible_subjects': len(groups),
                   'candidate_visits': len(candidates) + len(removed), 'dropped_clinical_visits': len(removed),
                   'excluded_factual_out_of_range': excluded, 'selected_subjects': len(cases),
                   'clinical_join': 'unchanged sorted positional visit join, all visits'}


def read_factual(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        raw = np.asarray(image)
        if not np.issubdtype(raw.dtype, np.floating):
            raise ValueError(f'Expected floating-point TIFF data: {path}')
        array = raw.astype(np.float32)
    if array.shape != (180, 180) or not np.isfinite(array).all() or array.min() < -1 or array.max() > 1:
        raise ValueError(f'Expected finite 180x180 float TIFF in [-1,1]: {path}')
    return Image.fromarray((((array + 1) / 2) * 255).clip(0, 255).astype(np.uint8)).convert('RGB')


def intervention_plan(levels):
    import numpy as np
    return [('vent_vol', float(value)) for value in np.linspace(.2, .8, levels)] + [
        ('brain_vol', .25), ('brain_vol', .75), ('age', .25), ('age', .75)]


def decoded_labels(label):
    from causal_datasets._constants import ADNI_MIN_MAX
    result = {'apoE': int(round(label[0] * 2 + label[1])), 'sex': int(round(label[3])),
              'slice': sum(value != 0 for value in label[6:])}
    for name, offset in [('age', 2), ('brain_vol', 4), ('vent_vol', 5)]:
        lo, hi = ADNI_MIN_MAX[name]
        result[name] = float(lo + label[offset] * (hi - lo))
    return result


def signed_difference(image, reference, limit=64.):
    import numpy as np
    delta = np.asarray(image.convert('L'), dtype=np.float32) - np.asarray(reference.convert('L'), dtype=np.float32)
    intensity = np.clip(delta / limit, -1, 1)
    rgb = np.full((*delta.shape, 3), 255., dtype=np.float32)
    rgb[..., 0] -= 255 * np.maximum(-intensity, 0)
    rgb[..., 1] -= 255 * np.abs(intensity)
    rgb[..., 2] -= 255 * np.maximum(intensity, 0)
    from PIL import Image
    return Image.fromarray(rgb.round().astype(np.uint8)), delta


def save_grid(directory, name, factual, reconstruction, variants):
    from PIL import Image, ImageDraw
    columns = [('original', factual), ('reconstruction', reconstruction)] + variants
    size, header, row_label = 256, 30, 22
    canvas = Image.new('RGB', (size * len(columns), header + 3 * (size + row_label)), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, (title, image) in enumerate(columns):
        draw.text((column * size + 5, 5), title, fill='black')
        for row, (label, shown) in enumerate([
            ('generated image', image),
            ('change vs original', signed_difference(image, factual)[0]),
            ('change vs reconstruction', signed_difference(image, reconstruction)[0]),
        ]):
            top = header + row * (size + row_label)
            draw.text((column * size + 5, top), label, fill='black')
            canvas.paste(shown, (column * size, top + row_label))
    canvas.save(directory / f'{name}.png')


def generate_case(assets, case, directory, transforms_pair, ddim_editing, sample, levels):
    import numpy as np
    import torch
    from PIL import Image
    directory.mkdir()
    benchmark = directory / 'benchmark_192'; benchmark.mkdir()
    differences = directory / 'differences'; differences.mkdir()
    factual = read_factual(case['image_path'])
    image_transform, original_transform = transforms_pair
    original = original_transform(factual.copy()).convert('RGB')
    image_t = image_transform(factual).unsqueeze(0).to(assets.device)
    if image_t.shape != (1, 3, 256, 256) or image_t.dtype != torch.float32 or not torch.isfinite(image_t).all() or image_t.min() < -1 or image_t.max() > 1:
        raise ValueError('Invalid diffusion input')
    label = torch.tensor([case['label']], dtype=torch.float32, device=assets.device)
    images, inverted, _, uncond = ddim_editing(
        assets.pipe, image_t, label.clone(), assets.presudo_token_ids, assets.prompt,
        num_steps=100, invert_guidance_scale=1., set_guidance_scale=1., return_PIL=True,
        null_optimization=False, pnp_inversion=False)
    if not inverted or not torch.isfinite(inverted[-1]).all():
        raise ValueError('Inversion produced empty/nonfinite latents')
    reconstruction = images[0].convert('RGB')
    entries = [('original', original), ('reconstruction', reconstruction)]
    metadata = {**case, 'factual_decoded': decoded_labels(case['label']), 'interventions': []}
    rows = {name: [] for name in ['vent_vol', 'brain_vol', 'age']}
    arrays = {'factual_label': np.array(case['label'], dtype=np.float32)}
    for index, (parent, value) in enumerate(intervention_plan(levels)):
        target = list(ATTRIBUTES).index(parent)  # six-variable SCM index, not encoded-vector offset
        edited, condition = sample(
            assets.pipe, assets.prompt, assets.presudo_token_ids, start_latents=inverted[-1].clone(),
            guidance_scale=1., num_inference_steps=100, num_images_per_prompt=1, negative_prompt=None,
            device=assets.device, label=label.clone(), intervention_indx=target,
            intervention_values=value, disentangle=False, return_PIL=True, uncond_embeddings=uncond)
        if condition.shape != (1, 16, 1) or not torch.isfinite(condition).all():
            raise ValueError('SCM returned invalid encoded labels')
        encoded = condition[0, :, 0].cpu().tolist()
        offset = {'age': 2, 'brain_vol': 4, 'vent_vol': 5}[parent]
        if abs(encoded[offset] - value) > 1e-5:
            raise ValueError(f'SCM did not retain requested {parent} intervention')
        result = edited[0].convert('RGB')
        name = f'{parent}_{index:02d}'
        entries.append((name, result))
        rows[parent].append((f'{parent} target={value:.3f}', result))
        arrays[name + '_label'] = np.array(encoded, dtype=np.float32)
        metadata['interventions'].append({'name': name, 'variable': parent, 'scm_index': target,
            'target_normalized': value, 'encoded_label': encoded, 'decoded_label': decoded_labels(encoded),
            'continuous_label_out_of_range': any(v < 0 or v > 1 for v in encoded[2:6])})
    for name, image in entries:
        if image.size != (256, 256):
            raise ValueError(f'{name}: expected 256x256 image, got {image.size}')
        image.save(directory / f'{name}.png')
        image.resize((192, 192), Image.Resampling.BICUBIC).convert('L').save(benchmark / f'{name}.png')
        for baseline_name, baseline in [('original', original), ('reconstruction', reconstruction)]:
            difference, values = signed_difference(image, baseline)
            difference.save(differences / f'{name}_vs_{baseline_name}.png')
            arrays[f'{name}_minus_{baseline_name}'] = values
    for parent, row in rows.items():
        save_grid(directory, parent + '_grid', original, reconstruction, row)
    metadata['reconstruction_mae_255'] = float(np.abs(arrays['reconstruction_minus_original']).mean())
    metadata['difference_legend'] = 'Blue: decrease; white: zero; red: increase. Display clips at +/-64 of 255; NPZ keeps full signed differences.'
    (directory / 'case.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    np.savez_compressed(directory / 'labels_and_differences.npz', **arrays)
    pngs = list(directory.rglob('*.png'))
    if not pngs or any(path.stat().st_size == 0 for path in pngs):
        raise ValueError('Missing/empty output images')
    return {'directory': directory.name, 'png_files': len(pngs), 'interventions': len(metadata['interventions']),
            'reconstruction_mae_255': metadata['reconstruction_mae_255']}


def write_gallery(output, completed):
    import html
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>ADNI qualitative results</title>',
            '<style>body{font-family:sans-serif;max-width:1500px;margin:30px auto}img{width:100%;height:auto}section{margin-bottom:40px}</style>',
            '<h1>ADNI qualitative interventions</h1><p>Released notebook SCM; qualitative demonstration, not a Table 3 evaluation.</p>',
            '<p>Rows: image, change versus original, change versus reconstruction. Blue = decrease; red = increase; white = zero. Difference display clips at +/-64 of 255.</p>']
    for case in completed:
        name = html.escape(case['directory'], quote=True)
        body.append(f'<section><h2>{name}</h2>')
        for parent in ['vent_vol', 'brain_vol', 'age']:
            body.append(f'<h3>{parent}</h3><a href="{name}/{parent}_grid.png"><img src="{name}/{parent}_grid.png" alt="{parent} intervention grid"></a>')
        body.append(f'<a href="{name}/case.json">Labels and source metadata</a></section>')
    (output / 'index.html').write_text('\n'.join(body) + '</html>\n')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Root containing preprocessed_data/')
    parser.add_argument('--adnimerge-csv', type=Path, help='Required if clinical export is ambiguous')
    parser.add_argument('--base-model', type=Path, required=True, help='Local miniSD snapshot')
    parser.add_argument('--weights-root', type=Path, required=True, help='Released weights ADNI/ directory')
    parser.add_argument('--output-root', type=Path, required=True, help='New directory; existing results refused')
    parser.add_argument('--subjects', type=int, default=4)
    parser.add_argument('--levels', type=int, default=5, help='Ventricular sweep points from normalized .2 to .8')
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--split', choices=['train', 'valid', 'test', 'all'], default='test')
    parser.add_argument('--preflight-only', action='store_true', help='Validate/select cases on CPU without loading diffusion weights')
    args = parser.parse_args()
    if args.subjects < 1 or not 2 <= args.levels <= 20 or args.seed < 0:
        parser.error('subjects must be positive, levels 2..20, seed nonnegative')
    for key in ['data_root', 'adnimerge_csv', 'base_model', 'weights_root', 'output_root']:
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    return args


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'validating', 'scope': 'qualitative_only', 'paper': PAPER,
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'started_utc': datetime.now(timezone.utc).isoformat(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'protocol': {'ddim_steps': 100, 'inversion_guidance': 1., 'forward_guidance': 1., 'generation_size': 256,
                     'benchmark_size': 192, 'seed': args.seed, 'scm': 'released notebook adapter SCM; not Deep-SCM benchmark flows',
                     'quantitative_reproduction': False, 'attention_guidance': False,
                     'latent_reuse': 'one inversion per factual; same inverted latent for every intervention',
                     'ddim_helper_seed': 'shared helper internally calls torch.manual_seed(0)'},
        'expected_release_revisions': {'adapter': ADAPTER_REVISION, 'base': BASE_REVISION}, 'completed_cases': []}
    def save_manifest():
        (args.output_root / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    try:
        paths = validate_inputs(args)
        cases, selection = select_cases(args.data_root, paths['clinical_csv'], args.split, args.subjects, args.seed)
        for case in cases:
            read_factual(case['image_path'])
            case['image_sha256'] = sha256(Path(case['image_path']))
        manifest.update(selection=selection, cases=cases, paths={key: str(value) for key, value in paths.items()},
                        clinical_csv_sha256=sha256(paths['clinical_csv']))
        manifest['git_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        manifest['git_diff'] = subprocess.check_output(['git', 'diff'], cwd=REPO, text=True)
        manifest['runner_sha256'] = sha256(Path(__file__))
        save_manifest()
        if args.preflight_only:
            manifest['status'] = 'inputs_validated_no_images_generated'
            print(f'Validated {len(cases)} subjects. Manifest: {args.output_root / "manifest.json"}', flush=True)
            return
        if not os.environ.get('SLURM_JOB_ID'):
            raise ValueError('GPU generation requires a SLURM allocation; submit qualitative_adni.sbatch')
        import numpy as np
        import torch
        import diffusers
        if REPO / 'diffusers' not in Path(diffusers.__file__).resolve().parents:
            raise ValueError('Install the repository fork with python -m pip install -e diffusers')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Exactly one visible CUDA GPU is required')
        gpu = torch.cuda.get_device_properties(0)
        if 'a100' not in gpu.name.lower() or not 35 <= gpu.total_memory / 1024 ** 3 <= 45:
            raise ValueError(f'Allocated {gpu.name} ({gpu.total_memory / 1024**3:.1f} GiB). Request one A100 40GB using the site partition/GRES/constraint.')
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        manifest['runtime'] = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': gpu.name,
                               'gpu_bytes': gpu.total_memory, 'python': sys.version, 'executable': sys.executable,
                               'diffusers_location': diffusers.__file__}
        manifest['packages'] = {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'numpy', 'pandas', 'Pillow', 'diffusers', 'transformers', 'accelerate']}
        manifest['artifacts'] = {name: identify(paths[name]) for name in ['base_model', 'controlnet', 'embeddings', 'scm']}
        manifest['status'] = 'running'
        save_manifest()
        from notebook_benchmarks.inference_utils import build_transforms, load_causal_adapter
        from causal_modules.ddim_modules import ddim_editing, sample
        with torch.inference_mode():
            assets = load_causal_adapter('ADNI', str(paths['base_model']), str(paths['controlnet']),
                        str(paths['embeddings']), scm_path=str(paths['scm']), device=torch.device('cuda'), torch_dtype=torch.float32)
            tokens = assets.tokenizer(assets.prompt, add_special_tokens=True).input_ids
            if [tokens.count(token) for token in assets.presudo_token_ids] != [1, 1, 10]:
                raise ValueError('ADNI pseudo-token multiplicities must be [1,1,10]; do not pad prompt twice')
            manifest['prompt'] = assets.prompt
            manifest['attribute_widths'] = ATTRIBUTES
            image_transform, original_transform, _ = build_transforms('ADNI', size=256)
            for index, case in enumerate(cases):
                start = time.monotonic()
                summary = generate_case(assets, case, args.output_root / f'case_{index:02d}',
                        (image_transform, original_transform), ddim_editing, sample, args.levels)
                summary['elapsed_seconds'] = time.monotonic() - start
                manifest['completed_cases'].append(summary)
                save_manifest()
                write_gallery(args.output_root, manifest['completed_cases'])
                print(f'Completed case {index + 1}/{len(cases)}; gallery: {args.output_root / "index.html"}', flush=True)
        manifest['status'] = 'qualitative_completed'
    except Exception as exc:
        manifest['status'], manifest['error'] = 'failed', str(exc)
        raise
    finally:
        manifest['ended_utc'] = datetime.now(timezone.utc).isoformat()
        save_manifest()


if __name__ == '__main__':
    main()
