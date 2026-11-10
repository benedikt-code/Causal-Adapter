"""ADNI-specific evaluation with explicit codecs, populations, and metric reductions.

Uses evaluate_SD_DSCM.produce_counterfactuals and its Deep-SCM flows. The
adapter SCM head is deliberately not substituted for the benchmark mechanisms.
"""
from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from scripts.adni.protocol import ATTRIBUTES, PARENTS, unique_checkpoint


def validate_evaluator_args(args):
    if args.editing != 'standard' or args.NTI or args.reverse:
        raise ValueError('Primary ADNI protocol requires standard editing, NTI=False, reverse=False; attention is unverified')
    if int(args.num_steps) != 100 or args.guidance_scale != 1 or args.resolution != 256:
        raise ValueError('ADNI paper protocol requires 100 DDIM steps, guidance 1.0, resolution 256')
    if args.bs < 1 or (args.max_batches is not None and args.max_batches < 1):
        raise ValueError('Batch size and max-batches must be positive')
    if args.interventions is None:
        args.interventions = PARENTS.copy()
    if args.interventions != PARENTS:
        raise ValueError(f'Paper evaluation interventions must be {PARENTS}')
    for name in ['data_root', 'flow_checkpoint_dir', 'predictor_checkpoint_dir', 'unconditional_vae']:
        if getattr(args, name, None) is None:
            raise ValueError(f'ADNI requires explicit --{name.replace("_", "-")}')
    if Path(__import__('os').environ.get('CAUSAL_ADAPTER_SD15_BASE_MODEL', '')).is_dir() is False:
        raise ValueError('Set CAUSAL_ADAPTER_SD15_BASE_MODEL to the local miniSD snapshot')
    # Validate all inputs before diffusion initialization, including direct CLI use.
    from scripts.adni.protocol import validate_paths
    args.preflight_audit = validate_paths({
        'data_root': str(args.data_root), 'adnimerge_csv': str(args.adnimerge_csv) if args.adnimerge_csv else None,
        'base_model': __import__('os').environ['CAUSAL_ADAPTER_SD15_BASE_MODEL'],
        'controlnet': args.controlnet_path, 'embeddings': args.mcpl_embedding_path,
        'flow_dir': str(args.flow_checkpoint_dir), 'predictor_dir': str(args.predictor_checkpoint_dir),
        'unconditional_vae': str(args.unconditional_vae)})
    if Path(args.output_root).joinpath('ADNI').exists():
        raise FileExistsError(f'ADNI outputs already exist under {args.output_root}')


def check_tensor(x, shape, low, high, name):
    if tuple(x.shape[1:]) != shape or x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError(f'{name}: invalid shape/dtype/nonfinite: {x.shape}, {x.dtype}')
    if x.min() < low - 1e-5 or x.max() > high + 1e-5:
        raise ValueError(f'{name}: range {x.min().item()}, {x.max().item()} outside [{low}, {high}]')


def labels(batch):
    return torch.cat([batch[key] for key in ATTRIBUTES], dim=1)


def check_labels(batch):
    encoded = labels(batch)
    if encoded.shape[1] != 16 or not torch.isfinite(encoded).all():
        raise ValueError('ADNI labels must be finite [B,16]')
    apo = batch['apoE']
    ordinal = batch['slice']
    if not (((apo == 0) | (apo == 1)).all() and (apo @ apo.new_tensor([2., 1.]) <= 2).all()):
        raise ValueError('Invalid binary APOE values')
    if not (((ordinal == 0) | (ordinal == 1)).all() and (ordinal[:, 1:] >= ordinal[:, :-1]).all() and (ordinal.sum(1) <= 9).all()):
        raise ValueError('Slice must be thermometer encoding of indices 0..9')
    if not ((batch['sex'] == 0) | (batch['sex'] == 1)).all():
        raise ValueError('Invalid sex code')


def sample_interventions(batch, indices, parent, dataset, seed, bins=None):
    """Uniform distinct held-out values, excluding the factual continuous bin.

    Independent random stream per sample/parent, unaffected by DDIM's internal
    manual_seed(0), batch size, or which metrics ran earlier.
    """
    possible = dataset.possible_values[parent].cpu().numpy()
    values = batch[parent].cpu().numpy()
    sampled = []
    for value, index in zip(values, indices):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(index), PARENTS.index(parent)]))
        if parent == 'slice':
            choices = possible[(possible != value).any(axis=1)]
        else:
            boundaries = (bins if bins is not None else dataset.bins)[parent]
            if not np.isfinite(boundaries).all():
                raise ValueError(f'Nonfinite training bins for {parent}; inspect clinical population')
            choices = possible[np.searchsorted(boundaries, possible) != np.searchsorted(boundaries, value.item())]
        if not len(choices):
            raise ValueError(f'No distinct intervention candidate for {parent}')
        sampled.append(choices[rng.integers(len(choices))])
    return {parent: torch.as_tensor(np.array(sampled).reshape(len(values), -1), device=batch[parent].device, dtype=torch.float32)}


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Empty or nonfinite metric')
    return {'mean': float(values.mean()), 'std': float(values.std()), 'n': int(values.size), 'std_definition': 'population standard deviation across samples (ddof=0)'}


def save_grid(directory, name, original, reconstructed, counterfactuals):
    images = [original, reconstructed, *counterfactuals]
    titles = ['original', 'reconstruction', *[f'do({p})' for p in PARENTS]]
    canvas = Image.new('RGB', (192 * len(images), 192 * 2 + 22), 'white')
    draw = ImageDraw.Draw(canvas)
    base = original[0].cpu()
    for i, (image, title) in enumerate(zip(images, titles)):
        array = (image[0].cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        preview = Image.fromarray(array).convert('RGB')
        difference = ((image[0].cpu() - base).abs().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        canvas.paste(preview, (i * 192, 22))
        canvas.paste(Image.fromarray(difference).convert('RGB'), (i * 192, 214))
        draw.text((i * 192 + 3, 3), title, fill='black')
    target = directory / f'{name}.png'
    canvas.save(target)
    if target.stat().st_size == 0:
        raise ValueError(f'Empty qualitative grid: {target}')


def kl_distance(first, second):
    """KL(first || second), averaged over latent coordinates; encoder emits log variance."""
    mu1, logvar1 = np.asarray(first)[..., 0, :], np.asarray(first)[..., 1, :]
    mu2, logvar2 = np.asarray(second)[..., 0, :], np.asarray(second)[..., 1, :]
    return .5 * np.mean(logvar2 - logvar1 + np.exp(logvar1 - logvar2) + (mu1 - mu2) ** 2 * np.exp(-logvar2) - 1, axis=-1)


def cld_scores(reference, reference_labels, factuals, counterfactuals, parents, factual_labels, counter_labels, bins):
    """Benchmark W1=.8/W2=.2 probability reduction with full held-out reference.

    Keep labels in continuous normalized units, and compare only the chosen
    parent. The legacy implementation truncated volumes to ints and confused
    posterior mean with log scale; those values cannot be interpreted as KL.
    """
    scores, skipped = [], 0
    for fac, cf, parent, original, target in zip(factuals, counterfactuals, parents, factual_labels, counter_labels):
        ref_labels = reference_labels[parent]
        if parent == 'slice':
            mask1, mask2 = ref_labels == original, ref_labels == target
        else:
            mask1 = np.searchsorted(bins[parent], ref_labels) == np.searchsorted(bins[parent], original)
            mask2 = np.searchsorted(bins[parent], ref_labels) == np.searchsorted(bins[parent], target)
        if not mask1.any() or not mask2.any():
            skipped += 1
            continue
        divergence = kl_distance(cf, fac)
        p1 = np.mean(kl_distance(reference[mask2], fac) <= divergence)
        p2 = np.mean(kl_distance(reference[mask1], fac) >= divergence)
        scores.append(np.log(.8 * np.exp(p1) + .2 * np.exp(p2)))
    return np.array(scores), skipped


@torch.inference_mode()
def evaluate_adni(evaluator, args, pipe, prompt, scm, train_set, test_set, config_cls):
    if dict(args.attribute_size) != ATTRIBUTES or list(args.attribute_size) != list(ATTRIBUTES):
        raise ValueError('ADNI attribute order/width does not match the released checkpoint')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = Path(evaluator.experiment_output_dir(args))
    output.mkdir(parents=True, exist_ok=False)
    for split, dataset in [('train', train_set), ('test', test_set)]:
        if len(dataset) != args.preflight_audit['splits'][split]['kept_slices']:
            raise ValueError(f'{split}: audit count does not match loaded dataset')
    (output / 'data_counts.json').write_text(json.dumps({'train': train_set.audit, 'test': test_set.audit}, indent=2))
    # Validate all required metric networks before running any diffusion edits.
    predictors = evaluator.build_predictors('adni', ATTRIBUTES, PARENTS, config_cls, args)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True, reduction='none').cuda().eval()
    from models.vaes import ADNICondVAE
    params = {'context_dim': 0, 'latent_dim': 100, 'hidden_dim': 512,
              'n_chan': [1, 16, 24, 32, 64, 128, 256], 'beta': 3, 'lr': 1e-3,
              'weight_decay': .01, 'fixed_logvar': 'False'}
    vae = ADNICondVAE(params, ATTRIBUTES, unconditional=True).cuda().eval()
    vae.load_state_dict(torch.load(args.unconditional_vae, map_location='cuda', weights_only=False)['state_dict'])
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(normalize=True).cuda() if 'fid' in args.metrics else None
    def features(image):
        mu, logvar = vae.encoder(image, None)
        return torch.stack([mu, logvar], dim=1).cpu().numpy()
    # Check pseudo-token multiplicities against the actual checkpoint tokenizer.
    tokenized = pipe.tokenizer(prompt, add_special_tokens=True).input_ids
    counts = [tokenized.count(token) for token in args.presudo_token_ids]
    if counts != [1, 1, 10]:
        raise ValueError(f'ADNI prompt pseudo-token counts must be [1,1,10], got {counts}')
    def resize(image):
        return evaluator.ddim_modules.resize_tensor(image, pipe, 'ADNI', return_tensor=True, mode='bicubic')
    def edit(batch, indices, parent, intervention=None):
        result = evaluator.produce_counterfactuals(pipe, prompt, batch, scm, parent, train_set, args=args,
                    interventions=intervention if intervention is not None else sample_interventions(batch, indices, parent, test_set, args.seed, bins=train_set.bins))
        check_tensor(result['image'], (1, 192, 192), 0, 1, 'counterfactual')
        return result
    loader = DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=args.dataloader_num_workers, drop_last=False)
    comp_mae, comp_lpips, comp_indices = [], [], []
    results = {p: {'predictions': {a: [] for a in PARENTS}, 'targets': {a: [] for a in PARENTS}, 'indices': []} for p in PARENTS}
    reference, reference_labels = [], {p: [] for p in PARENTS}
    fac_features, cf_features, parent_list, fac_labels, cf_labels, generated_images = [], [], [], [], [], []
    for batch_index, (batch, indices) in enumerate(loader):
        # CLD reference always uses the full held-out set, even for smoke/pilot.
        batch = {k: v.cuda() for k, v in batch.items()}
        check_tensor(batch['image'], (3, 256, 256), -1, 1, 'diffusion input')
        check_labels(batch)
        factual = resize(batch['image'])
        check_tensor(factual, (1, 192, 192), 0, 1, 'benchmark input')
        reference.append(features(factual))
        for parent in PARENTS:
            values = batch[parent].sum(1) if parent == 'slice' else batch[parent].reshape(-1)
            reference_labels[parent].extend(values.cpu().tolist())
        if args.max_batches is not None and batch_index >= args.max_batches:
            continue
        if 'composition' in args.metrics or args.smoke_samples:
            reconstructed, _, _, _ = evaluator.ddim_modules.ddim_editing(
                pipe, batch['image'], labels(batch), args.presudo_token_ids, prompt,
                num_steps=100, invert_guidance_scale=1., set_guidance_scale=1., return_PIL=False,
                null_optimization=False, pnp_inversion=False)
            reconstructed = resize(reconstructed)
            check_tensor(reconstructed, (1, 192, 192), 0, 1, 'reconstruction')
            comp_mae.extend((255 * (reconstructed - factual).abs().mean((1, 2, 3))).cpu().tolist())
            comp_lpips.extend(lpips(reconstructed.repeat(1, 3, 1, 1), factual.repeat(1, 3, 1, 1)).reshape(-1).cpu().tolist())
            comp_indices.extend(indices.tolist())
        generated_by_parent = {}
        for parent in PARENTS:
            cf = edit(batch, indices, parent)
            generated_by_parent[parent] = cf
            if 'effectiveness' in args.metrics:
                _, predicted = evaluator.effectiveness(cf, evaluator.unnormalize_adni, predictors, 'adni')
                results[parent]['indices'].extend(indices.tolist())
                for attr in PARENTS:
                    results[parent]['predictions'][attr].append(predicted['predictions'][attr])
                    results[parent]['targets'][attr].append(predicted['targets'][attr])
        # One deterministic uniformly selected image-parent intervention per test
        # image; selection does not depend on inference batch size.
        chosen = [int(np.random.default_rng(np.random.SeedSequence([args.seed, int(i), 99])).integers(3)) for i in indices]
        selected = torch.stack([generated_by_parent[PARENTS[j]]['image'][i] for i, j in enumerate(chosen)])
        if fid is not None:
            fid.update(selected.repeat(1, 3, 1, 1), real=False)
        generated_images.append(selected.cpu())
        if 'minimality' in args.metrics:
            fac_features.extend(features(factual))
            cf_features.extend(features(selected))
            for i, j in enumerate(chosen):
                parent = PARENTS[j]
                parent_list.append(parent)
                fac_labels.append(float(batch[parent][i].sum() if parent == 'slice' else batch[parent][i].item()))
                cf_labels.append(float(generated_by_parent[parent][parent][i].sum() if parent == 'slice' else generated_by_parent[parent][parent][i].item()))
        if args.smoke_samples and batch_index == 0:
            # Representative fixed traversals through each image parent's
            # observed range, using benchmark flow abduction for every point.
            traversals = output / 'traversals'
            traversals.mkdir()
            one = {k: v[:1] for k, v in batch.items()}
            for parent in PARENTS:
                candidates = test_set.possible_values[parent].cpu().numpy()
                if parent == 'slice':
                    candidates = candidates[np.argsort(candidates.sum(1))]
                positions = [0, len(candidates) // 2, len(candidates) - 1]
                row = []
                for position in positions:
                    intervention = {parent: torch.as_tensor(candidates[position], device='cuda', dtype=torch.float32).reshape(1, -1)}
                    traversed = edit(one, indices[:1], parent, intervention)
                    image = traversed['image'][0, 0].cpu().numpy()
                    row.append(Image.fromarray((image * 255).round().astype(np.uint8)))
                canvas = Image.new('L', (192 * len(row), 192))
                for i, image in enumerate(row):
                    canvas.paste(image, (i * 192, 0))
                canvas.save(traversals / f'{parent}.png')
                np.savez(traversals / f'{parent}.npz', values=candidates[positions], sample_index=int(indices[0]))
        if args.smoke_samples:
            qualitative = output / 'smoke'
            qualitative.mkdir(exist_ok=True)
            np.savez(qualitative / f'labels-batch-{batch_index}.npz', factual=labels(batch).cpu().numpy(),
                     **{p: labels(generated_by_parent[p]).cpu().numpy() for p in PARENTS})
            for i in range(min(args.smoke_samples, len(indices))):
                save_grid(qualitative, f'smoke-{int(indices[i])}', factual[i], reconstructed[i],
                          [generated_by_parent[p]['image'][i] for p in PARENTS])
    report = {'scope': 'smoke' if args.smoke_samples else ('pilot' if args.max_batches else 'full'),
              'test_samples': len(test_set), 'metrics': {}, 'interventions': PARENTS,
              'protocol_deviations': ['Dataset uses sorted-subject 70/15/15 split with all visits and positional clinical joins.',
                'Interventions use independent per-sample seeded draws rather than batch-dependent global RNG.',
                'CLD corrects the legacy posterior-KL and integer-volume-label bugs; reference is full held-out set.',
                'Paper does not specify the exact uncertainty chunks, F1 averaging, FID real split, or attention schedules.']}
    if comp_mae:
        np.savez(output / 'composition.npz', mae_255=comp_mae, lpips=comp_lpips, indices=comp_indices)
        report['metrics']['composition_mae_255'] = summary(comp_mae)
        report['metrics']['composition_lpips'] = summary(comp_lpips)
    if 'effectiveness' in args.metrics:
        for parent in PARENTS:
            entry = results[parent]
            for group in ['predictions', 'targets']:
                entry[group] = {key: np.concatenate(value) for key, value in entry[group].items()}
            directory = output / parent
            directory.mkdir()
            with (directory / 'final_results.pkl').open('xb') as stream:
                pickle.dump(entry, stream)
    if fid is not None:
        # Benchmark compute_FID.py compares against the training split. Generated
        # tensors are ALREADY [0,1]; never postprocess/denormalize them again.
        for batch, _ in DataLoader(train_set, batch_size=args.bs, num_workers=args.dataloader_num_workers):
            fid.update(resize(batch['image'].cuda()).repeat(1, 3, 1, 1), real=True)
        if sum(len(x) for x in generated_images) > 1:
            report['metrics']['fid'] = {'value': float(fid.compute()), 'real_n': len(train_set),
                                        'generated_n': sum(len(x) for x in generated_images), 'real_split': 'train'}
        else:
            report['metrics']['fid'] = {'value': None, 'reason': 'FID requires at least two generated images'}
    report['evaluated_samples'] = sum(len(x) for x in generated_images)
    if generated_images:
        torch.save(torch.cat(generated_images), output / 'counterfactual_tensors.pt')
    if 'minimality' in args.metrics:
        ref = np.concatenate(reference)
        ref_labels = {p: np.array(v) for p, v in reference_labels.items()}
        scores, skipped = cld_scores(ref, ref_labels, fac_features, cf_features, parent_list, fac_labels, cf_labels, train_set.bins)
        np.savez(output / 'cld.npz', scores=scores, skipped=skipped, factuals=fac_features,
                 counterfactuals=cf_features, parents=parent_list, factual_labels=fac_labels, counterfactual_labels=cf_labels)
        report['metrics']['cld_corrected'] = {**summary(scores), 'skipped': skipped, 'reference_n': len(ref)} if len(scores) else {'value': None, 'skipped': skipped}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
    return output
