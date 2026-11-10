#!/usr/bin/env python3
"""Summarize trusted local ADNI evaluator outputs and compare with Table 3 (v2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/compute_final_metrics'))
# Reuse the benchmark's stable sigmoid / result loader. The generic table helper
# dispatches to binary F1, which is inappropriate for ordinal ADNI labels.
from metrics_utils import load_results, sigmoid

PARENTS = ['brain_vol', 'vent_vol', 'slice']
TABLE3 = {
    'standard': {'brain_vol': [.09, .11, .11], 'vent_vol': [.03, .03, .03], 'slice': [.53, .55, .48],
                 'composition_mae_255': 3.54, 'composition_lpips': .035, 'fid': 9.130, 'cld_corrected': .346},
    'attention': {'brain_vol': [.10, .14, .14], 'vent_vol': [.10, .04, .10], 'slice': [.55, .57, .46],
                  'composition_mae_255': None, 'composition_lpips': None, 'fid': 9.066, 'cld_corrected': .332}}


def score_attribute(logits, targets, attr, chunk_size=256):
    logits, targets = np.asarray(logits), np.asarray(targets)
    if logits.shape != targets.shape or logits.ndim != 2 or not len(logits):
        raise ValueError(f'{attr}: incompatible/nonempty [N,width] predictions and targets required')
    if not np.isfinite(logits).all() or not np.isfinite(targets).all():
        raise ValueError(f'{attr}: nonfinite predictions/targets')
    probabilities = sigmoid(logits)
    if attr == 'slice':
        predicted = probabilities.argmax(-1)
        actual = np.count_nonzero(targets, axis=1)
        if np.any((actual < 0) | (actual > 9)):
            raise ValueError('slice labels must encode classes 0..9')
        errors = (predicted == actual).astype(float)
        metric = 'multiclass F1 micro (equal to accuracy)'
    else:
        errors = np.abs(probabilities - targets).mean(axis=1)
        metric = 'MAE in normalized [0,1] volume units'
    chunks = [float(errors[i:i + chunk_size].mean()) for i in range(0, len(errors), chunk_size)]
    return {'metric': metric, 'value': float(errors.mean()), 'n': len(errors),
            'chunk_mean': float(np.mean(chunks)), 'chunk_std': float(np.std(chunks)),
            'chunk_size': chunk_size, 'chunks': len(chunks), 'last_chunk_n': len(errors) % chunk_size or chunk_size,
            'uncertainty': 'population SD (ddof=0) across fixed ordered chunks; includes final partial chunk; not SE or across-seed SD'}


def aggregate(root, chunk_size=256):
    report = json.loads((root / 'metrics.json').read_text())
    out = {'scope': report['scope'], 'paper': 'https://arxiv.org/html/2509.24798v2',
           'effectiveness': {}, 'metrics': report['metrics'], 'missing': [], 'not_requested': [], 'deviations': report['protocol_deviations'],
           'reference_table3': TABLE3['standard']}
    for parent in PARENTS:
        entry = load_results(str(root / parent / 'final_results.pkl'))
        if entry is None:
            out['missing'].append(parent)
            continue
        indices = np.asarray(entry['indices'])
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f'{parent}: duplicate evaluated indices')
        if report['scope'] == 'full' and not np.array_equal(np.sort(indices), np.arange(report['test_samples'])):
            raise ValueError(f'{parent}: incomplete held-out population')
        out['effectiveness'][parent] = {}
        for attr in PARENTS:
            scores = score_attribute(entry['predictions'][attr], entry['targets'][attr], attr, chunk_size)
            if scores['n'] != len(indices):
                raise ValueError(f'{parent}/{attr}: prediction count differs from indices')
            scores['paper_value'] = TABLE3['standard'][attr][PARENTS.index(parent)]
            scores['delta'] = scores['value'] - scores['paper_value']
            out['effectiveness'][parent][attr] = scores
    for name in ['composition_mae_255', 'composition_lpips', 'fid', 'cld_corrected']:
        source_metric = 'composition' if name.startswith('composition') else ('minimality' if name == 'cld_corrected' else name)
        if source_metric not in report.get('requested_metrics', ['composition', 'effectiveness', 'fid', 'minimality']):
            out['not_requested'].append(name)
            continue
        if name not in out['metrics'] or out['metrics'][name].get('value', out['metrics'][name].get('mean')) is None:
            out['missing'].append(name)
    if (root / 'composition.npz').exists():
        with np.load(root / 'composition.npz') as comp:
            if report['scope'] == 'full' and not np.array_equal(np.sort(comp['indices']), np.arange(report['test_samples'])):
                raise ValueError('Composition has incomplete held-out population')
    out['evaluation_coverage_complete'] = report['scope'] == 'full' and not out['missing']
    out['paper_equivalence_confirmed'] = False
    return out


def render(report):
    lines = [f"ADNI {report['scope']} results — arXiv 2509.24798v2, Table 3", '',
             '| Target | Intervention | Measured | Paper | Delta | Chunk SD | N |',
             '|---|---|---:|---:|---:|---:|---:|']
    for parent, row in report['effectiveness'].items():
        for attr, score in row.items():
            lines.append(f"| {attr} | do({parent}) | {score['value']:.5f} | {score['paper_value']:.3f} | {score['delta']:+.5f} | {score['chunk_std']:.5f} | {score['n']} |")
    lines += ['', 'Volumes use normalized units. Slice uses micro F1. Chunk SD is descriptive, not a confidence interval.',
              'The paper reports SD but does not establish the chunking or F1 averaging; equivalence is unconfirmed.', '',
              '| Metric | Measured | Paper |', '|---|---:|---:|']
    for name in ['composition_mae_255', 'composition_lpips', 'fid', 'cld_corrected']:
        score = report['metrics'].get(name, {})
        value = score.get('value', score.get('mean'))
        lines.append(f"| {name} | {value if value is not None else ('not requested' if name in report.get('not_requested', []) else 'missing')} | {TABLE3['standard'][name]} |")
    lines += ['', 'CLD uses corrected Gaussian KL; its numeric equivalence to the paper implementation is unresolved.',
              'Missing: ' + (', '.join(report['missing']) or 'none'), '', *report['deviations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Leaf directory containing metrics.json')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory')
    parser.add_argument('--chunk-size', type=int, default=256)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error('chunk-size must be positive')
    result = aggregate(args.run_dir, args.chunk_size)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (args.output_dir / 'table3.md').write_text(render(result))
    print(render(result))


if __name__ == '__main__':
    main()
