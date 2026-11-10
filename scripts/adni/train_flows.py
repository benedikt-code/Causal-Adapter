#!/usr/bin/env python3
"""Separate, explicit fallback for missing benchmark flow checkpoints.

Never invoked by the reproduction runner. Uses the checked-out benchmark config
unchanged; it does not claim paper-matched training or train the image generator.
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'counterfactual-benchmark/counterfactual_benchmark'))
sys.path.insert(0, str(REPO / 'counterfactual-benchmark/counterfactual_benchmark/methods/deepscm'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--adnimerge-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    import os
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Training requires a separately authorized SLURM job')
    from causal_datasets.adni_dataset import ADNI, resolve_adnimerge
    resolve_adnimerge(args.data_root, args.adnimerge_csv)
    config = json.loads((REPO / 'counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/configs/adni/vae.json').read_text())
    from pytorch_lightning import seed_everything
    seed_everything(args.seed, workers=True)
    from training_scripts import train_flow
    from models.flows import BrainVolFlow, VentVolFlow
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'training_config.json').write_text(json.dumps({'seed': args.seed, 'config': config, 'data_root': str(args.data_root), 'clinical_csv': str(args.adnimerge_csv)}, indent=2))
    for variable, cls in [('brain_vol', BrainVolFlow), ('vent_vol', VentVolFlow)]:
        params = config['mechanism_models'][variable]['params']
        flow = cls(params=params, attr_size=config['attribute_size'])
        train_flow(flow, params, ADNI, config['causal_graph'], config['attribute_size'], str(args.output_dir),
                   dataset_dir=args.data_root, csv_path=args.adnimerge_csv, normalize_=True)


if __name__ == '__main__':
    main()
