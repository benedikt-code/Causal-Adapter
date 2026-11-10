#!/usr/bin/env python3
"""Populate public Torch predictor/metric caches on a network-enabled CPU node."""
import json
import os
from pathlib import Path


def main():
    if not os.environ.get('TORCH_HOME'):
        raise SystemExit('Set TORCH_HOME to the shared metric cache directory')
    from torchvision.models import resnet18, ResNet18_Weights
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image.fid import FrechetInceptionDistance
    resnet18(weights=ResNet18_Weights.DEFAULT)
    LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True, reduction='none')
    FrechetInceptionDistance(normalize=True)
    print(json.dumps({'torch_home': str(Path(os.environ['TORCH_HOME']).resolve()), 'status': 'metric constructors loaded on CPU'}))


if __name__ == '__main__':
    main()
