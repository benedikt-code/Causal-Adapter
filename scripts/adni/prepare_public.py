#!/usr/bin/env python3
"""Download only public released adapter/backbone weights at inspected revisions."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.adni.protocol import ADAPTER_REVISION, BASE_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True, help='New directory')
    parser.add_argument('--upstream-only', action='store_true', help='Fetch preprocessing source without downloading model weights')
    args = parser.parse_args()
    args.model_root.mkdir(parents=True, exist_ok=False)
    revision = '581dbb495421ca863c4a6e62ad759ac79de076e9'
    upstream = args.model_root / 'benchmark_preprocessing'
    upstream.mkdir()
    for name in ['README.md', 'main.py', 'get_config.py', 'image_preselection.py', 'move_selected_images.py']:
        url = f'https://raw.githubusercontent.com/gulnazaki/counterfactual-benchmark/{revision}/counterfactual_benchmark/datasets/adni/preprocessing/{name}'
        with urllib.request.urlopen(url, timeout=30) as response:
            (upstream / name).write_bytes(response.read())
    (args.model_root / 'upstream_revision.txt').write_text(revision + '\n')
    if args.upstream_only:
        print(upstream.resolve())
        return
    from huggingface_hub import snapshot_download
    snapshots = {}
    for name, repo, revision, patterns in [
        ('causal-adapter-weights', 'LeiTong/Causal-Adapter', ADAPTER_REVISION, ['ADNI/*']),
        ('miniSD-diffusers', 'lambda/miniSD-diffusers', BASE_REVISION, None),
    ]:
        target = args.model_root / name
        snapshots[name] = {'repo': repo, 'revision': revision, 'path': str(target.resolve())}
        snapshot_download(repo_id=repo, revision=revision, allow_patterns=patterns, local_dir=target)
    (args.model_root / 'public_models.json').write_text(json.dumps(snapshots, indent=2))
    print(json.dumps(snapshots, indent=2))


if __name__ == '__main__':
    main()
