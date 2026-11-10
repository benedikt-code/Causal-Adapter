#!/usr/bin/env python3
"""Run the inspected benchmark FSL/TIFF procedure in a new output directory.

Consumes selected image IDs from upstream image_preselection.py. Originals are
read from NIfTI files or ZIP members and never moved, deleted, or overwritten.
The upstream normalization/cropping/slice writers are reused verbatim.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.adni.metadata_from_xml import image_id, available_images
from scripts.adni.protocol import identify, sha256

UPSTREAM_REVISION = '581dbb495421ca863c4a6e62ad759ac79de076e9'


def locate(root, selected):
    locations = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        candidates = []
        if path.name.endswith(('.nii', '.nii.gz')):
            candidates = [(path.as_posix(), None, path.stat().st_size)]
        elif path.parent.name.startswith('batch_') and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                candidates = [(i.filename, i.filename, i.file_size) for i in archive.infolist() if i.filename.endswith(('.nii', '.nii.gz'))]
        for name, member, size in candidates:
            identity = image_id(name)
            if identity not in selected:
                continue
            if identity in locations:
                raise ValueError(f'Duplicate source image ID {identity}; supply an input root containing a single source per image')
            parts = Path(name).parts
            index = next((i for i, part in enumerate(parts) if re.fullmatch(r'\d{3}_S_\d{4}', part)), None)
            if index is None:
                raise ValueError('NIfTI source lacks benchmark subject/scan/date/acquisition hierarchy')
            relative = Path(*parts[index:])
            if len(relative.parts) != 5 or '..' in relative.parts:
                raise ValueError(f'Expected subject/scan/date/acquisition/file hierarchy for {identity}')
            locations[identity] = (path, member, relative, size)
    missing = selected - locations.keys()
    if missing:
        raise ValueError(f'{len(missing)} selected image IDs lack a unique NIfTI source')
    return locations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--selected-ids', type=Path, required=True)
    parser.add_argument('--upstream-preprocessing', type=Path, required=True, help='Pinned public source directory containing main.py/get_config.py')
    parser.add_argument('--output-root', type=Path, required=True, help='New preprocessing root; requires a separately supplied clinical CSV for evaluation')
    parser.add_argument('--max-scans', type=int, help='Preprocessing smoke cap; never use capped output for full evaluation')
    args = parser.parse_args()
    if args.max_scans is not None and args.max_scans < 1:
        parser.error('max-scans must be positive')
    tools = ['fslreorient2std', 'robustfov', 'bet', 'flirt']
    missing = [name for name in tools if shutil.which(name) is None]
    if missing:
        parser.error(f'Missing FSL tools: {missing}; initialize FSL before running')
    atlas = Path(os.environ.get('FSLDIR', '')) / 'data/standard/MNI152_T1_1mm_brain.nii.gz'
    if not atlas.is_file():
        parser.error('FSLDIR must contain the MNI152 1mm brain atlas')
    with args.selected_ids.open() as stream:
        selected = {row['Image Data ID'] for row in csv.DictReader(stream)}
    if not selected:
        parser.error('No selected image IDs')
    selected_for_run = selected
    if args.max_scans is not None:
        selected_for_run = set(sorted(selected & available_images(args.input_root))[:args.max_scans])
        if not selected_for_run:
            parser.error('No available selected scan for preprocessing smoke')
    locations = locate(args.input_root, selected_for_run)
    expected_sources = {'main.py': 'acea822f595acb4e8bb088bebb7b7d6e024a067b98a97732e6d03c570f6da0d3',
                        'get_config.py': 'ab93340b983060c50956361c4cb2aff88e75f842d55e7a98b1880afc1f2d889e'}
    for name, expected in expected_sources.items():
        if sha256(args.upstream_preprocessing / name) != expected:
            raise ValueError(f'Upstream {name} does not match inspected revision {UPSTREAM_REVISION}')
    sys.path.insert(0, str(args.upstream_preprocessing.resolve()))
    module_path = args.upstream_preprocessing / 'main.py'
    spec = importlib.util.spec_from_file_location('upstream_adni_preprocessing', module_path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    import numpy as np
    import SimpleITK as sitk
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'upstream_revision': UPSTREAM_REVISION, 'arguments': {k: str(v) for k, v in vars(args).items()},
                'upstream_sources': identify(args.upstream_preprocessing), 'atlas_sha256': sha256(atlas),
                'fsl_tools': {name: shutil.which(name) for name in tools},
                'fsl_version': (Path(os.environ['FSLDIR']) / 'etc/fslversion').read_text().strip(),
                'selected_scans': len(selected), 'processed_scans': 0,
                'fast_segmentation': False, 'note': 'FAST is commented out in inspected upstream main.py; source MRI files retained'}
    try:
        for index, identity in enumerate(sorted(locations)):
            if args.max_scans is not None and index >= args.max_scans:
                break
            source, member, relative, size = locations[identity]
            with tempfile.TemporaryDirectory(prefix='adni-', dir=args.output_root) as scratch:
                scratch = Path(scratch)
                original = scratch / ('source.nii.gz' if str(relative).endswith('.nii.gz') else 'source.nii')
                if member is None:
                    shutil.copyfile(source, original)
                else:
                    with zipfile.ZipFile(source) as archive, archive.open(member) as src, original.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                oriented, fov, brain, registered = [scratch / (name + '.nii.gz') for name in ['oriented', 'fov', 'brain', 'registered']]
                commands = [['fslreorient2std', str(original), str(oriented)],
                            ['robustfov', '-i', str(oriented), '-r', str(fov)],
                            ['bet', str(fov), str(brain), '-R'],
                            ['flirt', '-in', str(brain), '-ref', str(atlas), '-out', str(registered)]]
                for command in commands:
                    subprocess.run(command, check=True)
                array = sitk.GetArrayFromImage(sitk.ReadImage(str(registered)))
                cropped = upstream.cropping(upstream.intensity_normalization(array), axial_size=180, central_crop_along_z=True, central_crop_size=30)
                if cropped.shape != (30, 180, 180) or not np.isfinite(cropped).all() or cropped.min() < -1 or cropped.max() > 1:
                    raise ValueError('FSL output did not produce finite 30x180x180 slices in [-1,1]')
                destination = args.output_root / 'preprocessed_data' / relative.parent / (relative.name.split('.nii')[0] + '.nii.gz')
                destination.parent.mkdir(parents=True, exist_ok=True)
                upstream.save_2d(cropped, destination)
                if len(list(destination.parent.glob('*_slice*.tiff'))) != 30:
                    raise ValueError('Expected exactly 30 TIFFs in each acquisition folder')
            manifest['processed_scans'] += 1
            print(f'Processed scans: {manifest["processed_scans"]}/{len(selected)}', flush=True)
        manifest['status'] = 'smoke_completed' if args.max_scans else 'preprocessing_completed'
    except Exception as exc:
        manifest['status'], manifest['error'] = 'failed', str(exc)
        raise
    finally:
        (args.output_root / 'preprocessing_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
