#!/usr/bin/env python3
"""Inspect downloaded ADNI containers without extracting or transferring data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tarfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'input_root': str(args.input_root.resolve()), 'archives': [],
              'preprocessed_roots': [str(p) for p in args.input_root.rglob('preprocessed_data') if p.is_dir()],
              'clinical_csv_candidates': [str(p) for p in args.input_root.rglob('ADNIMERGE*.csv')]}
    all_files = [p for p in args.input_root.rglob('*') if p.is_file()]
    report['extracted_nifti_files'] = sum(p.name.endswith(('.nii', '.nii.gz')) for p in all_files)
    report['xml_metadata_files'] = sum(p.suffix.lower() == '.xml' for p in all_files)
    for path in sorted(args.input_root.glob('*/images/batch_*/*')):
        if not path.is_file():
            continue
        entry = {'path': str(path), 'bytes': path.stat().st_size, 'zip': zipfile.is_zipfile(path)}
        if entry['zip']:
            with zipfile.ZipFile(path) as archive:
                files = [i for i in archive.infolist() if not i.is_dir()]
                entry.update(members=len(files), extensions=dict(Counter(Path(i.filename).suffix for i in files)),
                             uncompressed_bytes=sum(i.file_size for i in files))
        report['archives'].append(entry)
    for path in args.input_root.glob('*/study_data/ADNIMERGE*.tar.gz'):
        with tarfile.open(path) as archive:
            report['legacy_merged_table_in_r_package'] = any(Path(n).name.lower() in {'adnimerge.rda', 'adnimerge.csv'} for n in archive.getnames())
    report['collection_csv_is_url_list'] = []
    for path in args.input_root.glob('downloader/*.csv'):
        with path.open() as stream:
            is_urls = stream.readline().startswith('https://')
        report['collection_csv_is_url_list'].append({'path': str(path), 'url_list': is_urls})
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
