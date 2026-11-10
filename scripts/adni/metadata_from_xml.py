#!/usr/bin/env python3
"""Convert IDA XML metadata to the benchmark collection CSV, without clinical joins.

All processed image IDs in the XML export are retained. Downloaded is marked
from NIfTI files/ZIP members actually present. This preserves upstream selection
over the full collection rather than choosing fallback scans when files are absent.
This is collection metadata, never an ADNIMERGE export.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

COLUMNS = ['Image Data ID', 'Subject', 'Visit', 'Acq Date', 'Description', 'Group',
           'Modality', 'Downloaded', 'Type', 'Format', 'Sex', 'Age']


def image_id(path):
    match = re.search(r'(?:^|[_/])I(\d+)(?:\D|$)', str(path))
    return 'I' + match.group(1) if match else None


def available_images(root):
    result = set()
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if path.name.endswith(('.nii', '.nii.gz')):
            identity = image_id(path)
            if identity:
                result.add(identity)
        elif path.parent.name.startswith('batch_') and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for entry in archive.infolist():
                    if entry.filename.endswith(('.nii', '.nii.gz')):
                        identity = image_id(entry.filename)
                        if identity:
                            result.add(identity)
    return result


def parse_record(path):
    root = ET.parse(path).getroot()
    for element in root.iter():
        element.tag = element.tag.rsplit('}', 1)[-1]
    subject = root.find('./project/subject')
    if subject is None:
        # Other XML files (e.g. acquisition sidecars) are not IDA exports.
        return None
    def text(parent, key):
        value = parent.findtext(key)
        if value is None or not value.strip():
            raise ValueError(f'Missing XML metadata field {key} in {path.name}')
        return value.strip()
    derived = subject.find('./study/series/seriesLevelMeta/derivedProduct')
    if derived is None:
        return None
    identity = text(derived, 'imageUID')
    identity = identity if identity.startswith('I') else 'I' + identity
    return {'Image Data ID': identity, 'Subject': text(subject, 'subjectIdentifier'),
            'Visit': text(subject, './visit/visitIdentifier'),
            'Acq Date': text(subject, './study/series/dateAcquired')[:10],
            'Description': text(derived, 'processedDataLabel'), 'Group': text(subject, 'researchGroup'),
            'Modality': text(subject, './study/series/modality'), 'Downloaded': 'Yes',
            'Type': 'Processed', 'Format': 'NIfTI', 'Sex': text(subject, 'subjectSex'),
            'Age': text(subject, './study/subjectAge')}


def convert(root, output):
    images = available_images(root)
    if not images:
        raise ValueError('No identifiable NIfTI image IDs in files or image ZIP archives')
    records, xml_count, duplicates, metadata_ids = {}, 0, 0, set()
    for path in root.rglob('*.xml'):
        xml_count += 1
        record = parse_record(path)
        if record is None:
            continue
        metadata_ids.add(record['Image Data ID'])
        identity = record['Image Data ID']
        record['Downloaded'] = 'Yes' if identity in images else 'No'
        if identity in records:
            if records[identity] != record:
                raise ValueError(f'Conflicting XML records for image ID {identity}')
            duplicates += 1
        records[identity] = record
    if images - records.keys():
        raise ValueError(f'{len(images - records.keys())} available NIfTI image IDs lack matching XML metadata')
    with output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records[key] for key in sorted(records))
    return {'xml_files': xml_count, 'identical_xml_duplicates': duplicates, 'available_image_ids': len(images),
            'collection_rows': len(records), 'metadata_image_ids': len(metadata_ids),
            'metadata_image_ids_without_available_nifti': len(metadata_ids - images),
            'clinical_export_created': False,
            'output': str(output.resolve()), 'source': 'IDA XML derivedProduct, subject and study/series fields'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = convert(args.input_root, args.output)
    with args.output.with_suffix('.manifest.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
