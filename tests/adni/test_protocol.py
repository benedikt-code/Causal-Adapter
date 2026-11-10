"""Synthetic loader/metric regressions. No ADNI participant data or GPU required."""
import ast
import csv
from types import SimpleNamespace
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import warnings
import zipfile
import sys

import numpy as np
from PIL import Image
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'counterfactual-benchmark/counterfactual_benchmark/methods/deepscm'))
from scripts.adni.protocol import ATTRIBUTES, audit_data, unique_checkpoint, validate_paths
from scripts.adni.aggregate import aggregate, score_attribute
from causal_datasets.adni_dataset import load_data, load_extra_attributes, resolve_adnimerge, split_subjects
from adni_support import check_labels, cld_scores, kl_distance, sample_interventions, save_grid
from scripts.adni.metadata_from_xml import convert
from scripts.adni.preprocess import locate


def benchmark_loader():
    path = REPO / 'counterfactual-benchmark/counterfactual_benchmark/ctf_datasets/adni/dataset_SD.py'
    spec = importlib.util.spec_from_file_location('adni_sd_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ADNI


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv = self.root / 'ADNIMERGE.csv'
        columns = ['PTID', 'VISCODE', 'EXAMDATE', 'APOE4', 'AGE', 'PTGENDER', 'WholeBrain', 'Ventricles']
        with self.csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for i in range(10):
                subject = f'{i:03d}_S_0001'
                writer.writerow(dict(zip(columns, [subject, 'bl', '2010-01-01', 2, 70, 'Female', 1000000, 50000])))
                visit = self.root / 'preprocessed_data' / subject / 'scan' / '2010-01-01' / 'series'
                visit.mkdir(parents=True)
                for index in [0, *range(10, 20), 29]:
                    Image.fromarray(np.full((180, 180), -.5, dtype=np.float32)).save(visit / f'image_slice{index}.tiff')

    def tearDown(self):
        self.temp.cleanup()

    def test_splits_and_audit_match_actual_loader(self):
        audit = audit_data(self.root, self.csv)
        self.assertEqual(audit['unassigned_subjects_due_to_floor'], 1)
        subjects = list(range(10))
        splits = [set(split_subjects(subjects, name)) for name in ['train', 'valid', 'test']]
        self.assertFalse(splits[0] & splits[1] or splits[1] & splits[2] or splits[0] & splits[2])
        for split, count in [('train', 70), ('valid', 10), ('test', 10)]:
            images, attributes, dates = load_data(self.root / 'preprocessed_data', num_of_slices=10, split=split)
            self.assertEqual(len(images), count)
            joined, dropped = load_extra_attributes(self.csv, ATTRIBUTES, attributes, dates)
            self.assertEqual(dropped, [])
            self.assertEqual(len(joined['brain_vol']), count)
            self.assertEqual(audit['splits'][split]['kept_slices'], count)
            self.assertEqual(np.sum(joined['slice'][0]), 0)
            self.assertEqual(np.sum(joined['slice'][-1]), 9)
        with warnings.catch_warnings():
            # Constant synthetic covariates intentionally leave some bins empty.
            warnings.simplefilter('ignore', RuntimeWarning)
            dataset = benchmark_loader()(ATTRIBUTES, split='test', dataset_dir=self.root, csv_path=self.csv)
        self.assertEqual(dataset.attrs.shape, (10, 16))
        image, attr, index = dataset[0]
        self.assertEqual(image.shape, (3, 256, 256))
        self.assertEqual(image.dtype, torch.float32)
        self.assertTrue(torch.isfinite(image).all())
        self.assertGreaterEqual(image.min(), -1)
        self.assertLessEqual(image.max(), 1)
        offsets = [0, 2, 3, 4, 5, 6, 16]
        batch = {name: dataset.attrs[:, offsets[i]:offsets[i + 1]] for i, name in enumerate(ATTRIBUTES)}
        check_labels(batch)
        self.assertTrue(torch.equal(batch['apoE'][0], torch.tensor([1., 0.])))

    def test_missing_subject_and_single_visit_join(self):
        attributes = {'subject': ['absent', '000_S_0001'], 'date': ['2010-01-01'] * 2, 'slice': [np.zeros(10)] * 2}
        result, dropped = load_extra_attributes(self.csv, ATTRIBUTES, attributes, {'absent': ['2010-01-01'], '000_S_0001': ['2010-01-01']})
        self.assertEqual(dropped, [0])
        self.assertEqual(result['age'], [70.])

    def test_ambiguous_csv_and_checkpoint_fail(self):
        (self.root / 'ADNIMERGE_other.csv').write_text('')
        with self.assertRaises(ValueError):
            resolve_adnimerge(self.root)
        self.assertEqual(resolve_adnimerge(self.root, self.csv), self.csv)
        with self.assertRaises(ValueError):
            unique_checkpoint(self.root, 'brain_vol')
        for name in ['brain_vol-a.ckpt', 'brain_vol-b.ckpt']:
            (self.root / name).touch()
        with self.assertRaises(ValueError):
            unique_checkpoint(self.root, 'brain_vol')
        with self.assertRaises(ValueError):
            split_subjects([], 'typo')

    def test_positional_date_mismatch_reported_without_repair(self):
        text = self.csv.read_text().replace('2010-01-01', '2009-12-31')
        self.csv.write_text(text)
        audit = audit_data(self.root, self.csv)
        self.assertEqual(audit['splits']['test']['date_mismatch_visits'], 1)
        self.assertEqual(audit['splits']['test']['kept_slices'], 10)

    def test_kl_uses_logvariance_and_broadcasting(self):
        first = np.array([[0., 0.], [0., 0.]])
        second = np.array([[0., 0.], [np.log(4.), np.log(4.)]])
        self.assertAlmostEqual(float(kl_distance(first, first)), 0)
        self.assertAlmostEqual(float(kl_distance(first, second)), .5 * (np.log(4) + .25 - 1))
        self.assertEqual(kl_distance(np.stack([first, second]), first).shape, (2,))
        scores, skipped = cld_scores(np.stack([first, second]), {'brain_vol': np.array([.2, .8])},
                                    [first], [second], ['brain_vol'], [.2], [.8], {'brain_vol': [.5]})
        self.assertEqual(skipped, 0)
        self.assertEqual(len(scores), 1)
        self.assertTrue(np.isfinite(scores).all())

    def test_interventions_independent_of_batching(self):
        class Dataset:
            possible_values = {'brain_vol': torch.tensor([.1, .3, .6, .9])}
            bins = {'brain_vol': [.25, .5, .75]}
        batch = {'brain_vol': torch.tensor([[.1], [.6]])}
        full = sample_interventions(batch, [12, 13], 'brain_vol', Dataset(), 40)['brain_vol']
        parts = torch.cat([sample_interventions({'brain_vol': batch['brain_vol'][i:i + 1]}, [12 + i], 'brain_vol', Dataset(), 40)['brain_vol'] for i in range(2)])
        self.assertTrue(torch.equal(full, parts))
        self.assertFalse(torch.equal(full, batch['brain_vol']))

    def test_aggregation_final_partial_chunk_and_singleton(self):
        logits = np.array([[0.], [0.], [0.]])
        targets = np.array([[0.], [0.], [.5]])
        result = score_attribute(logits, targets, 'brain_vol', 2)
        self.assertAlmostEqual(result['value'], 1 / 3)
        self.assertEqual(result['last_chunk_n'], 1)
        self.assertEqual(result['chunk_mean'], .25)
        ordinal = np.zeros((1, 10)); ordinal[0, -4:] = 1
        logits = np.zeros((1, 10)); logits[0, 4] = 10
        result = score_attribute(logits, ordinal, 'slice', 256)
        self.assertEqual(result['value'], 1.)
        self.assertEqual(result['n'], 1)

    def test_evaluator_edit_arguments_and_encoded_label_shapes(self):
        # Exercise the real evaluator function without importing CUDA-only model
        # dependencies. Its DDIM boundary must retain mandatory pseudo-token IDs.
        path = REPO / 'counterfactual-benchmark/counterfactual_benchmark/methods/deepscm/evaluate_SD_DSCM.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'produce_counterfactuals')
        node.returns = None
        for argument in node.args.args:
            argument.annotation = None
        def ddim(pipe, image, label, token_ids, prompt, **kwargs):
            self.assertEqual(token_ids, [12, 13, 14])
            self.assertEqual(prompt, 'padded prompt')
            target = kwargs['DSCM_labels']
            return image, None, target.unsqueeze(-1), None
        helpers = SimpleNamespace(get_dataset_attrs=lambda dataset: list(ATTRIBUTES),
                                  ddim_editing=ddim, resize_tensor=lambda image, *args, **kw: image)
        context = {'torch': torch, 'ddim_modules': helpers}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), context)
        class FlowSCM:
            def encode(self, **batch):
                return {k: v for k, v in batch.items() if k != 'image'}
            def decode(self, intervention, **noise):
                return {**noise, **intervention}
        for count in [1, 2]:
            batch = {name: torch.zeros(count, width) for name, width in ATTRIBUTES.items()}
            batch['image'] = torch.zeros(count, 3, 256, 256)
            for parent in ['brain_vol', 'slice']:
                target = torch.ones(count, ATTRIBUTES[parent])
                args = SimpleNamespace(dataset='ADNI', editing='standard', num_steps=100,
                                       guidance_scale=1., NTI=False, reverse=False,
                                       attribute_size=ATTRIBUTES, presudo_token_ids=[12, 13, 14])
                result = context['produce_counterfactuals'](None, 'padded prompt', batch, FlowSCM(), parent,
                                   None, device='cpu', args=args, interventions={parent: target})
                self.assertEqual(result[parent].shape, target.shape)
                self.assertTrue(torch.equal(result[parent], target))
                self.assertEqual(sum(result[k].shape[1] for k in ATTRIBUTES), 16)

    def test_xml_collection_conversion_and_archive_location(self):
        input_root = self.root / 'raw'
        batch = input_root / 'images' / 'batch_1'
        batch.mkdir(parents=True)
        archive = batch / 'ADNI1'
        name = 'ADNI/000_S_0001/scan/2010-01-01/I123/ADNI_I123.nii'
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr(name, b'synthetic nifti placeholder')
        xml = '<idaxs><project><subject><subjectIdentifier>000_S_0001</subjectIdentifier><researchGroup>CN</researchGroup><subjectSex>F</subjectSex><visit><visitIdentifier>screening</visitIdentifier></visit><study><subjectAge>70</subjectAge><series><dateAcquired>2010-01-01</dateAcquired><modality>MRI</modality><seriesLevelMeta><derivedProduct><imageUID>123</imageUID><processedDataLabel>MPR</processedDataLabel></derivedProduct></seriesLevelMeta></series></study></subject></project></idaxs>'
        (input_root / 'ida.xml').write_text(xml)
        output = self.root / 'collection.csv'
        report = convert(input_root, output)
        self.assertEqual(report['collection_rows'], 1)
        self.assertEqual(report['metadata_image_ids_without_available_nifti'], 0)
        self.assertFalse(report['clinical_export_created'])
        with output.open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row['Image Data ID'], 'I123')
        self.assertEqual(row['Description'], 'MPR')
        self.assertEqual(locate(input_root, {'I123'})['I123'][2].parts, tuple(name.split('/')[1:]))
        with self.assertRaises(FileExistsError):
            convert(input_root, output)
        # Missing files must remain visible in the full collection metadata,
        # so upstream preselection cannot silently choose fallback scans.
        (input_root / 'missing-image.xml').write_text(xml.replace('<imageUID>123', '<imageUID>124'))
        full_report = convert(input_root, self.root / 'full-collection.csv')
        self.assertEqual(full_report['collection_rows'], 2)
        self.assertEqual(full_report['metadata_image_ids_without_available_nifti'], 1)
        with self.assertRaisesRegex(ValueError, 'lack a unique NIfTI'):
            locate(input_root, {'I123', 'I124'})
        (input_root / 'conflict.xml').write_text(xml.replace('<researchGroup>CN', '<researchGroup>AD'))
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            convert(input_root, self.root / 'bad.csv')

    def test_smoke_grid_is_nonempty_and_has_differences(self):
        original = torch.zeros(1, 192, 192)
        edited = torch.ones(1, 192, 192)
        save_grid(self.root, 'synthetic-smoke', original, original, [edited] * 3)
        with Image.open(self.root / 'synthetic-smoke.png') as image:
            self.assertEqual(image.size, (960, 406))
            self.assertEqual(image.getpixel((400, 220)), (255, 255, 255))

    def test_aggregate_refuses_incomplete_full_population(self):
        report = {'scope': 'full', 'test_samples': 3, 'metrics': {}, 'protocol_deviations': []}
        (self.root / 'metrics.json').write_text(json.dumps(report))
        directory = self.root / 'brain_vol'; directory.mkdir()
        with (directory / 'final_results.pkl').open('wb') as stream:
            pickle.dump({'indices': [0, 1]}, stream)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            aggregate(self.root)


if __name__ == '__main__':
    unittest.main()
