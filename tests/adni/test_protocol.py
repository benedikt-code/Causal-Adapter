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
import subprocess

import numpy as np
from PIL import Image
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'counterfactual-benchmark/counterfactual_benchmark/methods/deepscm'))
from scripts.adni.protocol import ATTRIBUTES, audit_data, unique_checkpoint, validate_paths
from scripts.adni.aggregate import aggregate, score_attribute
from causal_datasets.adni_dataset import load_data, load_extra_attributes, resolve_adnimerge, split_subjects
from adni_support import check_labels, cld_scores, kl_distance, sample_interventions, save_grid, subject_balanced_indices
from scripts.adni.metadata_from_xml import convert
from scripts.adni.preprocess import locate
from scripts.adni.run import command
from scripts.adni.qualitative import select_cases, generate_case, read_factual, signed_difference, write_gallery


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

    def test_focused_subset_is_balanced_unique_and_repeatable(self):
        subjects = ['subject-a'] * 10 + ['subject-b'] * 10 + ['subject-c'] * 10
        indices = subject_balanced_indices(subjects, 8, 40)
        self.assertEqual(indices, subject_balanced_indices(subjects, 8, 40))
        self.assertEqual(len(set(indices)), 8)
        counts = [sum(subjects[i] == subject for i in indices) for subject in sorted(set(subjects))]
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertNotEqual(indices, subject_balanced_indices(subjects, 8, 41))
        with self.assertRaises(ValueError):
            subject_balanced_indices(subjects, 31, 40)

    def test_focused_command_omits_cld_and_keeps_paper_sampling(self):
        config = {key: 'explicit-path' for key in ['data_root', 'adnimerge_csv', 'controlnet', 'embeddings', 'flow_dir', 'predictor_dir']}
        config.update(metrics=['composition', 'effectiveness'], focused_samples=512)
        args = command(config, 'focused', self.root / 'result')
        self.assertEqual(args[args.index('--max-samples') + 1], '512')
        self.assertEqual(args[args.index('--num_steps') + 1], '100')
        self.assertNotIn('--unconditional-vae', args)
        self.assertNotIn('minimality', args)
        self.assertNotIn('fid', args)
        self.assertEqual(args[args.index('--NTI') + 1], 'False')

    def test_qualitative_selection_preserves_split_and_join(self):
        cases, audit = select_cases(self.root, self.csv, 'all', 4, 40)
        again, _ = select_cases(self.root, self.csv, 'all', 4, 40)
        self.assertEqual(cases, again)
        self.assertEqual(len({case['subject'] for case in cases}), 4)
        self.assertFalse(audit['held_out_claim'])
        for case in cases:
            self.assertEqual(case['stored_slice'], 15)
            self.assertEqual(len(case['label']), 16)
            self.assertTrue(case['scan_and_exam_dates_match'])
        held_out, audit = select_cases(self.root, self.csv, 'test', 1, 40)
        self.assertEqual(held_out[0]['subject'], '008_S_0001')
        self.assertTrue(audit['held_out_claim'])
        with self.assertRaisesRegex(ValueError, 'only 1'):
            select_cases(self.root, self.csv, 'test', 4, 40)

    def test_qualitative_generation_reuses_inversion_and_saves_all_outputs(self):
        from torchvision import transforms
        case = select_cases(self.root, self.csv, 'all', 1, 40)[0][0]
        prefix = [transforms.Pad(6), transforms.Resize((256, 256))]
        image_transform = transforms.Compose(prefix + [transforms.ToTensor(), transforms.Normalize([.5], [.5])])
        original_transform = transforms.Compose(prefix)
        factual = original_transform(read_factual(case['image_path']))
        assets = SimpleNamespace(pipe=None, prompt='padded ADNI prompt', presudo_token_ids=[12, 13, 14], device=torch.device('cpu'))
        calls = {'inversions': 0, 'samples': []}
        latent = torch.ones(1, 4, 32, 32)
        def invert(pipe, image, label, token_ids, prompt, **kwargs):
            calls['inversions'] += 1
            self.assertEqual(kwargs['num_steps'], 100)
            self.assertFalse(kwargs['pnp_inversion'])
            self.assertEqual(token_ids, assets.presudo_token_ids)
            return [factual], [latent], None, None
        def sample(pipe, prompt, token_ids, **kwargs):
            self.assertTrue(torch.equal(kwargs['start_latents'], latent))
            self.assertEqual(kwargs['num_inference_steps'], 100)
            self.assertEqual(kwargs['guidance_scale'], 1.)
            self.assertFalse(kwargs['disentangle'])
            index, value = kwargs['intervention_indx'], kwargs['intervention_values']
            # Return propagated labels, as the real helper does. The runner must
            # retain these instead of replacing them with the requested scalar.
            encoded = kwargs['label'].clone()
            offset = {1: 2, 3: 4, 4: 5}[index]
            encoded[0, offset] = value
            if index == 3:
                encoded[0, 5] = value / 2
            calls['samples'].append(index)
            return [Image.new('RGB', (256, 256), color=(int(value * 255),) * 3)], encoded.unsqueeze(-1)
        target = self.root / 'case_00'
        result = generate_case(assets, case, target, (image_transform, original_transform), invert, sample, 5)
        self.assertEqual(calls['inversions'], 1)
        self.assertEqual(calls['samples'], [4] * 5 + [3] * 2 + [1] * 2)
        self.assertEqual(result['png_files'], 47)
        self.assertEqual(result['interventions'], 9)
        metadata = json.loads((target / 'case.json').read_text())
        self.assertEqual(metadata['interventions'][5]['encoded_label'][5], .125)
        with np.load(target / 'labels_and_differences.npz') as values:
            self.assertTrue(np.all(values['reconstruction_minus_original'] == 0))
            self.assertEqual(values['vent_vol_00_minus_original'].shape, (256, 256))
        write_gallery(self.root, [result])
        self.assertIn('case_00/vent_vol_grid.png', (self.root / 'index.html').read_text())
        with Image.open(target / 'benchmark_192/original.png') as image:
            self.assertEqual(image.size, (192, 192))
            self.assertEqual(image.mode, 'L')
        with self.assertRaises(FileExistsError):
            generate_case(assets, case, target, (image_transform, original_transform), invert, sample, 5)

    def test_signed_difference_colors_preserve_raw_values(self):
        invalid = self.root / 'integer-input.tiff'
        Image.fromarray(np.zeros((180, 180), dtype=np.uint8)).save(invalid)
        with self.assertRaisesRegex(ValueError, 'floating-point'):
            read_factual(invalid)
        black = Image.new('RGB', (2, 2), 'black')
        white = Image.new('RGB', (2, 2), 'white')
        shown, raw = signed_difference(white, black)
        self.assertEqual(shown.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(float(raw[0, 0]), 255.)
        shown, raw = signed_difference(black, white)
        self.assertEqual(shown.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(float(raw[0, 0]), -255.)
        shown, raw = signed_difference(white, white)
        self.assertEqual(shown.getpixel((0, 0)), (255, 255, 255))
        self.assertTrue(np.all(raw == 0))

    def test_qualitative_cpu_preflight_needs_no_benchmark_models_or_cuda(self):
        base = self.root / 'base'
        base.mkdir()
        (base / 'model_index.json').write_text('{}')
        for name in ['tokenizer', 'text_encoder', 'unet', 'vae', 'scheduler']:
            (base / name).mkdir()
        weights = self.root / 'ADNI'
        control = weights / 'controlnet/controlnet-steps-100000.safetensors'
        control.mkdir(parents=True)
        (control / 'config.json').write_text(json.dumps({'_class_name': 'Causal_ControlNetModel', 'dataset': 'ADNI', 'task_cond': 'generation_text_global_after'}))
        (control / 'diffusion_pytorch_model.safetensors').write_bytes(b'synthetic placeholder, not model weights')
        (control.parent / 'learned_embeds-steps-100000.safetensors').write_bytes(b'synthetic placeholder')
        (weights / 'scm').mkdir()
        (weights / 'scm/best_model.pt').write_bytes(b'synthetic placeholder')
        output = self.root / 'preflight-output'
        cmd = [sys.executable, str(REPO / 'scripts/adni/qualitative.py'), '--data-root', str(self.root),
               '--base-model', str(base), '--weights-root', str(weights), '--output-root', str(output),
               '--subjects', '2', '--split', 'all', '--preflight-only']
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'inputs_validated_no_images_generated')
        self.assertFalse(manifest['protocol']['quantitative_reproduction'])
        self.assertEqual(len(manifest['cases']), 2)
        self.assertFalse(list(output.glob('case_*')))
        contents = (output / 'manifest.json').read_bytes()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((output / 'manifest.json').read_bytes(), contents)

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
