import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms
import os

MIN_MAX = {
    'image': [0.0, 255.0],
    'age': [55.1, 89.3],
    'brain_vol': [669364.0, 1350180.0],
    'vent_vol': [5834.0, 145115.0]
}

ATTRIBUTE_MAPPING = {
    "apoE": "APOE4",
    # "av45": "AV45",
    # "pT": "PTAU",
    "age": "AGE",
    "sex": "PTGENDER",
    "brain_vol": "WholeBrain",
    "vent_vol": "Ventricles",
    # "education": "PTEDUCAT",
    # "moc_score": "MOCA"
}

TOTAL_SLICES = 30

SPLIT_TRAIN_VAL_TEST = [
    ('train', 0.7),
    ('valid', 0.15),
    ('test', 0.15)
]

def split_subjects(subjects, split):
    offset = 0
    for split_, percent in SPLIT_TRAIN_VAL_TEST:
        split_size = int(percent * len(subjects))
        if split_ == split:
            return subjects[offset:offset+split_size]
        offset += split_size
    return subjects[offset:offset+split_size]

def bin_array(num, m=None, reverse=False):
    if reverse:
      return torch.matmul(num, (torch.flip(torch.arange(num.shape[1], dtype=torch.float32), [0]) + 1).to(num.device))
    else:
        return np.array(list(np.binary_repr(num).zfill(m))).astype(np.float32)

def ordinal_array(num, m=None, reverse=False, scale=1):
    if reverse:
        return scale * torch.count_nonzero(num, dim=1).to(num.device)
    else:
        return np.pad(np.ones(num), (m - num, 0), 'constant').astype(np.float32)

def encode_attribute(value, name, num_of_slices=30):
    if name == 'sex':
        return 0 if value == 'Female' else 1
    elif name == "apoE":
        return bin_array(int(value), 2)
    elif name == "slice":
        return ordinal_array(int(value), num_of_slices)
    else:
        return float(value)

def fix_age(initial_age, visit):
    return initial_age if visit == 'bl' else initial_age + float(visit[1:]) / 12
# adni(0,1) for contiuous
def normalize(value, name):
    if name not in MIN_MAX:
        return value
    # [min,max] -> [0,1]
    value = (value - MIN_MAX[name][0]) / (MIN_MAX[name][1] - MIN_MAX[name][0])
    return value

def unnormalize(value, name):
    if name not in MIN_MAX:
        return value
    # [0,1] -> [min,max]
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) +  MIN_MAX[name][0]
    return value

# Share the training loader's split and clinical pairing without changing either.
from causal_datasets.adni_dataset import load_data, load_extra_attributes, resolve_adnimerge

def transform_img_SD_input(img):
    if img.shape != (180, 180) or not np.isfinite(img).all() or img.min() < 0 or img.max() > 1:
        raise ValueError("Expected finite 180x180 ADNI slices in [0,1] after TIFF [-1,1] conversion")
    image_array = (img * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image_array, mode="L")
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image_transforms = transforms.Compose(
            [
                transforms.Pad(padding=6),
                transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )
    return image_transforms(image)

class ADNI(Dataset):
    def __init__(self, attribute_size, split='train', normalize_=True,
                 transform=None, transform_cls=None, num_of_slices=30, keep_only_screening=False,
                 dataset_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)),'preprocessing'), csv_path=None):
        super().__init__()
        self.has_valid_set = True
        self.transform = transform
        self.transform_cls = transform_cls
        # pad: 180x180 -> 192x192
        self.pad = transforms.Pad(padding=6)

        num_of_slices = attribute_size['slice']

        assert num_of_slices <= 30, "The 30 middle slices have been saved"
        data_dir = os.path.join(dataset_dir, 'preprocessed_data')
        images, attribute_dict, subject_dates_dict = load_data(data_dir, num_of_slices=num_of_slices,
                                                               split=split,
                                                               keep_only_screening=keep_only_screening)

        csv_path = resolve_adnimerge(dataset_dir, csv_path)
        self.attributes, indices_to_remove = load_extra_attributes(csv_path, attributes=attribute_size.keys(),
                                                                   attribute_dict=attribute_dict, subject_dates_dict=subject_dates_dict,
                                                                   keep_only_screening=keep_only_screening)
        self.attributes['slice'] = np.delete(self.attributes['slice'], indices_to_remove, axis=0)

        removed = set(indices_to_remove)
        self.images = [transform_img_SD_input(item) for i, item in enumerate(images) if i not in removed]
        if not self.images:
            raise ValueError(f'No usable ADNI slices in split {split} at {data_dir}')
        self.audit = {'loaded_slices': len(images), 'dropped_slices': len(removed),
                      'kept_slices': len(self.images), 'csv_path': str(csv_path)}
        
        #self.images = self.pad(torch.as_tensor(np.delete(images, indices_to_remove, axis=0).copy(), dtype=torch.float32).unsqueeze(1))

        if normalize_:
            self.attributes = {attr: normalize(torch.tensor(np.array(values), dtype=torch.float32), attr) for attr, values in self.attributes.items()}
        else:
            self.attributes = {attr: torch.tensor(np.array(values), dtype=torch.float32) for attr, values in self.attributes.items()}

        self.attrs = torch.cat([self.attributes[attr].unsqueeze(1) if len(self.attributes[attr].shape) == 1 else self.attributes[attr]
                                for attr in attribute_size.keys()], dim=1)

        self.possible_values = {attr: torch.unique(values, dim=0) for attr, values in self.attributes.items()}

        bins = np.linspace(0, 1, 5)
        self.bins = {}
        for attr, values in self.attributes.items():
            if attr not in ["sex", "apoE", "slice"]:
                data = values.numpy()
                digitized = np.digitize(data, bins)
                self.bins[attr] = [data[digitized == i].mean() for i in range(1, len(bins))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.images[idx], self.attrs[idx]),idx

        if self.transform_cls:
            return self.transform_cls(self.images[idx]), self.attrs[idx],idx

        return self.images[idx], self.attrs[idx],idx
