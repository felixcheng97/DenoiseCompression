import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from .utils import sRGBGamma, UndosRGBGamma

class SyntheticDataset(Dataset):
    def __init__(self, root, transform=None, split="train", dataset_opt=None):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = sorted([f for f in splitdir.iterdir() if f.is_file()])
        
        self.transform = transform
        self.split = split

        if self.split == 'test' or self.split == 'val':
            self.sigma_reads = [0.0068354, 0.01572141, 0.03615925, 0.08316627]
            self.sigma_shots = [0.05200081**2, 0.07886314**2, 0.11960187**2, 0.18138522**2]
            self.choices = len(self.sigma_reads)
        elif self.split == 'train':
            pass
        else:
            raise NotImplementedError('wrong split argument!')

    def __getitem__(self, index):
        gt = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            gt = self.transform(gt)

        # degamma
        noisy_degamma = UndosRGBGamma(gt)

        # sample read and shot noise
        if self.split == 'train':
            sigma_read = torch.from_numpy(
                np.power(10, np.random.uniform(-3.0, -1.5, (1, 1, 1)))
            ).type_as(noisy_degamma)
            sigma_shot = torch.from_numpy(
                np.power(10, np.random.uniform(-4.0, -2.0, (1, 1, 1)))
            ).type_as(noisy_degamma)
        else:
            sigma_read = torch.from_numpy(
                np.array([[[self.sigma_reads[index % self.choices]]]])
            ).type_as(noisy_degamma)
            sigma_shot = torch.from_numpy(
                np.array([[[self.sigma_shots[index % self.choices]]]])
            ).type_as(noisy_degamma)

        sigma_read_com = sigma_read.expand_as(noisy_degamma)
        sigma_shot_com = sigma_shot.expand_as(noisy_degamma)

        # apply formular in the paper
        if self.split == 'train':
            generator = None
        else:
            generator = torch.Generator()
            generator.manual_seed(index)
            
        noisy_degamma = torch.normal(noisy_degamma, 
            torch.sqrt(sigma_read_com ** 2 + noisy_degamma * sigma_shot_com),
            generator=generator
        ).type_as(noisy_degamma)

        # gamma
        noisy = sRGBGamma(noisy_degamma)

        # clamping
        noisy = torch.clamp(noisy, 0.0, 1.0)

        return gt, noisy

    def __len__(self):
        return len(self.samples)
