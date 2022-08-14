import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from .utils import sRGBGamma, UndosRGBGamma
from torchvision import transforms

class SyntheticTestDataset(Dataset):
    def __init__(self, dataset_opt):
        root = Path(dataset_opt['root'])
        if not root.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = sorted([f for f in root.iterdir() if f.is_file()])
        self.phase = dataset_opt['phase']
        self.transform = transforms.ToTensor()

        noise_level = dataset_opt['level']
        sigma_reads = [0.0068354, 0.01572141, 0.03615925, 0.08316627]
        sigma_shots = [0.05200081**2, 0.07886314**2, 0.11960187**2, 0.18138522**2]
        self.sigma_read = sigma_reads[noise_level-1]
        self.sigma_shot = sigma_shots[noise_level-1]

    def __getitem__(self, index):
        gt = Image.open(self.samples[index]).convert("RGB")
        gt = self.transform(gt)

        # degamma
        noisy_degamma = UndosRGBGamma(gt)

        # read and shot noise
        sigma_read = torch.from_numpy(
            np.array([[[self.sigma_read]]])
        ).type_as(noisy_degamma)
        sigma_shot = torch.from_numpy(
            np.array([[[self.sigma_shot]]])
        ).type_as(noisy_degamma)
        sigma_read_com = sigma_read.expand_as(noisy_degamma)
        sigma_shot_com = sigma_shot.expand_as(noisy_degamma)

        # apply formular in the paper
        generator = torch.Generator()
        generator.manual_seed(0)
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
