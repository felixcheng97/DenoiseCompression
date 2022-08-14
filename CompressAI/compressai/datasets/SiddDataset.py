import random
import os, glob
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SiddDataset(Dataset):
    def __init__(self, dataset_opt):
        self.root = dataset_opt['root']
        self.transform = transforms.ToTensor()
        self.patch_size = dataset_opt['patch_size']
        self.phase = dataset_opt['phase']
        if self.phase not in ['train', 'val', 'sidd']:
            raise NotImplementedError('wrong phase argument!')

        alpha = 60 if self.phase == 'train' else 1
        self.samples = []
        with open(self.root, 'r') as f:
            data = json.load(f)
            for i, scene in enumerate(sorted(data.keys())):
                all_scene_items = sorted(data[scene].items())
                for entry, entry_dict in all_scene_items:
                    for _ in range(alpha):
                        self.samples.append(entry_dict)
        

    def __getitem__(self, index):
        sample_dict = self.samples[index]

        if self.phase == 'train':
            img_dir = sample_dict['img_dir']
            gt_prefix = sample_dict['gt_prefix']
            noisy_prefix = sample_dict['noisy_prefix']            
            row, col = sample_dict['row'], sample_dict['col']
            h, w = sample_dict['h'], sample_dict['w']
            H, W = sample_dict['H'], sample_dict['W']

            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            r1 = rnd_h // h
            r2 = (rnd_h+self.patch_size-1) // h
            c1 = rnd_w // w
            c2 = (rnd_w+self.patch_size-1) // w
            rs = list(set({r1, r2}))
            cs = list(set({c1, c2}))
            rnd_h = rnd_h % h
            rnd_w = rnd_w % w
            # assert r1 < row and r2 < row and c1 < col and c2 < col, 'row={:d}, r1={:d}, r2={:d}; col={:d}, c1={:d}, c2={:d}'.format(row, r1, r2, col, c1, c2)

            gt = []
            noisy = []
            for r in rs:
                gt_r = []
                noisy_r = []
                for c in cs:
                    gt_path = os.path.join(img_dir, '{:s}_{:02d}_{:02d}.png'.format(gt_prefix, r+1, c+1))
                    gt_rc = Image.open(gt_path).convert("RGB")
                    gt_rc = self.transform(gt_rc)
                    gt_r.append(gt_rc)
                    
                    noisy_path = os.path.join(img_dir, '{:s}_{:02d}_{:02d}.png'.format(noisy_prefix, r+1, c+1))
                    noisy_rc = Image.open(noisy_path).convert("RGB")
                    noisy_rc = self.transform(noisy_rc)
                    noisy_r.append(noisy_rc)
                gt_r = torch.cat(gt_r, dim=2)
                gt.append(gt_r)
                noisy_r = torch.cat(noisy_r, dim=2)
                noisy.append(noisy_r) 
            gt = torch.cat(gt, dim=1)[:, rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size]
            noisy = torch.cat(noisy, dim=1)[:, rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size]
            return gt, noisy

        elif self.phase == 'val':
            gt = Image.open(sample_dict['gt_path']).convert("RGB")
            noisy = Image.open(sample_dict['noisy_path']).convert("RGB")
            gt = self.transform(gt)
            noisy = self.transform(noisy)
            return gt, noisy
        
        else:
            noisy = Image.open(sample_dict['noisy_path']).convert("RGB")
            noisy = self.transform(noisy)
            return noisy
            

    def __len__(self):
        return len(self.samples)

