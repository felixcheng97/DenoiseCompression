import os
import sys
import time
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
from torchvision.utils import make_grid
from shutil import get_terminal_size
from PIL import Image

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from compressai.zoo import models
from compressai.zoo.image import model_architectures as architectures
from typing import Tuple, Union
from pytorch_msssim import ms_ssim

# optimizers
def configure_optimizers(opt, net):
    parameters = [
        # p for n, p in net.named_parameters() if not n.endswith(".quantiles")
        p for n, p in net.named_parameters() if not n.endswith(".quantiles") and not "discriminator" in n
    ]
    D_parameters = [
        p for n, p in net.named_parameters() if not n.endswith(".quantiles") and "discriminator" in n
    ]
    aux_parameters = [
        p for n, p in net.named_parameters() if n.endswith(".quantiles")
    ]

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = set(parameters) & set(aux_parameters) & set(D_parameters)
    union_params = set(parameters) | set(aux_parameters) | set(D_parameters)

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    mode = opt['train']['mode']
    optimizer_dict = {}
    optimizer_dict['optimizer'] = torch.optim.Adam(
        (p for p in parameters if p.requires_grad),
        lr=opt['train'][mode]['lr'],
    )
    if len(D_parameters) > 0:
        optimizer_dict['D_optimizer'] = torch.optim.Adam(
            (p for p in D_parameters if p.requires_grad),
            lr=opt['train'][mode]['lr'],
        )
    optimizer_dict['aux_optimizer'] = torch.optim.Adam(
        (p for p in aux_parameters if p.requires_grad),
        lr=opt['train'][mode]['lr_aux'],
    )
    return optimizer_dict

def load_optimizer(optimizer_dict, name, checkpoint):
    optimizer = optimizer_dict.get(name, None)
    if optimizer is not None and checkpoint is not None:
        optimizer.load_state_dict(checkpoint[name])
    return optimizer

# schedulers
def configure_schedulers(opt, optimizer_dict):
    mode = opt['train']['mode']

    scheduler = opt['train'][mode]['lr_scheme']
    warm_up_counts = opt['train'][mode]['warm_up_counts']
    milestones = opt['train'][mode]['milestones']
    gamma = opt['train'][mode]['gamma']
    
    scheduler_dict = {}
    if scheduler == 'MultiStepLR':
        scheduler_dict['lr_scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_dict['optimizer'], 
            milestones=milestones,
            gamma=gamma
        )
        if 'D_optimizer' in optimizer_dict.keys():
            scheduler_dict['D_lr_scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_dict['D_optimizer'], 
                milestones=milestones,
                gamma=gamma
            )
        scheduler_dict['aux_lr_scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_dict['aux_optimizer'], 
            milestones=[],
            gamma=1.0
        )
    elif scheduler == 'LambdaLR':
        warm_up_with_multistep_lr = lambda i: (i + 1) / warm_up_counts if i < warm_up_counts else gamma**len([m for m in milestones if m <= i])
        scheduler_dict['lr_scheduler'] = torch.optim.lr_scheduler.LambdaLR(
            optimizer_dict['optimizer'], 
            lr_lambda=warm_up_with_multistep_lr
        )
        if 'D_optimizer' in optimizer_dict.keys():
            scheduler_dict['D_lr_scheduler'] = torch.optim.lr_scheduler.LambdaLR(
                optimizer_dict['D_optimizer'], 
                lr_lambda=warm_up_with_multistep_lr
            )
        warm_up_with_multistep_lr = lambda i: (i + 1) / warm_up_counts if i < warm_up_counts else 1.0
        scheduler_dict['aux_lr_scheduler'] = torch.optim.lr_scheduler.LambdaLR(
            optimizer_dict['aux_optimizer'], 
            lr_lambda=warm_up_with_multistep_lr
        )
    elif scheduler == 'ReduceLROnPlateau':
        scheduler_dict['lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_dict['optimizer'], "min")
        if 'D_optimizer' in optimizer_dict.keys():
            scheduler_dict['D_lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_dict['D_optimizer'], "min")
        scheduler_dict['aux_lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_dict['aux_optimizer'], "min")
    else:
        raise NotImplementedError('scheduler [{:s}] is not recognized.'.format(scheduler))

    return scheduler_dict

def load_scheduler(scheduler_dict, name, checkpoint):
    lr_scheduler = scheduler_dict.get(name, None)
    if lr_scheduler is not None and checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint[name])
        # print(lr_scheduler.state_dict())
        # lr_scheduler._step_count = checkpoint[name]['_step_count']
        # lr_scheduler.last_epoch = checkpoint['lr_scheduler']['last_epoch']
    return lr_scheduler

# model
def create_model(opt, checkpoint, rank):
    logger = logging.getLogger('base')
    model = opt['network']['model']
    quality = int(opt['network']['quality'])
    metric = opt['network']['criterions']['criterion_metric']
    pretrained = opt['network']['pretrained']

    if checkpoint is not None:
        m = architectures[model].from_state_dict(checkpoint['state_dict'], opt)
        m.update()
    else:
        m = models[model](quality=quality, metric=metric, pretrained=pretrained, opt=opt)

    print_network(m, rank)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

def print_network(net, rank):
    logger = logging.getLogger('base')
    if isinstance(net, nn.DataParallel) or isinstance(net, nn.parallel.DistributedDataParallel):
        net = net.module
    s = str(net)
    n = sum(map(lambda x: x.numel(), net.parameters()))
    if isinstance(net, nn.DataParallel) or isinstance(net, nn.parallel.DistributedDataParallel):
        net_struc_str = '{} - {}'.format(net.__class__.__name__,
                                            net.module.__class__.__name__)
    else:
        net_struc_str = '{}'.format(net.__class__.__name__)\

    m = 'structure: {}, with parameters: {:,d}'.format(net_struc_str, n)
    if rank <= 0:
        logger.info(m)
        logger.info(s)

# dataloader
def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = dataset_opt['use_shuffle']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers, sampler=sampler, drop_last=True,
                                            pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                           pin_memory=False)

# dataset
def create_dataset(dataset_opt):
    phase = dataset_opt['phase']
    if phase == 'train':
        transform = transforms.Compose(
            [transforms.RandomCrop(dataset_opt['patch_size']), transforms.ToTensor()]
        )
    elif phase == 'val':
        transform = transforms.Compose(
            [transforms.CenterCrop(dataset_opt['patch_size']), transforms.ToTensor()]
        )
    else:
        raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    mode = dataset_opt['name']
    if mode == 'image':
        from compressai.datasets import ImageFolder as D
    elif mode == 'sidd':
        transform = transforms.ToTensor()
        from compressai.datasets import SiddDataset as D
    elif mode == 'synthetic':
        from compressai.datasets import SyntheticDataset as D
    elif mode == 'synthetic-est':
        from compressai.datasets import SyntheticEstDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    
    dataset = D(dataset_opt['root'], transform, phase, dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

# related to compression
def torch2img(x: torch.Tensor) -> Image.Image:
    return transforms.ToPILImage()(x.clamp_(0, 1).squeeze()[0:3])

def compute_metrics(
    a: Union[np.array, Image.Image],
    b: Union[np.array, Image.Image],
    max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`. """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    m = ms_ssim(a, b, data_range=max_val).item()
    return p, m


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


####################
# image convert
####################

def tensor2img(tensor, out_type=np.uint8):
    tensor *= 255.0
    tensor = tensor.float().cpu().clamp_(0, 255.)
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC
    if out_type == np.uint8:
        img_np = img_np.round()
    return img_np.astype(out_type)

def tensor2imgs(tensor, out_type=np.uint8):
    tensor *= 255.0
    tensor = tensor.float().cpu().clamp_(0, 255.)
    imgs_np = tensor.numpy()
    imgs_np = np.stack(np.split(imgs_np, imgs_np.shape[0]//3, axis=0), axis=0)
    imgs_np = np.transpose(imgs_np, (0, 2, 3, 1))
    if out_type == np.uint8:
        imgs_np = imgs_np.round()
    return imgs_np.astype(out_type)

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


####################
# metric
####################

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

