import random
import numpy as np
import torch

def sRGBGamma(tensor):
    threshold = 0.0031308
    a = 0.055
    mult = 12.92
    gamma = 2.4
    res = torch.zeros_like(tensor)
    mask = tensor > threshold
    res[mask] = (1 + a) * torch.pow(tensor[mask] + 0.001, 1.0 / gamma) - a
    res[~mask] = tensor[~mask] * mult
    return res

def UndosRGBGamma(tensor):
    threshold = 0.0031308
    a = 0.055
    mult = 12.92
    gamma = 2.4
    res = torch.zeros_like(tensor)
    mask = tensor > threshold
    res[~mask] = tensor[~mask] / mult
    res[mask] = torch.pow(tensor[mask] + a, gamma) / (1 + a)
    return res

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed=seed)