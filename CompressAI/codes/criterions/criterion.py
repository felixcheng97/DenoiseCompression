import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

class Criterion(nn.Module):

    def __init__(self, opt):
        super(Criterion, self).__init__()
        self.opt = opt

        # criterions
        self.criterion_metric = opt['network']['criterions']['criterion_metric']
        self.criterion_fea = opt['network']['criterions']['criterion_fea']

        # lambdas
        self.lambda_metric = opt['network']['lambdas']['lambda_metric']
        self.lambda_fea = opt['network']['lambdas']['lambda_fea']

        self.metric_loss = RateDistortionLoss(lmbda=self.lambda_metric, criterion=self.criterion_metric)
        if self.criterion_fea:
            self.fea_loss = FeaLoss(lmbda=self.lambda_fea, criterion=self.criterion_fea)
        
    def forward(self, out_net, gt):
        out = {'loss': 0, 'rd_loss': 0}
        
        # bpp loss and metric loss
        out_metric = self.metric_loss(out_net, gt)
        out['loss'] += out_metric['bpp_loss']
        out['rd_loss'] += out_metric['bpp_loss']
        for k, v in out_metric.items():
            out[k] = v
            if 'weighted' in k:
                out['loss'] += v
                out['rd_loss'] += v

        # fea loss
        if self.criterion_fea:
            if 'y_inter' in out_net.keys():
                out_fea = self.fea_loss(out_net['y'], out_net['y_gt'], out_net['y_inter'], out_net['y_inter_gt'])
            else:
                out_fea = self.fea_loss(out_net['y'], out_net['y_gt'])
            for k, v in out_fea.items():
                out[k] = v
                if 'weighted' in k:
                    out['loss'] += v

        return out


# rate distortion loss
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, criterion='mse'):
        super().__init__()
        self.lmbda = lmbda
        self.criterion = criterion
        if self.criterion == 'mse':
            self.loss = nn.MSELoss()
        elif self.criterion == 'ms-ssim':
            from pytorch_msssim import ms_ssim
            self.loss = ms_ssim
        else:
            NotImplementedError('RateDistortionLoss criterion [{:s}] is not recognized.'.format(criterion))

    def forward(self, out_net, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_net["likelihoods"].values()
        )

        if self.criterion == 'mse':
            out["mse_loss"] = self.loss(out_net["x_hat"], target)
            out["weighted_mse_loss"] = self.lmbda * 255 ** 2 * out["mse_loss"]
        elif self.criterion == 'ms-ssim':
            out["ms_ssim_loss"] = 1 - self.loss(out_net["x_hat"], target, data_range=1.0)
            out["weighted_ms_ssim_loss"] = self.lmbda * out["ms_ssim_loss"]

        return out

# fea loss
class FeaLoss(nn.Module):
    def __init__(self, lmbda=1., criterion='l2'):
        super(FeaLoss, self).__init__()
        self.lmbda = lmbda
        self.criterion = criterion
        if self.criterion == 'l2':
            self.loss = nn.MSELoss()
        elif self.criterion == 'l1':
            self.loss = nn.L1Loss()
        else:
            NotImplementedError('FeaLoss criterion [{:s}] is not recognized.'.format(criterion))

    def forward(self, fea, fea_gt, fea_inter=None, fea_inter_gt=None):
        loss = self.loss(fea, fea_gt)
        if fea_inter is not None and fea_inter_gt is not None:
            loss += self.loss(fea_inter, fea_inter_gt)

        out = {
            'fea_loss': loss,
            'weighted_fea_loss': loss * self.lmbda,
        }
        return out

