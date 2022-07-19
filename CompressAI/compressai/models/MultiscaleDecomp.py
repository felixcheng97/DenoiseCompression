import torch
import torch.nn as nn
from torch.nn import functional as F
from .waseda import Cheng2020Anchor
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    conv1x1
)
import warnings

class MultiscaleDecomp(Cheng2020Anchor):
    def __init__(self, N=192, opt=None, **kwargs):
        super().__init__(N = N, **kwargs)
        self.use_fea = opt['network']['criterions']['criterion_fea']

        self.g_a = None
        self.g_a_block1 = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
        )
        self.g_a_block2 = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.denoise_module_1 = AttentionBlock(N)
        self.denoise_module_2 = AttentionBlock(N)

    def g_a_func(self, x, denoise=False):        
        x = self.g_a_block1(x)
        if denoise:
            x = self.denoise_module_1(x)
        y_inter = x

        x = self.g_a_block2(x)
        if denoise:
            x = self.denoise_module_2(x)
        y = x

        return y_inter, y

    def forward(self, x, gt=None):
        # g_a for noisy input
        y_inter, y = self.g_a_func(x, denoise=True)

        # g_a for clean input
        if self.use_fea and gt is not None:
            y_inter_gt, y_gt = self.g_a_func(gt)
        else:
            y_inter_gt, y_gt = None, None

        # h_a and h_s
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        # g_s
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "y_inter": y_inter,
            "y_inter_gt": y_inter_gt,
            "y": y,
            "y_gt": y_gt,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict, opt=None):
        """Return a new model instance from `state_dict`."""
        N = state_dict["h_a.0.weight"].size(0)
        net = cls(N, opt)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        _, y = self.g_a_func(x, denoise=True)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
