from typing import Union
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
import torch
import torch.nn as nn

from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel

from models.building_blocks.enc_dec_jura import JuraEncoder, JuraDecoder
from baselines.svgpvae.models.svgp import SVGP
from baselines.svgpvae.models.svgpvae_base_spatial_imp import SVGPVAESpatialImp

from utils.jura import load_jura
from utils.build_datasets import NNDataset


class SVGPVAEJura(SVGPVAESpatialImp):
    def __init__(self, M: int, file_path: Union[str, Path] = './data/jura', GP_joint=True, IP_joint=True,
                 device="cpu", sigma2_y=0.25, fix_variance=True, jitter=1e-6
    ):
        x_dim, y_dim, num_latents = 2, 3, 2
        N_train = 359
        encoder = JuraEncoder(y_dim=y_dim, num_latents=num_latents)
        decoder = JuraDecoder(y_dim=y_dim, num_latents=num_latents, sigma2_y=sigma2_y, fix_variance=fix_variance)

        scaled_kernel = ScaleKernel(RBFKernel(ard_num_dims=x_dim, batch_shape=torch.Size([1])))
        scaled_kernel.base_kernel.lengthscale = 0.1
        scaled_kernel.base_kernel.raw_lengthscale.requires_grad_(GP_joint)
        scaled_kernel.outputscale = 1.
        scaled_kernel.raw_outputscale.requires_grad_(GP_joint)

        data_X, masks, data_Ymiss, data_Yfull, stat = load_jura(file_path)
        data_dict = {"X": data_X, "Y": data_Ymiss, "masks": masks, "Y_full": data_Yfull}
        self.train_dataset = NNDataset(data_dict, missing=True, return_full=True, H=None)

        init_inducing_points = torch.tensor(kmeans2(data_X.numpy(), M, minit='points')[0],
                                            dtype=torch.get_default_dtype())

        inducing_points = nn.Parameter(init_inducing_points, requires_grad=IP_joint)

        svgp = SVGP(num_latents, scaled_kernel, inducing_points, N_train=N_train, jitter=jitter)

        super(SVGPVAEJura, self).__init__(encoder, decoder, svgp, device=device, jitter=jitter, geco=False)

        self.Ystat = stat 
