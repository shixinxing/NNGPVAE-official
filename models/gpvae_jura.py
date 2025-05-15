from typing import Union
from pathlib import Path

import torch
from torch import Tensor
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel

from models.building_blocks.gp import GP
from models.building_blocks.enc_dec_jura import JuraEncoder, JuraDecoder
from models.gpvae_base_spatial_imp import GPVAESpatialImp

from utils.jura import load_jura
from utils.build_datasets import NNDataset


class GPVAEJuraImpBase(GPVAESpatialImp):
    def __init__(self, H: int, GP_joint, build_sequential_first, file_path: Union[str, Path] = './data/jura',
                 search_device: str = 'cpu', data_device: str = 'cpu', module_device: str = 'cpu',
                 sigma2_y=0.25, fix_variance=False, jitter: float = 1e-6, geco=False
    ):
        x_dim, y_dim, num_latents = 2, 3, 2
        encoder = JuraEncoder(y_dim=y_dim, num_latents=num_latents)
        decoder = JuraDecoder(y_dim=y_dim, num_latents=num_latents, sigma2_y=sigma2_y, fix_variance=fix_variance)

        scaled_kernel = ScaleKernel(RBFKernel(ard_num_dims=x_dim, batch_shape=torch.Size([1])))
        scaled_kernel.base_kernel.lengthscale = 0.1
        scaled_kernel.base_kernel.raw_lengthscale.requires_grad_(GP_joint)
        scaled_kernel.outputscale = 1.
        scaled_kernel.raw_outputscale.requires_grad_(GP_joint)

        gp = GP(num_latents, scaled_kernel, mean=ZeroMean(batch_shape=torch.Size([num_latents])))
        for p in gp.parameters():
            p.requires_grad_(GP_joint)

        super(GPVAEJuraImpBase, self).__init__(encoder=encoder, decoder=decoder, gp=gp, H=H,
                                            search_device=search_device, data_device=data_device, module_device=module_device,
                                            jitter=jitter, geco=geco)

        data_X, masks, data_Ymiss, data_Yfull, stat = load_jura(file_path)
        data_dict = {"X": data_X, "Y": data_Ymiss, "masks": masks, "Y_full": data_Yfull}
        self.train_dataset = NNDataset(data_dict, missing=True, return_full=True, H=H, search_device=search_device,
                                       build_sequential_first=build_sequential_first)
        self.Ystat = stat 


class GPVAEJuraImp_SWS(GPVAEJuraImpBase):
    def __init__(self, *args, **kwargs):
        super(GPVAEJuraImp_SWS, self).__init__(*args, build_sequential_first=False, **kwargs)

    def train_gpvae(self, *args, **kwargs):
        super(GPVAEJuraImp_SWS, self).train_gpvae_sws(*args, **kwargs)

class GPVAEJuraImp_VNN(GPVAEJuraImpBase):
    def __init__(self, *args, **kwargs):
        super(GPVAEJuraImp_VNN, self).__init__(*args, build_sequential_first=True, **kwargs)

    def train_gpvae(self, *args, **kwargs):
        super(GPVAEJuraImp_VNN, self).train_gpvae_vnn(*args, **kwargs)


