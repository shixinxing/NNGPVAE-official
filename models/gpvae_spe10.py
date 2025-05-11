from typing import Union
from pathlib import Path
import time

import torch
from gpytorch.means import ZeroMean

from models.gpvae_base_spatial_imp import GPVAESpatialImp
from models.building_blocks.enc_dec_spe10 import SPE10Encoder, SPE10Decoder
from models.building_blocks.gp import GP
from utils.spe10 import NNDatasetSPE10, define_kernel


class GPVAESPE10Base(GPVAESpatialImp):
    def __init__(
            self, H, GP_joint=True, search_device: str = 'cpu',
            sigma2_y=1., fix_variance=True, kernel_type='rbf', lengthscale=2., jitter=1e-6
    ):
        latent_dims, x_dims, y_dims = 3, 3, 4
        encoder = SPE10Encoder(latent_dims, input_dims=y_dims)
        decoder = SPE10Decoder(latent_dims, output_dims=y_dims, sigma2_y=sigma2_y, fix_variance=fix_variance)

        kernel = define_kernel(kernel_type, latent_dims, x_dims)
        kernel.outputscale, kernel.base_kernel.lengthscale = 1., lengthscale
        for p in kernel.parameters():
            p.requires_grad_(GP_joint)
        if kernel_type == 'cauchy':
            kernel.base_kernel.alpha = 1.
            kernel.base_kernel.raw_alpha.requires_grad_(False)
        gp = GP(latent_dims, kernel, mean=ZeroMean(batch_shape=torch.Size([latent_dims])))

        super(GPVAESPE10Base, self).__init__(
            encoder, decoder, gp, H, search_device=search_device,
            data_device=search_device, jitter=jitter
        )
        self.train_dataset = None

    @torch.no_grad()
    def predict_gpvae(self, *args, **kwargs):
        return super(GPVAESPE10Base, self).predict_gpvae(*args, **kwargs)


class GPVAESPE10_SWS(GPVAESPE10Base):
    def __init__(self, file_path: Union[str, Path], *args, **kwargs):
        if isinstance(file_path, str):
            file_path = Path(file_path)

        super(GPVAESPE10_SWS, self).__init__(*args, **kwargs)
        self.train_dataset = NNDatasetSPE10(
            file_path, self.H, search_device=self.search_device, build_sequential_first=False
        )

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs: int, batch_size: int, max_norm=None,
            device="cpu", print_epochs=1
    ):
        return super(GPVAESPE10_SWS, self).train_gpvae_sws(
            optimizer, beta, epochs, batch_size, max_norm, device, print_epochs
        )


class GPVAESPE10_VNN(GPVAESPE10Base):
    def __init__(self, file_path: Union[str, Path], *args, **kwargs):
        if isinstance(file_path, str):
            file_path = Path(file_path)

        super(GPVAESPE10_VNN, self).__init__(*args, **kwargs)
        s = time.time()
        self.train_dataset = NNDatasetSPE10(
            file_path, self.H, search_device=self.search_device, build_sequential_first=True  # build nn structure first
        )
        print(f"==== NN structure built, cost {time.time() - s} sec. ===\n")

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs: int,
            kl_batch_size: int, expected_lk_batch_size=None, max_norm=None,
            device="cpu", print_epochs=1
    ):
        return super(GPVAESPE10_VNN, self).train_gpvae_vnn(
            optimizer, beta, epochs, kl_batch_size, expected_lk_batch_size, max_norm, device, print_epochs
        )



