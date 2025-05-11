from typing import Union
from pathlib import Path
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from torch import Tensor, LongTensor

from models.gpvae_base import GPVAEBase
from models.gpvae_hmnist import GPVAEHealingBase, GPVAEHealing_SWS, GPVAEHealing_VNN
from models.building_blocks.enc_dec_longhmnist import LongHMNISTEncoderDiagonal, LongHMNISTDecoder
from models.building_blocks.enc_dec_healing_mnist import HMNISTEncoderDiagonal, HMNISTDecoder
from models.building_blocks.gp import GP
from utils.healing_mnist import LazySeriesDataset


class GPVAELongHealingBase(GPVAEHealingBase, GPVAEBase):
    def __init__(
            self, H: int, file_path: Union[str, Path], GP_joint=True,
            search_device='cpu', jitter=1e-6
    ):
        latent_dims = 16  # 16 or 256
        encoder = LongHMNISTEncoderDiagonal(latent_dims=latent_dims)
        decoder = LongHMNISTDecoder(latent_dims=latent_dims)
        # encoder = HMNISTEncoderDiagonal(latent_dims=latent_dims)
        # decoder = HMNISTDecoder(latent_dims=latent_dims)

        kernel = ScaleKernel(RBFKernel(batch_shape=torch.Size([latent_dims])))
        kernel.outputscale, kernel.base_kernel.lengthscale = 1., 5.
        gp = GP(output_dims=latent_dims, kernel=kernel, mean=ZeroMean(batch_shape=torch.Size([latent_dims])))
        for p in gp.parameters():
            p.requires_grad_(GP_joint)

        super(GPVAEHealingBase, self).__init__(
            encoder, decoder, gp, H,
            search_device=search_device, data_device='cpu', module_device=search_device, jitter=jitter
        )

        self.train_dataset = None
        self.entire_train_dataset = LazySeriesDataset(
            file_path, train=True, return_full=False, val_split=None)  # [V,T,784]
        self.num_frames_per_series = self.entire_train_dataset[0][0].size(0)

    def expected_log_prob(self, y_batch_miss: Tensor, m_batch_miss: Tensor = None) -> Tensor:
        return super(GPVAELongHealingBase, self).expected_log_prob(y_batch_miss, m_batch_miss)

    @torch.no_grad()
    def predict_gpvae(
            self, test_file_path: Union[str, Path],
            series_batch_size: int, timestamp_batch_size: int, device: str = "cpu", num_samples=1
    ):
        return super(GPVAELongHealingBase, self).predict_gpvae(
            test_file_path, series_batch_size, timestamp_batch_size, device, num_samples)


class GPVAELongHealing_SWS(GPVAELongHealingBase, GPVAEHealing_SWS):
    def average_loss(self, vid_batch: Tensor, t_batch: Tensor, m_batch: Tensor, beta=1.) -> Tensor:
        return super().average_loss(vid_batch, t_batch, m_batch, beta)

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs: int,
            series_batch_size: int, timestamp_batch_size: int, max_norm: float = None,
            device="cpu", print_epochs=1  # device for training
    ):
        return super().train_gpvae(optimizer, beta, epochs, series_batch_size, timestamp_batch_size, max_norm, device)


class GPVAELongHealing_VNN(GPVAELongHealingBase, GPVAEHealing_VNN):
    def average_loss(self, vid_batch, m_batch,
                     current_kl_idx: LongTensor, seq_nn_structure: LongTensor, beta=1.) -> Tensor:
        return super().average_loss(vid_batch, m_batch, current_kl_idx, seq_nn_structure, beta)

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs: int,
            series_batch_size: int, timestamp_kl_batch_size: int, timestamp_expected_lk_batch_size: int = None,
            max_norm: float = None, device="cpu", print_epochs=1
    ):
        super().train_gpvae(
            optimizer, beta, epochs, series_batch_size, timestamp_kl_batch_size, timestamp_expected_lk_batch_size,
            max_norm, device, print_epochs
        )



