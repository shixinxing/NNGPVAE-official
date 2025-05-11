from typing import Union
from pathlib import Path
import warnings

import math
import torch
from torch import nn, Tensor, LongTensor
from torch.utils.data import DataLoader
from gpytorch.kernels import ScaleKernel
from models.building_blocks.rq_kernel import RQKernel
from gpytorch.means import ZeroMean

from models.gpvae_base import GPVAEBase
from models.building_blocks.enc_dec_healing_mnist import HMNISTEncoderDiagonal, HMNISTDecoder
from models.building_blocks.gp import GP
from utils.healing_mnist import generate_miniseries_hmnist_dict, LazySeriesDataset
from utils.build_datasets import NNDataset


class GPVAEHealingBase(GPVAEBase):
    def __init__(
            self, H, file_path: Union[str, Path], GP_joint=False,
            val_split=50000, search_device: str = "cpu", jitter=1e-6    # device for searching NNs
    ):
        encoder = HMNISTEncoderDiagonal(latent_dims=256, hidden_dims=(256, 256))
        decoder = HMNISTDecoder(latent_dims=256, hidden_dims=(256, 256, 256))

        # The original paper use Cauchy kernel with fixed lengthscales 2 and outputscales 1, no ARD,
        if not GP_joint:
            kernel = ScaleKernel(RQKernel(batch_shape=torch.Size([1])))
        else:
            kernel = ScaleKernel(RQKernel(batch_shape=torch.Size([256])))
        kernel.outputscale, kernel.base_kernel.lengthscale, kernel.base_kernel.alpha = 1, math.sqrt(2), 1
        # can be trained to compensate for fewer params
        kernel.raw_outputscale.requires_grad_(GP_joint)
        kernel.base_kernel.raw_lengthscale.requires_grad_(GP_joint)
        kernel.base_kernel.raw_alpha.requires_grad_(False)
        gp = GP(output_dims=256, kernel=kernel, mean=ZeroMean(batch_shape=torch.Size([256])))

        super(GPVAEHealingBase, self).__init__(
            encoder, decoder, gp, H,
            search_device=search_device, data_device='cpu', module_device=search_device, jitter=jitter
        )   # We still put data onto CPU
        self.train_dataset = None

        self.entire_train_dataset = LazySeriesDataset(
            file_path, train=True, val_split=val_split, return_full=False
        )   # [V, T, 784]
        self.num_frames_per_series = self.entire_train_dataset[0][0].size(0)  # 10

    # override, consider masks
    def expected_log_prob(self, y_batch_miss: Tensor, m_batch_miss: Tensor = None) -> Tensor:
        means, stds, y_rec_logits = self.forward(y_batch_miss)

        if m_batch_miss is None:
            m_batch_miss = torch.zeros_like(y_batch_miss, dtype=torch.bool)
        else:
            m_batch_miss = m_batch_miss.to(torch.bool)
        if self.decoder.output_distribution == 'bernoulli':
            expected_lk = - nn.functional.binary_cross_entropy_with_logits(
                input=y_rec_logits, target=y_batch_miss, reduction='none'
            )
        else:
            raise NotImplementedError(f'Unrecognized output distribution of the decoder.')

        expected_lk = torch.where(m_batch_miss, 0., expected_lk).sum()
        # we currently do not include missing rate, which is the same as the AISTATS 2020
        scale = y_batch_miss.shape[:len(self.train_dataset.series_shape) + 1].numel()
        scale = len(self.train_dataset) / scale  # averaged over series dim
        return scale * expected_lk

    @torch.no_grad()
    def predict_gpvae(
            self, test_file_path: Union[str, Path], series_batch_size: int, timestamp_batch_size: int,
            device: str = "cpu", num_samples=1
    ):
        if device != self.module_device:
            warnings.warn(f"Prediction on an other device {device} but self.module_device is {self.module_device}."
                          f"We have changed self.module_device to {device}.")
            self.module_device = device
        self.to(device)

        entire_test_dataset = LazySeriesDataset(test_file_path, train=False, return_full=True, val_split=None)
        series_dataloader = DataLoader(entire_test_dataset, batch_size=series_batch_size, shuffle=False)
        # We use the evaluation procedure of AISTATS 2020 paper: sample one `z`, metrics are computed on missing values
        # We don't use the Gaussian mixture as the predictive density here
        nll_series, mse_series, mse_series_non_round, img_rec_series = 0., 0., 0, []
        n_missings, total_pixels = 0., 0.
        for series_y_miss, m_miss, series_y_full in series_dataloader:
            n_missings = n_missings + m_miss.sum()
            total_pixels = total_pixels + series_y_full.numel()
            miniseries_dict = generate_miniseries_hmnist_dict(series_y_miss, m_miss, series_y_full)
            test_dataset = NNDataset(
                miniseries_dict, series_shape=series_y_miss.shape[:1], missing=True, return_full=True,
                data_device=self.data_device, H=None, search_device=None, build_sequential_first=False
            )  # we don't need nnutils
            time_loader = DataLoader(test_dataset, batch_size=timestamp_batch_size, shuffle=False)

            nll_batch, se_batch, se_non_round_batch, img_rec_batch = 0., 0., 0., []
            for y_test_miss, _t, m_test, y_test_full in time_loader:
                y_test_miss = y_test_miss.to(device).transpose(0, 1)
                m_test, y_test_full = m_test.to(device).transpose(0, 1), y_test_full.to(device).transpose(0, 1)

                # NLL from multiple samples
                enc_means, enc_stds = self.encoder(y_test_miss)
                eps = torch.randn([num_samples, *enc_means.shape], device=device)  # [s,V,T,...]
                latent_samples = enc_means + enc_stds * eps
                logits = self.decoder(latent_samples)
                log_p = - nn.functional.binary_cross_entropy_with_logits(
                    input=logits, target=y_test_full.expand_as(logits), reduction='none')
                log_p = - math.log(num_samples) + torch.logsumexp(log_p, dim=0)
                nll = torch.where(m_test, - log_p, 0.).sum()
                nll_batch += nll

                # MSE
                img_rec = nn.functional.sigmoid(self.decoder(enc_means))  # use latent mean directly
                img_rec_batch.append(img_rec)

                se_non_round = (img_rec - y_test_full).square().sum()
                se_non_round_batch += se_non_round
                pixels = torch.round(img_rec)  # round to the nearest integer
                se = torch.where(m_test, (pixels - y_test_full).square(), 0.).sum()
                se_batch += se
            nll_series += nll_batch
            mse_series_non_round += se_non_round_batch
            mse_series += se_batch
            img_rec_series.append(torch.cat(img_rec_batch, dim=1))  # [v,f,28,28]

        return (
            (nll_series/n_missings).item(), (mse_series/n_missings).item(), (mse_series_non_round/total_pixels).item(),
            torch.cat(img_rec_series, dim=0).cpu().numpy()
        )


class GPVAEHealing_SWS(GPVAEHealingBase):
    def average_loss(self, vid_batch: Tensor, t_batch: Tensor, m_batch: Tensor, beta=1.) -> Tensor:
        elbo = self.expected_log_prob(vid_batch, m_batch_miss=m_batch) - beta * self.kl_divergence_sws(t_batch)
        # We use sum loss, only averaging over video batch dim (not including the frame dim),
        # to remain consistent to the official code
        return - elbo  # / len(self.train_dataset)

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs: int,
            series_batch_size: int, timestamp_batch_size: int, max_norm: float = None,
            device="cpu", print_epochs=1   # device for training
    ):
        if device != self.module_device:
            warnings.warn(f"Training is not on self.module_device ({self.module_device}) set earlier. "
                          f"We have changed module device to {device}.")
            self.module_device = device
        self.to(device)

        series_dataloader = DataLoader(self.entire_train_dataset, batch_size=series_batch_size, shuffle=True)
        nn_util_cached = None
        for epoch in range(epochs):
            for series_y_miss, series_m_miss in series_dataloader:
                miniseries_dict = generate_miniseries_hmnist_dict(series_y_miss, series_m_miss, Y_batch_full=None)
                self.train_dataset = NNDataset(
                    miniseries_dict, series_shape=series_y_miss.shape[:1], missing=True, data_device=self.data_device,
                    H=None if nn_util_cached is not None else self.H, search_device=self.search_device,
                    build_sequential_first=False
                )
                if nn_util_cached is None:
                    nn_util_cached = self.train_dataset.nn_util
                else:
                    self.train_dataset.nn_util = nn_util_cached

                time_dataloader = DataLoader(self.train_dataset, batch_size=timestamp_batch_size, shuffle=True)
                for y_miss, t, m_miss in time_dataloader:
                    y_miss, t, m_miss = y_miss.to(device), t.to(device), m_miss.to(device)
                    y_miss, m_miss = y_miss.transpose(0, 1), m_miss.transpose(0, 1)  # [v,f,28,28], t: [f,1]
                    optimizer.zero_grad(set_to_none=True)
                    # faiss needs continuity (t is usually contiguous from dataloader)
                    loss = self.average_loss(y_miss, t.contiguous(), m_batch=m_miss, beta=beta)
                    loss.backward()
                    if max_norm is not None:
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.parameters(), max_norm=max_norm, error_if_nonfinite=True
                        )
                        if grad_norm > max_norm:
                            print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                    optimizer.step()
            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item(): .6f}',
                      f'global grad norm: {grad_norm}' if max_norm is not None else ' ', '\n')


class GPVAEHealing_VNN(GPVAEHealingBase):
    def average_loss(
            self, vid_batch: Tensor, m_batch: Tensor, current_kl_idx: LongTensor,
            seq_nn_structure: LongTensor, beta=1.
    ) -> Tensor:
        expected_lk = self.expected_log_prob(vid_batch, m_batch_miss=m_batch)
        kl = self.kl_divergence_vnn(current_kl_idx, seq_nn_structure)
        # We only average over video batch dim (not including the frame dim),
        # to remain consistent to the official code
        return - (expected_lk - beta * kl)  # / len(self.train_dataset)

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs: int,
            series_batch_size: int, timestamp_kl_batch_size: int, timestamp_expected_lk_batch_size: int = None,
            max_norm: float = None, device="cpu", print_epochs=1
    ):
        if device != self.module_device:
            warnings.warn(f"Training is not on self.module_device ({self.module_device}) set earlier. "
                          f"We have changed module device to {device}.")
            self.module_device = device
        self.to(device)
        self._set_training_iterator(timestamp_kl_batch_size, N=self.num_frames_per_series)

        series_dataloader = DataLoader(self.entire_train_dataset, batch_size=series_batch_size, shuffle=True)
        seq_nn_idx_cached, nn_util_cached = None, None  # We don't want to re-compute the timestamp structure
        for epoch in range(epochs):
            for series_y_miss, series_m_miss in series_dataloader:
                miniseries_dict = generate_miniseries_hmnist_dict(series_y_miss, series_m_miss, Y_batch_full=None)
                self.train_dataset = NNDataset(
                    miniseries_dict, series_shape=series_y_miss.shape[:1], missing=True, data_device=self.data_device,
                    H=None if nn_util_cached is not None else self.H, search_device=self.search_device,
                    build_sequential_first=True if seq_nn_idx_cached is None else False
                )  # got self.train_dataset.seq_nn_idx
                if nn_util_cached is None:
                    nn_util_cached = self.train_dataset.nn_util
                else:
                    self.train_dataset.nn_util = nn_util_cached
                if seq_nn_idx_cached is None:
                    seq_nn_idx_cached = self.train_dataset.seq_nn_idx
                # else:  # for completion
                #     self.train_dataset.seq_nn_idx = seq_nn_idx_cached

                if timestamp_expected_lk_batch_size is None:  # expected lk uses the same set of indices as the KL term
                    for _ in range(self._total_training_batches):
                        current_kl_indices = self._get_training_indices(timestamp_kl_batch_size)
                        y_miss, _t, m_miss = self.train_dataset[current_kl_indices]  # [v,b,28,28], [b,1], [v,b,28,28]
                        y_miss, m_miss = y_miss.to(device), m_miss.to(device)

                        optimizer.zero_grad(set_to_none=True)
                        loss = self.average_loss(y_miss, m_miss, current_kl_indices, seq_nn_idx_cached, beta=beta)
                        loss.backward()
                        if max_norm is not None:
                            grad_norm = nn.utils.clip_grad_norm_(
                                self.parameters(), max_norm=max_norm, error_if_nonfinite=True
                            )
                            if grad_norm > max_norm:
                                print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                        optimizer.step()
                else:
                    dataloader_lk = DataLoader(
                        self.train_dataset, batch_size=timestamp_expected_lk_batch_size, shuffle=True
                    )
                    for y_miss, _t, m_miss in dataloader_lk:
                        y_miss, m_miss = y_miss.to(device).transpose(0, 1), m_miss.to(device).transpose(0, 1)
                        current_kl_indices = self._get_training_indices(timestamp_kl_batch_size)

                        optimizer.zero_grad(set_to_none=True)
                        loss = self.average_loss(y_miss, m_miss, current_kl_indices, seq_nn_idx_cached, beta=beta)
                        loss.backward()
                        if max_norm is not None:
                            grad_norm = nn.utils.clip_grad_norm_(
                                self.parameters(), max_norm=max_norm, error_if_nonfinite=True
                            )
                            if grad_norm > max_norm:
                                print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                        optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item(): .6f}',
                      f'global grad norm: {grad_norm}' if max_norm is not None else ' ', '\n')

