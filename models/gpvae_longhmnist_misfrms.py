from typing import Union
from pathlib import Path
import warnings
import math

import torch
from torch import nn
from torch.utils.data import DataLoader
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ZeroMean

from models.gpvae_base_misfrms import GPVAEBaseMisFrms
from models.gpvae_sws_misfrms import GPVAEBaseMisFrms_SWS
from models.gpvae_vnn_misfrms import GPVAEBaseMisFrms_VNN
from models.building_blocks.enc_dec_longhmnist import LongHMNISTEncoderDiagonal, LongHMNISTDecoder
from models.building_blocks.enc_dec_healing_mnist import HMNISTEncoderDiagonal, HMNISTDecoder
from models.building_blocks.gp import GP
from utils.lazy_datasets_misfrms import (
    LazySeriesDatasetMisFrms, generate_test_shorter_miniseries_dict, generate_shorter_miniseries_dict, NNDatasetMisFrms)


class GPVAELongHMNISTMisFrms(GPVAEBaseMisFrms):
    def __init__(
            self, H: int, file_path: Union[str, Path], GP_joint=True, build_sequential_first=False,
            search_device='cpu', jitter=1e-5,
    ):
        latent_dims = 16
        encoder = LongHMNISTEncoderDiagonal(latent_dims)
        decoder = LongHMNISTDecoder(latent_dims)
        # encoder = HMNISTEncoderDiagonal(latent_dims)
        # decoder = HMNISTDecoder(latent_dims)

        kernel = ScaleKernel(RBFKernel(batch_shape=torch.Size([latent_dims])))
        kernel.outputscale, kernel.base_kernel.lengthscale = 1., 5.
        gp = GP(latent_dims, kernel, mean=ZeroMean(batch_shape=torch.Size([latent_dims])))
        for p in gp.parameters():
            p.requires_grad_(GP_joint)

        super(GPVAELongHMNISTMisFrms, self).__init__(
            encoder, decoder, gp, H,
            search_device=search_device, data_device='cpu', module_device=search_device, jitter=jitter
        )

        self.entire_train_dataset = LazySeriesDatasetMisFrms(
            file_path, train=True, H=self.H, search_device=search_device,
            build_sequential_first=build_sequential_first
        )
        assert len(self.entire_train_dataset[0:1][1].shape) == 3, "Only support auxiliary information with 3 dims."
        self.num_frames_before_clip = self.entire_train_dataset[0][0].size(0)

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

        entire_test_dataset = LazySeriesDatasetMisFrms(
            test_file_path, train=False, H=self.H, search_device=self.search_device, build_sequential_first=False
        )
        series_dataloader = DataLoader(entire_test_dataset, batch_size=series_batch_size, shuffle=False)

        nll_series, mse_series, mse_series_non_round = 0., 0., 0
        num_missing_frames = 0.
        img_rec_full = torch.as_tensor(entire_test_dataset.data_dict['seq_test_full'][:])
        img_rec_full = img_rec_full.clone().to(device)
        for miniseries_sq_full, miniseries_t_full, miniseries_t_mask, vid_ids in series_dataloader:
            # store the whole length for gathering neighbors later
            miniseries_whole_dict = generate_shorter_miniseries_dict(
                miniseries_sq_full, miniseries_t_full, miniseries_t_mask.to(torch.bool), clip=False
            )
            test_whole_dataset = NNDatasetMisFrms(
                miniseries_whole_dict, series_shape=miniseries_t_full.shape[:-2], data_device=self.data_device,
                search_device=self.search_device
            )
            # clip the test data for faster prediction
            miniseries_short_dict = generate_test_shorter_miniseries_dict(
                miniseries_sq_full, miniseries_t_full, ~miniseries_t_mask.to(torch.bool), clip=True
            )
            num_missing_frames += miniseries_short_dict['masks'].sum()

            test_dataset = NNDatasetMisFrms(
                miniseries_short_dict, series_shape=miniseries_t_full.shape[:-2], data_device=self.data_device,
                search_device=self.search_device
            )
            time_dataloader = DataLoader(test_dataset, batch_size=timestamp_batch_size, shuffle=False)

            nll_batch, se_batch, se_non_round_batch, img_rec_batch = 0., 0., 0., []
            for sq_short, t_short, m_short in time_dataloader:
                sq_short = sq_short.to(device).transpose(0, 1)               # [v, n, 28, 28]
                t_short = t_short.to(device).transpose(0, 1).contiguous()    # [v, n, 1]
                m_short = m_short.to(device).transpose(0, 1)[..., 0]         # [v,n], unobserved=1

                # find NN in test observations
                nn_idx = entire_test_dataset.nnutil_all.find_nn_idx_from_index_subsets(t_short, vid_ids, self.H)
                y_nn, t_nn = test_whole_dataset.gather(nn_idx)[:2]     # [v, n, H, D/1]
                y_nn, t_nn = y_nn.to(device), t_nn.to(device)
                # First mask and then feed to avoid numerical warnings from predictions of already observed timestamps
                enc_means, enc_stds = self.encoder(y_nn[m_short])      # [<v*n,H,L]

                # Compute marginal rather than conditional
                post_mean, post_cov = self.gp.posterior(
                    t_short[m_short], t_nn[m_short], enc_means, std_Z_near=enc_stds, jitter_val=self.jitter
                )  # [<v*n,L]
                assert torch.isfinite(post_cov).all() and (post_cov > 0.).all(), f"post covar min: {post_cov.min()}."

                # NLL
                eps = torch.randn([num_samples, *post_mean.shape], device=device, dtype=torch.get_default_dtype())
                latent_samples = post_mean + post_cov.sqrt() * eps
                logits = self.decoder(latent_samples)
                assert torch.isfinite(logits).all()
                log_p = - nn.functional.binary_cross_entropy_with_logits(
                    input=logits, target=sq_short[m_short].expand_as(logits), reduction='none'
                )  # [s, <v*n, ...]
                assert torch.isfinite(log_p).all(), "log p in prediction is not finite."
                log_p = - math.log(num_samples) + torch.logsumexp(log_p, dim=0)
                nll_batch += - log_p.sum()

                # MSE
                img_rec = nn.functional.sigmoid(self.decoder(post_mean))  # [<v*n,28,28]
                img_rec_2 = torch.zeros_like(sq_short)                    # [v,n,28,28]
                img_rec_2[m_short] = img_rec                              # assign
                img_rec_batch.append(img_rec_2)

                se_non_round_batch += (img_rec - sq_short[m_short]).square().sum()
                pixels = torch.round(img_rec)
                se_batch += (pixels - sq_short[m_short]).square().sum()
            nll_series += nll_batch
            mse_series += se_batch
            mse_series_non_round += se_non_round_batch
            img_rec_series = torch.cat(img_rec_batch, dim=1)        # [v,f,28,28]

            # Fill in the reconstructed
            frame_loc = ~miniseries_t_mask[..., 0].to(torch.bool)   # [v,f]
            frame_short_loc = miniseries_short_dict['masks'][..., 0]
            vid_coords = vid_ids.unsqueeze(-1).expand_as(frame_loc)
            frms_coords = torch.arange(miniseries_t_full.size(-2)).expand_as(frame_loc)
            img_rec_full[vid_coords[frame_loc], frms_coords[frame_loc]] = img_rec_series[frame_short_loc]

        prediction_coll = {
            'images': img_rec_full.cpu().numpy(),                      # ndarray: [V,T,28,28]
            'masks': entire_test_dataset.data_dict['m_test_miss'][:],  # ndarray: [V,T,1]
            'aux_data': entire_test_dataset.data_dict['t_test_full'][:]
        }
        return (nll_series/num_missing_frames, mse_series/num_missing_frames, mse_series_non_round/num_missing_frames,
                prediction_coll)


class GPVAELongHMNISTMisFrms_SWS(GPVAELongHMNISTMisFrms, GPVAEBaseMisFrms_SWS):
    def __init__(self, *args, **kwargs):
        super(GPVAELongHMNISTMisFrms_SWS, self).__init__(*args, build_sequential_first=False, **kwargs)


class GPVAELongHMNISTMisFrms_VNN(GPVAELongHMNISTMisFrms, GPVAEBaseMisFrms_VNN):
    def __init__(self, *args, **kwargs):
        super(GPVAELongHMNISTMisFrms_VNN, self).__init__(*args, build_sequential_first=True, **kwargs)
















