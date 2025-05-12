from typing import Union
from pathlib import Path
import warnings

import torch
from torch.utils.data import DataLoader
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.means import ZeroMean

from models.building_blocks.enc_dec_mujoco import MujocoEncoder, MujocoDecoder
from models.building_blocks.gp import GP
from models.gpvae_base_misfrms import GPVAEBaseMisFrms
from models.gpvae_sws_misfrms import GPVAEBaseMisFrms_SWS
from models.gpvae_vnn_misfrms import GPVAEBaseMisFrms_VNN
from utils.lazy_datasets_misfrms import LazySeriesDatasetMisFrms, generate_shorter_miniseries_dict
from utils.mujoco_action import NNDatasetMujoco
from utils.mujoco_metric import predict_y

class GPVAEMujocoBase(GPVAEBaseMisFrms):
    def __init__(self, H: int, file_path: Union[str, Path], GP_joint=True, fix_decoder_variance=False,
                 build_sequential_first=False, y_dim=14, init_lengthscale=50., search_device='cpu', jitter=1e-6,
    ):
        num_latents = 15
        encoder = MujocoEncoder(y_dim=y_dim, num_latents=num_latents)
        decoder = MujocoDecoder(y_dim=y_dim, num_latents=num_latents, fix_variance=fix_decoder_variance)

        kernel = ScaleKernel(MaternKernel(nu=1.5, batch_shape=torch.Size([num_latents])))
        kernel.outputscale, kernel.base_kernel.lengthscale = 1., init_lengthscale
        gp = GP(num_latents, kernel, mean=ZeroMean(batch_shape=torch.Size([num_latents])))
        for p in gp.parameters():
            p.requires_grad_(GP_joint)

        super(GPVAEMujocoBase, self).__init__(
            encoder, decoder, gp, H,
            search_device=search_device, data_device='cpu', module_device=search_device, jitter=jitter
        )

        self.entire_train_dataset = LazySeriesDatasetMisFrms(
            file_path, data_split="train", H=self.H, search_device=search_device, build_sequential_first=build_sequential_first
        )
        self.file_path = file_path
        assert len(self.entire_train_dataset[0:1][1].shape) == 3, "Only support auxiliary information with 3 dims."
        self.num_frames_before_clip = self.entire_train_dataset[0][0].size(0)

    @torch.no_grad()
    def predict_gpvae(self, file_path: Union[str, Path], data_split: str, series_batch_size: int, timestamp_batch_size: int,
                      device: str="cpu", num_samples=20, return_pred=True, diagnosis=False
    ):
        if device != self.module_device:
            warnings.warn(f"Prediction on an other device {device} but self.module_device is {self.module_device}."
                          f"We have changed self.module_device to {device}.")
            self.module_device = device
        self.to(device)

        entire_test_dataset = LazySeriesDatasetMisFrms(
            file_path, data_split=data_split, H=self.H, search_device=self.search_device, build_sequential_first=False,
        )

        series_dataloader = DataLoader(entire_test_dataset, batch_size=series_batch_size, shuffle=False)

        all_nll, all_se, all_pred_mean, all_pred_std = [], [], [], []

        if diagnosis:
            all_latent_mean, all_latent_std = [], []

        for miniseries_sq_full, miniseries_t_full, miniseries_t_mask, vid_ids in series_dataloader:
            # store the full data for extracting NNs
            miniseries_dict_full = generate_shorter_miniseries_dict(
                miniseries_sq_full, miniseries_t_full, miniseries_t_mask, clip=False
            )

            test_dataset = NNDatasetMujoco(
                miniseries_dict_full, series_shape=miniseries_t_full.shape[:-2], data_device=self.data_device
            )

            time_dataloader = DataLoader(test_dataset, batch_size=timestamp_batch_size, shuffle=False)

            miniseries_nll, miniseries_se, miniseries_pred_mean, miniseries_pred_std = [], [], [], []

            if diagnosis:
                miniseries_latent_mean, miniseries_latent_std = [], []

            for sq_short, t_short, t_mask_short in time_dataloader:
                sq_short = sq_short.to(device).transpose(0, 1)  # [v, n, D]
                t_short = t_short.to(device).transpose(0, 1).contiguous()  # [v, n, 1]
                t_mask_short = t_mask_short.to(device).transpose(0, 1)[..., 0]  # [v,n], unobserved=0

                # find NN in test observations
                nn_idx = entire_test_dataset.nnutil_all.find_nn_idx_from_index_subsets(t_short, vid_ids, self.H)
                y_nn, t_nn = test_dataset.gather(nn_idx)[:2]  # [v, n, H, D/1]
                y_nn, t_nn = y_nn.to(device), t_nn.to(device)
                enc_means, enc_stds = self.encoder(y_nn)  # [v,n,H,L]

                post_mean, post_cov = self.gp.posterior(t_short, t_nn, enc_means, enc_stds)  # [v,n,L]
                if diagnosis:
                    r_dict = predict_y(post_mean, torch.clamp(post_cov.sqrt(), max=0.05), self.decoder, sq_short, s=num_samples)
                else:
                    r_dict = predict_y(post_mean, post_cov.sqrt(), self.decoder, sq_short, s=num_samples)

                miniseries_nll.append(r_dict["nll"])
                miniseries_se.append(r_dict["se"])

                if diagnosis:
                    miniseries_latent_mean.append(post_mean)
                    miniseries_latent_std.append(post_cov.sqrt())

                if return_pred:
                    miniseries_pred_mean.append(r_dict["pred_mean"])
                    miniseries_pred_std.append(r_dict["pred_std"])

            all_se.append(torch.cat(miniseries_se, dim=-2))
            all_nll.append(torch.cat(miniseries_nll, dim=-2))

            if return_pred:
                all_pred_mean.append(torch.cat(miniseries_pred_mean, dim=-2))
                all_pred_std.append(torch.cat(miniseries_pred_std, dim=-2))

            if diagnosis:
                all_latent_mean.append(torch.cat(miniseries_latent_mean, dim=-2))  # along time dim
                all_latent_std.append(torch.cat(miniseries_latent_std, dim=-2))

        all_se, all_nll = torch.cat(all_se, dim=0), torch.cat(all_nll, dim=0)  # [V,T,D]

        mean_rmse, mean_nll = all_se.mean(dim=(-1, -2)).sqrt().mean(), all_nll.mean()

        if not return_pred:
            return mean_rmse, mean_nll
        else:
            all_pred_mean, all_pred_std = torch.cat(all_pred_mean, dim=0), torch.cat(all_pred_std, dim=0)

            if not diagnosis:
                return mean_rmse, mean_nll, all_pred_mean, all_pred_std

            else:
                all_latent_mean, all_latent_std = torch.cat(all_latent_mean, dim=0), torch.cat(all_latent_std, dim=0)
                return mean_rmse, mean_nll, all_pred_mean, all_pred_std, all_latent_mean, all_latent_std

class GPVAEMujoco_SWS(GPVAEMujocoBase, GPVAEBaseMisFrms_SWS):
    def __init__(self, *args, **kwargs):
        super(GPVAEMujoco_SWS, self).__init__(*args, build_sequential_first=False, **kwargs)

class GPVAEMujoco_VNN(GPVAEMujocoBase, GPVAEBaseMisFrms_VNN):
    def __init__(self, *args, **kwargs):
        super(GPVAEMujoco_VNN, self).__init__(*args, build_sequential_first=True, **kwargs)