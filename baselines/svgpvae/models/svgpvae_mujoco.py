import warnings
from typing import Union
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn

from gpytorch.kernels import ScaleKernel, MaternKernel

from baselines.svgpvae.models.svgp import SVGP
from baselines.svgpvae.models.svgpvae_base import SVGPVAEBase
from baselines.svgpvae.utils.elbo_utils import negative_gaussian_cross_entropy

from models.building_blocks.enc_dec_mujoco import MujocoEncoder, MujocoDecoder
from utils.lazy_datasets_misfrms import LazySeriesDatasetMisFrms, generate_shorter_miniseries_dict
from utils.mujoco_action import NNDatasetMujoco
from utils.mujoco_metric import predict_y


class SVGPVAEMujoco(SVGPVAEBase):
    """
    SVGPVAE for Mujoco, with missing frames.
    """
    def __init__(self, M, file_path: Union[str, Path], fix_decoder_variance=False,
                 GP_joint=False, IP_joint=False, y_dim=14, init_lengthscale=50., device: str = 'cpu', jitter=1e-6
    ):  
        num_latents = 15
        encoder = MujocoEncoder(y_dim=y_dim, num_latents=num_latents)
        decoder = MujocoDecoder(y_dim=y_dim, num_latents=num_latents, fix_variance=fix_decoder_variance)

        kernel = ScaleKernel(MaternKernel(nu=1.5, batch_shape=torch.Size([num_latents])))
        kernel.outputscale, kernel.base_kernel.lengthscale = 1., init_lengthscale
        for p in kernel.parameters():
            p.requires_grad_(GP_joint)
        
        self.entire_train_dataset = LazySeriesDatasetMisFrms(
            file_path, data_split="train", H=None,
        )
        self.file_path = file_path  
        self.num_all_frames = self.entire_train_dataset[0][0].size(0)  # 1000
        inducing_points = nn.Parameter(
            torch.linspace(0, self.num_all_frames - 1, M).unsqueeze(-1).repeat(num_latents, 1, 1), requires_grad=IP_joint
        )
        svgp = SVGP(num_latents, kernel, inducing_points, N_train=self.num_all_frames, jitter=jitter)

        super(SVGPVAEMujoco, self).__init__(
            encoder, decoder, svgp, device=device, jitter=jitter, geco=False
        )
    
    # override
    def expected_log_prob(self, mu_qs: Tensor, vars_qs_diag_clipped: Tensor, y_batch: Tensor, masks: Tensor):
        # y_batch: [(v),b,14] and masks: [(v),b], 0 refers to missing
        eps = torch.rand_like(mu_qs)
        latent_sample = (mu_qs + eps * torch.sqrt(vars_qs_diag_clipped)).mT  # [(v),b,L]
        y_rec = self.decoder(latent_sample)  # [(v),b,14]
        if self.decoder.output_distribution == 'normal':
            sigma2_y = self.decoder.sigma2_y
            expected_lk = (y_batch - y_rec).square() / sigma2_y + torch.log(2 * torch.pi * sigma2_y)
            expected_lk = - 1 / 2 * expected_lk
            expected_lk = expected_lk * masks.unsqueeze(-1)  # [(v),b,14]
        else:
            raise NotImplementedError

        return expected_lk  # [(v),b,14]
    
    # override
    def average_loss(
            self, vid_batch: Tensor, masks: Tensor, t_batch: Tensor,
            clip_qs_var_min=None, clip_qs_var_max=None, beta=1., N_train=None
    ):
        # vid_batch: [(v),b,14]; masks: [(v),b]; t_batch: [(v),b,1]， N_train: [(v)]
        assert not self.geco, "Do not support GECO in this class."
        assert N_train is not None, "N_train is required, as every vid has different number of observed frames."

        if masks.ndim == 3:
            masks = masks.squeeze(-1)
        else:
            assert masks.ndim == 2, f"masks MUST have 2 or 3 dims!, but get {masks.ndim}"

        qnet_mus, qnet_vars = self.build_MLP_inference_graph(vid_batch)  # [(v),b,L]
        # get q(Z_U) and q_s(Z): [(v),L,b]
        mu_qs, cov_qs_diag, mu, A = self.svgp.approx_posterior_params(
            t_batch, qnet_mus, qnet_vars, x_test=t_batch, diag_cov_qs=True, masks=masks, N_train=N_train
        )

        # term 1/3 expected-log-likelihood
        expected_lk = self.expected_log_prob(
            mu_qs, torch.clamp(cov_qs_diag, min=clip_qs_var_min, max=clip_qs_var_max), vid_batch, masks
        )  # [(v),b,D=14]

        # term 2/3 cross-entropy
        cross_entry = negative_gaussian_cross_entropy(mu_qs, cov_qs_diag, qnet_mus.mT, qnet_vars.mT)  # [(v),L,b]
        cross_entry = (cross_entry * masks.unsqueeze(-2)).mT  # [(v),b,L]

        # term 3/3 - L_H
        inside_elbo_batch_log, inside_elbo_KL = self.svgp.variational_loss(t_batch, qnet_mus, qnet_vars, mu, A,
                                                                       masks=masks)  # [(v)], [(v)]

        v = qnet_mus.shape[:-2].numel()
        eff_batch_sizes = masks.sum(axis=-1)  # [(v)], effective batch size
        inv_eff_batch_sizes = torch.where(eff_batch_sizes > 0, 1 / eff_batch_sizes, 0.)
        scale = N_train * inv_eff_batch_sizes  # [(v)], num of obs in whole vid / num of obs in mini-batch

        L_H = (scale * inside_elbo_batch_log).sum() - inside_elbo_KL.sum()
        expected_lk = (scale * expected_lk.sum(dim=(-1, -2))).sum()
        KL = (scale * cross_entry.sum(dim=(-1, -2))).sum() - L_H
        elbo = (expected_lk - beta * KL) / v  # elbo per vid
        return - elbo

    def train_gpvae(self, optimizer: torch.optim.Optimizer, beta: float, epochs: int,
                    timestamp_batch_size: int, series_batch_size: int, max_norm: float=None, device='cpu',
                    print_epochs=1, validation=True, num_samples=20, save_folder=""
    ):
        assert not self.geco, "The current training does not support GECO."
        self.to(device)

        best_val_rmse, best_val_nll = torch.inf, torch.inf

        series_dataloader = DataLoader(self.entire_train_dataset, batch_size=series_batch_size, shuffle=True)
        for epoch in range(epochs):
            for miniseries_sq_full, miniseries_t_full, miniseries_t_mask, vid_ids in series_dataloader:
                N_train = miniseries_t_mask.sum((-1, -2)).to(device)  # miniseries_t_mask: [v,T,1] --> [v]
                # or N_train = None (don't forget to change predict_gpvae!)
                miniseries_dict = generate_shorter_miniseries_dict(
                    miniseries_sq_full, miniseries_t_full, miniseries_t_mask, clip=True  # or True?
                )
                train_dataset = NNDatasetMujoco(
                    miniseries_dict, series_shape=miniseries_t_full.shape[:-2], 
                )

                time_dataloader = DataLoader(train_dataset,  batch_size=timestamp_batch_size, shuffle=True)
                for sq_short, t_short, t_mask_short in time_dataloader:
                    sq_short = sq_short.to(device).transpose(0, 1)  # [v, T, D]
                    t_short = t_short.to(device).transpose(0, 1).contiguous()    # [v, T, 1]
                    t_mask_short = t_mask_short.to(device).transpose(0, 1)

                    optimizer.zero_grad(set_to_none=True)
                    loss = self.average_loss(sq_short, t_mask_short, t_short, clip_qs_var_min=1e-4,
                                             clip_qs_var_max=1e+3, beta=beta, N_train=N_train)
                    if max_norm is not None:
                        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm,
                                                             error_if_nonfinite=True)
                        # if grad_norm > max_norm:
                        #     print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                    loss.backward()
                    optimizer.step()
            
            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.detach().item():.6f}')

                if validation and (epoch > 50):
                    val_rmse, val_nll = self.predict_gpvae(data_split="val", timestamp_batch_size=timestamp_batch_size,
                                                            series_batch_size=series_batch_size, device=device,
                                                            num_samples=num_samples, return_pred=False)
                    if val_rmse < best_val_rmse:
                        print(f"Saving best val rmse model, with val rmse: {val_rmse}")
                        torch.save(self.state_dict(), f"{save_folder}/best_val_rmse.pt")
                        best_val_rmse = val_rmse
                    
                    if val_nll < best_val_nll:
                        print(f"Saving best val nll model, with val nll: {val_nll}")
                        torch.save(self.state_dict(), f"{save_folder}/best_val_nll.pt")
                        best_val_nll = val_nll
        
        return best_val_rmse, best_val_nll
    
    @torch.no_grad()
    def predict_gpvae(self, data_split: str, timestamp_batch_size: int, series_batch_size: int, 
                      device: str = "cpu", num_samples=20, return_pred: bool = False
    ):
        self.to(device)

        entire_test_dataset = LazySeriesDatasetMisFrms(
            self.file_path, data_split=data_split, H=None,
        )

        series_dataloader = DataLoader(entire_test_dataset, batch_size=series_batch_size, shuffle=False)

        all_nll, all_se, all_pred_mean, all_pred_std = [], [], [], []

        for miniseries_sq_full, miniseries_t_full, miniseries_t_mask, vid_ids in series_dataloader:
            N_train = miniseries_t_mask.sum((-1, -2)).to(device)  # miniseries_t_mask: [v,T,1] --> [v]
            # or N_train = None (don't forget to change train_gpvae!)

            # store the full data for extracting NNs
            miniseries_dict_full = generate_shorter_miniseries_dict(
                miniseries_sq_full, miniseries_t_full, miniseries_t_mask, clip=False 
            )
            test_dataset = NNDatasetMujoco(
                miniseries_dict_full, series_shape=miniseries_t_full.shape[:-2]
            )

            time_dataloader = DataLoader(test_dataset, batch_size=timestamp_batch_size, shuffle=False)

            # mini-batch encoding
            tmp_store = [[] for _ in range(5)]
            for sq_short, t_short, t_mask_short in time_dataloader:
                sq_short = sq_short.to(device).transpose(0, 1)                    # [v, b, D]
                t_short = t_short.to(device).transpose(0, 1).contiguous()         # [v, b, 1]
                t_mask_short = t_mask_short.to(device).transpose(0, 1)[..., 0]    # [v, b]
                qnet_mus, qnet_vars = self.build_MLP_inference_graph(sq_short)    # [(v),b,L]

                for i, item in enumerate([sq_short, t_short, t_mask_short, qnet_mus, qnet_vars]):
                    tmp_store[i].append(item)

            sq_long, t_long, t_mask_long, qnet_mus, qnet_vars = [
                torch.cat(x, dim=1) for x in tmp_store
            ]

            mu_qs, cov_qs_diag, mu, A = self.svgp.approx_posterior_params(
                t_long, qnet_mus, qnet_vars, x_test=t_long, diag_cov_qs=True,
                masks=t_mask_long, N_train=N_train,
            )  # [(v),L,T], [(v),L,T], [(v),L,M], [(v),L,M,M]
            r_dict = predict_y(mu_qs.mT, cov_qs_diag.mT.sqrt(), self.decoder, sq_long, s=num_samples)
            all_se.append(r_dict["se"])
            all_nll.append(r_dict["nll"])

            if return_pred:
                all_pred_mean.append(r_dict["pred_mean"])
                all_pred_std.append(r_dict["pred_std"])
        
        all_se, all_nll = torch.cat(all_se, dim=0), torch.cat(all_nll, dim=0)

        mean_rmse, mean_nll = all_se.mean(dim=(-1, -2)).sqrt().mean(), all_nll.mean()

        if not return_pred:
            return mean_rmse, mean_nll
        else:
            all_pred_mean, all_pred_std = torch.cat(all_pred_mean, dim=0), torch.cat(all_pred_std, dim=0)
            return mean_rmse, mean_nll, all_pred_mean, all_pred_std
