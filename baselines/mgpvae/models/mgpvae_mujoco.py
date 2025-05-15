from typing import Union
from pathlib import Path

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from gpytorch.kernels import MaternKernel

from baselines.mgpvae.models.mgpvae_base import MGPVAEBase
from baselines.mgpvae.models.ssm import SSM
from baselines.mgpvae.models.building_blocks.kernels import Matern32SSM
from baselines.svgpvae.utils.elbo_utils import negative_gaussian_cross_entropy

from models.building_blocks.enc_dec_mujoco import MujocoEncoder, MujocoDecoder
from utils.lazy_datasets_misfrms import LazySeriesDatasetMisFrms
from utils.mujoco_metric import predict_y

class MGPVAEMujoco(MGPVAEBase):
    """
    MGPVAE for Mujoco, with missing frames.
    NOTE: no mini-batch along frames axis.
    """
    def __init__(
            self, file_path: Union[str, Path], GP_joint=True, fix_decoder_var=False, y_dim=14, init_lengthscale=50.,
            device='cpu', jitter: float=1e-6,
    ):  
        num_latents = 15
        encoder = MujocoEncoder(y_dim=y_dim, num_latents=num_latents)
        decoder = MujocoDecoder(y_dim=y_dim, num_latents=num_latents, fix_variance=fix_decoder_var)

        kernel = Matern32SSM(base_kernel=MaternKernel(nu=1.5, batch_shape=torch.Size([num_latents])))
        for p in kernel.parameters():
            p.requires_grad_(GP_joint)
        kernel.outputscale, kernel.base_kernel.lengthscale = 1., init_lengthscale
        ssm = SSM(num_latents, kernel, jitter=jitter)

        super(MGPVAEMujoco, self).__init__(encoder, decoder, ssm, device=device)

        self.entire_train_dataset = LazySeriesDatasetMisFrms(
            file_path, data_split="train", H=None, fill_miss_t=True,
        )
        self.file_path = file_path
        self.num_frames_per_series = self.entire_train_dataset[0][0].size(0)  # 1000

    # override
    def forward(self, vid_batch_miss_f: Tensor, t_batch: Tensor, m_batch_miss_f: Tensor):
        """
        Support missing frames
        vid_batch_miss_f: [v,T,14]; t_batch: [v,T,1]; m_batch_miss_f: [v,T]
        """
        qnet_mus, qnet_vars = self.build_MLP_inference_graph(vid_batch_miss_f) 
        dt_nonzero = torch.diff(t_batch, dim=-2).unsqueeze(-1).unsqueeze(-1)
        _, _, m_ss_Z, P_ss_Z, sum_log_p = self.ssm.update_posterior(dt_nonzero, qnet_mus, qnet_vars, m=m_batch_miss_f)
        m_ss_Z = m_ss_Z.mT
        P_ss_Z = P_ss_Z.mT

        latent_samples = m_ss_Z + P_ss_Z.sqrt() * torch.randn_like(m_ss_Z)
        rec_vid = self.decoder(latent_samples) 
        # rec_vid = torch.where(m_batch_miss_f.unsqueeze(-1).bool(), rec_vid, 0.)
        return qnet_mus, qnet_vars, m_ss_Z, P_ss_Z, rec_vid, sum_log_p

    # override
    def expected_log_prob(self, y_recon: Tensor, y_batch: Tensor, m_batch_miss_f: Tensor, **kwargs):
        """Compute E2 in the paper: E_q(Z)[log p(Y|Z)], with missing frames"""
        if self.decoder.output_distribution == 'normal':
            sigma2_y = self.decoder.sigma2_y
            _exp_lk = - 1 / 2 * ((y_batch - y_recon).square() / sigma2_y + torch.log(2 * torch.pi * sigma2_y))
            exp_lk = torch.where(m_batch_miss_f.unsqueeze(-1).bool(), _exp_lk, 0) 
        else:
            raise NotImplementedError

        return exp_lk.sum() 

    # override
    def average_loss(self, vid_batch_miss_f: Tensor, t_batch: Tensor, m_batch_miss_f: Tensor, beta=1.):
        # vid_batch_miss_f: [v,T,14]; t_batch: [v,T,1]; m_batch_miss_f: [v,T]
        if m_batch_miss_f.ndim == 3:
            m_batch_miss_f = m_batch_miss_f.squeeze(-1)
        else:
            assert m_batch_miss_f.ndim == 2, "Mask MUST has 2 or 3 dims!"
        qnet_mus, qnet_vars, m_ss_Z, P_ss_Z, rec_vid, sum_log_p = self.forward(vid_batch_miss_f, t_batch, m_batch_miss_f)

        # E2
        exp_lk_observed = self.expected_log_prob(rec_vid, vid_batch_miss_f, m_batch_miss_f)

        # E3
        E3 = sum_log_p.sum() 

        # E1 negative cross entropy
        # m_ss_Z, P_ss_Z, qnet_mus, qnet_vars: [v,T,L]
        E1 = negative_gaussian_cross_entropy(m_ss_Z, P_ss_Z, qnet_mus, qnet_vars)
        E1 = torch.where(m_batch_miss_f.unsqueeze(-1).bool(), E1, 0.)
        E1 = E1.sum()

        KL = E1 - E3

        # scale = m_batch_miss_f.sum()  # to compute loss per frame
        scale = m_batch_miss_f.size(0)  # to compute loss per vid
        elbo = (exp_lk_observed - beta * KL) / scale
        return - elbo

    def train_gpvae(self, optimizer: torch.optim.Optimizer, beta, epochs,
                    series_batch_size: int, max_norm: float = None, device='cpu', print_epochs=1,
                    validation=True, num_samples=20, save_folder=""
    ):
        self.to(device)

        best_val_rmse, best_val_nll = torch.inf, torch.inf

        series_dataloader = DataLoader(self.entire_train_dataset, batch_size=series_batch_size, shuffle=True)
        for epoch in range(epochs):
            for miniseries_sq_full, _, miniseries_fill_t_full, miniseries_t_mask, vid_ids in series_dataloader:
                miniseries_sq_full, miniseries_fill_t_full, miniseries_t_mask = miniseries_sq_full.to(device), miniseries_fill_t_full.to(device), miniseries_t_mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = self.average_loss(vid_batch_miss_f=miniseries_sq_full, t_batch=miniseries_fill_t_full, m_batch_miss_f=miniseries_t_mask, beta=beta)
                loss.backward()
                if max_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm, error_if_nonfinite=True)
                    # if grad_norm > max_norm:
                    #     print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item(): .6f}',
                      f'global grad norm: {grad_norm}' if max_norm is not None else ' ', '\n')

                if validation and (epoch > 50):
                    val_rmse, val_nll = self.predict_gpvae(self.file_path, data_split="val", series_batch_size=series_batch_size,
                                                            device=device, num_samples=num_samples, return_pred=False)
                    if val_rmse < best_val_rmse:
                        print(f"Saving best val rmse model, with val rmse: {val_rmse:.6f}")
                        torch.save(self.state_dict(), f"{save_folder}/best_val_rmse.pt")
                        best_val_rmse = val_rmse

                    if val_nll < best_val_nll:
                        print(f"Saving best val nll model, with val nll: {val_nll:.6f}")
                        torch.save(self.state_dict(), f"{save_folder}/best_val_nll.pt")
                        best_val_nll = val_nll

        return best_val_rmse, best_val_nll

    @torch.no_grad()
    def predict_gpvae(
            self, file_path: Union[str, Path], data_split: str, series_batch_size: int, device='cpu', num_samples=20, return_pred = False
    ):
        self.to(device)

        test_dataset = LazySeriesDatasetMisFrms(
            file_path, data_split=data_split, H=None, fill_miss_t=True,
        )
        series_dataloader = DataLoader(test_dataset, batch_size=series_batch_size, shuffle=False)

        all_se, all_nll, all_pred_mean, all_pred_std = [], [], [], []

        for miniseries_sq_full, miniseries_t_full, miniseries_fill_t_full, miniseries_t_mask, vid_ids in series_dataloader:
            miniseries_sq_full, miniseries_t_full, miniseries_fill_t_full, miniseries_t_mask = (
                miniseries_sq_full.to(device), miniseries_t_full.to(device),
                miniseries_fill_t_full.to(device), miniseries_t_mask.to(device))
            qnet_mus, qnet_vars = self.build_MLP_inference_graph(miniseries_sq_full)
            # latent GP prediction
            pred_m_Zs, pred_cov_Zs = self.ssm.predict(qnet_mus, qnet_vars, miniseries_t_mask, miniseries_fill_t_full, miniseries_t_full)
            r_dict = predict_y(pred_m_Zs.mT, pred_cov_Zs.mT.sqrt(), self.decoder, miniseries_sq_full, s=num_samples)

            all_se.append(r_dict["se"])
            all_nll.append(r_dict["nll"])

            if return_pred:
                all_pred_mean.append(r_dict["pred_mean"])
                all_pred_std.append(r_dict["pred_std"])

        all_se, all_nll = torch.cat(all_se, dim=0), torch.cat(all_nll, dim=0)
        mean_rmse, mean_nll, = all_se.mean(dim=(-1, -2)).sqrt().mean(), all_nll.mean()

        if not return_pred:
            return mean_rmse, mean_nll
        else:
            all_pred_mean = torch.cat(all_pred_mean, dim=0).to('cpu')
            all_pred_std = torch.cat(all_pred_std, dim=0).to('cpu')

            return mean_rmse, mean_nll, all_pred_mean, all_pred_std
