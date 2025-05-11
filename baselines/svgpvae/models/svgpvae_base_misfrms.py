import warnings

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from baselines.svgpvae.models.svgpvae_base import SVGPVAEBase
from baselines.svgpvae.utils.elbo_utils import negative_gaussian_cross_entropy
from utils.lazy_datasets_misfrms import NNDatasetMisFrms, generate_shorter_miniseries_dict


class SVGPVAEBaseMisFrms(SVGPVAEBase):
    def __init__(self, *args, **kwargs):
        super(SVGPVAEBaseMisFrms, self).__init__(*args, **kwargs)

    # override
    def expected_log_prob(self, mu_qs: Tensor, vars_qs_diag_clipped: Tensor, y_batch: Tensor, m_batch: Tensor = None):
        """y_batch: [(v),b,28,28] and masks: [(v),b,1], 0 refers to missing"""
        if m_batch is None:
            m_batch = torch.ones(
                y_batch.shape[:len(self.train_dataset.series_shape)+1], dtype=torch.bool, device=y_batch.device)
        else:
            m_batch = m_batch[..., 0].to(torch.bool)                         # [(v),b]
            if torch.all(~m_batch):
                warnings.warn("Got a batch without observed frames.")
                return torch.tensor(0., device=self.module_device)

        eps = torch.rand_like(mu_qs)
        latent_sample = (mu_qs + eps * torch.sqrt(vars_qs_diag_clipped)).mT  # [(v),b,L]
        y_rec = self.decoder(latent_sample)                                  # [(v),b,28,28]
        if self.decoder.output_distribution == 'normal':
            sigma2_y = self.decoder.sigma2_y
            expected_lk = (y_batch - y_rec).square() / sigma2_y + torch.log(2 * torch.pi * sigma2_y)
            expected_lk = - 1 / 2 * expected_lk
        elif self.decoder.output_distribution == 'bernoulli':
            expected_lk = - nn.functional.binary_cross_entropy_with_logits(
                input=y_rec, target=y_batch, reduction='none'
            )
        else:
            raise NotImplementedError

        # expected_lk = expected_lk * m_batch.reshape(
        #     [*m_batch.shape, *(1 for _ in range(y_rec.ndim - m_batch.ndim))])    # [(v),b,28,28]
        expected_lk = torch.where(
            m_batch.reshape([*m_batch.shape, *(1 for _ in range(y_rec.ndim - m_batch.ndim))]),
            expected_lk, 0.
        )                                                                        # [(v),b,28,28]
        expected_lk = expected_lk.sum(
            dim=[i for i in range(-1, -(y_rec.ndim - m_batch.ndim + 2), -1)])    # [(v)]
        # scaling
        len_batch_inv = torch.where(m_batch.sum(-1) > 0, 1/m_batch.sum(-1), 0.)  # [(v)]
        scale = self.train_dataset.masks.to(m_batch.device).sum(dim=(-1, -2)) * len_batch_inv
        lik = torch.sum(expected_lk * scale)
        return lik / self.train_dataset.series_shape.numel()  # sum over f, average over v

    # override
    def average_loss(
            self, vid_batch: Tensor, t_batch: Tensor, m_batch: Tensor = None,
            clip_qs_var_min=None, clip_qs_var_max=None, beta=1.
    ):
        """vid_batch: [(v),b,28,28]; m_batch: [(v),b,1]; t_batch: [(v),b,1]."""
        if m_batch is None:
            N_train = self.N_train
        else:
            N_train = self.train_dataset.masks.to(m_batch.device).sum(dim=(-1, -2))  # [v]

        qnet_mus, qnet_vars = self.build_MLP_inference_graph(vid_batch)  # [(v),b,L]
        # get q(Z_U) and q_s(Z): [(v),L,b]
        mu_qs, cov_qs_diag, mu, A = self.svgp.approx_posterior_params(
            t_batch, qnet_mus, qnet_vars, x_test=t_batch, diag_cov_qs=True,
            m_train_batch=m_batch, N_train=N_train
        )

        # scaling
        len_batch_inv = torch.where(m_batch.sum(dim=(-1, -2)) > 0, 1 / m_batch.sum(dim=(-1, -2)), 0.)  # [(v)]
        scale = self.train_dataset.masks.to(m_batch.device).sum(dim=(-1, -2)) * len_batch_inv

        # term 1/3 - expected-log-likelihood
        if clip_qs_var_max is None and clip_qs_var_min is None:
            clip_qs_var_max = 1e10
        expected_lk = self.expected_log_prob(
            mu_qs, torch.clamp(cov_qs_diag, min=clip_qs_var_min, max=clip_qs_var_max), vid_batch, m_batch=m_batch
        )

        # term 2/3 - cross-entropy
        cross_entry = negative_gaussian_cross_entropy(mu_qs, cov_qs_diag, qnet_mus.mT, qnet_vars.mT)  # [(v),L,b]
        # cross_entry = (cross_entry * m_batch.squeeze(-1).unsqueeze(-2)).mT
        cross_entry = torch.where(m_batch.squeeze(-1).unsqueeze(-2), cross_entry, 0.).mT              # [(v),b,L]
        cross_entry = torch.sum(cross_entry.sum(dim=(-1, -2)) * scale)
        cross_entry = cross_entry / self.train_dataset.series_shape.numel()

        # term 3/3 - L_H
        inside_elbo_batch_log, inside_elbo_KL = self.svgp.variational_loss(
            t_batch, qnet_mus, qnet_vars, mu, A, m_train_batch=m_batch)    # [(v)], [(v)]
        assert inside_elbo_batch_log.shape == self.train_dataset.series_shape
        assert inside_elbo_KL.shape == self.train_dataset.series_shape
        L_H = (scale * inside_elbo_batch_log).mean() - inside_elbo_KL.mean()

        KL = cross_entry - L_H
        assert torch.isfinite(expected_lk)
        assert torch.isfinite(KL)
        elbo = expected_lk - beta * KL
        return - elbo   # sum over f, average over v

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs,
            series_batch_size: int, timestamp_batch_size: int, max_norm: float = None,
            device='cpu', print_epochs=1
    ):
        self.to(device)

        series_dataloader = DataLoader(self.entire_train_dataset, batch_size=series_batch_size, shuffle=True)
        for epoch in range(epochs):
            for miniseries_sq_full, miniseries_t_full, miniseries_t_mask, _ in series_dataloader:
                miniseries_shorter_dict = generate_shorter_miniseries_dict(
                    miniseries_sq_full, miniseries_t_full, miniseries_t_mask, clip=True
                )
                self.train_dataset = NNDatasetMisFrms(
                    miniseries_shorter_dict, series_shape=miniseries_t_full.shape[:-2], data_device=self.data_device
                )

                time_dataloader = DataLoader(self.train_dataset, batch_size=timestamp_batch_size, shuffle=True)
                for sq_short, t_short, m_short in time_dataloader:
                    sq_short = sq_short.to(device).transpose(0, 1)  # [v, T, D]
                    t_short = t_short.to(device).transpose(0, 1).contiguous()  # [v, T, 1]
                    m_short = m_short.to(device).transpose(0, 1)

                    optimizer.zero_grad(set_to_none=True)
                    loss = self.average_loss(
                        sq_short, t_short, m_short, beta=beta, clip_qs_var_min=None, clip_qs_var_max=None
                    )
                    loss.backward()
                    if max_norm is not None:
                        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm, error_if_nonfinite=True)
                        if grad_norm > max_norm:
                            print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                    optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.detach().item(): .6f}',
                      f'global grad norm: {grad_norm}\n' if max_norm is not None else '\n')

