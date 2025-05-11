import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from baselines.svgpvae.models.svgpvae_base import SVGPVAEBase
from baselines.svgpvae.utils.elbo_utils import negative_gaussian_cross_entropy
from utils.spe10 import predict_y


class SVGPVAESpatialImp(SVGPVAEBase):
    """Used for spatial datasets"""
    # override, using mean loss here
    def expected_log_prob(
            self, mu_qs: Tensor, vars_qs_diag_clipped: Tensor, y_batch_miss, m_batch_miss=None
    ):  # [L,b], [b,D_y]
        if m_batch_miss is None:
            m_batch_miss = torch.zeros_like(y_batch_miss, dtype=torch.bool)
        elif m_batch_miss.dtype != torch.bool:
            m_batch_miss = m_batch_miss.to(torch.bool)

        latent_samles = mu_qs + torch.randn_like(mu_qs) * torch.sqrt(vars_qs_diag_clipped)
        latent_samles = latent_samles.mT
        y_rec = self.decoder(latent_samles)

        if self.decoder.output_distribution == 'normal':
            sigma2_y = self.decoder.sigma2_y
            expected_lk = (y_batch_miss - y_rec).square() / sigma2_y + torch.log(2 * torch.pi * sigma2_y)
            expected_lk = - 0.5 * expected_lk  # [b, D]
        else:
            raise NotImplementedError
        expected_lk = torch.where(m_batch_miss, expected_lk, 0.).sum()
        scale = y_batch_miss.shape[:len(self.train_dataset.series_shape) + 1].numel()
        return expected_lk / scale

    def average_loss(
            self, y_batch: Tensor, x_batch: Tensor, m_batch=None, clip_qs_var_min=None, clip_qs_var_max=None, beta=1.
    ):
        qnet_mus, qnet_vars = self.build_MLP_inference_graph(y_batch)  # [b,L]
        # get q(Z_U) and q_s(Z): [L,b]
        mu_qs, cov_qs_diag, mu, A = self.svgp.approx_posterior_params(
            x_batch, qnet_mus, qnet_vars, x_test=x_batch, diag_cov_qs=True
        )

        # term 1/3 - expected-log-likelihood
        expected_lk = self.expected_log_prob(
            mu_qs, torch.clamp(cov_qs_diag, min=clip_qs_var_min, max=clip_qs_var_max), y_batch, m_batch
        )

        # term 2/3 - cross-entropy
        cross_entry = negative_gaussian_cross_entropy(mu_qs, cov_qs_diag, qnet_mus.mT, qnet_vars.mT)  # [L,b]
        cross_entry = cross_entry.sum()

        # term 3/3 - L_H
        inside_elbo_batch_log, inside_elbo_KL = self.svgp.variational_loss(x_batch, qnet_mus, qnet_vars, mu, A)
        L_H = self.N_train / x_batch.size(-2) * inside_elbo_batch_log.sum() - inside_elbo_KL.sum()

        v, b = qnet_mus.shape[:-2].numel(), qnet_vars.shape[-2]
        kl = cross_entry / b - L_H / self.N_train
        elbo = expected_lk - beta * kl / v  # mean loss, averaged over N
        return - elbo

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs, batch_size: int, max_norm: float = None,
            device='cpu', print_epochs=1
    ):
        self.to(device)
        self.train_dataset.return_full = False
        dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for y_batch, x_batch, m_batch in dataloader:
                y_batch, x_batch = y_batch.to(device), x_batch.to(device)
                m_batch = m_batch.to(device=device, dtype=torch.bool)
                optimizer.zero_grad(set_to_none=True)
                # faiss needs continuity (t is usually contiguous from dataloader)
                loss = self.average_loss(
                    y_batch, x_batch, m_batch=m_batch, clip_qs_var_min=1e-4, clip_qs_var_max=1e3, beta=beta
                )
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

    @torch.no_grad()
    def predict_gpvae(
            self, batch_size: int, stat: dict = None, device: str = "cpu",
            num_samples=1, return_pred=False
    ):
        self.to(device)
        self.train_dataset.return_full = True
        dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)

        all_nll, all_se, all_ae, all_pred_mean, all_pred_std = [], [], [], [], []
        for y_batch, x_batch, m_batch, y_batch_full in dataloader:
            y_batch, x_batch, y_batch_full = y_batch.to(device), x_batch.to(device), y_batch_full.to(device)
            m_batch = m_batch.to(device=device, dtype=torch.bool)

            qnet_mus, qnet_vars = self.build_MLP_inference_graph(y_batch)
            # we seem not being able to use the whole series as it's large
            mu_qs, cov_qs, _, _ = self.svgp.approx_posterior_params(
                x_batch, qnet_mus, qnet_vars, x_test=x_batch, diag_cov_qs=True
            )
            full_p_mu, full_p_var = mu_qs.mT, cov_qs.mT   # [b,L]
            r_dict = predict_y(full_p_mu, full_p_var.sqrt(), self.decoder, y_batch_full, s=num_samples, stat=stat)

            all_se.append(r_dict['se'][~m_batch])
            all_ae.append(r_dict['ae'][~m_batch])
            all_nll.append(r_dict['nll'][~m_batch])
            all_pred_mean.append(r_dict['pred_mean'])
            all_pred_std.append(r_dict['pred_std'])

        all_se = torch.cat(all_se, dim=0)
        all_ae = torch.cat(all_ae, dim=0)
        all_nll = torch.cat(all_nll, dim=0)
        all_pred_mean = torch.cat(all_pred_mean, dim=0)  # [n_test, D_y]
        all_pred_std = torch.cat(all_pred_std, dim=0)

        if not return_pred:
            return all_se.mean().item(), all_ae.mean().item(), all_nll.mean().item()
        else:
            return (all_se.mean().item(), all_ae.mean().item(), all_nll.mean().item(),
                    all_pred_mean.cpu().numpy(), all_pred_std.cpu().numpy())


