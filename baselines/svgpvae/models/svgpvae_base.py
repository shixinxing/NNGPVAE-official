# Adapted from SVGPVAE's official code: https://github.com/ratschlab/SVGP-VAE/blob/main/SVGPVAE_model.py
# replace TensorFlow 1.x with Pytorch version

import torch
from torch import nn, Tensor
from linear_operator.utils.cholesky import psd_safe_cholesky

from baselines.svgpvae.models.svgp import SVGP
from baselines.svgpvae.utils.elbo_utils import negative_gaussian_cross_entropy


class SVGPVAEBase(nn.Module):
    def __init__(
            self, encoder: nn.Module, decoder: nn.Module, svgp: SVGP,
            device: str = 'cpu', jitter: float = 0., geco=False
    ):
        super(SVGPVAEBase, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.svgp = svgp
        self.M, self.N_train = svgp.inducing_points.size(-2), svgp.N_train

        self.train_dataset = None

        self.latent_dims = self.svgp.output_dims
        self.device = device
        self.jitter = jitter
        self.geco = geco

    def build_MLP_inference_graph(self, y_batch: Tensor):  # returns vars instead of stds
        qnet_mus, qnet_stds = self.encoder(y_batch)
        qnet_vars = qnet_stds.square()
        return qnet_mus, qnet_vars  # [(v),n,L]

    def forward(self, vid_batch: Tensor, t_batch: Tensor, cov_latent_diag=True, num_samples=1, **kwargs):
        qnet_mus, qnet_vars = self.build_MLP_inference_graph(vid_batch)
        mu_qs, cov_qs, _, _ = self.svgp.approx_posterior_params(
            t_batch, qnet_mus, qnet_vars, x_test=t_batch, diag_cov_qs=cov_latent_diag
        )  # using the full batch here
        full_p_mu = mu_qs.mT
        full_p_var = cov_qs.mT if cov_latent_diag else torch.diagonal(cov_qs, dim1=-2, dim2=-1).mT  # [v,f,2]

        if cov_latent_diag:  # as the official code
            eps = torch.randn([num_samples, *full_p_mu.shape], dtype=full_p_mu.dtype, device=full_p_mu.device)
            latent_samples = full_p_mu + torch.sqrt(torch.clamp(full_p_var, 1e-4, 1e3)) * eps
        else:
            L_cov_latent = psd_safe_cholesky(cov_qs)     # [v,2,f,f]
            eps = torch.randn([num_samples, *mu_qs.shape], dtype=mu_qs.dtype, device=mu_qs.device)
            latent_samples = mu_qs + L_cov_latent @ eps  # [s,v,2,f]
            latent_samples = latent_samples.mT

        recon_imgs = self.decoder(latent_samples)
        return full_p_mu, full_p_var, recon_imgs

    def expected_log_prob(self, mu_qs: Tensor, vars_qs_diag_clipped: Tensor, y_batch: Tensor, **kwargs):  # [(v),L,b]
        eps = torch.randn_like(mu_qs)
        latent_sample = (mu_qs + eps * torch.sqrt(vars_qs_diag_clipped)).mT  # [(v),b,L]
        recon_imgs = self.decoder(latent_sample)                             # [(v),b,32,32]
        if self.decoder.output_distribution == 'bernoulli':
            expected_lk = - nn.functional.binary_cross_entropy(input=recon_imgs, target=y_batch, reduction='sum')
        else:
            raise NotImplementedError
        return expected_lk  # v*b

    def average_loss(
            self, vid_batch: Tensor, t_batch: Tensor, clip_qs_var_min=None, clip_qs_var_max=None, beta=1., **kwargs
    ):
        assert not self.geco, "Do not support GECO in this base class."
        qnet_mus, qnet_vars = self.build_MLP_inference_graph(vid_batch)  # [(v),b,L]
        # get q(Z_U) and q_s(Z): [(v),L,b]
        mu_qs, cov_qs_diag, mu, A = self.svgp.approx_posterior_params(
            t_batch, qnet_mus, qnet_vars, x_test=t_batch, diag_cov_qs=True
        )

        # term 1/3 expected-log-likelihood
        expected_lk = self.expected_log_prob(
            mu_qs, torch.clamp(cov_qs_diag, min=clip_qs_var_min, max=clip_qs_var_max), vid_batch
        )

        # term 2/3 cross-entropy
        cross_entry = negative_gaussian_cross_entropy(mu_qs, cov_qs_diag, qnet_mus.mT, qnet_vars.mT)  # [(v),L,b]
        cross_entry = cross_entry.sum()

        # term 3/3 - L_H
        inside_elbo_batch_log, inside_elbo_KL = self.svgp.variational_loss(t_batch, qnet_mus, qnet_vars, mu, A)
        L_H = self.N_train / t_batch.size(-2) * inside_elbo_batch_log.sum() - inside_elbo_KL.sum()

        v, b = qnet_mus.shape[:-2].numel(), qnet_vars.shape[-2]
        kl = self.N_train / b * cross_entry - L_H
        elbo = expected_lk / (v * b) - beta * kl / (v * self.N_train)
        return - elbo

