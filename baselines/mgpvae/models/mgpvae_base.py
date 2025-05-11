import torch
from torch import nn, Tensor

from baselines.mgpvae.models.ssm import SSM
from baselines.svgpvae.utils.elbo_utils import negative_gaussian_cross_entropy


class MGPVAEBase(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, ssm: SSM):
        super(MGPVAEBase, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.ssm = ssm

        self.latent_dims = ssm.output_dims

    def build_MLP_inference_graph(self, y_batch):
        """
        Get pseudo y and noise through the encoder.
        `y_batch`: [v,T,...], mini-batches of sequences containing all time steps (frames).
        """
        # if we follow the official code to use "natural parameterization"
        # lambda1, lambda2 = self.encoder(y_batch)
        # lambda2 = - lambda2
        # qnet_mus = lambda1 * lambda2
        # qnet_vars = - 0.5 * lambda2

        qnet_mus, qnet_stds = self.encoder(y_batch)
        qnet_vars = qnet_stds.square()
        return qnet_mus, qnet_vars   # [v,T,L]

    def forward(self, vid_batch: Tensor, t_batch: Tensor, num_samples=1, **kwargs):
        """
        `vid_batch`: [v,T,...]; `t_batch`: [t,T,1]`
        return (posterior) smoothing distribution, reconstruction, and log likelihood.
        """
        qnet_mus, qnet_vars = self.build_MLP_inference_graph(vid_batch)       # [v,T,L]
        dt_nonzero = torch.diff(t_batch, dim=-2).unsqueeze(-1).unsqueeze(-1)  # [v,T-1,1,1,1]
        _, _, m_ss_Z, P_ss_Z, sum_log_p = self.ssm.update_posterior(dt_nonzero, qnet_mus, qnet_vars)  # [v,L,T]
        m_ss_Z = m_ss_Z.mT    # [v,T,L]
        P_ss_Z = P_ss_Z.mT

        eps = torch.randn([num_samples, *m_ss_Z.shape], device=m_ss_Z.device)
        latent_samples = m_ss_Z + P_ss_Z.sqrt() * eps
        rec_vid = self.decoder(latent_samples)  # [n, v,T,...]
        return qnet_mus, qnet_vars, m_ss_Z, P_ss_Z, rec_vid, sum_log_p

    def expected_log_prob(self, m_ss_Z, P_ss_Z, y_batch, **kwargs):  # [v,T,L]
        """Compute E2 in the paper: E_q(Z)[log p(Y|Z)]"""
        latent_samples = m_ss_Z + P_ss_Z.sqrt() * torch.randn_like(m_ss_Z)
        rec_vid = self.decoder(latent_samples)  # [v,T,...]

        if self.decoder.output_distribution == 'bernoulli':
            exp_lk = - nn.functional.binary_cross_entropy(input=rec_vid, target=y_batch, reduction='sum')
        else:
            raise NotImplementedError(f"{self.decoder.output_distribution} not implemented")
        return exp_lk  # sum over v*T

    def average_loss(self, vid_batch, t_batch, beta=1., **kwargs):  # [v,T,...], [v,T,1]
        qnet_mus, qnet_vars, m_ss_Z, P_ss_Z, _, sum_log_p = self.forward(vid_batch, t_batch)

        # E2
        exp_lk = self.expected_log_prob(m_ss_Z, P_ss_Z, vid_batch)

        # E3
        E3 = sum_log_p.sum()   # sum over v*T

        # E1 negative cross entropy: int q(Z)log N(tilde_Y|Z,tilde_var), mean/cov: [v,T,L]
        E1 = negative_gaussian_cross_entropy(m_ss_Z, P_ss_Z, qnet_mus, qnet_vars)
        E1 = E1.sum()

        KL = E1 - E3
        v, T = t_batch.shape[:-2].numel(), t_batch.size(-2)
        elbo = (exp_lk - beta * KL) / v  # only average over video dim
        return - elbo






