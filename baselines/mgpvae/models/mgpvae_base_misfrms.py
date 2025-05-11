import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from baselines.mgpvae.models.mgpvae_base import MGPVAEBase
from baselines.svgpvae.utils.elbo_utils import negative_gaussian_cross_entropy


class MGPVAEBaseMisFrms(MGPVAEBase):
    # override
    def forward(self, vid_batch: Tensor, t_batch: Tensor, m_batch: Tensor = None, num_samples=1):
        """
        `vid_batch`: [v,T,...]; `t_batch`/`m_batch`: [v,T,1]`
        return (posterior) smoothing distribution [v,T,L], reconstruction, and log likelihood.
        """
        if m_batch is None:
            m = torch.ones_like(t_batch, dtype=torch.bool)
        else:
            m = m_batch.to(torch.bool) if m_batch.dtype != torch.bool else m_batch
        m = m.squeeze(-1)

        # diffent from the official code which makes the unobserved frames all zeros,
        # we compute as few frames as possible
        pseudo_y = torch.zeros(
            *t_batch.shape[:-1], self.latent_dims, device=vid_batch.device, requires_grad=True).clone()
        pseudo_var = torch.zeros_like(pseudo_y, requires_grad=True).clone()
        qnet_mus, qnet_vars = self.build_MLP_inference_graph(vid_batch[m])       # [<v*T,L]
        pseudo_y[m], pseudo_var[m] = qnet_mus, qnet_vars

        dt_nonzero = torch.diff(t_batch, dim=-2)                              # [v,T-1,1]
        # follow the official code agian, making the unobserved part of `dt` all zeros
        dt_nonzero = (dt_nonzero * m[..., 1:].unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)   # [v,T-1,1,1,1]
        if m_batch is not None:
            m_nonzero = m_batch[..., 1:, :].unsqueeze(-1).unsqueeze(-1)
        else:
            m_nonzero = None
        _, _, m_ss_Z, P_ss_Z, sum_log_p = self.ssm.update_posterior(
            dt_nonzero, pseudo_y, pseudo_var, m_batch=m_nonzero, return_state=False
        )                     # [v,L,T]
        m_ss_Z = m_ss_Z.mT    # [v,T,L]
        P_ss_Z = P_ss_Z.mT

        eps = torch.randn([num_samples, *m_ss_Z.shape], device=m_ss_Z.device)
        latent_samples = m_ss_Z + P_ss_Z.sqrt() * eps
        rec_vid = self.decoder(latent_samples)  # [n, v,T,...]
        return pseudo_y, pseudo_var, m_ss_Z, P_ss_Z, rec_vid, sum_log_p

    def expected_log_prob(self, m_ss_Z, P_ss_Z, y_batch, m_batch=None, **kwargs):  # [v,T,L], [v,T,1]
        """Compute E2 in the paper: E_q(Z)[log p(Y|Z)]"""
        if m_batch is None:
            m_batch = torch.ones([*m_ss_Z.shape[:-1], 1], device=m_ss_Z.device, dtype=torch.bool)
        elif m_batch.dtype != torch.bool:
            m_batch = m_batch.to(torch.bool)
        m = m_batch.squeeze(-1)

        mean_masked, cov_masked = m_ss_Z[m], P_ss_Z[m]
        eps = torch.randn_like(mean_masked)
        latent_samples = mean_masked + cov_masked.sqrt() * eps
        rec_vid_logits = self.decoder(latent_samples)  # [<v*T,...]

        if self.decoder.output_distribution == 'bernoulli':
            exp_lk = - nn.functional.binary_cross_entropy_with_logits(
                input=rec_vid_logits, target=y_batch[m], reduction='none')  # [<v*T,28,28]
        else:
            raise NotImplementedError
        return exp_lk.sum()  # sum over v*T

    # override, using sum loss
    def average_loss(self, vid_batch, t_batch, m_batch=None, beta=1.) -> Tensor:
        pseudo_y, pseudo_var, m_ss_Z, P_ss_Z, _, sum_log_p = self.forward(vid_batch, t_batch, m_batch, num_samples=1)

        # E2 with mask
        exp_lk_observed = self.expected_log_prob(m_ss_Z, P_ss_Z, vid_batch, m_batch)

        # E3
        E3 = sum_log_p.sum()  # sum over v*T

        # E1 negative cross entropy: int q(Z)log N(tilde_Y|Z,tilde_var), mean/cov: [v,T,L]
        if m_batch is None:
            E1 = negative_gaussian_cross_entropy(m_ss_Z, P_ss_Z, pseudo_y, pseudo_var)
        else:
            assert m_batch.dtype == torch.bool
            m = m_batch.squeeze(-1)
            E1 = negative_gaussian_cross_entropy(m_ss_Z[m], P_ss_Z[m], pseudo_y[m], pseudo_var[m])

        E1 = E1.sum()

        KL = E1 - E3
        elbo = (exp_lk_observed - beta * KL) / t_batch.shape[:-2].numel()  # averaged over v
        return - elbo

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs,
            series_batch_size: int, max_norm: float = None, device='cpu', print_epochs=1
    ):
        self.to(device)
        series_dataloader = DataLoader(self.entire_train_dataset, batch_size=series_batch_size, shuffle=True)
        for epoch in range(epochs):
            for series_sq_full, series_t_full, series_t_mask, vid_ids in series_dataloader:
                # We use the whole sequence
                series_sq_full = series_sq_full.to(device)  # [v,T,28,28], t: [v,T,1]
                series_t_full = series_t_full.to(device)
                series_t_mask = series_t_mask.to(device=device, dtype=torch.bool)

                optimizer.zero_grad(set_to_none=True)
                loss = self.average_loss(series_sq_full, series_t_full, series_t_mask, beta=beta)
                loss.backward()
                if max_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm, error_if_nonfinite=True)
                    if grad_norm > max_norm:
                        print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item(): .6f}',
                      f'global grad norm: {grad_norm}' if max_norm is not None else ' ', '\n')






