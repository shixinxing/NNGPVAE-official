import warnings
import torch
from torch import LongTensor, Tensor, nn
from torch.utils.data import DataLoader
from linear_operator.utils.cholesky import psd_safe_cholesky

from models.gpvae_base_misfrms import GPVAEBaseMisFrms
from utils.lazy_datasets_misfrms import NNDatasetMisFrms, generate_shorter_miniseries_dict


class GPVAEBaseMisFrms_SWS(GPVAEBaseMisFrms):
    # override
    def kl_divergence_sws(self, t_batch: Tensor, m_batch: Tensor = None, vid_ids: LongTensor = None, **kwargs):
        """
        t_batch, i.e., query_t: [(v),t<Tmax,D=1], vid_ids: [v] used in `self.get_nn_data`.
        We currently consider all kl indices in range(Tmax), so some KL indices might have no frame in some videos,
        but their neighbors are restricted to the observed frames.
        """
        if m_batch is None:
            m_batch = torch.ones_like(t_batch)
        m_batch = m_batch[..., 0].to(torch.bool)     # [v,t]
        if torch.all(~m_batch):
            warnings.warn("Got a batch without observed frames.")
            return torch.tensor(0., device=self.module_device)

        y_nn, t_nn = self.get_nn_data(t_batch, H=self.H, vid_ids=vid_ids)[:2]  # [v,t,H,32,32], [(v),t,H,D]
        y_nn, t_nn = y_nn.to(self.module_device), t_nn.to(self.module_device)
        t_nn_pick = t_nn[m_batch]                    # [<v*t,H,1]

        mean_q, std_q = self.encoder(y_nn[m_batch])  # [<v*t,H,L]
        mean_q, std_q = mean_q.mT, std_q.mT          # [<v*t,L,H]
        mean_p, cov_p = self.gp.prior(t_nn_pick, are_neighbors=False)          # [<v*t,L,H], [<v*t,L,H,H]

        L = psd_safe_cholesky(cov_p + self.jitter * torch.eye(self.H, device=self.module_device))  # [<v*t,L,H,H]
        mean_diff = (mean_q - mean_p).unsqueeze(-1)           # [<v*t,L,H,1]
        mahalanobis = torch.linalg.solve_triangular(L, mean_diff, upper=False)
        mahalanobis = mahalanobis.square().sum(dim=(-1, -2))  # [<v*t,L]

        L_inv = torch.linalg.solve_triangular(L, torch.eye(L.size(-1), device=self.module_device), upper=False)
        tmp = L_inv * std_q.unsqueeze(-2)                     # [<v*t,L,H,H]
        trace = tmp.square().sum(dim=(-1, -2))                # [<v*t,L]

        log_det_cov_p = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # [<v*t,L]
        log_det_cov_q = std_q.log().sum(dim=-1)               # [<v*t,L]
        log_det = log_det_cov_p - log_det_cov_q               # broadcast

        kl = torch.sum(0.5 * (mahalanobis + trace - self.H) + log_det, dim=-1)         # [<v*t]
        kl_mat = torch.zeros_like(m_batch, requires_grad=True, dtype=t_batch.dtype)    # [v,t]
        kl_mat = kl_mat.clone()  # in-place operation not allowed
        kl_mat[m_batch] = kl     # assign
        len_batch_inv = torch.where(m_batch.sum(dim=-1) > 0, 1 / m_batch.sum(-1), 0.)  # [v]
        kl = torch.sum(kl_mat * len_batch_inv.unsqueeze(-1))

        # test, jitter will have effect
        true_kl, num_obs_frames = 0., 0.
        for v in range(self.train_dataset.series_shape.numel()):
            m = m_batch[v]
            if torch.all(~m):
                warnings.warn("Got a batch without observed frames.")
                continue
            num_obs_frames += m.sum()

            t_nnn, y_nnn = t_nn[v], y_nn[v]      # [t,H,1/D]
            mean_q, std_q = self.encoder(y_nnn)  # [t,H,L]
            mean_q, std_q = mean_q.mT, std_q.mT  # [t,L,H]
            q = torch.distributions.MultivariateNormal(loc=mean_q[m], covariance_matrix=torch.diag_embed(std_q[m].square()))

            mean_p, cov_p = self.gp.prior(t_nnn, are_neighbors=False)  # [t,L,H]
            p = torch.distributions.MultivariateNormal(
                loc=mean_p[m], covariance_matrix=cov_p[m] + self.jitter * torch.eye(self.H, device=self.module_device))
            res_v = torch.distributions.kl.kl_divergence(q, p)
            true_kl = true_kl + res_v.sum() / m.sum()
        print(f"num observed frames: {num_obs_frames}, true kl: {true_kl/self.train_dataset.series_shape.numel()}.")
        print(f"my num observed frames: {m_batch.sum()}, my kl: {kl/self.train_dataset.series_shape.numel()}.\n")

        return kl / self.train_dataset.series_shape.numel()  # average over v

    def average_loss_sws(
            self, vid_batch_short: Tensor, t_batch_short: Tensor, t_mask_short: Tensor,
            vid_ids: LongTensor, beta=1.
    ):
        lik = self.expected_log_prob(vid_batch_short, t_mask_short)
        kl = self.kl_divergence_sws(t_batch_short, t_mask_short, vid_ids=vid_ids)
        assert torch.isfinite(lik).all(), "Likelihood now is not finite."
        assert torch.isfinite(kl).all(), "KL divergence now is not finite."
        elbo = lik - beta * kl
        return - elbo

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs: int,
            series_batch_size: int, timestamp_batch_size: int, max_norm: float = None,
            device="cpu", print_epochs=1
    ):
        if device != self.module_device:
            warnings.warn(f"Training is not on self.module_device ({self.module_device}) set earlier. "
                          f"We have changed module device to {device} and train the model on it.")
            self.module_device = device
        self.to(device)

        series_dataloader = DataLoader(self.entire_train_dataset, batch_size=series_batch_size, shuffle=True)
        for epoch in range(epochs):
            for miniseries_sq_full, miniseries_t_full, miniseries_t_mask, vid_ids in series_dataloader:
                miniseries_shorter_dict = generate_shorter_miniseries_dict(
                    miniseries_sq_full, miniseries_t_full, miniseries_t_mask, clip=True
                )
                self.train_dataset = NNDatasetMisFrms(
                    miniseries_shorter_dict, series_shape=miniseries_t_full.shape[:-2], data_device=self.data_device,
                    search_device=self.search_device
                )

                time_dataloader = DataLoader(self.train_dataset, batch_size=timestamp_batch_size, shuffle=True)
                for sq_short, t_short, m_short in time_dataloader:
                    sq_short = sq_short.to(device).transpose(0, 1)  # [v, T, D]
                    t_short = t_short.to(device).transpose(0, 1).contiguous()  # [v, T, 1]
                    m_short = m_short.to(device).transpose(0, 1)

                    optimizer.zero_grad(set_to_none=True)
                    loss = self.average_loss_sws(sq_short, t_short, m_short, vid_ids, beta=beta)
                    loss.backward()
                    if max_norm is not None:
                        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm, error_if_nonfinite=True)
                        if grad_norm > max_norm:
                            print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                    optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item(): .6f}',
                      f'global grad norm: {grad_norm}\n' if max_norm is not None else '\n')
