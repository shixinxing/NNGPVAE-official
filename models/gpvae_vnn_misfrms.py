import warnings
import torch
from torch import LongTensor, Tensor, nn
from torch.utils.data import DataLoader
from linear_operator.utils.cholesky import psd_safe_cholesky

from models.gpvae_base_misfrms import GPVAEBaseMisFrms
from utils.lazy_datasets_misfrms import NNDatasetMisFrms, generate_shorter_miniseries_dict


class GPVAEBaseMisFrms_VNN(GPVAEBaseMisFrms):
    # override
    def kl_divergence_vnn(self, kl_indices: LongTensor, vid_ids: LongTensor, **kwargs):
        """
        kl_indices: indices of current batch data, [J]; vid_ids indicates the Index ids.
        sequential_nn_structure [v,Tmax-H,H] is pre-computed in `self.entire_train_dataset` and padded with zeros,
        """
        # get current data batch [v,J<Tmax,D/1], mask is along [v,J].
        y_batch, t_batch, m_batch = self.train_dataset[kl_indices][:3]   # [v,J,D/1]
        y_batch, t_batch = y_batch.to(self.module_device), t_batch.to(self.module_device)
        m_batch = m_batch.to(self.module_device)[..., 0]                 # [v, J]
        if torch.all(~m_batch):
            warnings.warn("Got a batch without observed frames.")
            return torch.tensor(0., device=self.module_device)
        t_batch_pick = t_batch[m_batch]                                  # [<v*J,1]

        seq_nn_struct_pad = self.entire_train_dataset.seq_nn_idx         # [V,Tmax-H,H]
        if seq_nn_struct_pad.device != t_batch.device:
            warnings.warn(f"Please move the sequential NN structure to {self.module_device} before training.")
            self.entire_train_dataset.seq_nn_idx = seq_nn_struct_pad.to(self.module_device)

        flag_first_H_idx = True if kl_indices[0] < self.H else False
        if not flag_first_H_idx:
            # encoder output, mask first to save computation
            mean_q, std_q = self.encoder(y_batch[m_batch])             # [<v*J,L]
            cov_q = std_q.square()

            # get neighbors
            seq_nn_idx = seq_nn_struct_pad[vid_ids.unsqueeze(-1), kl_indices - self.H, :]  # [v,J,H]
            y_nn, t_nn = self.train_dataset.gather(seq_nn_idx)[:2]                         # [v,J,H,D], [v,J,H,1]
            y_nn, t_nn = y_nn.to(self.module_device), t_nn.to(self.module_device)
            t_nn_pick = t_nn[m_batch]                                  # [<v*J, H, 1]

            # latent representations of nearest neighbors
            mean_q_nn, std_q_nn = self.encoder(y_nn[m_batch])          # [<v*J,H,L]
            mean_q_nn, std_q_nn = mean_q_nn.mT, std_q_nn.mT            # [<v*J,L,H]

            # preparation
            mean_prior_nn, K_prior_nn = self.gp.prior(t_nn_pick, are_neighbors=False)     # [<v*J,L/1,H,H], no transpose
            k_nj = self.gp.kernel(
                t_nn_pick.unsqueeze(-3), t_batch_pick.unsqueeze(-2).unsqueeze(-2)         # [<v*J,L/1,H,1]
            ).to_dense()

            L = psd_safe_cholesky(
                K_prior_nn + self.jitter * torch.eye(self.H, device=self.module_device)   # [<v*J,L/1,H,H]
            )
            # better than `torch.cholesky_solve`
            L_nn_inv_k_nj = torch.linalg.solve_triangular(L, k_nj, upper=False)
            K_nn_inv_k_nj = torch.linalg.solve_triangular(L.mT, L_nn_inv_k_nj, upper=True)  # [<v*J,L/1,H,1]
            k_jj = self.gp.kernel(t_batch_pick.unsqueeze(-3), diag=True).mT                 # [<v*J,L/1]
            Sigma_p = k_jj - (K_nn_inv_k_nj * k_nj).sum(dim=(-1, -2))                       # [<v*J,L/1]
            assert (Sigma_p > 0.).all(), f"Sigma_p min: {Sigma_p.min()} is less than 0."

            mahalanobis_1_sqrt = std_q_nn * K_nn_inv_k_nj.squeeze(-1)                       # [<v*J,L,H]
            mahalanobis_1 = mahalanobis_1_sqrt.square().sum(dim=-1)                         # [<v*J,L]
            mahalanobis_2 = (K_nn_inv_k_nj.squeeze(-1) * (mean_q_nn - mean_prior_nn)).sum(dim=-1)
            mahalanobis_2 = mahalanobis_2 + self.gp.mean(t_batch_pick.unsqueeze(-3)).mT - mean_q
            mahalanobis = (mahalanobis_1 + mahalanobis_2.square()) / Sigma_p                # [<v*J, L]

            trace = cov_q / Sigma_p
            log_det = 0.5 * Sigma_p.log() - std_q.log()

            kl = torch.sum(0.5 * (mahalanobis + trace - 1.) + log_det, dim=-1)              # [<v*J]
            kl_mat = torch.zeros_like(m_batch, requires_grad=True, dtype=t_batch.dtype)     # [v,J]
            kl_mat = kl_mat.clone()  # in-place operation not allowed
            kl_mat[m_batch] = kl     # assign

            len_batch_inv = torch.where(m_batch.sum(dim=-1) > 0, 1 / m_batch.sum(-1), 0.)          # [v]
            scale = self.train_dataset.masks.to(m_batch.device).sum(dim=(-1, -2)) * len_batch_inv  # [v]
            kl = torch.sum(kl_mat * scale.unsqueeze(-1))

            return kl / self.train_dataset.series_shape.numel()

        else:
            assert torch.all(m_batch), f"The first H-{self.H} KL indices must be all observed."
            # encoder output
            mean_q, std_q = self.encoder(y_batch)  # [v,J,L]
            mean_q, std_q = mean_q.mT, std_q.mT    # [v,L,J]
            cov_q = std_q.square()

            mean_p, cov_p = self.gp.prior(t_batch, are_neighbors=False)       # [v,L,H=J], [v,L/1,H,H]
            L = psd_safe_cholesky(cov_p + self.jitter * torch.eye(self.H, device=self.module_device))

            mahalanobis = torch.linalg.solve_triangular(L, (mean_p - mean_q).unsqueeze(-1), upper=False)
            mahalanobis = mahalanobis.square().sum(dim=(-1, -2))              # [v,L]

            L_inv = torch.linalg.solve_triangular(L, torch.eye(self.H, device=self.module_device), upper=False)
            trace = (L_inv * std_q.unsqueeze(-2)).square().sum(dim=(-1, -2))  # [v,L]

            log_det_cov_p = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)    # [v,L/1]
            log_det_cov_q = std_q.log().sum(dim=-1)
            log_det = log_det_cov_p - log_det_cov_q

            kl = torch.sum(0.5 * (mahalanobis + trace - self.H) + log_det, dim=-1)  # [v]
            scale = self.train_dataset.masks.to(kl.device).sum(dim=(-1, -2)) / self.H        # [v]
            kl = torch.sum(kl * scale)

            return kl / self.train_dataset.series_shape.numel()

    def average_loss_vnn(
            self, vid_batch_short: Tensor, t_mask_short: Tensor, current_kl_indices: LongTensor,
            vid_ids: LongTensor = None, beta=1.
    ):
        lik = self.expected_log_prob(vid_batch_short, t_mask_short)
        kl = self.kl_divergence_vnn(current_kl_indices, vid_ids=vid_ids)
        assert torch.isfinite(lik).all(), "Likelihood now is not finite."
        assert torch.isfinite(kl).all(), "KL divergence now is not finite."
        elbo = lik - beta * kl
        return - elbo

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs: int,
            series_batch_size: int, timestamp_kl_batch_size: int, timestamp_expected_lk_batch_size: int = None,
            max_norm: float = None, device="cpu", print_epochs=1
    ):
        if device != self.module_device:
            warnings.warn(f"Training is not on self.module_device ({self.module_device}) set earlier. "
                          f"We have changed module device to {device} and train the model on it.")
            self.module_device = device
        self.to(device)
        # move NN structure onto GPU only once
        self.entire_train_dataset.seq_nn_idx = self.entire_train_dataset.seq_nn_idx.to(self.module_device)

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
                self._set_training_iterator(timestamp_kl_batch_size, N=len(self.train_dataset))  # get shorter length

                if timestamp_expected_lk_batch_size is None:
                    for _ in range(self._total_training_batches):
                        current_kl_indices = self._get_training_indices(timestamp_kl_batch_size)
                        sq_short, _, m_short = self.train_dataset[current_kl_indices]
                        sq_short, m_short = sq_short.to(device), m_short.to(device)

                        optimizer.zero_grad(set_to_none=True)
                        loss = self.average_loss_vnn(sq_short, m_short, current_kl_indices, vid_ids, beta=beta)
                        loss.backward()
                        if max_norm is not None:
                            grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm, error_if_nonfinite=True)
                            if grad_norm > max_norm:
                                print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                        optimizer.step()
                else:
                    dataloader_lk = DataLoader(
                        self.train_dataset, batch_size=timestamp_expected_lk_batch_size, shuffle=True
                    )
                    for sq_short, _, m_short in dataloader_lk:
                        sq_short, m_short = sq_short.to(device).transpose(0, 1), m_short.to(device).transpose(0, 1)
                        current_kl_indices = self._get_training_indices(timestamp_kl_batch_size)

                        optimizer.zero_grad(set_to_none=True)
                        loss = self.average_loss_vnn(sq_short, m_short, current_kl_indices, vid_ids, beta=beta)
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
                      f'global grad norm: {grad_norm}\n' if max_norm is not None else '\n')


