import torch
from torch import Tensor, LongTensor
from torch import nn
from linear_operator.utils.cholesky import psd_safe_cholesky

from models.building_blocks.gp import GP


class GPVAEBase(nn.Module):
    def __init__(
            self, encoder: nn.Module, decoder: nn.Module, gp: GP,
            H: int, search_device: str = 'cpu', data_device: str = 'cpu', module_device: str = 'cpu',
            jitter: float = 0., geco=False
    ):
        super(GPVAEBase, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.gp = gp

        self.H = H
        self.latent_dims = self.gp.output_dims
        self.search_device, self.data_device, self.module_device = search_device, data_device, module_device
        self.jitter, self.geco = jitter, geco

        self.train_dataset = None           # defined in subclass, some exps generate data during training

    def get_nn_data(self, test_x: Tensor, H: int, **kwargs):
        """
        find H-nearest training data for test_x with shape [..., n, D],
        return (Y_near, X_near, (mask_near, Y_full_near)) with shape [...,n,H,...]
        """
        nn_indices = self.train_dataset.nn_util.find_nn_idx(test_x, k=H)  # [..., n, H]
        return self.train_dataset.gather(nn_indices)

    # VNNGP utility methods for kl indices
    def _set_training_iterator(self, batch_size_kl: int, N: int) -> None:
        """set the KL indices for training, the first batch has H data. N is the number of all timestamps"""
        self._training_ordinal = 0
        if self.H == N:
            self._training_idx_iterator = (torch.arange(self.H),)
        elif self.H < N:
            training_idx = torch.randperm(N - self.H) + self.H
            self._training_idx_iterator = (torch.arange(self.H),) + training_idx.split(batch_size_kl)
        else:
            raise ValueError(f'Got H ({self.H}) > N ({N}).')
        self._total_training_batches = len(self._training_idx_iterator)

    def _get_training_indices(self, batch_size_kl: int) -> LongTensor:
        self.current_training_indices = self._training_idx_iterator[self._training_ordinal]

        if self._training_ordinal != self._total_training_batches - 1:
            assert len(self.current_training_indices) == batch_size_kl or len(self.current_training_indices) == self.H
            self._training_ordinal += 1
        else:
            assert len(self.current_training_indices) <= batch_size_kl
            # shuffle again, now train_dataset is not None
            self._set_training_iterator(batch_size_kl, N=len(self.train_dataset))
        return self.current_training_indices

    # Computational graph
    def forward(self, y_batch: Tensor):
        means, stds = self.encoder(y_batch)
        eps = torch.randn_like(means)
        latent_samples = means + stds * eps
        recon_imgs = self.decoder(latent_samples)
        return means, stds, recon_imgs

    def expected_log_prob(self, y_batch: Tensor, **kwargs) -> Tensor:  # only for non-missing data
        means, stds, y_rec = self.forward(y_batch)

        out_dist = self.decoder.output_distribution
        if out_dist == 'normal':
            sigma2_y = self.decoder.sigma2_y
            expected_lk = (y_batch - y_rec).square() / sigma2_y + torch.log(2 * torch.pi * sigma2_y)
            expected_lk = - 1 / 2 * expected_lk.sum()
        elif out_dist == 'bernoulli':
            expected_lk = - nn.functional.binary_cross_entropy(input=y_rec, target=y_batch, reduction='sum')
        else:
            raise NotImplementedError(f'Unrecognized output distribution {out_dist}.')

        scale = (y_batch.shape[:len(self.train_dataset.series_shape)+1]).numel()
        scale = len(self.train_dataset) / scale    # N/B, and averaged over series dim
        return scale * expected_lk

    def kl_divergence_sws(self, x_batch: Tensor, **kwargs):                        # [(v),n,D]
        y_nn, x_nn = self.get_nn_data(x_batch, H=self.H, **kwargs)[:2]             # [v,n,H,32,32], [(v),n,H,D]
        y_nn, x_nn = y_nn.to(self.module_device), x_nn.to(self.module_device)
        v, n = self.train_dataset.series_shape.numel(), x_batch.size(-2)

        mean_q, std_q = self.encoder(y_nn)                               # [v,n,H,L]
        mean_q = mean_q.permute(*range(mean_q.ndim - 3), -1, -3, -2)     # [v,L,n,H]
        std_q = std_q.permute(*range(std_q.ndim - 3), -1, -3, -2)
        mean_p, cov_p = self.gp.prior(x_nn, are_neighbors=True)          # [(v),L,n,H], [(v),L/1,n,H,H]

        L = psd_safe_cholesky(cov_p + self.jitter * torch.eye(self.H, device=self.module_device))  # [(v),L/1,n,H,H]
        mean_diff = (mean_q - mean_p).unsqueeze(-1)                      # [v,L,n,H,1]
        mahalanobis = torch.linalg.solve_triangular(L, mean_diff, upper=False)
        mahalanobis = mahalanobis.square().sum(dim=(-1, -2))             # [v,L,n]

        L_inv = torch.linalg.solve_triangular(L, torch.eye(L.size(-1), device=self.module_device), upper=False)
        tmp = L_inv * std_q.unsqueeze(-2)                                # [v,L,n,H,H]
        trace = tmp.square().sum(dim=(-1, -2))                           # [v,L,n]

        log_det_cov_p = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)   # [(v),L/1,n]
        log_det_cov_q = std_q.log().sum(dim=-1)                          # [(v),L,n]
        log_det = log_det_cov_p - log_det_cov_q                          # broadcast

        res = 0.5 * (mahalanobis + trace - self.H) + log_det
        res = res.sum() / (v * n)  # averaged over v * n
        return res

    def kl_divergence_vnn(self, kl_indices: LongTensor, sequential_nn_structure: LongTensor, **kwargs):
        """kl_indices: indices of current batch data, [J]; sequential_nn_structure: [(v), N-H, H] """
        vid_batch, t_batch = self.train_dataset[kl_indices][:2]   # use the same kl indices for series in one batch
        vid_batch, t_batch = vid_batch.to(self.module_device), t_batch.to(self.module_device)   # [v,J,32,32], [(v),J,D]
        sequential_nn_structure = sequential_nn_structure.to(self.module_device)

        # encoder output
        mean_q, std_q = self.encoder(vid_batch)                                   # [v,J,L]
        mean_q, std_q = mean_q.mT, std_q.mT                                       # [v,L,J]
        cov_q = std_q.square()
        v, J = self.train_dataset.series_shape.numel(), t_batch.size(-2)

        flag_first_H_idx = True if kl_indices[0] < self.H else False
        if not flag_first_H_idx:
            t_batch = t_batch.unsqueeze(-3)                                       # [(v),1,J,D], allow for kernel batch
            # get neighbors
            seq_nn_idx = sequential_nn_structure[..., kl_indices - self.H, :]     # [(v),J,H]
            vid_nn, t_nn = self.train_dataset.gather(seq_nn_idx)[:2]              # [v,J,H,32,32], [(v),J,H,D]
            vid_nn, t_nn = vid_nn.to(self.module_device), t_nn.to(self.module_device)

            # latent representations of nearest neighbors
            mean_q_nn, std_q_nn = self.encoder(vid_nn)                             # [v,J,H,L]
            mean_q_nn = mean_q_nn.permute(*range(mean_q_nn.ndim - 3), -1, -3, -2)  # [v,L,J,H]
            std_q_nn = std_q_nn.permute(*range(std_q_nn.ndim - 3), -1, -3, -2)     # [v,L,J,H]

            # preparation
            mean_prior_nn, K_prior_nn = self.gp.prior(t_nn, are_neighbors=True)    # [(v),L/1,J,H,H]
            k_nj = self.gp.kernel(t_nn.unsqueeze(-3), t_batch.transpose(-2, -3).unsqueeze(-2))   # [(v),J,L/1,H,1]
            k_nj = k_nj.to_dense().transpose(-3, -4)                                             # [(v),L/1,J,H,1]

            L = psd_safe_cholesky(
                K_prior_nn + self.jitter * torch.eye(self.H, device=self.module_device)          # [(v),L/1,J,H,H]
            )
            # better than K_nn_inv_k_nj = torch.cholesky_solve(k_nj, L, upper=False)
            K_nn_inv_k_nj = torch.linalg.solve_triangular(L, k_nj, upper=False)                  # [(v),L/1,J,H,1]
            K_nn_inv_k_nj = torch.linalg.solve_triangular(L.mT, K_nn_inv_k_nj, upper=True)       # [(v),L/1,J,H,1]
            k_jj = self.gp.kernel(t_batch, diag=True)                              # [(v),L/1,J]
            Sigma_p = k_jj - (K_nn_inv_k_nj * k_nj).sum(dim=(-1, -2))

            mahalanobis_1_sqrt = std_q_nn * K_nn_inv_k_nj.squeeze(-1)              # [v,L,J,H]
            mahalanobis_1 = mahalanobis_1_sqrt.square().sum(dim=-1)                # [v,L,J]
            mahalanobis_2 = (K_nn_inv_k_nj.squeeze(-1) * (mean_q_nn - mean_prior_nn)).sum(dim=-1)
            mahalanobis_2 = mahalanobis_2 + self.gp.mean(t_batch) - mean_q
            mahalanobis_2 = mahalanobis_2.square()                                 # [v,L,J]
            mahalanobis = (mahalanobis_1 + mahalanobis_2) / Sigma_p

            trace = cov_q / Sigma_p
            log_det = 0.5 * Sigma_p.log() - std_q.log()

            res = 0.5 * (mahalanobis + trace - 1.) + log_det

            return len(self.train_dataset) * res.sum() / (v * J)

        else:
            mean_p, cov_p = self.gp.prior(t_batch, are_neighbors=False)            # [(v),L,H=J], [(v),L/1,H,H]
            L = psd_safe_cholesky(cov_p + self.jitter * torch.eye(self.H, device=self.module_device))

            mahalanobis = torch.linalg.solve_triangular(L, (mean_p - mean_q).unsqueeze(-1), upper=False)
            mahalanobis = mahalanobis.square().sum(dim=(-1, -2))                   # [(v),L]

            L_inv = torch.linalg.solve_triangular(L, torch.eye(self.H, device=self.module_device), upper=False)
            trace = (L_inv * std_q.unsqueeze(-2)).square().sum(dim=(-1, -2))       # [(v),L]

            log_det_cov_p = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)         # [(v),L/1]
            log_det_cov_q = std_q.log().sum(dim=-1)
            log_det = log_det_cov_p - log_det_cov_q

            res = 0.5 * (mahalanobis + trace - J) + log_det

            return len(self.train_dataset) * res.sum() / (v * J)

    @staticmethod
    def GECO_loss(
            recon_error: Tensor, kl: Tensor, Lagrange_lamb=1., C_ma=None,
            lr_C_ma=1., alpha=0.99, kappa=0.1
    ):
        """
        GECO algorithm from Taming VAE(2018), introducing Lagrange multiplier to balance ELBO terms;
        The objective is: min{ lamda * (recon_error - kappa ** 2) + KL }
        """
        constraint = recon_error - kappa ** 2
        geco_loss = Lagrange_lamb * constraint + kl
        with torch.no_grad():
            C_ma = constraint if C_ma is None else C_ma
            C_ma = alpha * C_ma + (1 - alpha) * constraint
            Lagrange_lamb = Lagrange_lamb * torch.exp(lr_C_ma * C_ma)
        return geco_loss, Lagrange_lamb, C_ma

