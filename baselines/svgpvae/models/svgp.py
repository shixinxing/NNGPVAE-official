# Adapted from https://github.com/ratschlab/SVGP-VAE/blob/main/SVGPVAE_model.py#L174

import torch
from torch import Tensor

from gpytorch.kernels import Kernel
from gpytorch.means import ZeroMean
from linear_operator.utils.cholesky import psd_safe_cholesky

from models.building_blocks.gp import GP


class SVGP(GP):
    """SVGP component used in AISTATS 2021"""
    def __init__(
            self, output_dims: int, kernel: Kernel,
            inducing_points: torch.nn.Parameter, N_train: int = None, jitter=1e-6
    ):
        """If N_train is None: missing-frame case, N_train depends on mini-series."""
        super(SVGP, self).__init__(output_dims, kernel, mean=ZeroMean(batch_shape=torch.Size([output_dims])))
        # The official model doesn't share for moving ball (it sets two completely independent SVGPs)
        # but it does share for MNIST
        self.inducing_points = inducing_points  # [(L), M, D]
        self.N_train = N_train                  # N or None
        self.jitter = jitter

    def _inv_K(self, K: Tensor):
        L = psd_safe_cholesky(K + self.jitter * torch.eye(K.shape[-1], device=K.device))
        K_inv = torch.cholesky_solve(torch.eye(K.shape[-1], device=K.device), L, upper=False)
        return K_inv

    def approx_posterior_params(
            self, X_train_batch, qnet_mus, qnet_vars, x_test, diag_cov_qs=True,
            m_train_batch: Tensor = None, N_train=None
    ):
        """
        Return variational distribution q(Z_U)=N(mu_b, A_b) (i.e., Eq(9)): [(v),L,M]
        and q_s(Z)=int p(Z|Z_U)q(Z_U) dZ_U: [(v),L,n] ;
        X_train_batch: [(v), b, D];
        m_train_batch: [(v), b, 1], 0 indicating missing frame;
        qnet_mus, qnet_var: tilde latent dataset from inference network [(v), b, L],
        x_test: [(v), n, D]
        N_train=None or [(v)] total num of non-missing frames per vid

        We refer to https://github.com/ratschlab/SVGP-VAE/blob/main/SVGPVAE_model.py#L303 even though there might be
        a more efficient way to do the algebra here.
        """
        if N_train is None:
            assert self.N_train is not None, "self.N_train can't be None if N_train is not provided."
            N_train = self.N_train
        else:
            N_train = N_train.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # [(v),1,1,1]
        if m_train_batch is None:
            batch_size = X_train_batch.size(-2)
            scale = N_train / batch_size
        else:
            batch_size = m_train_batch.sum(dim=(-1, -2))
            batch_size = batch_size.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            scale = torch.where(batch_size > 0, N_train / batch_size, 0.)
        X_b = X_train_batch.unsqueeze(-3)                                   # [(v),1,b,D]

        Kuu = self.kernel(self.inducing_points).to_dense()                  # [L,M,M]
        Kbu = self.kernel(X_b, self.inducing_points).to_dense()             # [(v),L,b,M]
        tilde_noise_inv = 1 / qnet_vars.mT.unsqueeze(-1)                    # [(v),L,b,1]
        if m_train_batch is not None:
            # tilde_noise_inv = tilde_noise_inv * m_train_batch.unsqueeze(-3)
            tilde_noise_inv = torch.where(m_train_batch.unsqueeze(-3), tilde_noise_inv, 0.)
        tilde_Y = qnet_mus.mT.unsqueeze(-1)

        # q(Z_U)
        Sigma = Kuu + scale * (Kbu.mT @ (tilde_noise_inv * Kbu))   # [(v),L,M,M]
        L_Sigma = psd_safe_cholesky(
            Sigma + self.jitter * torch.eye(Sigma.size(-1), dtype=torch.get_default_dtype(), device=Sigma.device)
        )

        mu_b = Kuu @ torch.cholesky_solve(Kbu.mT @ (tilde_noise_inv * tilde_Y), L_Sigma)   # vector multiplication first
        mu_b = (scale * mu_b).squeeze(-1)                          # [(v),L,M]
        A_b_sqrt = torch.linalg.solve_triangular(L_Sigma, Kuu, upper=False)
        A_b = A_b_sqrt.mT @ A_b_sqrt

        # q_s(Z)
        x_n = x_test.unsqueeze(-3)                                          # [(v),1,n,D]
        Kxu = self.kernel(x_n, self.inducing_points).to_dense()             # [(v),L,n,M]
        mu_qs = Kxu @ torch.cholesky_solve(Kbu.mT @ (tilde_noise_inv * tilde_Y), L_Sigma)  # similarly
        mu_qs = (scale * mu_qs).squeeze(-1)             # [(v),L,n]

        Kxx = self.kernel(x_n, diag=diag_cov_qs).to_dense()
        L_uu = psd_safe_cholesky(
            Kuu + self.jitter * torch.eye(Sigma.size(-1), dtype=torch.get_default_dtype(), device=Kuu.device)
        )
        Kxu_Sigma_inv = torch.cholesky_solve(Kxu.mT, L_Sigma).mT
        Kxu_Kuu_inv = torch.cholesky_solve(Kxu.mT, L_uu).mT
        if diag_cov_qs:
            cov_qs = Kxx + torch.sum((Kxu_Sigma_inv - Kxu_Kuu_inv) * Kxu, dim=-1)  # [(v),L,n]
        else:
            cov_qs = Kxx + (Kxu_Sigma_inv - Kxu_Kuu_inv) @ Kxu.mT                  # [(v),L,n,n]

        return mu_qs, cov_qs, mu_b, A_b

    def variational_loss(self, X_train_batch, qnet_mus, qnet_vars, mu, A, m_train_batch=None):
        """
        Compute L_H = Sum of log Normal - KL for the current batch `X_train_batch`.

        (X_train_batch: [(v), b, D]; qnet_mus, qnet_var: [(v), b, L]) produced by inference network using mini-batch
        (mu, A): [(v), L, M] from q(Z_U)=N(mu, A), the variational distribution over inducing Z_U;

        We refer to https://github.com/ratschlab/SVGP-VAE/blob/main/SVGPVAE_model.py#L220 even
        though there might be a more efficient way to do the algebra here.
        """
        if m_train_batch is not None:
            assert m_train_batch.ndim > 2
            m_train_batch = m_train_batch[..., 0]
        X_b = X_train_batch.unsqueeze(-3)                   # [(v),1,b,D]

        Kuu = self.kernel(self.inducing_points).to_dense()  # [L,M,M]
        L_uu = psd_safe_cholesky(Kuu + self.jitter * torch.eye(Kuu.shape[-1], device=Kuu.device))
        Kbu = self.kernel(X_b, self.inducing_points).to_dense()
        bb_x_b = torch.cholesky_solve(Kbu.mT, L_uu).mT      # [(v),L,b,M]

        # term 1/3 - log Normal
        mean = bb_x_b @ mu.unsqueeze(-1)                    # [(v),L,b,1]
        log_Normal = - 0.5 * (qnet_mus.mT - mean.squeeze(-1)).square() / qnet_vars.mT  # [(v),L,b]
        log_Normal = log_Normal - 0.5 * torch.log(2 * torch.tensor(torch.pi, device=log_Normal.device))
        log_Normal = log_Normal - 0.5 * torch.log(qnet_vars.mT)
        if m_train_batch is not None:
            # log_Normal = log_Normal * m_train_batch.unsqueeze(-2)
            log_Normal = torch.where(m_train_batch.unsqueeze(-2), log_Normal, 0.)

        # term 2/3 - trace
        K_tilde = self.kernel(X_b, diag=True) - torch.sum(bb_x_b * Kbu, dim=-1)         # [(v),L,b]
        Lamda = bb_x_b.unsqueeze(-1) @ bb_x_b.unsqueeze(-1).mT                          # [(v),L,b,M,M]
        trace = torch.diagonal(A.unsqueeze(-3) @ Lamda, dim1=-1, dim2=-2).sum(-1)       # [(v),L,b]
        trace = - 0.5 * (K_tilde + trace) / qnet_vars.mT
        if m_train_batch is not None:
            # trace = trace * m_train_batch.unsqueeze(-2)
            trace = torch.where(m_train_batch.unsqueeze(-2), trace, 0.)

        # term 3/3 - KL
        KL_mahalanobis = mu * torch.cholesky_solve(mu.unsqueeze(-1), L_uu).squeeze(-1)  # [(v),L,M]
        KL_mahalanobis = torch.sum(KL_mahalanobis, dim=-1)
        L_Ab = psd_safe_cholesky(A + self.jitter * torch.eye(A.shape[-1], device=A.device))  # [(v),L,M,M]
        KL_trace_term = torch.linalg.solve_triangular(L_uu, L_Ab, upper=False)
        KL_trace_term = KL_trace_term.square().sum(dim=(-1, -2))                        # [(v),L]
        KL_log_det = L_uu.diagonal(dim1=-1, dim2=-2).log().sum(-1) - L_Ab.diagonal(dim1=-1, dim2=-2).log().sum(-1)
        KL = 0.5 * (KL_mahalanobis - self.inducing_points.size(-2) + KL_trace_term) + KL_log_det  # [(v),L]

        return (log_Normal + trace).sum(dim=(-1, -2)), KL.sum(-1)  # [(v)], [(v)]



