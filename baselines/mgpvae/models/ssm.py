import torch
from torch import Tensor
from torch import nn
from torch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean
from linear_operator.utils.cholesky import psd_safe_cholesky

from models.building_blocks.gp import GP
from baselines.mgpvae.models.building_blocks.kernels import Matern32SSM


class SSM(nn.Module):
    def __init__(
            self, output_dims: int, kernel: Matern32SSM, jitter: float = 0,
    ):
        super(SSM, self).__init__()
        self.gp = GP(output_dims, kernel, mean=ZeroMean(batch_shape=kernel.batch_shape))  # let it check the args
        self.kernel = kernel  # self.gp.kernel

        self.output_dims = output_dims
        self.jitter = jitter

    @staticmethod
    def process_noise_covar(A, P0):
        Q = P0 - A @ P0 @ A.mT
        return Q

    def _filter(self, m_f, P_f, A, Q, H, pseudo_y, pseudo_var):  # [...,2,2/1]
        # prediction N(m_p, P_p)
        m_p = A @ m_f
        P_p = A @ P_f @ A.mT + Q

        # likelihood p(y_i|y_{1:i-1})=N(hat_y, S)
        hat_y = H @ m_p
        HP = H @ P_p
        S = HP @ H.mT + pseudo_var
        dist = MultivariateNormal(hat_y.squeeze(-1), covariance_matrix=S)  # [v,L,1]
        assert (S.diagonal(dim1=-1, dim2=-2) > 0.).all(), f"S min: {S.diagonal(dim1=-1, dim2=-2).min()}"
        log_p = dist.log_prob(pseudo_y.squeeze(-1))                        # [v,L]

        # filtering N(m_f,P_f)
        chol_S = psd_safe_cholesky(S + self.jitter * torch.eye(S.shape[-1], device=S.device, dtype=S.dtype))
        W = torch.cholesky_solve(HP, chol_S).mT
        m_f = m_p + W @ (pseudo_y - hat_y)
        P_f = P_p - W @ HP

        return m_f, P_f, log_p

    def filter(self, dt: Tensor, pseudo_y: Tensor, pseudo_var: Tensor, mask: Tensor = None):
        """
        Kalman filter. refer to https://github.com/harrisonzhu508/MGPVAE/blob/main/mgpvae/ops.py#L313
        dt: [v,T,1/L,1,1], starting with `0`, mask: [v,T,1,1,1]
        pseudo_y/var: padded, [v,T,L]
        return filtering mean/cov over s: [v,L,T,2,1/2]
        """
        T, L = pseudo_y.size(-2), pseudo_y.size(-1)

        if mask is not None:
            assert mask.dtype == torch.bool
            Tmax = mask.sum(dim=(-1, -2, -3, -4)).max()
        else:
            Tmax = T

        pseudo_y = pseudo_y.transpose(-1, -2)                # [v,L,T]
        pseudo_var = pseudo_var.transpose(-1, -2)

        _, _, H, P_0 = self.kernel.kernel_to_state_space()   # [L,2,2]
        As = self.kernel.state_transition(dt)                # [v,T,L,2,2]
        Qs = self.process_noise_covar(As, P_0)               # [v,T,L,2,2]
        As, Qs = As.transpose(-3, -4), Qs.transpose(-3, -4)  # [v,L,T,2,2]
        m_0 = torch.zeros([*P_0.shape[:-1], 1], device=P_0.device, dtype=P_0.dtype)  # [L,2,1]

        pseudo_y = pseudo_y.unsqueeze(-1).unsqueeze(-1)      # [v,L,T,1,1]
        pseudo_var = pseudo_var.unsqueeze(-1).unsqueeze(-1)

        m_f, P_f = m_0, P_0                                  # [L,2,2/1]
        m_fs, P_fs, sum_log_p = [], [], 0.                   # record for smoothing later
        for t in range(T):  # along T
            A = As[..., t, :, :]                             # [v,L,2,2]
            Q = Qs[..., t, :, :]
            pse_y = pseudo_y[..., t, :, :]
            pse_var = pseudo_var[..., t, :, :]
            if mask is None:
                m_f, P_f, log_p = self._filter(m_f, P_f, A, Q, H, pse_y, pse_var)     # [v,L,2,1/2], [v,L]
            elif t < Tmax:
                _m_f, _P_f, _log_p = self._filter(m_f, P_f, A, Q, H, pse_y, pse_var)  # [v,L,2,1/2], [v,L]
                # copy the filtering and log_p=0 at unobserved time
                mask_at_t = mask[..., t, :, :, :]            # [v,1,1,1]
                m_f, P_f = torch.where(mask_at_t, _m_f, m_f), torch.where(mask_at_t, _P_f, P_f)
                log_p = torch.where(mask_at_t.squeeze(-1).squeeze(-1), _log_p, 0.)
            else:
                m_f, P_f, log_p = m_f, P_f, torch.zeros([*m_f.shape[:-3], L], device=m_f.device, dtype=m_f.dtype)
            m_fs.append(m_f)
            P_fs.append(P_f)
            # print(f"log_p: {log_p.sum()}")
            sum_log_p = sum_log_p + log_p
        m_fs = torch.stack(m_fs, dim=-3)                     # [v,L,T,2,1]
        P_fs = torch.stack(P_fs, dim=-3)                     # [v,L,T,2,2]
        assert m_fs.size(-4) == L
        return m_fs, P_fs, sum_log_p

    def _smoother(self, m_f, P_f, A, Q, m_s, P_s):           # at t:(m_f, P_f), at t+1:(m_s, P_s)
        # prediction at t+1: (m^p_{t+1}, P^p_{t+1})
        m_p = A @ m_f
        A_Pf = A @ P_f
        P_p = A_Pf @ A.mT + Q

        # smoothing at t: (m^s_{t}, P^s_{t+1}) ~ N(m_s, P_s)
        chol_G = psd_safe_cholesky(P_p + self.jitter * torch.eye(P_p.shape[-1], device=P_p.device, dtype=P_p.dtype))
        G = torch.cholesky_solve(A_Pf, chol_G).mT
        m_s = m_f + G @ (m_s - m_p)
        m_P = P_f - G @ (P_s - P_p) @ G.mT
        return m_s, m_P, G

    def smoother(self, dt: Tensor, filter_mean: Tensor, filter_cov: Tensor, mask: Tensor = None):
        """
        RTS smoother. refer to https://github.com/harrisonzhu508/MGPVAE/blob/main/mgpvae/ops.py#L351
        `dt`: [v,T,1/L,1,1], ending with `0`, `filter_mean/cov`: [v,L,T,2,1/2], same filtering at unobserved frames
        return smoother mean/cov over Z: [v,L,T]
        """
        if mask is not None:
            assert mask.dtype == torch.bool
            Tmax = mask.sum(dim=(-1, -2, -3, -4)).max()
        else:
            Tmax = filter_mean.size(-3)

        T = filter_mean.size(-3)
        _, _, H, P_0 = self.kernel.kernel_to_state_space()   # [L,1/2,2]
        As = self.kernel.state_transition(dt)                # [v,T,L,2,2]
        Qs = self.process_noise_covar(As, P_0)               # [v,T,L,2,2]
        As, Qs = As.transpose(-3, -4), Qs.transpose(-3, -4)  # [v,L,T,2,2]

        m_s, P_s, G = filter_mean[..., -1, :, :], filter_cov[..., -1, :, :], 0.
        m_ss_S, P_ss_S, Gs = [], [], []
        m_ss_Z, P_ss_Z = [], []       # over Z
        for t in range(T-1, -1, -1):  # backward
            m_f, P_f = filter_mean[..., t, :, :], filter_cov[..., t, :, :]
            A = As[..., t, :, :]  # [v,L,2,2]
            Q = Qs[..., t, :, :]
            if mask is None:
                m_s, P_s, G = self._smoother(m_f, P_f, A, Q, m_s, P_s)     # [v,L,2,1/2]
            elif t >= Tmax:
                m_s, P_s, G = m_f, P_f, torch.zeros_like(P_s)
            else:
                _m_s, _P_s, _G = self._smoother(m_f, P_f, A, Q, m_s, P_s)  # [v,L,2,1/2]
                mask_at_t = mask[..., t, :, :, :]   # [v,1,1,1]
                m_s, P_s = torch.where(mask_at_t, _m_s, m_s), torch.where(mask_at_t, _P_s, P_s)
                G = torch.where(mask_at_t, _G, G)
            m_ss_S.append(m_s)
            P_ss_S.append(P_s)
            Gs.append(G)

            m_s_Z = H @ m_s                                  # [v,L,1,1]
            P_s_Z = H @ P_s @ H.mT
            m_ss_Z.append(m_s_Z.squeeze(-1).squeeze(-1))     # [v,L]
            P_ss_Z.append(P_s_Z.squeeze(-1).squeeze(-1))
        for item in (m_ss_S, P_ss_S, Gs, m_ss_Z, P_ss_Z):
            item.reverse()
        m_ss_S = torch.stack(m_ss_S, dim=-3)                 # [v,L,T,2,1/2]
        P_ss_S = torch.stack(P_ss_S, dim=-3)
        Gs = torch.stack(Gs, dim=-3)
        m_ss_Z = torch.stack(m_ss_Z, dim=-1)                 # [v,L,T]
        P_ss_Z = torch.stack(P_ss_Z, dim=-1)

        return m_ss_Z, P_ss_Z, m_ss_S, P_ss_S, Gs

    def update_posterior(
            self, dt_nonzero, pseudo_y, pseudo_var, m_batch=None, return_state=False
    ):
        """
        Forward filtering and backward smoothing.
        `dt_nonzero`: [v,T-1,1/L,1,1] not beginning with `0`, all zeros at unobserved time;
        `pseudo_Y/var   : [v,T,L], all zeros at unobserved time;
        `m_batch`: [v,T-1,1/L,1,1]. pseudo_Y/var will be padded according to the mask.
        return filtering mean/cov over `s`: [v,L,T,2,1/2], smoothing mean/cov over `Z`: [v,L,T]
        """
        zeros = torch.zeros(
            [*dt_nonzero.shape[:-4], 1, *dt_nonzero.shape[-3:]], device=dt_nonzero.device, dtype=dt_nonzero.dtype
        )
        ones = torch.ones_like(zeros, dtype=torch.bool)

        dt = torch.cat([zeros, dt_nonzero], dim=-4)  # [v,T,1,1,1]
        if m_batch is not None:
            assert m_batch.dtype == torch.bool
            m_batch = torch.cat([ones, m_batch], dim=-4)

        m_fs, P_fs, sum_log_p = self.filter(dt, pseudo_y, pseudo_var, m_batch)
        sum_log_p = torch.sum(sum_log_p, dim=-1)     # [v]

        dt = torch.cat([dt_nonzero, zeros], dim=-4)
        m_ss_Z, P_ss_Z, m_ss_S, P_ss_S, Gs = self.smoother(dt, m_fs, P_fs, m_batch)
        if return_state:
            return m_fs, P_fs, m_ss_Z, P_ss_Z, m_ss_S, P_ss_S, Gs, sum_log_p  # used in prediction
        else:
            return m_fs, P_fs, m_ss_Z, P_ss_Z, sum_log_p

    @torch.no_grad()
    def predict(self, t_full, m_batch, pseudo_y, pseudo_var):
        """
        Prediction using adjacent timestamps and states.
        pseudo_y, pseudo_var: [v,T,L]; t_full/m_batch: [v,T,1]. mask indicates the boundary.
        In `t_full`, observations is on the left, missing timestamps on the right.
        return predicted mean/var at all locations: [v,L,Tmax]
        """
        # refer to https://github.com/harrisonzhu508/MGPVAE/blob/main/mgpvae/model.py#L460
        # find adjacent timestamps
        v, T = t_full.size(-3), t_full.size(-2)
        Tmax_test = (~m_batch).sum(dim=(-1, -2)).max()

        t_train = t_full.clone()                             # [v,T,1]
        t_train[~m_batch] = t_full.max()                     # assign
        dt_nonzero = torch.diff(t_train, dim=-2)             # [v,T-1,1]
        dt_nonzero = dt_nonzero.unsqueeze(-1).unsqueeze(-1)  # [v,T-1,1,1,1]
        m_batch = m_batch[..., 1:, :].unsqueeze(-1).unsqueeze(-1)
        _, _, _, _, m_ss_S, P_ss_S, Gs, _ = self.update_posterior(
            dt_nonzero, pseudo_y, pseudo_var, m_batch=m_batch, return_state=True
        )                                                    # [v,L,T,2,1/2]

        t_aug = torch.cat([
            torch.tensor([-1e10], device=t_train.device).expand(v, 1),
            t_train[..., 0],
            torch.tensor([1e10], device=t_train.device).expand(v, 1)
        ], dim=-1).unsqueeze(-1)                             # [v,T+2,1]

        t_test = t_full[:, -Tmax_test:]                      # [v,Tmax,1]
        nn_idx = torch.searchsorted(
            t_aug[..., 0].contiguous(), t_test[..., 0].contiguous(), right=False
        ) - 1    # [v, Tmax], contiguous to avoid warnings from PyTorch

        t_left = torch.gather(t_aug, dim=-2, index=nn_idx.unsqueeze(-1))             # [v,Tmax,1]
        t_right = torch.gather(t_aug, dim=-2, index=(nn_idx+1).unsqueeze(-1))

        dt_fwd = (t_test - t_left).unsqueeze(-1).unsqueeze(-1)                       # [v,Tmax,1,1,1]
        dt_back = (t_right - t_test).unsqueeze(-1).unsqueeze(-1)
        A_fwd = self.kernel.state_transition(dt_fwd)                                 # [v,Tmax,L,2,2]
        A_back = self.kernel.state_transition(dt_back)
        A_fwd = A_fwd.transpose(-3, -4)                                              # [v,L,Tmax,2,2]
        A_back = A_back.transpose(-3, -4)

        # refer to https://github.com/harrisonzhu508/MGPVAE/blob/main/mgpvae/util.py#L252
        # extract adjacent quantities
        m_ss_aug = torch.cat([
            torch.zeros([*m_ss_S.shape[:-3], 1, *m_ss_S.shape[-2:]], device=m_ss_S.device, dtype=m_ss_S.dtype),
            m_ss_S,
            torch.zeros([*m_ss_S.shape[:-3], 1, *m_ss_S.shape[-2:]], device=m_ss_S.device, dtype=m_ss_S.dtype)
        ], dim=-3)                                                                   # [v,L,T+2,2,1]

        _, _, H, P_0 = self.kernel.kernel_to_state_space()                           # [L,2,2]
        P_0 = P_0.unsqueeze(-3)                                                      # [L,1,2,2]
        H = H.unsqueeze(-3)
        P_ss_aug = torch.cat([
            P_0.expand(v, *P_0.shape), P_ss_S, P_0.expand(v, *P_0.shape)
        ], dim=-3)                                                                   # [v,L,T+2,2,2]
        Gs_aug = torch.cat([
            torch.zeros([*Gs.shape[:-3], 1, *Gs.shape[-2:]], device=Gs.device, dtype=Gs.dtype), Gs
        ], dim=-3)                                                                   # [v,L,T+1,2,2]

        mean_pred, cov_s_pred = self._predict_from_state(
            nn_idx, m_ss_aug, P_ss_aug, Gs_aug, A_fwd, A_back, P_0)
        mean_pred_Z = (H @ mean_pred).squeeze(-1).squeeze(-1)     # [v,L,Tmax]
        cov_pred_Z = (H @ cov_s_pred @ H.mT).squeeze(-1).squeeze(-1)

        return mean_pred_Z, cov_pred_Z

    def _predict_from_state(self, nn_idx, m_ss_aug, P_ss_aug, Gs_aug, A_fwd, A_back, P0):
        """
        refer to https://github.com/harrisonzhu508/MGPVAE/blob/main/mgpvae/util.py#L239
        and then https://github.com/harrisonzhu508/MGPVAE/blob/main/mgpvae/util.py#L191

        m_ss_aug/P_ss_aug: [v,L,T+2,2,2/1]; Gs_aug: [v,L,T+1,2,2/1];
        nn_idx: [v,Tmax]; A_fwd/A_back: [v,L,Tmax,2,2]; P0: [L,2,2].
        return q(s*): [v,L,Tmax,2,1/2]
        """
        # P and T
        Q_fwd = self.process_noise_covar(A_fwd, P0)
        Q_back = self.process_noise_covar(A_back, P0)
        A_back_Q_fwd = A_back @ Q_fwd
        Q_mp = Q_back + A_back @ A_back_Q_fwd.mT

        chol_Q_mp = psd_safe_cholesky(Q_mp + self.jitter * torch.eye(Q_mp.size(-2), device=Q_mp.device))
        Q_mp_inv_A_back = torch.cholesky_solve(A_back, chol_Q_mp)

        T = Q_fwd - A_back_Q_fwd.mT @ Q_mp_inv_A_back @ Q_fwd
        P_2 = Q_fwd @ Q_mp_inv_A_back.mT
        P = torch.cat([A_fwd - P_2 @ A_back @ A_fwd, P_2], dim=-1)  # [v,L,Tmax,2,4]

        # mean_joint, cov_joint
        nn_idx_expand = nn_idx.unsqueeze(-2).unsqueeze(-1).unsqueeze(-1)       # [v,1,Tmax,1,1]
        m_s_left = torch.take_along_dim(m_ss_aug, nn_idx_expand, dim=-3)       # [v,L,Tmax,2,1]
        m_s_right = torch.take_along_dim(m_ss_aug, nn_idx_expand+1, dim=-3)
        m_s_joint = torch.cat([m_s_left, m_s_right], dim=-2)

        P_s_left = torch.take_along_dim(P_ss_aug, nn_idx_expand, dim=-3)       # [v,L,Tmax,2,2]
        P_s_right = torch.take_along_dim(P_ss_aug, nn_idx_expand+1, dim=-3)
        G_left = torch.take_along_dim(Gs_aug, nn_idx_expand, dim=-3)
        G_P = G_left @ P_s_right
        P_s_joint = torch.cat([
            torch.cat([P_s_left, G_P], dim=-1),
            torch.cat([G_P.mT, P_s_right], dim=-1)
        ], dim=-2)    # [v,L,Tmax,4,4]

        return P @ m_s_joint, P @ P_s_joint @ P.mT + T


