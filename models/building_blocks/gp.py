import warnings

import torch
from torch import Tensor
from torch import nn

from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from linear_operator.utils.cholesky import psd_safe_cholesky


class GP(nn.Module):
    """
    Base GP class over the latent space, using auxiliary information X as inputs and VAE latent variables Z as outputs;
    The output GPs are independent along `output_dims`.
    """
    def __init__(self, output_dims: int, kernel: Kernel, mean: Mean):
        super(GP, self).__init__()
        self.output_dims = output_dims  # L

        kernel_batch_shape = kernel.batch_shape
        # kernel_batch should be [1] or equal to output dims [L]
        if len(kernel_batch_shape) == 0:
            warnings.warn(f"Got a kernel without batch dim, we added an extra batch dim.")
            kernel = kernel.expand_batch(torch.Size([1]))
        elif len(kernel_batch_shape) > 1:
            raise ValueError(f"Only support one batch dim, but got batch shape {kernel_batch_shape}.")
        elif len(kernel_batch_shape) == 1 and kernel_batch_shape[0] != 1:
            assert kernel_batch_shape[0] == output_dims, f"kernel batch doesn't match the GP output dims {output_dims}."

        # Mean function must have batch size = output dims L
        assert len(mean.batch_shape) == 1, f"Mean function of GP must have 1-D batch dim but got {mean.batch_shape}."
        assert mean.batch_shape[0] == output_dims, f"The batch dim of Mean must be equal to output_dims {output_dims}."

        self.kernel = kernel
        self.mean = mean

    def prior(self, x: Tensor, are_neighbors=False):
        """
        Compute the prior, x can be the data [(v), J, D] (are_neighbors=False) or
                            the nearest data [(v), n, H, D] when training.
        kernel batch `L` will be inserted before dim n/J and after dim v:
        return prior mean [(v), L, J] / [(v), L, n, H] and prior cov [(v), L/1, J, J] / [(v), L/1, n, H, H]
        """
        x = x.unsqueeze(-3)
        prior_mean = self.mean(x)                     # [(v), L, J] or [(v), n, L, H]
        prior_cov = self.kernel(x).to_dense()
        if are_neighbors:
            prior_mean = prior_mean.transpose(-2, -3)
            prior_cov = prior_cov.transpose(-3, -4)   # [(v), L/1, J, J] or [(v), L/1, n, H, H]
        return prior_mean, prior_cov

    def posterior(
            self, x: Tensor, X_near: Tensor, Z_near: Tensor, std_Z_near: Tensor = None,
            jitter_val: float = 1e-6
    ):
        """
        :return: if std_Z_near=None, return conditional p(z|Z_near, X_near, x) with shape [n, L];
                 otherwise, return marginal posterior by integrating p(z|Z_near,x)q(Z_near)dZ_near
        :param x: test auxiliary input [(v), n, D]
        :param X_near: nearest X [(v), n, H, D]
        :param Z_near: nearest Z [(v), n, H, L], independent along the last dim; or mean of q(Z_near|Y) from encoder
        :param std_Z_near: can be stds of q(Z_near|Y) from encoder with shape [n, H, L] or `0`
        :param jitter_val: jitter value [default: 1e-6]
        """
        assert Z_near.size(-1) == self.output_dims

        x = x.unsqueeze(-2).unsqueeze(-2)                          # [n,1,1,D]
        X_near = X_near.unsqueeze(-3)                              # [n,1,H,D]
        K_nn = self.kernel(X_near, X_near).add_jitter(jitter_val)           # [(v), n, L, H, H]
        K_nx = self.kernel(X_near, x).to_dense()                            # [(v), n, L, H, 1]
        L = psd_safe_cholesky(K_nn.to_dense())
        L_inv_Knx = torch.linalg.solve_triangular(L, K_nx, upper=False)  # [(v), n, L, H, 1]
        K_nn_inv_Knx = torch.linalg.solve_triangular(L.mT, L_inv_Knx, upper=True)

        mean_diff = (Z_near.mT - self.mean(X_near)).unsqueeze(-1)  # [(v),n,L,H,1]
        post_mean = (K_nn_inv_Knx * mean_diff).sum(dim=(-1, -2)) + self.mean(x).squeeze(-1)

        post_cov = self.kernel(x).to_dense() - (K_nx * K_nn_inv_Knx).sum(dim=-2, keepdim=True)
        post_cov = post_cov.squeeze(-1).squeeze(-1)                # [(v), n, L]
        if std_Z_near is not None:
            std_Z_near = std_Z_near.mT.unsqueeze(-1)
            S_K_nn_inv_Knx = std_Z_near * K_nn_inv_Knx
            post_cov = post_cov + S_K_nn_inv_Knx.square().sum(dim=(-1, -2))

        return post_mean, post_cov

