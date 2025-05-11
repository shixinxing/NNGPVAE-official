import torch
import torch.nn as nn
import numpy as np
from gpytorch.kernels import ScaleKernel
from linear_operator.utils.cholesky import psd_safe_cholesky

from baselines.bae.gp.likelihoods import Gaussian


class BSGP(nn.Module):
    def __init__(
            self, output_dims: int, kernel: ScaleKernel, likelihood: Gaussian,
            inducing_points: nn.Parameter,       # [(L), M, D]
            inducing_values: nn.Parameter,       # [(v), L, M]
            N_train: int, prior_S_type: str,
            prior_lengthscale=2., prior_variance=0.05, prior_lik_var=0.05  # hyper-params of the prior distributions.
    ):
        """
        The inducing points in this class can be shared or not, but the official implementation seems only
        using shared kernel params and shared inducing locations.
        """
        super(BSGP, self).__init__()
        self.output_dims = output_dims  # L
        self.kernel = kernel
        assert len(kernel.batch_shape) == 1, f"Kernel must have one batch dimension, but got {kernel.batch_shape}"
        self.likelihood = likelihood  # with noise var as params

        self.N_train, self.M = N_train, inducing_points.size(-2)  # M
        self.S = inducing_points                      # inducing locations: [(L), M, D]
        self.U = inducing_values                      # inducing variables: [(v), L, M]
        self.Lm = None

        self.prior_S_type = prior_S_type              # prior over inducing locations
        self.prior_lengthscale = prior_lengthscale    # the hyper-params in the prior distribution of kernel params
        self.prior_variance = prior_variance
        self.prior_lik_var = prior_lik_var

    def conditional(self, X):
        """
        compute Gaussian conditional: p(f_n|U) at X: [(v), N, D], U: [M, L],
        return f_mean, f_var: [(v), N, L]
        note that this GP uses diagonal covars and whiten
        adapted from https://github.com/tranbahien/sgp-bae/blob/master/gp/conditionals.py#L32
        """
        X = X.unsqueeze(-3)                                      # [(v),1,N,D]
        Kmn = self.kernel(self.S, X).to_dense()                  # [(v),L,M,N]
        Kmm = self.kernel(self.S).to_dense()                     # [L/1,M,M]
        Lm = psd_safe_cholesky(Kmm + 1e-6 * torch.eye(self.M, dtype=torch.get_default_dtype(), device=self.S.device))

        A = torch.linalg.solve_triangular(Lm, Kmn, upper=False)
        fmean = A.mT @ self.U.unsqueeze(-1)                      # [(v),L,N,1], whiten
        fmean = fmean.squeeze(-1).mT                             # [(v),N,L]
        fvar = self.kernel(X, diag=True) - (A ** 2).sum(dim=-2)  # [(v),L,N]
        fvar = fvar.mT                                           # [(v),N,L]
        self.Lm = Lm
        return fmean, fvar

    def predict(self, X):
        f_mean, f_var = self.conditional(X)
        z_mean, z_var = self.likelihood.predict_mean_and_var(f_mean, f_var)
        return z_mean, z_var                                     # [(v),N,L]

    def log_prior_S(self):
        """ prior over the inducing locations S"""
        if self.prior_S_type == "uniform":
            return 0.
        if self.prior_S_type == "normal":
            return - torch.sum(torch.square(self.S)) / 2.0
        # if self.Lm is not None: # determinantal;
        if self.prior_S_type == "determinantal":
            self.Lm = torch.linalg.cholesky(
                self.kernel(self.S) + 1e-7 * torch.eye(self.M, dtype=torch.get_default_dtype(), device=self.S.device)
            )
            log_prob = torch.sum(torch.log(torch.square(torch.diagonal(self.Lm))))
            return log_prob
        else:
            raise Exception("Invalid prior type")

    def log_prior_U(self):
        """ prior over the inducing variables U ~ N(0, Kss) or U ~ N(0, I) if whiten """
        return - torch.sum(torch.square(self.U)) / 2.0

    def log_prior_hyper(self):
        """ prior over the hyper-params in kernel and GP likelihood: """
        log_lengthscales = torch.log(self.kernel.base_kernel.lengthscale)
        log_variance = torch.log(self.kernel.outputscale)
        log_lik_var = torch.log(self.likelihood.variance.get())

        log_prob = - torch.sum(torch.square(log_lengthscales - np.log(self.prior_lengthscale))) / 2.
        log_prob += - torch.sum(torch.square(log_variance - np.log(self.prior_variance))) / 2.
        log_prob += - torch.sum(torch.square(log_lik_var - np.log(self.prior_lik_var))) / 2.
        return log_prob

    def log_prior(self):
        return self.log_prior_S() + self.log_prior_U() + self.log_prior_hyper()

    def log_likelihood(self, X, Z):
        """the GP's output: sum_n log p(f_n|U) p(z_n|f_n) df_n = p(z_n|U), where p(f_n|U) is conditional Gaussian"""
        f_mean, f_var = self.conditional(X)
        log_likelihood = torch.sum(self.likelihood.predict_density(f_mean, f_var, Z))
        return log_likelihood

    def log_prob(self, X, Z):
        # log p(Z|U), sum along axis "v"
        log_likelihood = (self.N_train / X.size(-2)) * self.log_likelihood(X, Z)
        # log p(\psi)
        log_prior = self.log_prior()

        log_prob = log_likelihood + log_prior
        return log_prob


if __name__ == "__main__":
    from gpytorch.kernels import RBFKernel
    from baselines.bae.gp.likelihoods import Gaussian

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    L = 2
    k = ScaleKernel(RBFKernel(batch_shape=torch.Size([L])))
    k.outputscale = 1.
    k.base_kernel.lengthscale = 1.
    N, M, D = 7, 3, 5
    S = torch.arange(M * D).to(torch.get_default_dtype()).reshape(M, D) / 100
    S[0, 0], S[1, 0] = torch.tensor(33.), torch.tensor(-1.)
    U = torch.zeros(L, M)

    model = BSGP(L, k, Gaussian(), S, U, N_train=N, prior_S_type="uniform")
    xx = torch.arange(N * D).to(torch.get_default_dtype()).reshape(D, N).mT / 100

    mean, var = model.conditional(xx)

    print(f'mean ({mean.shape}): \n{mean}')
    print(f'var ({var.shape}): \n{var}\n')
    print(f'model params:\n{dict(model.named_parameters())}')

