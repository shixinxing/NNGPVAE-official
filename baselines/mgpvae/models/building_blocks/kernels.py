# Markovian kernels with state transition
# inspired by https://github.com/harrisonzhu508/MGPVAE/blob/main/mgpvae/kernels.py#L116

import math

import torch
from gpytorch.kernels import ScaleKernel, MaternKernel


class Matern32SSM(ScaleKernel):
    """SSM form of Matern32 kernel."""
    def __init__(self, base_kernel: MaternKernel, **kwargs):
        assert base_kernel.nu == 1.5, f"nu  should be 1.5, but got {base_kernel.nu}."
        assert base_kernel.raw_lengthscale.size(-1) == 1
        self.nu = base_kernel.nu
        self.state_dim = 2
        super(Matern32SSM, self).__init__(base_kernel, **kwargs)

    @property
    def lamb(self):
        return math.sqrt(3) / self.base_kernel.lengthscale

    def stationary_covariance(self):   # P_0
        variance = self.outputscale.unsqueeze(-1).unsqueeze(-1)
        zeros = torch.zeros_like(variance, dtype=variance.dtype, device=variance.device)
        P_0 = torch.cat([
            torch.cat([variance, zeros], dim=-1),
            torch.cat([zeros, variance * self.lamb.square()], dim=-1)
        ], dim=-2)
        return P_0  # [L,2,2]

    def kernel_to_state_space(self):
        lam = self.lamb   # [L,1,1]
        zeros_ones = torch.tensor([0., 1.], dtype=lam.dtype, device=lam.device).expand(*lam.shape[:-1], 2)  # [L,1,2]
        F = torch.cat([
            zeros_ones,
            torch.cat([-lam.square(), -2*lam], dim=-1)
        ], dim=-2)   # [L,2,2]
        L = torch.tensor(
            [[0.], [1.]], dtype=lam.dtype, device=lam.device).expand(*self.batch_shape, 2, 1)  # [L,2,1]
        H = torch.tensor(
            [[1., 0.]], dtype=lam.dtype, device=lam.device).expand(*self.batch_shape, 1, 2)    # [L,1,2]
        P_0 = self.stationary_covariance()
        return F, L, H, P_0

    def state_transition(self, dt):
        """
        Calculate state transition matrix A = expm(FΔt): [T,L,2,2] with dt: [...,T,1/L,1,1]
        """
        lam = self.lamb
        mat = torch.cat([
            torch.cat([lam, torch.ones_like(lam)], dim=-1),
            torch.cat([-lam.square(), -lam], dim=-1)
        ], dim=-2)
        A = torch.exp(- dt * lam) * (dt * mat + torch.eye(2, dtype=dt.dtype, device=dt.device))
        return A


class Matern52SSM(ScaleKernel):
    """SSM form of Matern52 kernel."""
    def __init__(self, base_kernel: MaternKernel, **kwargs):
        assert base_kernel.nu == 2.5, f"nu  should be 1.5, but got {base_kernel.nu}."
        assert base_kernel.raw_lengthscale.size(-1) == 1
        self.nu = base_kernel.nu
        self.state_dim = 3
        super(Matern52SSM, self).__init__(base_kernel, **kwargs)

    @property
    def lamb(self):
        return math.sqrt(5) / self.base_kernel.lengthscale

    def stationary_covariance(self):  # P_0
        variance = self.outputscale.unsqueeze(-1).unsqueeze(-1)
        zeros = torch.zeros_like(variance, dtype=variance.dtype, device=variance.device)
        kappa = variance * self.lamb.square() / 3
        P_0 = torch.cat([
            torch.cat([variance, zeros, -kappa], dim=-1),
            torch.cat([zeros, kappa, zeros], dim=-1),
            torch.cat([-kappa, zeros, variance * self.lamb.pow(4)], dim=-1)
        ], dim=-2)
        return P_0  # [L,3,3]

    def kernel_to_state_space(self):
        lam = self.lamb  # [L,1,1]
        zeros_ones_0 = torch.tensor(
            [0., 1., 0.], dtype=lam.dtype, device=lam.device).expand(*lam.shape[:-1], 3)  # [L,1,3]
        zeros_ones_1 = torch.tensor([0., 0., 1.], dtype=lam.dtype, device=lam.device).expand_as(zeros_ones_0)
        F = torch.cat([
            zeros_ones_0, zeros_ones_1,
            torch.cat([-lam.pow(3), -3*lam.square(), -3*lam], dim=-1)
        ], dim=-2)  # [L,3,3]
        L = torch.tensor(
            [[0.], [0.], [1.]], dtype=lam.dtype, device=lam.device).expand(*self.batch_shape, 3, 1)  # [L,3,1]
        H = torch.tensor(
            [[1., 0., 0.]], dtype=lam.dtype, device=lam.device).expand(*self.batch_shape, 1, 3)  # [L,1,3]
        P_0 = self.stationary_covariance()
        return F, L, H, P_0

    def state_transition(self, dt):
        """
        Calculate state transition matrix A = expm(FΔt): [T,L,3,3] with dt: [...,T,1/L,1,1]
        """
        lam = self.lamb
        dtlam = dt * lam
        mat = torch.cat([
            torch.cat([lam*(0.5*dtlam+1), dtlam+1, (0.5*dt).expand_as(dtlam)], dim=-1),
            torch.cat([-0.5*dtlam*lam.square(), lam*(1-dtlam), 1-0.5*dtlam], dim=-1),
            torch.cat([lam.pow(3)*(0.5*dtlam-1), lam.square()*(dtlam-3), lam*(0.5*dtlam-2)], dim=-1)
        ], dim=-2)
        A = torch.exp(- dtlam) * (dt * mat + torch.eye(3, dtype=dt.dtype, device=dt.device))
        return A


if __name__ == "__main__":
    def test(nu=1.5):
        base_k = MaternKernel(nu=nu, batch_shape=torch.Size([3]))
        k = Matern32SSM(base_kernel=base_k) if nu == 1.5 else Matern52SSM(base_kernel=base_k)
        k.outputscale, k.base_kernel.lengthscale = 1., 1.
        print(k.stationary_covariance())

        F, L, H, P_0 = k.kernel_to_state_space()
        dt = torch.rand(8).reshape(2, 4, 1, 1, 1)
        A = k.state_transition(dt)

        print(f"F: {F}")
        print(f"L: {L}")
        print(f"H: {H}")
        print(f"P_0: {P_0}")
        print(f"A (shape: {A.shape}): {A}")
        print(f"K(x,x) shape: {k(torch.randn(3, 1, 1)).shape}")


    test(nu=2.5)



