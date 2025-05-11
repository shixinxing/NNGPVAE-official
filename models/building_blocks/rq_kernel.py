# a slight modification of gpytorch's RQKernel due to its dealing with the batch shape of `alpha`
# For detail, see https://github.com/cornellius-gp/gpytorch/blob/main/gpytorch/kernels/rq_kernel.py#L68

from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Optional
import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive, Interval


class RQKernel(Kernel):
    """
    Cauchy kernel used in Healing MNIST experiment. There is no need for `alpha` to unsqueeze such that the kernel batch
    becomes the outermost dim of `alpha`
    """

    has_lengthscale = True

    def __init__(self, alpha_constraint: Optional[Interval] = None, **kwargs):
        super(RQKernel, self).__init__(**kwargs)
        # We don't need `alpha` to have a batch dim
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(1, 1)))
        if alpha_constraint is None:
            alpha_constraint = Positive()

        self.register_constraint("raw_alpha", alpha_constraint)

    def forward(self, x1, x2, diag=False, **params):
        def postprocess_rq(dist_mat):
            alpha = self.alpha
            for _ in range(1, len(dist_mat.shape) - len(self.batch_shape)):
                alpha = alpha.unsqueeze(-1)
            return (1 + dist_mat.div(2 * alpha)).pow(-alpha)

        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        return postprocess_rq(
            self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params),
        )

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))
