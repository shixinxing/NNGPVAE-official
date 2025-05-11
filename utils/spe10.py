from typing import Union
from pathlib import Path
import math
import numpy as np
import torch
from torch import Tensor, nn
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from models.building_blocks.rq_kernel import RQKernel

from matplotlib import pyplot as plt

from utils.build_datasets import NNDataset


class NNDatasetSPE10(NNDataset):
    def __init__(
            self, file_path: Union[str, Path], H: int = None,
            search_device: str = 'cpu', build_sequential_first=False
    ):
        if isinstance(file_path, str):
            file_path = Path(file_path)

        data_dict = np.load(file_path)   # {'X': [N,3], 'Y'/'mask': [N,4]}
        super(NNDatasetSPE10, self).__init__(
            data_dict, series_shape=torch.Size([]), missing=True, return_full=True, data_device=search_device,
            H=H, search_device=search_device, build_sequential_first=build_sequential_first
        )


@torch.no_grad()
def predict_y(Z_ms: Tensor, Z_stds: Tensor, decoder: nn.Module, obs: Tensor, s: int = 20, stat: dict = None):
    """
    Z_ms, Z_stds: [b, L], obs: [b, D]
    for Jura, stat['mean'], stat['std'] (both [D]) should rescale predictive dist.
    """
    assert decoder.output_distribution == 'normal'
    assert Z_ms.shape[:-1] == Z_stds.shape[:-1] == obs.shape[:-1]
    if stat is not None:
        Y_mean = torch.as_tensor(stat['mean'], device=Z_ms.device)
        Y_std = torch.as_tensor(stat['std'], device=Z_ms.device)
    else:
        Y_mean, Y_std = 0., 1.

    eps = torch.randn([s, *Z_ms.shape], device=Z_ms.device)
    Z_samples = Z_ms + Z_stds * eps
    sample_means = decoder(Z_samples)  # [s,b,D]
    sample_means = sample_means * Y_std + Y_mean
    sigma2_y = decoder.sigma2_y * (Y_std ** 2)

    # nll, obtained from a mixture of the output distributions
    dist = torch.distributions.Normal(loc=sample_means, scale=sigma2_y.sqrt())
    log_p = dist.log_prob(obs * Y_std + Y_mean)
    nll = - (torch.logsumexp(log_p, dim=0) - math.log(s))  # [b,D]

    # se/ae, obtained by feeding latent mean
    delta = (decoder(Z_ms) - obs) * Y_std
    se, ae = delta.square(), delta.abs()

    # adopt MGPVAE's pred_mean, pred_std
    pred_mean = sample_means.mean(dim=0)  # [b,D]
    cond_expectation_squared = sample_means ** 2                              # {E[Y]}^2
    expected_y_squared = (sigma2_y + cond_expectation_squared).mean(dim=0)  # E[Y^2]
    pred_std = (expected_y_squared - pred_mean ** 2).sqrt()

    r_dict = {
        'nll': nll, 'se': se, 'ae': ae, 'pred_mean': decoder(Z_ms), 'pred_std': pred_std
    }  # results dictionary

    return r_dict


def define_kernel(kernel_type: str, latent_dims, x_dims):
    bs = torch.Size([]) if latent_dims is None else torch.Size([latent_dims])

    if kernel_type == 'rbf':
        kernel = ScaleKernel(RBFKernel(batch_shape=bs, ard_num_dims=x_dims))
    elif kernel_type == 'matern32':
        kernel = ScaleKernel(MaternKernel(nu=1.5, batch_shape=bs, ard_num_dims=x_dims))
    elif kernel_type == 'matern12':
        kernel = ScaleKernel(MaternKernel(nu=0.5, batch_shape=bs, ard_num_dims=x_dims))
    elif kernel_type == 'cauchy':
        if len(bs) > 0:
            kernel = ScaleKernel(RQKernel(batch_shape=bs, ard_num_dims=x_dims))
        else:
            kernel = ScaleKernel(gpytorch.kernels.RQKernel(batch_shape=bs, ard_num_dims=x_dims))
    else:
        raise NotImplementedError(f'Kernel type {kernel_type} not implemented')
    return kernel


def plot_spe10(data_dict_true: dict, pred_mean, layer_idx=20, nll=None, mse=None, y_dim_idx=0):
    Nx, Ny, Nz = 30, 110, 43
    Y_full = data_dict_true['Y_full']
    Y = data_dict_true['Y']
    show_res = f' nll={nll:.4f}' if nll is not None else ''
    show_res = show_res + f' mse={mse:.4f}' if mse is not None else ''

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 8))
    vmin = min(Y.min(), Y_full.min(), pred_mean.min())
    vmax = max(Y.max(), Y_full.max(), pred_mean.max())

    im = axs[0].imshow(Y.reshape(Nx, Ny, Nz, 4)[..., layer_idx, y_dim_idx], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[0].set_title('Missing Y')

    axs[1].imshow(Y_full.reshape(Nx, Ny, Nz, 4)[..., layer_idx, y_dim_idx], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[1].set_title('Full Y')

    axs[2].imshow(pred_mean.reshape(Nx, Ny, Nz, 4)[..., layer_idx, y_dim_idx], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[2].set_title('Predicted mean' + show_res)

    cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Value Range')

    return fig, axs




