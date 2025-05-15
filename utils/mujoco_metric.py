from typing import Union
import math
import torch
from torch import Tensor

from models.building_blocks.enc_dec_jura import JuraDecoder
from models.building_blocks.enc_dec_mujoco import MujocoDecoder


@torch.no_grad()
def predict_y(Z_ms: Tensor, Z_stds: Tensor, decoder: Union[MujocoDecoder, JuraDecoder], obs: Tensor, s: int=20, stat: dict=None):
    """
    Z_ms, Z_stds: [v, t/T, L=15], obs: [v, t/T, D=14]
    for Jura, stat['mean'], stat['std'] (both [D]) should rescale predictive dist.
    """
    assert decoder.output_distribution == 'normal'
    assert Z_ms.shape[:-1] == Z_stds.shape[:-1] == obs.shape[:-1]

    eps = torch.randn(s, *Z_ms.shape, device=Z_ms.device)
    Z_samples = Z_ms + Z_stds * eps
    recon_means = decoder(Z_samples) 

    if stat is not None:
        assert "mean" in stat and "std" in stat
        recon_means = (recon_means * stat['std']) + stat['mean']
        sigma2_y = (stat['std'] * decoder.sigma2_y.detach().sqrt()) ** 2
    else:
        sigma2_y = decoder.sigma2_y.detach()

    pred_mean = recon_means.mean(axis=0) 
    # pred_mean = decoder(Z_ms)

    # nll
    dist = torch.distributions.Normal(loc=recon_means, scale=sigma2_y.sqrt())
    logp = dist.log_prob(obs.unsqueeze(0))
    nll = - (torch.logsumexp(logp, dim=0) - math.log(s))

    # se
    se = (pred_mean - obs).square()

    # ae
    ae = (pred_mean - obs).abs()

    # adopt MGPVAE's pred_mean, pred_std
    cond_expectation_squared = recon_means ** 2
    expected_y_squared = (sigma2_y + cond_expectation_squared).mean(axis=0)
    pred_std = (expected_y_squared - pred_mean ** 2).sqrt()

    r_dict = {}  # results dictionary
    r_dict['nll'] = nll 
    r_dict['se'] = se
    r_dict['ae'] = ae 
    r_dict['pred_mean'] = pred_mean
    r_dict['pred_std'] = pred_std

    return r_dict

