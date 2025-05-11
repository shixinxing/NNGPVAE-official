import math
import torch
from torch import Tensor

from baselines.bae.distributions.conditional.normal import ConditionalMeanNormal


@torch.no_grad()
def predict_y(self, y_miss: Tensor, m: Tensor, Y_full: Tensor, stat: dict = None):
    """
    Used in predict_sgpbae; y_miss: [b,...], m: [b,...], Y_full: [b,...];
    stat will rescale predictive dist. Note: y_miss is in normalized scale, Y_full is in un-normalized (original) scale.
    m: 0 indicates missing (to impute)
    """
    assert isinstance(self.decoder, ConditionalMeanNormal), "Need to be checked for other decoder likelihoods."
    assert y_miss.shape == m.shape == Y_full.shape
    if m.dtype != torch.bool:
        m = m.bool()  # recall that 0 indicates missing, where we used to compute metric

    Y_samples, Y_means, logps = [], [], []

    # refer to https://github.com/tranbahien/sgp-bae/blob/master/models/mask_sgpbae.py#L49
    for i in range(len(self.decoder_samples)):  # s samples
        decoder_params, gp_params = self.load_samples(i)
        self.decoder_params = decoder_params
        self.gp_params = gp_params

        Z_sample, _ = self.encode(y_miss)  # [b,L]
        cond_dist = self.decoder.cond_dist(Z_sample)  # p(y|z), [b,...]

        # transform back
        if stat is not None:
            cond_dist.loc = cond_dist.mean * stat['std'] + stat['mean']
            cond_dist.scale = cond_dist.stddev * stat['std']

        Y_means.append(cond_dist.mean)
        Y_samples.append(cond_dist.sample())
        logps.append(cond_dist.log_prob(Y_full))

    Y_samples = torch.stack(Y_samples, dim=0)  # [s,b,...]
    Y_means = torch.stack(Y_means, dim=0)  # [s,b,...]
    # ⚠️ We here regard the decoder output as a mixture to get nll, which is different from the official code in:
    # https://github.com/tranbahien/sgp-bae/blob/master/dsgpbae_experiment.py#L195, where they got mean/var directly
    logps = torch.stack(logps, dim=0)  # [s,b,...]

    # se: square error
    Y_means_mean = Y_means.mean(dim=0)  # [b,...]
    _se = (Y_means_mean - Y_full).square()  # [b,...]
    se = _se[~m]  # Compute metric only on missing part

    # ae: absolute error
    _ae = (Y_means_mean - Y_full).abs()  # [b,...]
    ae = _ae[~m]

    # nll
    _nll = - (torch.logsumexp(logps, dim=0) - math.log(logps.size(0)))  # [b,...]
    nll = _nll[~m]

    # predictive mean/std, mainly for plot
    # ⚠️⚠️ inconsistent std computation ?!
    Y_samples_mean, Y_samples_std = Y_samples.mean(dim=0), Y_samples.std(dim=0)

    r_dict = {
        'nll': nll, 'se': se, 'ae': ae, 'pred_mean': Y_means_mean, 'pred_std': Y_samples_std
    }

    return r_dict
