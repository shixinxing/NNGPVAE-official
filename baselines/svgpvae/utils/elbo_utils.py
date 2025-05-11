import torch


def negative_gaussian_cross_entropy(mu1, var1, mu2, var2):
    """
    refer to https://github.com/scrambledpie/GPVAE/blob/master/GPVAEmodel.py#L17
    Computes the element-wise cross entropy
    Given q(z) ~ N(z| mu1, var1)， returns E_q[ log N(z| mu2, var2) ]
    args:
        mu1 / var1 / mu2 / var2: (batch, tmax, 2)
    returns:
        cross_entropy: (batch, tmax, 2)
    """
    term0 = torch.log(torch.tensor(2 * torch.pi))  # log(2*pi)
    term1 = torch.log(var2)
    term2 = (var1 + mu1 ** 2 - 2 * mu1 * mu2 + mu2 ** 2) / var2
    cross_entropy = - 0.5 * (term0 + term1 + term2)
    return cross_entropy
