import os
import math
import numpy as np
import pandas as pd
import torch


def load_jura(folder_dir=None, fill_value: float=0.):
    folder_dir = "./data/jura" if folder_dir is None else folder_dir
    _train = pd.read_csv(os.path.join(folder_dir, 'prediction.dat')).set_index(['Xloc', 'Yloc'])
    _test = pd.read_csv(os.path.join(folder_dir, 'validation.dat')).set_index(['Xloc', 'Yloc'])

    _data_Yfull = pd.concat([_train[['Ni', 'Zn', 'Cd']], _test[['Ni', 'Zn', 'Cd']]])
    _data_Ymiss = pd.concat([_train[['Ni', 'Zn', 'Cd']], _test[['Ni', 'Zn']]])

    Y_mean = torch.as_tensor(_data_Ymiss.mean().to_numpy(), dtype=torch.get_default_dtype())
    Y_std = torch.as_tensor(_data_Ymiss.std().to_numpy(), dtype=torch.get_default_dtype())

    data_X = torch.as_tensor([[i, j] for (i, j) in _data_Ymiss.index], dtype=torch.get_default_dtype())
    data_Yfull = torch.as_tensor(_data_Yfull.to_numpy(), dtype=torch.get_default_dtype())
    _data_Ymiss_numpy = _data_Ymiss.to_numpy()
    _masks = ~np.isnan(_data_Ymiss_numpy)  # False (0) indicates missing
    masks = torch.as_tensor(_masks, dtype=torch.bool)
    data_Ymiss = torch.as_tensor(_data_Ymiss_numpy, dtype=torch.get_default_dtype())

    data_Ymiss[~masks] = fill_value  # fill before data normalization
    # normalize data_Ymiss NOTE: data_Yfull is NOT normalized
    data_Ymiss = (data_Ymiss - Y_mean) / Y_std

    stat = {'mean': Y_mean, 'std': Y_std}  # keys should be 'mean' and 'std'

    return data_X, masks, data_Ymiss, data_Yfull, stat


def gaussian_nll(pred_mean, pred_std, Y, eps=1e-6):
    """
    nll by assuming Gaussian distribution
    pred_mean, pred_std, Y: [n_test]
    Y: ground truth
    return: mean nll
    """
    pred_std = torch.clamp(pred_std, min=eps)
    var = pred_std ** 2
    nll = 0.5 * torch.log(2 * math.pi * var) + 0.5 * ((Y - pred_mean) ** 2 / var)

    return nll.mean()