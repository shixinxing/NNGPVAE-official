import warnings

import torch
from torch.utils.data import Dataset

from utils.nnutils import NNUtil_GPVAE


class NNDataset(Dataset):
    """
    Mini-batch along the time dim; Support Y with nan values, and nan is filled with `0`.
    """
    def __init__(
            self, data_dict: dict, series_shape=torch.Size([]), missing=False, return_full=False, data_device='cpu',
            H: int = None, search_device: str = None, build_sequential_first=False
    ):
        """
        data_dict:
        'images/Y': Numpy array or Tensor of shape ((B), N, C, H, W).
        'aux_data/X': ((B), N, D). For regular timestamps, it may not have batch dim for efficient NN searching.
        'masks':  shape same as `images/Y` or None (i.e., no mask, or we have to find it ourselves).
        'imges_full/Y_full': complete array without missing values.
        Usually, getting nn_idx can be done on GPU (requires putting X onto the GPU), and gather data on CPU;
        For small Y, both processes can be done on GPU.
        """
        super(NNDataset, self).__init__()
        self.Y = torch.as_tensor(
            data_dict['images'] if 'images' in data_dict.keys() else data_dict['Y'],
            dtype=torch.get_default_dtype(), device=data_device
        )
        self.X = torch.as_tensor(
            data_dict['aux_data'] if 'aux_data' in data_dict.keys() else data_dict['X'],
            dtype=torch.get_default_dtype(), device=data_device
        ).contiguous()

        self.series_shape = series_shape   # batch dim B in Y
        assert self.X.shape[:-2] == torch.Size([]) or self.series_shape == self.X.shape[:-2], f'Got wrong series shape.'
        assert self.series_shape == self.Y.shape[:len(self.series_shape)], f"Got wrong series shape."

        self.data_device, self.search_device = data_device, None if H is None else search_device

        # handle missing values
        self.missing, self.return_full = missing, return_full
        if missing:
            # construct masks
            if 'masks' in data_dict.keys():
                self.masks = torch.as_tensor(data_dict['masks'], dtype=torch.bool, device=data_device)
            else:
                warnings.warn(f"We did not find 'masks' in data_dict, so nan values is set to zeros.")
                self.masks = torch.isnan(self.Y)
                self.Y[self.masks] = 0.

            self.return_full = False if not return_full else True  # whether to return full data for evaluation
            if self.return_full:
                if 'images_full' in data_dict.keys() or 'Y_full' in data_dict.keys():
                    self.Y_full = torch.as_tensor(
                        data_dict['images_full'] if 'images_full' in data_dict.keys() else data_dict['Y_full'],
                        dtype=torch.get_default_dtype(), device=data_device
                    )
                else:
                    warnings.warn("This dataset has missing values but does not have full images/Y for evaluation.")
                    self.return_full = False

        # construct NN utils
        self.nn_util, self.seq_nn_idx = None, None
        if H is not None:
            self.nn_util = NNUtil_GPVAE(
                k=H, dim=self.X.size(-1), anchor_batch_shape=self.X.shape[:-2], search_device=search_device
            )
            if build_sequential_first:
                self.seq_nn_idx = self.nn_util.build_sequential_nn_idx(self.X, faiss_reset=False)
                self.nn_util.set_nn_idx(anchor_x=None)
            else:
                # we don't need to set up each time we initialize this class, for some exps
                self.nn_util.set_nn_idx(anchor_x=self.X)

    def __len__(self):
        return self.Y.size(len(self.series_shape))   # we don't consider extra data batch dim

    def __getitem__(self, idx):                      # idx will be an int
        if self.series_shape == torch.Size([]):
            sample, label = self.Y[idx], self.X[idx]
            if not self.missing:
                return sample, label
            elif not self.return_full:
                return sample, label, self.masks[idx]
            return sample, label, self.masks[idx], self.Y_full[idx]

        # index along the dim after extra batch dims, each data has shape [v, 32, 32]
        slices_y = [slice(None)] * len(self.series_shape) + [idx]
        sample = self.Y[slices_y]
        label = self.X[idx] if self.X.ndim == 2 else self.X[slices_y]
        if not self.missing:
            return sample, label
        elif not self.return_full:
            return sample, label, self.masks[slices_y]
        return sample, label, self.masks[slices_y], self.Y_full[slices_y]

    def gather(self, nn_idx):
        """
        idx [...,n, H] is on `search_device`, and self.Y/X is on `data_device`.
        We put both the nn_idx onto the `data_device` , and then select NNs.
        """
        def _gather(idx, X_or_Y):
            # Y: [...,N,1,28,28] or X: [...,N,D], idx [..., n, H] should have the same batch dim as X_or_Y
            idx_ndim = idx.ndim
            new_shape = idx.shape + torch.Size([1 for _ in range(X_or_Y.ndim + 1 - idx_ndim)])
            idx_expand_dims = idx.reshape(new_shape)
            data_expand_dims = X_or_Y.unsqueeze(idx.ndim - 2)
            data_gather = torch.take_along_dim(data_expand_dims, indices=idx_expand_dims, dim=idx_ndim - 1)
            return data_gather  # [...,n, H, ...]

        nn_idx = nn_idx.to(self.data_device)
        nn_idx_expand_as_Y = nn_idx.expand(*self.series_shape, -1, -1)
        y_gather = _gather(nn_idx_expand_as_Y, self.Y.to(self.data_device))
        x_gather = _gather(nn_idx, self.X.to(self.data_device))
        if not self.missing:
            return y_gather, x_gather
        else:
            m_gather = _gather(nn_idx_expand_as_Y, self.masks.to(self.data_device))
            if not self.return_full:
                return y_gather, x_gather, m_gather
            full_gather = _gather(nn_idx_expand_as_Y, self.Y_full.to(self.data_device))
            return y_gather, x_gather, m_gather, full_gather

