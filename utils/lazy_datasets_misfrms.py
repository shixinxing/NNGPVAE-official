from typing import Union, Optional
from pathlib import Path
import h5py
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils.nnutils_mask import NNUtilMasked_GPVAE
from utils.build_datasets import NNDataset


class LazySeriesDatasetMisFrms(Dataset):
    """
    Sequence data [N,T,D...] with frame masks [N,T,1]. data dict has
    {'seq_train/test_full', 't_train/test_full', 'm_train/test_miss'}, where observations are moved to the left and
    masks can indicate the boundary of the observed and the missing.

    NN structure over timestamps for all videos is built here (before the training).
    """
    def __init__(
            self, file_path: Union[str, Path], train: bool, H: Optional[int],
            search_device='cpu', build_sequential_first=False
    ):
        super(LazySeriesDatasetMisFrms, self).__init__()
        self.file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        self.train, self.H = train, H

        if self.file_path.suffix == ".npz":
            self.data_dict = np.load(file_path)
        elif self.file_path.suffix == ".h5":
            self.data_dict = h5py.File(file_path, "r")
        else:
            raise ValueError(f"Unsupported file type: {self.file_path.suffix}")

        # build Index and NN structure before training if necessary.
        if train:
            t_all, m_all = self.data_dict["t_train_full"][:], self.data_dict["m_train_miss"][:]
        else:
            t_all, m_all = self.data_dict["t_test_full"][:], self.data_dict["m_test_miss"][:]
        t_all = torch.as_tensor(t_all, dtype=torch.get_default_dtype(), device=search_device)
        m_all = torch.as_tensor(m_all, dtype=torch.get_default_dtype(), device=search_device)

        self.nnutil_all, self.seq_nn_idx = None, None
        if H is not None:
            self.nnutil_all = NNUtilMasked_GPVAE(
                H, t_all, m_all, preferred_nnlib='faiss', metric='L2', search_device=search_device
            )

            # Since the timestamps are sorted, we may construct NN structure directly to accelerate
            if build_sequential_first:
                self.seq_nn_idx = self.nnutil_all.build_sequential_nn_idx(faiss_reset=False, padding=True)
                self.nnutil_all.set_nn_idx(faiss_reset=False)  # actually do nothing since the above step already did
            else:
                self.nnutil_all.set_nn_idx(faiss_reset=True)

    def __getitem__(self, idx):
        if self.train:
            seq_full = self.data_dict["seq_train_full"][idx]      # [T,D]
            t_full = self.data_dict["t_train_full"][idx]
            t_mask = self.data_dict["m_train_miss"][idx]
        else:
            seq_full = self.data_dict["seq_test_full"][idx]
            t_full = self.data_dict["t_test_full"][idx]
            t_mask = self.data_dict["m_test_miss"][idx]
        return (
            torch.as_tensor(seq_full), torch.as_tensor(t_full), torch.as_tensor(t_mask), idx
        )  # idx will be collated into an int64 tensor, used for locating Index when finding NNs.

    def __len__(self):
        return self.data_dict['t_train_full'].shape[0] if self.train else self.data_dict["t_test_full"].shape[0]


def generate_shorter_miniseries_dict(
        vid_batch_full: Tensor, t_batch_full: Tensor, mask_batch: Tensor, clip=True
):
    """
    Shorten the series ([v,T,D/1]) length since the right is missing frames. Clip according to the longest series.
    """
    assert t_batch_full.ndim == 3 and mask_batch.ndim == 3, "Auxiliary information must have 3 dims."
    if clip:
        seq_length = mask_batch.sum(dim=(-1, -2)).to(torch.int64)
        seq_length_max = seq_length.max().item()
    else:
        seq_length_max = t_batch_full.size(-2)
    data_dict = {
        'images': vid_batch_full[:, :seq_length_max],   # [v,Tmax,...]
        'masks': mask_batch[:, :seq_length_max],        # [v,Tmax,1]
        'aux_data': t_batch_full[:, :seq_length_max]
    }
    return data_dict


def generate_test_shorter_miniseries_dict(
        vid_batch_full: Tensor, t_batch_full: Tensor, mask_batch_test: Tensor, clip=True
):
    """
    When testing, clip the left and remain the right. Note that now `True` in mask represents the unobserved frames
    """
    assert t_batch_full.ndim == 3 and mask_batch_test.ndim == 3, "Auxiliary information must have 3 dims."
    if clip:
        seq_length = mask_batch_test.sum(dim=(-1, -2)).to(torch.int64)
        seq_length_max = seq_length.max().item()
    else:
        seq_length_max = t_batch_full.size(-2)
    data_dict = {
        'images': vid_batch_full[:, -seq_length_max:],   # [v,T-Tmin,...]
        'masks': mask_batch_test[:, -seq_length_max:],   # [v,T-Tmin,1]
        'aux_data': t_batch_full[:, -seq_length_max:]
    }
    return data_dict


class NNDatasetMisFrms(NNDataset):
    """
    tailored mini_series for missing-frame cases,
    data stored in this class are clipped when constructing data_dict.
    """
    def __init__(
            self, data_dict: dict, series_shape=torch.Size([]), data_device='cpu', search_device: str = None
    ):
        super(NNDatasetMisFrms, self).__init__(
            data_dict, series_shape, missing=True, return_full=False, data_device=data_device,
            H=None  # nn_utils already set in the `entire dataset`
        )
        self.search_device = search_device

        self.total_observed_frames_per_miniseries = self.masks.sum()
        self.seq_len_mean = self.total_observed_frames_per_miniseries / series_shape.numel()




