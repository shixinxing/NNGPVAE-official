from typing import Union
from pathlib import Path
import urllib.request
import numpy as np
import h5py

import torch
from torch import Tensor
from torch.utils.data import Dataset

from matplotlib import pyplot as plt


def download_healing_mnist(data_dir='../data/healing_mnist', random_mechanism="mnar"):
    """
    original code is from https://github.com/ratschlab/GP-VAE/blob/master/data/load_hmnist.sh;
    transformed it into python; The keys of .npz file are
    ['x_train_full',' y_train', 'x_test_full','y_test',              # (60000 / 10000, 10, 784)
     'x_train_miss', 'm_train_miss', 'x_test_miss', 'm_test_miss']
    The unobserved pixels are masked by `1`. dtype is `np.float32`
    """
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Define the URLs for different random mechanisms
    urls = {
        "mnar": "https://www.dropbox.com/s/xzhelx89bzpkkvq/hmnist_mnar.npz?dl=1",
        "spatial": "https://www.dropbox.com/s/jiix44usv7ibv1z/hmnist_spatial.npz?dl=1",
        "random": "https://www.dropbox.com/s/7s5y70f4idw9nei/hmnist_random.npz?dl=1",
        "temporal_neg": "https://www.dropbox.com/s/fnqi4rv9wtt2hqo/hmnist_temporal_neg.npz?dl=1",
        "temporal_pos": "https://www.dropbox.com/s/tae3rdm9ouaicfb/hmnist_temporal_pos.npz?dl=1"
    }

    # Get the URL for the selected random mechanism
    url = urls.get(random_mechanism)
    if not url:
        raise ValueError(f"Unknown random mechanism: {random_mechanism}")
    # Define the output file path
    file_path = path / f"hmnist_{random_mechanism}.npz"
    if file_path.exists():
        print(f"File already exists, skipping download.\n")
    else:  # Download the file
        print(f"===== Start Downloading hmnist_{random_mechanism}.npz =====")
        urllib.request.urlretrieve(url, file_path)
        print(f"===== File has downloaded to {file_path} =====")


class LazySeriesDataset(Dataset):
    """
    Given an extremely large set of N series [N, T, H, W] with length T, instead of putting the entire data to memory,
    use lazy loading when getting mini-batch series [V, T, H, W], V << N
    """
    def __init__(
            self, file_path: Union[str, Path], train: bool,
            val_split: int = None, return_full: bool = False
    ):  # training or test set
        super(LazySeriesDataset, self).__init__()
        self.file_path = file_path if isinstance(file_path, Path) else Path(file_path)

        if self.file_path.suffix == ".npz":
            self.data_dict = np.load(file_path, mmap_mode='r')   # use memory-mapping
        elif self.file_path.suffix == ".h5":
            self.data_dict = h5py.File(file_path, 'r')
        else:
            raise ValueError(f"Unsupported file type: {self.file_path.suffix}")
        self.len_train = self.data_dict['x_train_miss'].shape[0]
        self.len_test = self.data_dict['x_test_miss'].shape[0]

        self.train = train
        self.val_split = val_split
        if val_split is not None:
            if train:
                assert val_split <= self.len_train, f"Got wrong val_split {val_split} in training set."
            else:
                assert val_split <= self.len_test, f"Got wrong val_split {val_split} in test set."
        self.return_full = return_full

    def __getitem__(self, idx):
        if self.train:
            y_miss = self.data_dict['x_train_miss'][idx]
            mask = self.data_dict['m_train_miss'][idx]
        else:
            y_miss = self.data_dict['x_test_miss'][idx]
            mask = self.data_dict['m_test_miss'][idx]
        if not self.return_full:
            return torch.from_numpy(y_miss), torch.from_numpy(mask)
        y_full = self.data_dict['x_train_full'][idx] if self.train else self.data_dict['x_test_full'][idx]
        return torch.from_numpy(y_miss), torch.from_numpy(mask), torch.from_numpy(y_full)  # default np.float32

    def __len__(self):  # by setting dataset length, we can control the random idx range the dataloader can generate
        if self.train:
            return self.len_train if self.val_split is None else self.val_split
        else:
            return self.len_test if self.val_split is None else self.val_split


def generate_miniseries_hmnist_dict(Y_batch: Tensor, mask_batch: Tensor, Y_batch_full: Tensor = None):
    """
    Y: [v,T,784], v can be the number of miniseries
    Generate a dict {'images': [v,T,28,28], 'aux_data': [T,1], 'masks':[v,T,28,28], 'images_full': [v,T,28,28]};
    """
    v, T = Y_batch.shape[-3], Y_batch.shape[-2]
    t = torch.arange(0, T, dtype=Y_batch.dtype)   # regular timestamps
    data_dict = {   # reshape tensors in GPU maybe for faster computation ?
        'images': Y_batch.reshape(v, T, 28, 28),
        'masks': mask_batch.reshape(v, T, 28, 28),
        'aux_data': t.reshape(T, 1)   # We don't give an extra series shape for efficient NN search
    }
    # print(data_dict['aux_data'].is_contiguous())  # True
    if Y_batch_full is not None:
        data_dict['images_full'] = Y_batch_full.reshape(v, T, 28, 28)
    return data_dict


def plot_hmnist(y_miss: np.ndarray, y_rec: np.ndarray, y_full: np.ndarray):   # [n_seqs,10,28,28]
    num_seqs, num_steps = y_miss.shape[0], y_miss.shape[1]
    fig, axs = plt.subplots(
        nrows=3*num_seqs, ncols=num_steps, sharex=True, sharey=True, figsize=(3*num_steps, 3*3*num_seqs), constrained_layout=True
    )
    for i in range(num_seqs):
        for j in range(num_steps):
            axs[3*i, j].imshow(y_miss[i, j], cmap='gray')
            axs[3*i+1, j].imshow(y_rec[i, j], cmap='gray')
            axs[3*i+2, j].imshow(y_full[i, j], cmap='gray')
            axs[3*i, j].axis('off')
            axs[3*i+1, j].axis('off')
            axs[3*i+2, j].axis('off')
    return fig, axs

