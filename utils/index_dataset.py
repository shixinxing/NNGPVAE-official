# produce mini-batches of indices for large matrices stored as h5
import numpy as np
from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(self, N: int):
        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return idx


def get_submatrix_from_h5(file_mat, idx_0, idx_1):
    """given coordinates, extract submatrix from h5 file """
    yy, xx = len(idx_0), len(idx_1)
    res = np.empty((yy, xx, *file_mat.shape[2:]))
    for i in range(yy):
        for j in range(xx):
            res[i, j] = file_mat[idx_0[i], idx_1[j]]

    return res


