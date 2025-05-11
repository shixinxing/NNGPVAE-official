# A neater version adapted from https://github.com/cornellius-gp/gpytorch/blob/main/gpytorch/utils/nearest_neighbors.py
# Provide customized implementation, also utilize faiss: https://github.com/facebookresearch/faiss/wiki/Getting-started
# TODO: simply use `topk`

import warnings
import numpy as np
import torch
from torch import Tensor


class NNUtil_GPVAE(torch.nn.Module):
    """
    database contains 'anchored' vectors from which we search nearest neighbors, if it :
    (1) [..., B, N, D], i.e, has batch shape, then query vectors should have the same batch shape;
    (2) [N, D], i.e., does not have batch shape, then query vectors can have any batch dimension.
    """
    def __init__(
            self, k, dim, anchor_batch_shape=torch.Size([]), preferred_nnlib='faiss',
            metric='L2', search_device: str = 'cpu'
    ):
        """
        preferred_nnlib can be 'faiss', 'sklearn', 'customized'
        metric can be 'L2' (Euclidean), 'L1' (Manhattan), 'Cosine', ...
        """
        super(NNUtil_GPVAE, self).__init__()
        assert k > 0, f'k must be greater than 0, but got {k}.'
        assert isinstance(anchor_batch_shape, torch.Size), f'The anchor_batch_shape must be torch.Size.'
        assert search_device in ('cpu', 'cuda'), f'search_device must be either cpu or cuda, but got {search_device}.'
        self.k, self.D, self.anchor_batch_shape, self.N = k, dim, anchor_batch_shape, None
        self.nnlib, self.metric = preferred_nnlib, metric
        self.search_device = search_device

        if preferred_nnlib == 'faiss':
            import faiss
            import faiss.contrib.torch_utils   # allow and return torch.Tensor in faiss

            self.num_index = anchor_batch_shape.numel()  # torch.Size([]).numel() = 1
            # see metric types in https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
            if search_device == 'cpu':
                if metric == 'L2':
                    self.index = [faiss.IndexFlatL2(self.D) for _ in range(self.num_index)]
                else:  # TODO
                    raise NotImplementedError
            elif search_device == 'cuda':
                res = faiss.StandardGpuResources()
                if metric == 'L2':
                    self.index = [faiss.GpuIndexFlatL2(res, self.D) for _ in range(self.num_index)]
                else:  # TODO
                    raise NotImplementedError

        elif preferred_nnlib == 'sklearn':
            if search_device != 'cpu':
                warnings.warn(f"Using sklearn need to transform Tensors to np.ndarray, it can only be done on CPU.")
                self.search_device = 'cpu'
            self.num_index = anchor_batch_shape.numel() if len(anchor_batch_shape) != 0 else 1
            self.train_neighbors = []

        elif preferred_nnlib == 'customized':  # TODO
            if metric not in ('L2', 'L1'):
                raise NotImplementedError
            self.anchor_x = None

        else:
            raise ValueError(f"Invalid preferred_nnlib option: {preferred_nnlib}")

    def _check_query_shape(self, query_x: Tensor):
        assert len(query_x.size()) >= 2, 'query_x must have at least 2 dimensions.'
        if len(self.anchor_batch_shape) != 0:
            assert query_x.shape[:-2] == self.anchor_batch_shape, (
                f"The database has batch dims, "
                f"so queries' batch shape must be equal to self.batch_shape: {self.anchor_batch_shape}, "
                f"but got batch shape {query_x.shape[:-2]}."
            )
        return query_x

    @torch.no_grad()
    def build_sequential_nn_idx(self, anchor_x: Tensor, faiss_reset=False):
        """
        anchor x: [..., (B), N, D], return structure: [..., N-k, k]. This method also utilizes the Index in `__init__`.
        Usually, if this method is called before `set_nn_idx` method, and the anchored database is not changed,
        we don't need to reset Index in `set_nn_idx` method.
        """
        assert anchor_x.shape[:-2] == self.anchor_batch_shape
        N = anchor_x.size(-2)

        if self.nnlib == 'faiss':
            anchor_x = anchor_x.to(self.search_device).reshape(self.num_index, anchor_x.size(-2), self.D)
            seq_nn_idx = torch.empty(
                self.num_index, N - self.k, self.k, dtype=torch.int64, device=self.search_device
            )
            for i in range(self.num_index):
                self.index[i].reset()
                self.index[i].add(anchor_x[i, :self.k])
                for r in range(self.k, N):
                    row = anchor_x[i, r:r+1]
                    seq_nn_idx[i, r-self.k] = self.index[i].search(row, self.k)[1]
                    self.index[i].add(row)
            if faiss_reset:  # clear Index
                for idx in self.index:
                    idx.reset()
            seq_nn_idx = seq_nn_idx.reshape(*self.anchor_batch_shape, N - self.k, self.k)
            return seq_nn_idx

        elif self.nnlib == 'sklearn':
            from sklearn.neighbors import NearestNeighbors
            if self.metric == 'L2':
                metric = 'euclidean'
            elif self.metric == 'L1':
                raise NotImplementedError

            anchor_x = anchor_x.cpu().numpy()
            anchor_x = anchor_x.reshape(self.num_index, anchor_x.shape[-2], self.D)
            seq_nn_idx = np.empty([self.num_index, N - self.k, self.k], dtype=int)
            for i in range(self.num_index):
                for r in range(self.k, N):
                    train_neighbor = NearestNeighbors(
                        n_neighbors=self.k, metric=metric, algorithm='auto'
                    ).fit(anchor_x[i, :r])
                    seq_nn_idx[i, r-self.k] = train_neighbor.kneighbors(anchor_x[i, r:r+1])[1]
            seq_nn_idx = torch.from_numpy(seq_nn_idx).long()
            return seq_nn_idx

        elif self.nnlib == 'customized':
            raise NotImplementedError

    @torch.no_grad()
    def set_nn_idx(self, anchor_x: Tensor = None):   # [..., N, D]
        """
        If we are using faiss, we may save set-up time by not resetting faiss Index
        after we use `build_sequential_nn_idx` and decide not to change the anchor vectors (i.e., anchor=None)
        To use faiss, added inputs have to be contiguous
        """
        if anchor_x is None:
            assert self.nnlib == 'faiss', f"Only Using faiss allow anchor_x=None in this method."
            assert self.index[0].ntotal != 0, f"The Index is empty. Please provide the database."
        else:
            assert anchor_x.shape[:-2] == self.anchor_batch_shape

        if self.nnlib == 'faiss':
            if anchor_x is not None:  # clear the Index
                anchor_x = anchor_x.to(self.search_device).reshape(self.num_index, anchor_x.size(-2), self.D)
                for i in range(self.num_index):
                    self.index[i].reset()
                    self.index[i].add(anchor_x[i])
            self.N = self.index[0].ntotal

        elif self.nnlib == 'sklearn':
            from sklearn.neighbors import NearestNeighbors

            if self.metric == 'L2':
                metric = 'euclidean'
            else:
                raise NotImplementedError

            anchor_x = anchor_x.reshape(-1, anchor_x.size(-2), self.D).cpu().numpy()
            for i in range(self.num_index):
                self.train_neighbors.append(   # set up
                    NearestNeighbors(n_neighbors=self.k, metric=metric, algorithm='auto').fit(anchor_x[i])
                )
            self.N = anchor_x.shape[-2]

        elif self.nnlib == 'customized':  # TODO
            assert anchor_x.shape[:-2] == self.anchor_batch_shape
            self.N = anchor_x.size(-2)
            self.anchor_x = anchor_x.to(self.search_device)
            raise NotImplementedError

    @torch.no_grad()
    def find_nn_idx(self, query_x: Tensor, k=None):
        """query: [..., n, D], return nn indices: [..., n, k]"""
        k = self.k if k is None else k
        assert k > 0, f'k must be greater than 0, but got {k}.'
        assert k <= self.N, f'k must be smaller than N: {self.N}, but got {k}.'

        query_x = self._check_query_shape(query_x)
        n = query_x.size(-2)

        if self.nnlib == 'faiss':
            query_x = query_x.to(self.search_device)
            query_batch_shape = query_x.shape[:-2]
            if self.num_index == 1:  # compare with only one batch of anchor_x
                query_x = query_x.reshape(-1, self.D)
                nn_idx = self.index[0].search(query_x, k)[1].reshape(*query_batch_shape, n, k)
            else:    # each batch has an Index for nn searching
                nn_idx = torch.empty(self.num_index, n, k, dtype=torch.int64, device=self.search_device)
                query_x = query_x.reshape(self.num_index, n, self.D)
                for i in range(self.num_index):
                    nn_idx[i] = self.index[i].search(query_x[i], k)[1]
                nn_idx = nn_idx.reshape(*self.anchor_batch_shape, n, k)

            return nn_idx

        if self.nnlib == 'sklearn':
            query_x = query_x.cpu().numpy()
            if self.num_index == 1:
                query_batch_shape = query_x.shape[:-2]
                query_x = query_x.reshape(-1, self.D)
                nn_idx = self.train_neighbors[0].kneighbors(query_x, n_neighbors=k)[1].reshape(*query_batch_shape, n, k)
            else:
                nn_idx = np.empty([self.num_index, n, k], dtype=int)
                query_x = query_x.reshape(self.num_index, n, self.D)
                for i in range(self.num_index):
                    nn_idx[i] = self.train_neighbors[i].kneighbors(query_x[i])[1][:, :k]

            nn_idx = torch.from_numpy(nn_idx).long()  # torch.int64
            return nn_idx

        if self.nnlib == 'customized':  # TODO cdist, pdist, topk, PairwiseDistance?
            raise NotImplementedError


