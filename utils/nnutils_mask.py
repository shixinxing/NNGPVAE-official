import warnings
from typing import Union, Iterable
import torch
from torch import Tensor, LongTensor


class NNUtilMasked_GPVAE(torch.nn.Module):
    """
    database contains `anchored` video timestamps of varying lengths [v, T, ...].
    The unobserved frames are indicated by masks. There should be one "search engine" for each video timestamps.
    """
    def __init__(
            self, k: int, vid_timestamps: Tensor, time_masks: Tensor,
            preferred_nnlib='faiss', metric='L2', search_device='cpu'
    ):
        """
        `vid_timestamps` [v, T, 1] with observations moved to the left. `time_masks` [v, T, 1] indicate the boundary.
        """
        super(NNUtilMasked_GPVAE, self).__init__()
        assert k > 0, f'k must be greater than 0, but got {k}.'
        if vid_timestamps.ndim != 3:
            raise ValueError(f"vid_timestamps must have dimension 3, but got ndim {vid_timestamps.ndim}.")
        self.k = k
        self.num_videos, self.num_frames, self.D = vid_timestamps.shape

        self.nnlib, self.metric = preferred_nnlib, metric
        self.search_device = search_device

        if preferred_nnlib == 'faiss':
            import faiss
            import faiss.contrib.torch_utils

            if search_device == 'cpu':
                if metric == 'L2':
                    self.index = [faiss.IndexFlatL2(self.D) for _ in range(self.num_videos)]
                else:  # TODO
                    raise NotImplementedError
            elif search_device == 'cuda':
                res = faiss.StandardGpuResources()
                if metric == 'L2':
                    self.index = [faiss.GpuIndexFlatL2(res, self.D) for _ in range(self.num_videos)]
                else:  # TODO
                    raise NotImplementedError
            else:
                raise ValueError(f'search_device must be either "cpu" or "cuda", but got {search_device}.')

        elif preferred_nnlib == 'sklearn':
            raise NotImplementedError
            # if search_device != 'cpu':
            #     warnings.warn(f"Using sklearn need to transform Tensors to np.ndarray, it can only be done on CPU.")
            #     self.search_device = 'cpu'
            # self.train_neighbors = []

        else:
            raise NotImplementedError(f"{preferred_nnlib} is not supported.")

        self.vid_timestamps = vid_timestamps.to(self.search_device)  # [v,T,1]
        self.time_masks = time_masks.to(device=self.search_device, dtype=torch.bool)
        self.seq_length = self.time_masks.sum(dim=(-1, -2))  # [v]
        assert k <= self.seq_length.min(), f"k {k} must be smaller than the minimum seq_length {self.seq_length.min()}."

    @torch.no_grad()
    def build_sequential_nn_idx(self, faiss_reset=False, padding=True):
        """
        loop over the video dim to find the NN structure for each series.
        For a series of length T', its NN structure is [T'-k,k]. Fill in zero to make it a tensor of size [v,Tmax-k,k]
        """
        if self.nnlib == 'faiss':
            seq_nn_idx = [torch.empty(
                [self.seq_length[i] - self.k, self.k], dtype=torch.int64, device=self.search_device
            ) for i in range(self.num_videos)]  # can be shape like [0, k]
            for i in range(self.num_videos):
                self.index[i].reset()
                anchor_t = self.vid_timestamps[i, self.time_masks[i, :, 0]]  # [T', 1]
                self.index[i].add(anchor_t[:self.k])
                for r in range(self.k, self.seq_length[i]):  # Only the observed timestamps are put into database
                    row = anchor_t[r:r+1]
                    seq_nn_idx[i][r-self.k] = self.index[i].search(row, self.k)[1]
                    self.index[i].add(row)
            if faiss_reset:  # after getting seq_n_idx, clear Index
                for idx in self.index:
                    idx.reset()

            if not padding:
                return seq_nn_idx   # [v][T'-k, k]
            else:
                sq_len_max = self.seq_length.max()
                seq_nn_idx_pad = torch.zeros(
                    [self.num_videos, sq_len_max-self.k, self.k], dtype=torch.int64, device=self.search_device
                )
                for i in range(self.num_videos):
                    seq_nn_idx_pad[i, :seq_nn_idx[i].size(-2), :seq_nn_idx[i].size(-1)] = seq_nn_idx[i]
                return seq_nn_idx_pad
        else:
            raise NotImplementedError("nnlibs other than faiss are not supported.")

    @torch.no_grad()
    def set_nn_idx(self, faiss_reset=False):
        """
        after we use `build_sequential_nn_idx` and decide not to change the anchor vectors (i.e., `faiss_reset`=False).
        Otherwise, we will reset the Index using timestamps (e.g., `faiss_reset`=True in SWS).
        """
        if not faiss_reset:
            assert self.nnlib == 'faiss', "Only `faiss` is supported now."
            assert self.index[-1].ntotal != 0, f"The index is empty. Please provide the database."
        else:   # build the database
            if self.nnlib == 'faiss':
                for i in range(self.num_videos):
                    self.index[i].reset()
                    anchor_t = self.vid_timestamps[i, self.time_masks[i, :, 0]]  # only put in observed time vectors
                    self.index[i].add(anchor_t)
            else:
                raise NotImplementedError

    @staticmethod
    def _check_query_shape(query_t: Tensor, vid_batch_size: int):  # [(v),n,D=1]
        assert query_t.ndim == 2 or query_t.ndim == 3, f'query_t must have 2 or 3 dimensions, but got {query_t.ndim}.'
        if query_t.ndim == 2:
            query_t = query_t.expand(vid_batch_size, -1, -1)
        elif query_t.ndim == 3:
            assert query_t.size(-3) == vid_batch_size, f"query's batch size must be {vid_batch_size}."
        return query_t

    @torch.no_grad()
    def find_nn_idx(self, query_t: Tensor, vid_indices=None, k=None):
        """query: [v, n, D=1], return nn indices: [(v), n, k], `vid_indices` indicate which Index should be used"""
        k = self.k if k is None else k
        assert k > 0, f'k must be greater than 0, but got {k}.'
        assert k <= self.seq_length.min(), f"k {k} must be smaller than the minimum seq_length {self.seq_length.min()}."

        vid_indices = range(self.num_videos) if vid_indices is None else vid_indices
        vid_batch_size, n = len(vid_indices), query_t.size(-2)
        assert vid_batch_size <= self.num_videos

        query_t = self._check_query_shape(query_t, vid_batch_size)  # [v, n, D=1]
        if self.nnlib == 'faiss':
            query_t = query_t.to(self.search_device)
            nn_idx = torch.empty(
                [vid_batch_size, n, k], dtype=torch.int64, device=self.search_device
            )
            for i, idx in enumerate(vid_indices):
                nn_idx[i] = self.index[idx].search(query_t[i], k)[1]
            return nn_idx
        else:
            raise NotImplementedError

    @torch.no_grad()
    def find_nn_idx_from_index_subsets(
            self, query_t: Tensor, index_subsets_ids: Union[Iterable[int], LongTensor], k: int = None
    ):
        k = self.k if k is None else k
        v = len(index_subsets_ids)
        query_t = query_t.to(self.search_device).expand(v, -1, -1)  # [v, n, D=1]
        nn_idx = torch.empty(
            [v, query_t.size(-2), k], dtype=torch.int64, device=self.search_device
        )
        for i in range(v):
            nn_idx[i] = self.index[index_subsets_ids[i]].search(query_t[i], k)[1]
        return nn_idx
