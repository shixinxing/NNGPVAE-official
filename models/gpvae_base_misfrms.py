from typing import Union, Iterable
import warnings

import torch
from torch import Tensor, LongTensor, nn

from models.gpvae_base import GPVAEBase


class GPVAEBaseMisFrms(GPVAEBase):
    """
    Base class for GPVAEs handling missing-frame cases. In this scenario, we have several videos [v,T,D...]
    of varying lengths due to different degrees of missingness on frames.
    Each video has observations on the left and unobserved (missing) frames on the right.
    Timestamp mask [v, T, 1] indicates the boundary.
    """
    def __init__(self, *args, **kwargs):
        super(GPVAEBaseMisFrms, self).__init__(*args, **kwargs)

        self.entire_train_dataset = None      # NN structure will be constructed in subclasses.
        self.train_dataset = None             # for mini_series
        self.num_frames_before_clip = None    # store T

    # override
    def get_nn_data(self, query_t: Tensor, H: int, vid_ids: Union[Iterable[int], LongTensor] = None, **kwargs):
        """
        extract H-nearest data of `query_t` [(v),n,1] from the stored clipped data, looping over video dim v.
        return (Y_near, X_near, mask_near) with shape [v,n,H,...]
        """
        nn_indices = self.entire_train_dataset.nnutil_all.find_nn_idx_from_index_subsets(query_t, vid_ids, k=H)
        return self.train_dataset.gather(nn_indices)

    def expected_log_prob(self, y_batch: Tensor, m_batch: Tensor = None):
        """
        y: [v, t<Tmax, D], m: [v, t<Tmax, 1], `True` in the mask represents observed, `
        """
        if m_batch is None:
            m_batch = torch.ones(
                y_batch.shape[:len(self.train_dataset.series_shape)+1], dtype=torch.bool, device=y_batch.device)
        else:
            m_batch = m_batch[..., 0].to(torch.bool)          # [v,t]
            if torch.all(~m_batch):
                warnings.warn("Got a batch without observed frames.")
                return torch.tensor(0., device=self.module_device)
            assert m_batch.shape[:-1] == self.train_dataset.series_shape

        # mask first and then feed to save computation
        y_pick = y_batch[m_batch]                             # [<v*t, D]
        means, stds, y_rec = self.forward(y_pick)

        if self.decoder.output_distribution == 'normal':
            sigma2_y = self.decoder.sigma2_y
            expected_lk = (y_pick - y_rec).square() / sigma2_y + torch.log(2 * torch.pi * sigma2_y)
            expected_lk = - 0.5 * expected_lk
        elif self.decoder.output_distribution == 'bernoulli':
            expected_lk = - nn.functional.binary_cross_entropy_with_logits(
                input=y_rec, target=y_pick, reduction='none'  # [<v*t, D]
            )
        else:
            raise NotImplementedError(f'Unrecognized output distribution {self.decoder.output_distribution}.')

        expected_lk = expected_lk.sum(dim=[i for i in range(1, expected_lk.ndim)])            # [<v*t]
        expected_lk_mat = torch.zeros_like(m_batch, requires_grad=True, dtype=y_batch.dtype)  # [v,t]
        expected_lk_mat = expected_lk_mat.clone()    # in-place operation not allowed by leaf variables
        expected_lk_mat[m_batch] = expected_lk       # assign

        # scaling of each video
        len_batch_inv = torch.where(m_batch.sum(dim=-1) > 0, 1/m_batch.sum(-1), 0.)  # [v] in case of no observations
        scale = self.train_dataset.masks.to(m_batch.device).sum(dim=(-1, -2)) * len_batch_inv   # [v]
        lik = torch.sum(expected_lk_mat * scale.unsqueeze(-1))
        return lik / self.train_dataset.series_shape.numel()  # average over v


