import torch
from torch import nn
from torch import Tensor


# based on class GPVAEBase
class VAEBase(nn.Module):
    def __init__(
            self, encoder, decoder, latent_dims: int, extra_data_batch_shape: torch.Size = torch.Size([])
    ):
        super(VAEBase, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims

        self.extra_data_batch_shape = extra_data_batch_shape

    def forward(self, y_batch: Tensor):
        means, stds = self.encoder(y_batch)
        eps = torch.randn_like(means)
        latent_samples = means + stds * eps
        recon_imgs = self.decoder(latent_samples)
        return means, stds, recon_imgs

    def expected_log_prob(self, y_batch: Tensor, **kwargs) -> Tensor:
        means, stds, y_rec = self.forward(y_batch)
        out_dist = self.decoder.output_distribution
        if out_dist == 'normal':
            sigma2_y = self.decoder.sigma2_y
            expected_lk = (y_batch - y_rec).square() / sigma2_y + torch.log(2 * torch.pi * sigma2_y)
            expected_lk = - 1 / 2 * expected_lk.sum()
        elif out_dist == 'bernoulli':
            expected_lk = - nn.functional.binary_cross_entropy(input=y_rec, target=y_batch, reduction='sum')
        else:
            raise NotImplementedError(f'Unrecognized output distribution {out_dist}.')

        return expected_lk

    def kl_divergence(self, y_batch: Tensor) -> Tensor:
        """KL is factorized across data points"""
        mean_q, std_q = self.encoder(y_batch)
        mahalanobis = mean_q.square().sum(dim=-1)
        trace = std_q.square().sum(dim=-1)
        log_det_cov_q = std_q.log().sum(dim=-1)

        res = 0.5 * (mahalanobis + trace - self.latent_dims) - log_det_cov_q
        # test
        # print(f"my kl divergence (shape: {res.shape}): {res.sum()}")
        # p = torch.distributions.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.))
        # q = torch.distributions.Normal(loc=mean_q, scale=std_q)
        # res_true = torch.distributions.kl.kl_divergence(q, p)
        # print(f"true kl divergence: {res_true.sum()}\n")
        return res

    def average_loss(self, y_batch: Tensor, beta=1., **kwargs) -> Tensor:
        scale = y_batch.shape[:len(self.extra_data_batch_shape) + 1].numel()  # v*b
        # mean loss
        return - (self.expected_log_prob(y_batch) - beta * self.kl_divergence(y_batch).sum()) / scale

    def train_vae(self, *args, **kwargs):
        raise NotImplementedError

