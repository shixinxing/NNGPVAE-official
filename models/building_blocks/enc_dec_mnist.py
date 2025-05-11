# adapt from https://github.com/ratschlab/SVGP-VAE/blob/main/VAE_utils.py#L99
import torch
from torch import Tensor
import torch.nn as nn


class MNISTEncoder(nn.Module):
    """
    NIPS 2018 paper, Figure S1, q(z|y)
    """
    def __init__(self, latent_dims=16, kernel_size=3):
        super(MNISTEncoder, self).__init__()
        self.latent_dims = latent_dims

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=kernel_size, stride=2, padding=1),  # [8,14,14]
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, stride=2, padding=1),  # [8,7,7]
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, stride=2, padding=1),  # [8,4,4]
            nn.ELU(),
            nn.Flatten()
        )
        self.dense_zm = nn.Linear(in_features=8 * 4 * 4, out_features=self.latent_dims)
        self.dense_zs = nn.Linear(in_features=8 * 4 * 4, out_features=self.latent_dims)
        self.softplus = nn.Softplus(beta=1, threshold=20)

    def forward(self, imgs: Tensor):
        before_dense = self.encoder(imgs)
        zm = self.dense_zm(before_dense)
        zs = self.softplus(self.dense_zs(before_dense))  # std
        return zm, zs


class MNISTDecoder(nn.Module):
    """
    NIPS 2018 paper, Figure S1
    """
    def __init__(self, latent_dims=16, kernel_size=3, sigma2_y=0.1, out_dist='normal'):
        super(MNISTDecoder, self).__init__()
        if not (out_dist == 'normal' or out_dist == 'bernoulli'):
            raise NotImplementedError("Invalid out distribution type")
        self.output_distribution = out_dist

        self.latent_dims = latent_dims
        if out_dist == 'normal':
            self.register_buffer('sigma2_y', torch.tensor(sigma2_y))  # what's the appropriate value?
            # self.register_parameter('sigma2_y', nn.Parameter(torch.tensor(sigma2_y), requires_grad=True))

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dims, out_features=8 * 4 * 4),
            nn.Unflatten(1, (8, 4, 4)),
            nn.Upsample(scale_factor=2),
            # different from S1, use `decoder` in https://github.com/ratschlab/SVGP-VAE/blob/main/VAE_utils.py#L99
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, stride=1, padding='same'),  # [8,8,8]
            nn.ELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, stride=1),  # [8,14,14]
            nn.ELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=kernel_size, stride=1, padding='same'),  # [1,28,28]
            nn.Sigmoid()
        )

    def forward(self, z: Tensor):
        return self.decoder(z)


if __name__ == "__main__":
    enc = MNISTEncoder()
    x = torch.randn(10, 1, 28, 28)
    print(enc(x)[0].shape, enc(x)[1].shape)
    # x = torch.randn(7, 10, 1, 28, 28)  # not support 2 batch dims
    # print(enc(x)[0].shape, enc(x)[1].shape)

    dec = MNISTDecoder()
    z = torch.randn(10, 16)
    print(dec(z).shape, '\n')

