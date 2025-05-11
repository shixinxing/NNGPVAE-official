# To design the encoder/decoder structure for long healing MNIST,
# we follow the appendix B.1 of MGPVAE (ICML 2023) (i.e., Page 19) to design similar structures
# (The paper gives an encoder/decoder producing images with shape [32,32] rather than [28,28])

import torch
from torch import nn, Tensor


class LongHMNISTEncoderDiagonal(nn.Module):
    def __init__(self, latent_dims=16, hidden_dims: tuple = ()):
        super(LongHMNISTEncoderDiagonal, self).__init__()
        self.latent_dims = latent_dims

        # CNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),   # [...,32,14,14]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # [...,64,7,7]
            nn.ReLU(),
            nn.Flatten(start_dim=-3, end_dim=-1)                                             # [...,64*7*7=1568]
        )

        # FCNN
        layer_dims = [64 * 7 * 7, *hidden_dims, 2 * latent_dims]
        self.fnn = nn.Sequential()
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            self.fnn.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.fnn.append(nn.ReLU())

    def forward(self, vid_batch: Tensor):
        """ vid_batch: [v, f, 28, 28] -> [v, f, L] or [v, n, H, 28, 28] -> [v, n, H, L]"""
        vid_batch_reshaped = vid_batch.reshape(-1, 1, *vid_batch.shape[-2:])  # Conv2D only accepts 4-D inputs
        x = self.conv_layers(vid_batch_reshaped)
        x = self.fnn(x.reshape(*vid_batch.shape[:-2], 64 * 7 * 7))
        q_means = x[..., :self.latent_dims]
        q_stds = nn.functional.softplus(x[..., self.latent_dims:])
        # q_stds = torch.exp(0.5 * x[..., self.latent_dims:])
        return q_means, q_stds


class LongHMNISTDecoder(nn.Module):
    def __init__(self, latent_dims=16, hidden_dims: tuple = ()):
        super(LongHMNISTDecoder, self).__init__()
        self.output_distribution = 'bernoulli'
        self.latent_dims = latent_dims

        # FCNN
        layer_dims = [latent_dims, *hidden_dims, 32 * 7 * 7]
        self.fnn = nn.Sequential()
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            self.fnn.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.fnn.append(nn.ReLU())

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1  # [64, 14, 14]
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1  # [32, 28, 28]
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0   # [1, 28, 28]
            )
        )

    def forward(self, z: Tensor):
        x = self.fnn(z)
        x = x.reshape(-1, 32, 7, 7)
        logits = self.deconv_layers(x)  # [-1,32,7,7] -> [-1,32,28,28]
        logits = logits.reshape(*z.shape[:-1], 28, 28)
        return logits


if __name__ == '__main__':
    img = torch.randn([32, 10, 5, 28, 28])
    L = 4

    enc = LongHMNISTEncoderDiagonal(latent_dims=L)
    print(f"The output of encoder mean: {enc(img)[0].shape}, std: {enc(img)[1].shape}")

    dec = LongHMNISTDecoder(latent_dims=L)
    print(f"The output of decoder probs: {dec(enc(img)[0]).shape}")






