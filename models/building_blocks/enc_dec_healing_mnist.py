from torch import Tensor
from torch import nn


class HMNISTEncoderDiagonal(nn.Module):
    """
    Same structure as https://github.com/ratschlab/GP-VAE/blob/master/lib/models.py#L16 from AISTATS 2020
    preprocessed by a CNN;
    mean-field, not joint for a complete series; not using 1-D Conv;
    """
    def __init__(self, latent_dims=256, hidden_dims: tuple = (256, 256), kernel_size=3):
        super(HMNISTEncoderDiagonal, self).__init__()
        self.latent_dims = latent_dims

        # CNN as preprocessor
        self.preprocessor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=kernel_size, padding='same'),
            nn.ReLU()
        )

        # following TensorFlow's default initialization (Xavier_uniform), which is different from that of PyTorch
        nn.init.xavier_uniform_(self.preprocessor[0].weight)
        nn.init.zeros_(self.preprocessor[0].bias)
        nn.init.xavier_uniform_(self.preprocessor[2].weight)
        nn.init.zeros_(self.preprocessor[2].bias)

        # ForwardNN
        layer_dims = [28 * 28, *hidden_dims, 2 * latent_dims]
        self.fnn = nn.Sequential()
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i+1])
            # similarly, using Xavier_uniform initialization
            nn.init.xavier_uniform_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)

            self.fnn.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.fnn.append(nn.ReLU())

    def forward(self, vid_batch: Tensor):
        """ vid_batch: [v, f, 28, 28] -> [v, f, 256] or [v, n, H, 28, 28] -> [v, n, H, 256]"""
        vid_batch_reshaped = vid_batch.reshape(-1, 1, *vid_batch.shape[-2:])  # Conv2D only accepts 4-D inputs
        x = self.preprocessor(vid_batch_reshaped).reshape(*vid_batch.shape[:-2], 28 * 28)
        x = self.fnn(x)
        q_means = x[..., :self.latent_dims]
        q_stds = nn.functional.softplus(x[..., self.latent_dims:])
        return q_means, q_stds


class HMNISTDecoder(nn.Module):
    """ AISTATS 2020 paper """
    def __init__(self, latent_dims=256, hidden_dims=(256, 256, 256)):
        super(HMNISTDecoder, self).__init__()
        self.output_distribution = 'bernoulli'
        self.latent_dims = latent_dims

        layer_dims = [latent_dims, *hidden_dims, 28 * 28]
        self.decoder = nn.Sequential()
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            # which initialization?
            # std = 1. / np.sqrt(layer_dims[i])
            # nn.init.trunc_normal_(linear_layer.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.xavier_uniform_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)

            self.decoder.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.decoder.append(nn.ReLU())

    def forward(self, latent_samples: Tensor):
        """ latent_samples: [v, f, 256] -> [v, f, 28, 28],
        we follow the official code to use logits instead of using sigmoid directly"""
        logits = self.decoder(latent_samples)
        logits = logits.reshape(*logits.shape[:-1], 28, 28)
        return logits


if __name__ == '__main__':
    import torch

    enc = HMNISTEncoderDiagonal()
    x = torch.randn(32, 10, 28, 28)
    print(enc(x)[0].shape, enc(x)[1].shape, '\n')
    x = torch.randn(32, 10, 5, 28, 28)
    print(enc(x)[0].shape, enc(x)[1].shape, '\n')

    dec = HMNISTDecoder(latent_dims=256)
    z = torch.randn(32, 10, 256)
    print(dec(z).shape, '\n')



