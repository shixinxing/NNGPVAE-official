import numpy as np
import torch
from torch import nn
from torch import Tensor


class BallEncoder(nn.Module):
    """
    Pearce's AABI 2019 paper, for structure details, see Table A.1 from the AISTATS 2021 paper
    or https://github.com/scrambledpie/GPVAE/blob/master/GPVAEmodel.py#L60
    """
    def __init__(self, img_size: int = 32, hidden_dims: tuple = (500,)):
        super(BallEncoder, self).__init__()

        layer_dims = [img_size ** 2]
        for dim in hidden_dims:
            layer_dims.append(dim)
        layer_dims.append(4)

        self.encoder = nn.Sequential()
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            # use the same initialization as Pearce's model
            std = 1. / np.sqrt(layer_dims[i])
            nn.init.trunc_normal_(linear_layer.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(linear_layer.bias)
            self.encoder.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.encoder.append(nn.Tanh())

    def forward(self, vid_batch: Tensor):
        """
        :param vid_batch: [v, f, 32, 32] -> [v, f, 32*32]
        :return: means and stds with shape [v, f, 2]
        """
        px, py = vid_batch.shape[-2], vid_batch.shape[-1]
        x = vid_batch.reshape(*vid_batch.shape[:-2], px * py)
        x = self.encoder(x)
        q_means = x[..., :2]
        q_vars = torch.exp(x[..., 2:])  # The AABI paper uses vars rather than stds
        q_stds = torch.sqrt(q_vars)     # to remain consistent with the paper's design
        return q_means, q_stds


class BallDecoder(nn.Module):
    """
    A Fully connected decoder that goes from 2-D latent space to a Bernoulli distribution for each pixel
    or refer to https://github.com/scrambledpie/GPVAE/blob/master/GPVAEmodel.py#L104
    """
    def __init__(self, img_size: int = 32, hidden_dims: tuple = (500,)):
        super(BallDecoder, self).__init__()
        self.output_distribution = 'bernoulli'
        self.img_size = img_size

        layer_dims = [2]
        layer_dims.extend([dim for dim in hidden_dims])
        layer_dims.append(img_size ** 2)

        self.decoder = nn.Sequential()
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            # use the same initialization as Pearce's model
            std = 1. / np.sqrt(layer_dims[i])
            nn.init.trunc_normal_(linear_layer.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(linear_layer.bias)
            self.decoder.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.decoder.append(nn.Tanh())
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent_samples: Tensor):
        """
        :param latent_samples: [v, f, 2]
        :return: params of Bernoulli distributions with shape [v, f, 32, 32]
        """
        logits = self.decoder(latent_samples)
        out = self.sigmoid(logits).reshape(*logits.shape[:-1], self.img_size, self.img_size)
        return out


if __name__ == '__main__':
    enc = BallEncoder(img_size=28, hidden_dims=(50, 20))
    x = torch.randn(2, 3, 28, 28)
    print(enc(x)[0].shape, enc(x)[1].shape, '\n')

    dec = BallDecoder(img_size=28, hidden_dims=(25, 30))
    z = torch.randn(6, 10, 2)
    print(dec(z).shape, '\n')
