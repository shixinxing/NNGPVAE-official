import torch
from torch import nn, Tensor
from gpytorch.utils.transforms import inv_softplus


class SPE10Encoder(nn.Module):
    def __init__(self, latent_dims=3, hidden_dims: tuple = (256, 64), input_dims=4):
        super(SPE10Encoder, self).__init__()
        self.latent_dims = latent_dims

        layer_dims = [input_dims, *hidden_dims, 2*latent_dims]
        self.fnn = nn.Sequential()
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            self.fnn.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.fnn.append(nn.ReLU())

    def forward(self, y: Tensor):
        y = self.fnn(y)
        q_means = y[..., :self.latent_dims]
        q_stds = nn.functional.softplus(y[..., self.latent_dims:])
        return q_means, q_stds


class SPE10Decoder(nn.Module):
    def __init__(
            self, latent_dims=3, hidden_dims: tuple = (64, 256), output_dims=4,
            sigma2_y: float = 1., fix_variance: bool = True
    ):
        super(SPE10Decoder, self).__init__()
        self.output_distribution = 'normal'
        if fix_variance:
            self.register_buffer('sigma2_y', torch.tensor(sigma2_y))
        else:
            self.register_parameter(
                'raw_sigma2_y', nn.Parameter(inv_softplus(torch.tensor(sigma2_y)), requires_grad=True))
        self.latent_dims = latent_dims

        layer_dims = [latent_dims, *hidden_dims, output_dims]
        self.fnn = nn.Sequential()
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            self.fnn.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.fnn.append(nn.ReLU())

    def forward(self, z: Tensor):
        return self.fnn(z)


if __name__ == '__main__':
    enc = SPE10Encoder()
    x = torch.randn(15, 5, 4)
    print(enc(x)[0].shape, enc(x)[1].shape, '\n')

    dec = SPE10Decoder()
    zz = torch.randn(15, 5, 3)
    print(f"{dec(zz).shape}")


