import torch
from torch import Tensor, nn
from gpytorch.utils.transforms import inv_softplus

class MLP(nn.Module):
    """
    adapted from https://github.com/tranbahien/sgp-bae/blob/master/nn/nets/mlp.py#L10,
    Used as component for BAE Jura.
    """
    def __init__(self, input_size, output_size, hidden_units):
        super(MLP, self).__init__()

        layers = []

        if len(hidden_units) > 1:
            for in_size, out_size in zip([input_size] + hidden_units[:-1], hidden_units):
                layers.append(nn.Linear(in_size, out_size))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hidden_units[-1], output_size))
        else:
            layers.append(nn.Linear(input_size, hidden_units[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_units[0], output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        return self.layers(X)

class JuraEncoder(nn.Module):
    # ICML 2023 <Fully Bayesian Autoencoders with Latent Sparse Gaussian Processes> Table 8
    def __init__(self, y_dim: int = 3, hidden_dims: tuple = (20,), num_latents: int = 2):
        super(JuraEncoder, self).__init__()

        self.num_latents = num_latents
        layer_dims = [y_dim]
        for dim in hidden_dims:
            layer_dims.append(dim)
        layer_dims.append(num_latents * 2)  # first half for mean, last half for std

        self.encoder = nn.Sequential()
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            self.encoder.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.encoder.append(nn.ReLU())

    def forward(self, y_batch: Tensor):
        """
        :param y_batch: [(n), b, y_dim]
        :return means and stds of shape [(n), b, num_latents]
        """
        x = self.encoder(y_batch)
        q_means = x[..., :self.num_latents]
        q_stds = nn.functional.softplus(x[..., self.num_latents:])

        return q_means, q_stds

class JuraDecoder(nn.Module):
    # ICML 2023 <Fully Bayesian Autoencoders with Latent Sparse Gaussian Processes> Table 8
    def __init__(self, y_dim: int = 3, hidden_dims: tuple = (5, 5), num_latents: int = 2, sigma2_y: float = 0.25,
                 fix_variance: bool = True):
        super(JuraDecoder, self).__init__()
        self.output_distribution = 'normal'
        self.fix_variance = fix_variance
        self.num_latents = num_latents

        if fix_variance:
            self.register_buffer('_sigma2_y', torch.tensor(sigma2_y))
        else:
            self.register_parameter(
                'raw_sigma2_y', nn.Parameter(inv_softplus(torch.tensor(sigma2_y)), requires_grad=True))
        layer_dims = [num_latents]
        for dim in hidden_dims:
            layer_dims.append(dim)
        layer_dims.append(y_dim)

        self.decoder = nn.Sequential()
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            self.decoder.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.decoder.append(nn.ReLU())

    @property
    def sigma2_y(self):
        if self.fix_variance:
            return self._sigma2_y
        else:
            return nn.functional.softplus(self.raw_sigma2_y)

    def forward(self, z: Tensor):
        return self.decoder(z)


if __name__ == '__main__':
    enc = JuraEncoder()
    Y = torch.randn([10, 359, 5, 3])  # H=5
    z_mean, z_stds = enc(Y)
    print(z_mean.shape, z_stds.shape)

    dec = JuraDecoder()
    Z = torch.randn([10, 359, 5, 2])
    Y = dec(Z)
    print(Y.shape)







