import torch
from torch import Tensor, nn
from gpytorch.utils.transforms import inv_softplus


class MujocoEncoder(nn.Module):
    """
    Follows MGPVAE's encoder https://github.com/harrisonzhu508/MGPVAE/blob/main/example_mujoco.ipynb
    """
    def __init__(
            self, y_dim: int = 14, hidden_dims: tuple = (32,), num_latents: int = 15,
            natural: bool = True, init="Xavier_normal"
    ):  # choose from ['Xavier_normal', 'Xavier_uniform', 'kaiming']
        super(MujocoEncoder, self).__init__()

        self.num_latents = num_latents
        self.natural = natural
        self.Softplus = torch.nn.Softplus()
        layer_dims = [y_dim]
        for dim in hidden_dims:
            layer_dims.append(dim)
        layer_dims.append(num_latents * 2)  # first half for mean, last half for std

        self.encoder = nn.Sequential()
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])

            if init == "Xavier_normal":
                nn.init.xavier_normal_(linear_layer.weight)
                nn.init.zeros_(linear_layer.bias)
            elif init == "Xavier_uniform":
                nn.init.xavier_uniform_(linear_layer.weight)
                nn.init.zeros_(linear_layer.bias)
            elif init == "kaiming":
                pass
            else:
                raise NotImplementedError(
                    f"{init} is not yet implemented. Please choose from Xavier_normal, Xavier_uniform and kaiming.")

            self.encoder.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.encoder.append(nn.ReLU())

    def forward(self, y_batch: Tensor):
        """
        :param y_batch: [(n), t, 14/20]
        return means and stds with shape [(n), t, 15]
        """
        if not self.natural:
            # mean-std parametrisation
            x = self.encoder(y_batch)
            q_means = x[..., :self.num_latents]
            q_stds = self.Softplus(x[..., self.num_latents:])

        else:
            # natural parametrisation
            x = self.encoder(y_batch)
            lambda1 = x[..., :self.num_latents]
            lambda2_inv = -self.Softplus(x[..., self.num_latents:])  # numerical concern?
            # q_means = -0.5 * lambda1 * lambda2_inv
            q_means = lambda1 * lambda2_inv  # follows https://github.com/harrisonzhu508/MGPVAE/blob/main/mgpvae/model.py#L130
            q_stds = torch.sqrt(-0.5 * lambda2_inv)

        return q_means, q_stds


class MujocoDecoder(nn.Module):
    """
    Follows MGPVAE's encoder https://github.com/harrisonzhu508/MGPVAE/blob/main/example_mujoco.ipynb
    """
    def __init__(
            self, y_dim: int = 14, hidden_dims: tuple = (16,), num_latents: int = 15, sigma2_y: float = 0.2,
            fix_variance: bool = True, init="Xavier_normal"
    ):  # choose from ['Xavier_normal', 'Xavier_uniform', 'kaiming']
        super(MujocoDecoder, self).__init__()
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

            if init == "Xavier_normal":
                nn.init.xavier_normal_(linear_layer.weight)
                nn.init.zeros_(linear_layer.bias)
            elif init == "Xavier_uniform":
                nn.init.xavier_uniform_(linear_layer.weight)
                nn.init.zeros_(linear_layer.bias)
            elif init == "kaiming":
                pass
            else:
                raise NotImplementedError(
                    f"{init} is not yet implemented. Please choose from Xavier_normal, Xavier_uniform and kaiming.")

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
    enc = MujocoEncoder()
    Y = torch.randn(3, 50, 4, 14)  # H=4
    z_mean, z_std = enc(Y)
    print(z_mean.shape, z_std.shape)

    dec = MujocoDecoder()
    Z = torch.randn(3, 50, 4, 15)
    Y = dec(Z)
    print(Y.shape)

