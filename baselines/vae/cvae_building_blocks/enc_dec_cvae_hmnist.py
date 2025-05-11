from torch import nn, Tensor
from models.building_blocks.enc_dec_longhmnist import LongHMNISTDecoder


class CVAELongHMNISTEncoderDiagonal(nn.Module):
    """
    same structure as other baselines, but does not use softplus at the encoder output
    """
    def __init__(self, latent_dims=16, hidden_dims: tuple = ()):
        super(CVAELongHMNISTEncoderDiagonal, self).__init__()
        self.latent_dims = latent_dims

        # CNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),   # [...,32,14,14]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # [...,64,7,7]
            nn.ReLU(),
            nn.Flatten(start_dim=-3, end_dim=-1)                                             # [...,64*7*7=1568]
        )

        # nn.init.xavier_uniform_(self.conv_layers[0].weight)
        # nn.init.zeros_(self.conv_layers[0].bias)
        # nn.init.xavier_uniform_(self.conv_layers[2].weight)
        # nn.init.zeros_(self.conv_layers[2].bias)

        # FCNN
        layer_dims = [64 * 7 * 7, *hidden_dims, 2 * latent_dims]
        self.fnn = nn.Sequential()
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            # ⚠️ using JAX initialization ???
            # nn.init.xavier_uniform_(linear_layer.weight)
            # nn.init.zeros_(linear_layer.bias)
            self.fnn.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.fnn.append(nn.ReLU())

    def forward(self, vid_batch: Tensor):
        """ vid_batch: [v, f, 28, 28] -> [v, f, L] or [v, n, H, 28, 28] -> [v, n, H, L]"""
        vid_batch_reshaped = vid_batch.reshape(-1, 1, *vid_batch.shape[-2:])  # Conv2D only accepts 4-D inputs
        x = self.conv_layers(vid_batch_reshaped)
        x = self.fnn(x.reshape(*vid_batch.shape[:-2], 64 * 7 * 7))
        return x


class CVAELongHMNISTDecoder(LongHMNISTDecoder):
    pass




