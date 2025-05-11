from torch import nn, Tensor


class SPE10EncoderBAE(nn.Module):
    """Use a similar structure as others but this encoder accepts extra randomness as input and returns samples of Z"""

    def __init__(self, latent_dims=3, hidden_dims: tuple = (256, 64), input_dims=4):
        super(SPE10EncoderBAE, self).__init__()
        self.latent_dims = latent_dims

        layer_dims = [2*input_dims, *hidden_dims, latent_dims]
        self.fnn = nn.Sequential()
        for i in range(len(layer_dims) - 1):
            linear_layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            self.fnn.append(linear_layer)
            if i < len(layer_dims) - 2:
                self.fnn.append(nn.ReLU())

    def forward(self, y: Tensor):
        return self.fnn(y)

