import numpy as np
import torch
from torch import nn
from torch import Tensor


class BallEncoderBAE(nn.Module):
    """
    similar to Pearce's AABI 2019 paper but this net returns posterior samples, see Table 6 for structure details
    """
    def __init__(self, img_size: int = 32, hidden_dims: tuple = (500,)):
        super(BallEncoderBAE, self).__init__()

        layer_dims = [2 * img_size ** 2]
        for dim in hidden_dims:
            layer_dims.append(dim)
        layer_dims.append(2)

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

    def forward(self, vid_batch_flatten: Tensor):
        """
        :param vid_batch_flatten: [v, f, 2*32*32]
        :return: approximate samples with shape [v, f, 2]
        """
        return self.encoder(vid_batch_flatten)


if __name__ == '__main__':
    enc = BallEncoderBAE(img_size=28, hidden_dims=(50, 20))
    x = torch.randn(2, 3, 28*28*2)
    print(enc(x).shape, '\n')

