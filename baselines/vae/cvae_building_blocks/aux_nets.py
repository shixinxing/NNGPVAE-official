import torch
from torch import nn, Tensor


class AuxEncoderHMNISTMisFrms(nn.Module):
    def __init__(self, cond_dims=16):
        super(AuxEncoderHMNISTMisFrms, self).__init__()
        self.cond_dims = cond_dims
        self.linear = nn.Linear(2, 50)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(50, cond_dims)


    def forward(self, t_batch: Tensor):  # [v,T,1]
        assert t_batch.size(-1) == 1
        t_scaled = (t_batch - 50) / 50 * torch.pi  # -pi ~ pi, inspired by Casale NIPS 2018
        c_batch = torch.cat([torch.sin(t_scaled), torch.cos(t_scaled)], dim=-1)
        return self.linear2(self.relu(self.linear(c_batch)))  # [v,T,2]