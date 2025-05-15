from typing import Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baselines.vae.standard_vae import VAEBase
from models.building_blocks.enc_dec_jura import JuraEncoder, JuraDecoder

from utils.jura import load_jura
from utils.build_datasets import NNDataset
from utils.mujoco_metric import predict_y


class VAEJura(VAEBase):
    def __init__(self, file_path: Union[Path, str]='./data/jura', sigma2_y=0.25,  fix_variance=True):
        x_dims, y_dim, num_latents = 2, 3, 2
        encoder = JuraEncoder(y_dim=y_dim, num_latents=num_latents)
        decoder = JuraDecoder(y_dim=y_dim, num_latents=num_latents, sigma2_y=sigma2_y, fix_variance=fix_variance)
        super(VAEJura, self).__init__(encoder, decoder, num_latents, extra_data_batch_shape=torch.Size([]))

        data_X, masks, data_Ymiss, data_Yfull, stat = load_jura(file_path)
        data_dict = {"X": data_X, "Y": data_Ymiss, "masks": masks, "Y_full": data_Yfull}
        self.train_dataset = NNDataset(data_dict, missing=True, return_full=True, H=None)
        self.Ystat = stat 

    def train_vae(self, optimizer: torch.optim.Optimizer, batch_size: int, epochs: int, beta: float=1.,
                  device='cpu', print_epochs=10,
    ):
        self.to(device)
        dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.train_dataset.return_full = False
        for epoch in range(epochs):
            for y_miss_b, _, _ in dataloader:
                optimizer.zero_grad(set_to_none=True)
                y_miss_b = y_miss_b.to(device)
                loss = self.average_loss(y_miss_b, beta)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}')

    @torch.no_grad()
    def predict_vae(self, batch_size: int, stat: dict=None, device='cpu',
                    num_samples=20, return_pred=False
    ):
        self.to(device)
        self.train_dataset.return_full = True
        test_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)

        all_ae, all_se, all_nll, all_pred_mean, all_pred_std = [], [], [], [], []

        for y_miss_b, _, m_b, y_full_b in test_dataloader:
            y_miss_b, m_b, y_full_b = y_miss_b.to(device), m_b.to(device=device, dtype=torch.bool), y_full_b.to(device)
            enc_means, enc_stds = self.encoder(y_miss_b) 
            r_dict = predict_y(enc_means, enc_stds, self.decoder, y_full_b, s=num_samples, stat=stat)

            all_se.append(r_dict['se'][~m_b])
            all_ae.append(r_dict['ae'][~m_b]) 
            all_nll.append(r_dict['nll'][~m_b])
            all_pred_mean.append(r_dict['pred_mean'])
            all_pred_std.append(r_dict['pred_std'])

        all_ae = torch.cat(all_ae, dim=0)
        all_se = torch.cat(all_se, dim=0)
        all_nll = torch.cat(all_nll, dim=0)
        all_pred_mean = torch.cat(all_pred_mean, dim=0)
        all_pred_std = torch.cat(all_pred_std, dim=0)

        if not return_pred:
            return all_se.mean().item(), all_ae.mean().item(), all_nll.mean().item()
        else:
            return all_se.mean().item(), all_ae.mean().item(), all_nll.mean().item(), all_pred_mean.cpu(), all_pred_std.cpu()
