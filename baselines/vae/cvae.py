# Conditional VAE from "Learning Structured Output Representation using Deep Conditional Generative Models(NIPS 2015)"
# CVAE incorporates auxiliary information in both the encoder and the decoder

import torch
from torch import nn, Tensor


class CVAEBase(nn.Module):
    def __init__(
            self, encoder: nn.Module, decoder: nn.Module,
            aux_encoder: nn.Module, aux_decoder: nn.Module
    ):
        """
        refer to Casale 2018 paper Page 6 and its appendix Page 4.
        X is transformer by `aux_encoder` and then concatenated with the latent representation from `encoder`.
        They together are fed to an MLP to get the final representation Z;
        The same is for vectors before fed into the decoder.
        We set the cond_dim = latent_dim
        """
        super(CVAEBase, self).__init__()
        self.latent_dims = encoder.latent_dims

        self.encoder = encoder
        self.aux_encoder = aux_encoder
        assert aux_encoder.cond_dims == self.latent_dims
        self.before_z = nn.Sequential(
            nn.Linear(3 * self.latent_dims, self.latent_dims),
            nn.ReLU(),
            nn.Linear(self.latent_dims, 2 * self.latent_dims)
        )

        self.decoder = decoder
        self.aux_decoder = aux_decoder
        assert aux_decoder.cond_dims == self.latent_dims
        self.after_z = nn.Sequential(
            nn.Linear(2 * self.latent_dims, self.latent_dims),
            nn.ReLU(),
            nn.Linear(self.latent_dims, self.latent_dims)
        )

    def encode(self, y_batch: Tensor, c_batch: Tensor):  # c is the conditional vector
        aux_enc_out = self.aux_encoder(c_batch)
        encoder_out = self.encoder(y_batch)
        before_z = torch.cat([encoder_out, aux_enc_out], dim=-1)
        z = self.before_z(before_z)
        means, stds = z[..., :self.latent_dims], nn.functional.softplus(z[..., self.latent_dims:])
        return means, stds

    def decode(self, z_batch: Tensor, c_batch: Tensor):
        aux_dec_out = self.aux_decoder(c_batch)
        after_z = torch.cat([z_batch, aux_dec_out], dim=-1)
        z = self.after_z(after_z)
        logits = self.decoder(z)
        return logits

    def forward(self, y_batch: Tensor, c_batch: Tensor):
        means, stds = self.encode(y_batch, c_batch)
        latent_samples = means + stds * torch.randn_like(means)
        recon_imgs = self.decode(latent_samples, c_batch)
        return means, stds, recon_imgs




