# Adapted from the official code of BAE (ICML 2023) https://github.com/tranbahien/sgp-bae/blob/master/models/sgpbae.py

from pathlib import Path
import warnings
import torch
from torch import nn
from torch.nn import functional as F

from baselines.bae.utils.util import get_all_files
from baselines.bae.distributions.conditional.base import ConditionalDistribution
from baselines.bae.priors.fixed_priors import PriorModule
from baselines.bae.models.bsgp import BSGP


class SGPBAE(nn.Module):
    def __init__(
            self, encoder: nn.Module, decoder: ConditionalDistribution, decoder_prior: PriorModule,
            bsgp: BSGP, sample_dir: Path = Path('.'), device='cpu'
    ):
        super(SGPBAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder               # this class has `.log_prob` method, will be used later
        self.decoder_prior = decoder_prior   # the prior over the decoder weights

        self.bsgp = bsgp

        self.decoder_samples = None          # store samples or their file paths
        self.gp_samples = None
        self.loaded_samples = False          # load samples into memo or just save file paths
        self.sample_dir = sample_dir
        self.sample_dir.mkdir(exist_ok=True, parents=False)

        self.Z = None                        # output of the encoder: [(v), N, L]
        self.Y_tilde = None
        self.device = device

    def encode(self, Y_batch):
        """concat noise and Y before fed into the encoder; concat along after the batch dim"""
        # return Z, Y_tilde
        raise NotImplementedError("Implicit network concatenating noise and Y before fed into the encoder.")

    def init_z(self, Y):
        """
        Go through the encoder once, as said by the authors,
        if the encoder is MLP, concatenate into a long vector; if is CNN, concat along the channel dimension
        """
        Z, Y_tilde = self.encode(Y)
        Z = Z.detach()
        self.Y_tilde = Y_tilde.detach()

        if self.Z is None:
            self.Z = torch.nn.Parameter(Z.data, requires_grad=True).to(Y.device)  # will be registered automatically
        else:
            self.Z.data = Z.data

    def get_parameters(self):
        """return variables whose distributions we want to sample from"""
        params = list(self.decoder.net.parameters()) + list([self.Z]) + list(self.bsgp.parameters())
        return params

    @property
    def params(self):
        return self.state_dict()

    @params.setter
    def params(self, params):
        self.load_state_dict(params)

    @property
    def decoder_params(self):
        return self.decoder.net.state_dict()

    @decoder_params.setter
    def decoder_params(self, params):  # decoder weights
        self.decoder.net.load_state_dict(params)

    @property
    def gp_params(self):
        return self.bsgp.state_dict()

    @gp_params.setter
    def gp_params(self, params):
        self.bsgp.load_state_dict(params)

    def decode(self, Z, randomness=False, return_std=False):
        """compute log p(Y|Z, \theta), there is randomness in decoder weights"""
        if randomness and (self.decoder_samples is not None):
            Y = []
            for i in range(len(self.decoder_samples)):
                decoder_params, _ = self.load_samples(i)
                self.decoder_params = decoder_params
                Y.append(self.decoder(Z))

            Y = torch.stack(Y, dim=0)
            Y_std = torch.std(Y, dim=0)
            Y = torch.mean(Y, dim=0)

            if return_std:
                return Y, Y_std
        else:
            Y = self.decoder.net(Z)
        return Y

    def log_prob(self, Y_batch, X_batch, m_batch=None):
        """ Compute the overall log likelihood: i.e., -U(\theta), X_batch:[(v),n,D]"""
        # log p(Y|Z, \theta)
        if m_batch is None:
            m_batch = torch.ones_like(Y_batch, dtype=torch.bool)
        elif m_batch.dtype != torch.bool:
            m_batch = m_batch.to(torch.bool)

        scale = self.bsgp.N_train / X_batch.size(-2)
        log_lik = scale * torch.sum(m_batch * self.decoder.log_prob(Y_batch, context=self.Z, return_sum=False))

        # log p(\theta) over decoder weights
        log_prior = torch.sum(self.decoder_prior.log_prob(self.decoder.net))

        # log p(\psi) + N/B\sum\log p(Z|U)
        gp_log_lik = self.bsgp.log_prob(X_batch, self.Z)  # official code used X.double() and Z.double()

        log_prob = log_lik + log_prior + gp_log_lik
        return log_prob, log_lik, log_prior, gp_log_lik

    def z_loss(self):
        loss = F.mse_loss(self.encoder(self.Y_tilde), self.Z)
        return loss

    def save_sample(self, idx):
        torch.save(self.decoder_params, self.sample_dir/f"decoder_{idx}.pth")
        torch.save(self.gp_params, self.sample_dir/f"gp_{idx}.pth")

    def set_samples(self, cache=False):
        decoder_files = get_all_files(self.sample_dir/"decoder*")
        gp_files = get_all_files(self.sample_dir/"gp*")

        if cache:  # whether to load the samples into memory
            self.decoder_samples = []
            self.gp_samples = []

            for i in range(len(decoder_files)):
                self.decoder_samples.append(torch.load(decoder_files[i], map_location=self.device))
                self.gp_samples.append(torch.load(gp_files[i], map_location=self.device))
        else:     # only load file paths
            self.decoder_samples = decoder_files  # list
            self.gp_samples = gp_files

        self.loaded_samples = cache   # get samples or sample file path

    def load_samples(self, idx):
        if self.loaded_samples:       # we have loaded the samples into the memory
            decoder_params = self.decoder_samples[idx]
            gp_params = self.gp_samples[idx]
        else:                         # we only have a list containing the file paths
            decoder_params = torch.load(self.decoder_samples[idx], map_location=self.device)
            gp_params = torch.load(self.gp_samples[idx], map_location=self.device)

        return decoder_params, gp_params

    # haven't used in current exps!!
    def conditional_generate(self, X):
        if self.decoder_samples is None:
            warnings.warn("We don't have samples yet!")
            Z_mean = self.bsgp.predict(X)[0]
            Y = self.decoder.net(Z_mean)
            return Y, None
        else:
            Y = []
            for i in range(len(self.decoder_samples)):
                decoder_params, gp_params = self.load_samples(i)
                self.decoder_params = decoder_params
                self.gp_params = gp_params

                Z_mean = self.bsgp.predict(X)[0]
                Y.append(self.decoder.net(Z_mean))

            Y = torch.stack(Y, dim=0)
            Y_var = torch.var(Y, dim=0)
            Y = torch.mean(Y, dim=0)
            return Y, Y_var

    def predict(self, Y, randomness=False, get_mean=True):
        if randomness:
            Y_pred = []
            for i in range(len(self.decoder_samples)):
                decoder_params, _ = self.load_samples(i)
                self.decoder_params = decoder_params
                Y_pred.append(self.decoder(self.encoder(Y)))
            Y_pred = torch.stack(Y_pred, dim=0)
            if not get_mean:
                return Y_pred

            Y_pred = torch.mean(Y_pred, dim=0)
        else:
            Z = self.encoder(Y)
            Y_pred = self.decoder(Z)

        return Y_pred

