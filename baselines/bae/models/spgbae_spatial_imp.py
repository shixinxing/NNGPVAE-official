import torch
from torch.utils.data import DataLoader

from baselines.bae.models.sgpbae import SGPBAE
from baselines.bae.adaptive_sghmc import AdaptiveSGHMC
from baselines.bae.utils.util import inf_loop
from baselines.bae.utils.spe10 import predict_y


class SGPBAESpatialImp(SGPBAE):
    """
    Spatial Imputation, like Jura.
    """
    def __init__(self, *args, **kwargs):
        super(SGPBAESpatialImp, self).__init__(*args, **kwargs)
        self.train_dataset = None

    def encode(self, Y):
        """concat noise and obs to a long tensor before fed into the encoder"""
        Y_tilde = torch.cat([Y, torch.randn_like(Y)], dim=-1)
        Z = self.encoder(Y_tilde)
        return Z, Y_tilde  # [b,L], [b,...]

    def train_sgpbae(
            self, bae_sampler: AdaptiveSGHMC, encoder_optimizer: torch.optim.Optimizer,
            n_burnin_iters: int, collect_every: int, n_samples: int,
            batch_size=30, SGHMC_steps=30, encoder_steps=50,
            num_print_iters=1, device='cpu'
    ):
        all_iters = n_burnin_iters + n_samples * collect_every
        self.train_dataset.return_full = False
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        # Start sampling
        self.to(device)
        iter = 0
        sample_idx = 0
        for y_miss_b, x_b, m_b in inf_loop(train_dataloader):  # m_b: 0 indicates missing
            if iter >= all_iters:
                break

            y_miss_b, x_b, m_b = y_miss_b.to(device), x_b.to(device), m_b.to(device)
            self.init_z(y_miss_b)      # update `self.Z`, `self.Y_tilde`

            # Sample the latent variables and decoder params
            for k in range(SGHMC_steps):
                log_prob, log_lik, log_prior, gp_log_lik = self.log_prob(y_miss_b, x_b, m_batch=m_b)
                bae_sampler.zero_grad()
                bae_loss = - log_prob / len(self.train_dataset)
                bae_loss.backward()
                bae_sampler.step()

            # Update the encoder
            for j in range(encoder_steps):
                encoder_optimizer.zero_grad()
                z_loss = self.z_loss()
                z_loss.backward()
                encoder_optimizer.step()

            # Collect sample
            if (iter+1) > n_burnin_iters and (iter+1-n_burnin_iters) % collect_every == 0:
                print(f"\nCollecting sample {sample_idx}...")
                self.save_sample(sample_idx)
                sample_idx += 1
                self.set_samples(cache=False)  # load sample dir path

            if (iter+1) % num_print_iters == 0:
                log_lik = log_lik.detach().cpu() / len(self.train_dataset)
                log_prior = log_prior.detach().cpu() / len(self.train_dataset)
                gp_log_lik = gp_log_lik.detach().cpu() / len(self.train_dataset)
                print(f"{iter+1}/{all_iters}, BAE loss: {bae_loss.detach().item():.5f}, "
                      f"log_lik: {log_lik.detach().item():.5f}, decoder log_prior : {log_prior.detach().item():.5f}, "
                      f"GP_log_lik: {gp_log_lik.detach().item():.5f}, z_loss: {z_loss.detach().item():.5f}")
            iter += 1

    @torch.no_grad()
    def predict_sgpbae(
            self, batch_size: int, stat: dict = None, device='cpu', return_pred=False
    ):
        """Don't provide test_dataset, as in Jura, the test samples are the missing elements in train_dataset."""
        self.to(device)
        self.train_dataset.return_full = True
        test_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=False
        )

        all_ae, all_se, all_nll, all_pred_mean, all_pred_std = [], [], [], [], []
        for y_miss_b, x_b, m_b, y_full_b in test_dataloader:  # NOTE: y_miss_b are normalized, but y_full_b are not.
            y_miss_b, x_b, m_b, y_full_b = y_miss_b.to(device), x_b.to(device), m_b.to(device), y_full_b.to(device)
            r_b = predict_y(self, y_miss_b, m_b, y_full_b, stat=stat)

            all_se.append(r_b['se'])  # 1d
            all_ae.append(r_b['ae'])
            all_nll.append(r_b['nll'])
            all_pred_mean.append(r_b['pred_mean'])  # [b,...]
            all_pred_std.append(r_b['pred_std'])

        all_se = torch.cat(all_se, dim=0)
        all_ae = torch.cat(all_ae, dim=0)
        all_nll = torch.cat(all_nll, dim=0)
        all_pred_mean = torch.cat(all_pred_mean, dim=0)  # [N, ...]
        all_pred_std = torch.cat(all_pred_std, dim=0)

        if not return_pred:
            return all_se.mean().item(), all_ae.mean().item(), all_nll.mean().item()
        else:
            return (all_se.mean().item(), all_ae.mean().item(), all_nll.mean().item(),
                    all_pred_mean.cpu().numpy(), all_pred_std.cpu().numpy())




