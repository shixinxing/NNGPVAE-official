import warnings
import torch
from torch import Tensor, LongTensor, nn
from torch.utils.data import DataLoader

from models.gpvae_base import GPVAEBase
from utils.spe10 import predict_y


class GPVAESpatialImp(GPVAEBase):
    """
    Spatial Imputation, such as Jura
    """
    # overrides
    def expected_log_prob(self, y_batch_miss: Tensor, m_batch_miss: Tensor = None):
        """
        y: [b, D], m: [b, D], 'True' in the mask represents observed.
        """
        assert y_batch_miss.ndim == 2 and m_batch_miss.ndim == 2
        if m_batch_miss is None:
            m_batch_miss = torch.ones_like(y_batch_miss, dtype=torch.bool)
        elif m_batch_miss.dtype != torch.bool:
            m_batch_miss = m_batch_miss.to(torch.bool)

        means, stds, y_rec = self.forward(y_batch_miss)

        if self.decoder.output_distribution == 'normal':
            sigma2_y = self.decoder.sigma2_y
            expected_lk = (y_batch_miss - y_rec).square() / sigma2_y + torch.log(2 * torch.pi * sigma2_y)
            expected_lk = - 0.5 * expected_lk  # [b, D]
        else:
            raise NotImplementedError

        # We will use mean loss later
        expected_lk = torch.where(m_batch_miss, expected_lk, 0.).sum()
        scale = y_batch_miss.shape[:len(self.train_dataset.series_shape) + 1].numel()
        return expected_lk / scale

    def loss_sws(self, y_miss_batch: Tensor, x_batch: Tensor, m_batch: Tensor, beta=1.):
        lik = self.expected_log_prob(y_miss_batch, m_batch)
        # We use mean loss here
        kl = self.kl_divergence_sws(x_batch) / len(self.train_dataset)
        elbo = lik - beta * kl
        return - elbo

    def train_gpvae_sws(
            self, optimizer: torch.optim.Optimizer, beta: float, epochs: int, batch_size: int, max_norm: float = None,
            device="cpu", print_epochs=1, validate=True
    ):
        if device != self.module_device:
            warnings.warn(f"Training is not on self.module_device ({self.module_device}) set earlier. "
                          f"We have changed module device to {device}.")
            self.module_device = device
        self.to(device)

        self.train_dataset.return_full = False
        dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for y_miss_b, x_b, m_b in dataloader:  # [b,D_y], [b,D_x], [b,D_y],
                optimizer.zero_grad(set_to_none=True)
                y_miss_b, x_b, m_b = y_miss_b.to(device), x_b.to(device), m_b.to(device=device, dtype=torch.bool)
                loss = self.loss_sws(y_miss_b, x_b, m_b, beta=beta)
                loss.backward()
                if max_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.parameters(), max_norm=max_norm, error_if_nonfinite=True
                    )
                    if grad_norm > max_norm:
                        print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item(): .6f}',
                      f'global grad norm: {grad_norm}' if max_norm is not None else ' ', '\n')
                if validate:
                    se, ae, nll = self.predict_gpvae(batch_size, None, device, num_samples=20, return_pred=False)
                    print(f'SE: {se}, AE: {ae}, NLL: {nll}\n')
                    self.train_dataset.return_full = False

    def loss_vnn(self, y_miss_b: Tensor, m_b: Tensor, kl_indices: LongTensor, seq_nn_idx: LongTensor, beta=1.):
        lik = self.expected_log_prob(y_miss_b, m_b)
        # We use mean loss here
        kl = self.kl_divergence_vnn(kl_indices, seq_nn_idx) / len(self.train_dataset)
        elbo = lik - beta * kl
        return - elbo

    def train_gpvae_vnn(
            self, optimizer: torch.optim.Optimizer, beta: float, epochs: int,
            kl_batch_size: int, expected_lk_batch_size=None, max_norm: float = None,
            device="cpu", print_epochs=1, validate=True
    ):
        if device != self.module_device:
            warnings.warn(f"Training is not on self.module_device ({self.module_device}) set earlier. "
                          f"We have changed module device to {device}.")
            self.module_device = device
        self.to(device)

        seq_nn_idx = self.train_dataset.seq_nn_idx
        self.train_dataset.return_full = False
        for epoch in range(epochs):
            self._set_training_iterator(kl_batch_size, N=len(self.train_dataset))

            if expected_lk_batch_size is None:
                for _ in range(self._total_training_batches):
                    current_kl_indices = self._get_training_indices(kl_batch_size)
                    y_miss_b, _, m_b = self.train_dataset[current_kl_indices]
                    y_miss_b, m_b = y_miss_b.to(device), m_b.to(device=device, dtype=torch.bool)

                    optimizer.zero_grad(set_to_none=True)
                    loss = self.loss_vnn(y_miss_b, m_b, current_kl_indices, seq_nn_idx, beta=beta)
                    loss.backward()
                    if max_norm is not None:
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.parameters(), max_norm=max_norm, error_if_nonfinite=True
                        )
                        if grad_norm > max_norm:
                            print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                    optimizer.step()

            else:
                dataloader_lk = DataLoader(
                    self.train_dataset, batch_size=expected_lk_batch_size, shuffle=True
                )
                for y_miss_b, _, m_b in dataloader_lk:
                    y_miss_b, m_b = y_miss_b.to(device), m_b.to(device)
                    current_kl_indices = self._get_training_indices(kl_batch_size)

                    optimizer.zero_grad(set_to_none=True)
                    loss = self.loss_vnn(y_miss_b, m_b, current_kl_indices, seq_nn_idx, beta=beta)
                    loss.backward()
                    if max_norm is not None:
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.parameters(), max_norm=max_norm, error_if_nonfinite=True
                        )
                        if grad_norm > max_norm:
                            print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                    optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item(): .6f}',
                      f'global grad norm: {grad_norm}' if max_norm is not None else ' ', '\n')
                if validate:
                    se, ae, nll = self.predict_gpvae(kl_batch_size, None, device, num_samples=20, return_pred=False)
                    print(f'SE: {se}, AE: {ae}, NLL: {nll}\n')
                    self.train_dataset.return_full = False

    @torch.no_grad()
    def predict_gpvae(
            self, batch_size: int, stat: dict = None, device="cpu",
            num_samples=1, return_pred=False
    ):
        """
        Don't provide test_dataset, as for spatial imputation, the test samples are
        the missing elements in the train_dataset.
        for Jura, stat['mean'] and stat['std'] should rescale predictive dist in predict_y.
        """
        if device != self.module_device:
            warnings.warn(f"Prediction on an other device {device} but self.module_device is {self.module_device}."
                          f"We have changed self.module_device to {device}.")
            self.module_device = device
        self.to(device)

        self.train_dataset.return_full = True
        test_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)

        all_nll, all_se, all_ae, all_pred_mean, all_pred_std = [], [], [], [], []
        for y_miss_b, _, m_b, y_full_b in test_dataloader:  # [b,D_y] NOTE: y_miss_b are normalized but y_full_b are not
            y_miss_b, m_b, y_full_b = y_miss_b.to(device), m_b.to(dtype=torch.bool, device=device), y_full_b.to(device)
            enc_means, enc_stds = self.encoder(y_miss_b)    # [b,L]
            r_dict = predict_y(enc_means, enc_stds, self.decoder, y_full_b, s=num_samples, stat=stat)

            all_se.append(r_dict['se'][~m_b])
            all_ae.append(r_dict['ae'][~m_b])  # only compute metric on missing parts, [n_test]
            all_nll.append(r_dict['nll'][~m_b])
            all_pred_mean.append(r_dict['pred_mean'])
            all_pred_std.append(r_dict['pred_std'])

        all_se = torch.cat(all_se, dim=0)
        all_ae = torch.cat(all_ae, dim=0)
        all_nll = torch.cat(all_nll, dim=0)
        all_pred_mean = torch.cat(all_pred_mean, dim=0)  # [n_test, D_y]
        all_pred_std = torch.cat(all_pred_std, dim=0)

        if not return_pred:
            return all_se.mean().item(), all_ae.mean().item(), all_nll.mean().item()
        else:
            return (all_se.mean().item(), all_ae.mean().item(), all_nll.mean().item(),
                    all_pred_mean.cpu().numpy(), all_pred_std.cpu().numpy())
