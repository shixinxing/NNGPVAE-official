import torch
from torch import Tensor
from torch.utils.data import DataLoader

from baselines.vae.standard_vae_jura import VAEJura

class HIVAEJura(VAEJura):
    # override
    def expected_log_prob(self, y_batch: Tensor, m_batch: Tensor=None) -> Tensor:
        if m_batch is None:
            m_batch = torch.ones_like(y_batch, dtype=torch.bool)
        elif m_batch.dtype != torch.bool:
            m_batch = m_batch.to(dtype=torch.bool)

        means, stds, y_rec = self.forward(y_batch)
        if self.decoder.output_distribution == 'normal':
            sigma2_y = self.decoder.sigma2_y
            expected_lk = (y_batch - y_rec).square() / sigma2_y + torch.log(2 * torch.pi * sigma2_y)
            expected_lk = - 1 / 2 * expected_lk
        else:
            raise NotImplementedError

        expected_lk_masked = torch.where(m_batch, expected_lk, 0.).sum()
        return expected_lk_masked

    # override
    def average_loss(self, y_batch: Tensor, m_batch: Tensor=None, beta=1.) -> Tensor:
        scale = y_batch.shape[:len(self.extra_data_batch_shape) + 1].numel()
        exp_lk_masked = self.expected_log_prob(y_batch, m_batch)
        kl = self.kl_divergence(y_batch).sum()
        elbo = exp_lk_masked - beta * kl
        return - elbo / scale

    # override
    def train_vae(self, optimizer: torch.optim.Optimizer, batch_size: int, epochs: int, beta: float=1.,
                  device='cpu', print_epochs=10,
    ):
        self.to(device)
        dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.train_dataset.return_full = False
        for epoch in range(epochs):
            for y_miss_b, _, m_b in dataloader:
                optimizer.zero_grad(set_to_none=True)
                y_miss_b, m_b = y_miss_b.to(device), m_b.to(device=device, dtype=torch.bool)
                loss = self.average_loss(y_miss_b, m_b, beta=beta)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}')
