import torch
from torch import Tensor
from torch.utils.data import DataLoader

from baselines.vae.standard_vae_spe10 import VAESPE10


class HIVAESPE10(VAESPE10):
    # override
    def expected_log_prob(self, y_batch: Tensor, m_batch: Tensor = None) -> Tensor:
        if m_batch is None:
            m_batch = torch.ones_like(y_batch, dtype=torch.bool)
        elif m_batch.dtype != torch.bool:
            m_batch = m_batch.to(torch.bool)

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
    def average_loss(self, y_batch: Tensor, m_batch: Tensor = None, beta=1.) -> Tensor:
        scale = y_batch.shape[:len(self.extra_data_batch_shape)+1].numel()
        exp_lk_masked = self.expected_log_prob(y_batch, m_batch)
        kl = self.kl_divergence(y_batch).sum()
        elbo = exp_lk_masked - beta * kl
        return - elbo / scale

    # override
    def train_vae(
            self, optimizer: torch.optim.Optimizer, beta: float, epochs: int, batch_size: int,
            device='cpu', print_epochs=1
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


if __name__ == '__main__':
    from datetime import datetime
    import time
    import numpy as np
    import argparse
    import pickle
    from pathlib import Path

    from utils.spe10 import plot_spe10

    def run_hivae_spe10(args):
        current_time = datetime.now().strftime("%m%d_%H_%M_%S")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float32)
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        print(
            f"HI-VAE on SPE10, Time: {current_time}, seed: {args.seed}, device: {device}, "
            f"lr={args.lr}, beta={args.beta}, sigma2_y={args.sigma2_y}, {torch.get_default_dtype()}."
        )

        file_path = Path("../data/SPE10/spe10.npz")

        model = HIVAESPE10(file_path, sigma2_y=args.sigma2_y, fix_variance=True)
        # train
        s = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
        model.train_vae(
            optimizer, args.beta, epochs=args.num_epochs, batch_size=args.batch_size,
            device=device, print_epochs=args.num_print_epochs
        )
        print(f"\nTotal Training time: {time.time() - s}")
        print(f"Total parameters: {sum([p.shape.numel() for p in model.parameters() if p.requires_grad])}\n")

        # Testing
        mse, mae, nll, pred_mean, pred_std = model.predict_vae(
            args.batch_size, stat=None, device=device,
            num_samples=20, return_pred=True
        )
        print(f'\nnll: {nll}, mse: {mse}, mae: {mae}.\n')

        # Plot
        data_dict_true = np.load(file_path)
        fig, ax = plot_spe10(data_dict_true, pred_mean, nll=nll, mse=mse)

        experiment_dir = Path(f'./exp_spe10_HIVAE')
        if args.save:
            experiment_dir.mkdir(exist_ok=True)
            trial_dir = experiment_dir / f'{current_time}'
            trial_dir.mkdir(parents=False, exist_ok=True)

            torch.save(model.state_dict(), trial_dir / f'model_spe10.pth')  # model

            img_path = trial_dir / f'img_hivae_spe10.pdf'  # images
            fig.savefig(img_path, bbox_inches='tight')

            everything_for_imgs = {
                # 'y_test_full': y_test_full, 'y_test_miss': y_test_miss, # to save storage
                'y_rec_mean': pred_mean, 'y_rec_std': pred_std,
                'NLL': nll, 'MSE': mse, 'MAE': mae
            }
            imgs_pickle_path = trial_dir / 'everything_for_imgs.pkl'
            with imgs_pickle_path.open('wb') as f:
                pickle.dump(everything_for_imgs, f)

            print("The experiment results are saved at:")
            print(f"{trial_dir.resolve()}")  # get absolute path for moving the log file
        else:
            print("The experimental results are not saved.")


    parser = argparse.ArgumentParser(description="HI-VAE on SPE10")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--sigma2_y', type=float, default=1., help='Standard deviation of GP prior')
    parser.add_argument('--beta', type=float, default=0.8, help='beta in the ELBO')

    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='Number of printed epochs')
    parser.add_argument('--save', action='store_true')

    args_hivae_spe10 = parser.parse_args()
    run_hivae_spe10(args_hivae_spe10)

