from typing import Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from baselines.vae.standard_vae import VAEBase
from models.building_blocks.enc_dec_spe10 import SPE10Encoder, SPE10Decoder
from utils.spe10 import NNDatasetSPE10, predict_y, plot_spe10


class VAESPE10(VAEBase):
    def __init__(self, file_path: Union[Path, str], sigma2_y=1., fix_variance=True):
        latent_dims, y_dims, x_dims = 3, 4, 3
        encoder = SPE10Encoder(latent_dims, input_dims=y_dims)
        decoder = SPE10Decoder(latent_dims, output_dims=y_dims, sigma2_y=sigma2_y, fix_variance=fix_variance)
        super(VAESPE10, self).__init__(
            encoder, decoder, latent_dims, extra_data_batch_shape=torch.Size([])
        )

        if isinstance(file_path, str):
            file_path = Path(file_path)
        self.train_dataset = NNDatasetSPE10(file_path, H=None)

    def train_vae(
            self, optimizer: torch.optim.Optimizer, beta: float, epochs: int, batch_size: int,
            device='cpu', print_epochs=1
    ):
        self.to(device)
        dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.train_dataset.return_full = False
        for epoch in range(epochs):
            for y_miss_b, _, _ in dataloader:
                optimizer.zero_grad(set_to_none=True)
                y_miss_b = y_miss_b.to(device)
                loss = self.average_loss(y_miss_b, beta=beta)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}')

    @torch.no_grad()
    def predict_vae(
            self, batch_size: int, stat: dict = None, device="cpu",
            num_samples=1, return_pred=False
    ):
        self.to(device)
        self.train_dataset.return_full = True
        test_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)

        all_nll, all_se, all_ae, all_pred_mean, all_pred_std = [], [], [], [], []
        for y_miss_b, _, m_b, y_full_b in test_dataloader:  # [b,D_y] NOTE: y_miss_b are normalized but y_full_b are not
            y_miss_b, m_b, y_full_b = y_miss_b.to(device), m_b.to(dtype=torch.bool, device=device), y_full_b.to(device)
            enc_means, enc_stds = self.encoder(y_miss_b)  # [b,L]
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


if __name__ == '__main__':
    from datetime import datetime
    import time
    import numpy as np
    import argparse
    import pickle

    def run_vae_spe10(args):
        current_time = datetime.now().strftime("%m%d_%H_%M_%S")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float32)
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        print(
            f"Standard VAE on SPE10, Time: {current_time}, seed: {args.seed}, device: {device}, "
            f"lr={args.lr}, beta={args.beta}, sigma2_y={args.sigma2_y}, {torch.get_default_dtype()}."
        )

        file_path = Path("../data/SPE10/spe10.npz")

        model = VAESPE10(file_path, sigma2_y=args.sigma2_y, fix_variance=True)
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

        experiment_dir = Path(f'./exp_spe10_VAE')
        if args.save:
            experiment_dir.mkdir(exist_ok=True)
            trial_dir = experiment_dir / f'{current_time}'
            trial_dir.mkdir(parents=False, exist_ok=True)

            torch.save(model.state_dict(), trial_dir / f'model_spe10.pth')  # model

            img_path = trial_dir / f'img_vae_spe10.pdf'  # images
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


    parser = argparse.ArgumentParser(description="Standard VAE on SPE10")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--sigma2_y', type=float, default=1., help='Standard deviation of GP prior')
    parser.add_argument('--beta', type=float, default=0.8, help='beta in the ELBO')

    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='Number of printed epochs')
    parser.add_argument('--save', action='store_true')

    args_vae_spe10 = parser.parse_args()
    run_vae_spe10(args_vae_spe10)

