import math
from typing import Union
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from gpytorch.kernels import MaternKernel

from baselines.mgpvae.models.mgpvae_base import MGPVAEBase
from baselines.mgpvae.models.ssm import SSM
from baselines.mgpvae.models.building_blocks.kernels import Matern32SSM
from models.building_blocks.enc_dec_healing_mnist import HMNISTEncoderDiagonal, HMNISTDecoder
from utils.build_datasets import NNDataset
from utils.healing_mnist import LazySeriesDataset, generate_miniseries_hmnist_dict
from baselines.svgpvae.utils.elbo_utils import negative_gaussian_cross_entropy


class MGPVAEHealing(MGPVAEBase):
    def __init__(
            self, file_path: Union[str, Path], GP_joint=True, jitter: float = 1e-6,
    ):
        self.entire_train_dataset = LazySeriesDataset(file_path, train=True, return_full=False, val_split=50000)
        self.num_videos = len(self.entire_train_dataset)
        self.num_frames_per_series = self.entire_train_dataset[0][0].size(0)

        latent_dims = 256
        encoder = HMNISTEncoderDiagonal(latent_dims=latent_dims)
        decoder = HMNISTDecoder(latent_dims=latent_dims)

        kernel = Matern32SSM(base_kernel=MaternKernel(
            nu=1.5, batch_shape=torch.Size([latent_dims]) if GP_joint else torch.Size([1])
        ))
        kernel.outputscale, kernel.base_kernel.lengthscale = 1., math.sqrt(2)
        for p in kernel.parameters():
            p.requires_grad_(GP_joint)
        ssm = SSM(latent_dims, kernel, jitter=jitter)

        super(MGPVAEHealing, self).__init__(encoder, decoder, ssm)
        self.train_dataset = None

    # override
    def expected_log_prob(self, m_ss_Z, P_ss_Z, y_batch_miss, m_batch_miss=None):  # [v,T,L]
        """Compute E2 with corrupted frames in the paper: E_q(Z)[log p(Y|Z)]"""
        latent_samples = m_ss_Z + P_ss_Z.sqrt() * torch.randn_like(m_ss_Z)
        logits = self.decoder(latent_samples)  # [v,T,...]

        if m_batch_miss is None:
            m_batch_miss = torch.zeros_like(y_batch_miss, dtype=torch.bool, device=y_batch_miss.device)
        else:
            m_batch_miss = m_batch_miss.to(torch.bool)

        assert self.decoder.output_distribution == 'bernoulli'
        exp_lk = - nn.functional.binary_cross_entropy_with_logits(input=logits, target=y_batch_miss, reduction='none')
        exp_lk = torch.where(m_batch_miss, 0., exp_lk).sum()
        scale = y_batch_miss.shape[:len(self.train_dataset.series_shape)].numel()
        return exp_lk / scale  # sum over T (no mini-batch along T), average over v

    # override, using sum loss
    def average_loss(self, vid_batch_miss, t_batch, m_batch_miss=None, beta=1.):
        qnet_mus, qnet_vars, m_ss_Z, P_ss_Z, _, sum_log_p = self.forward(vid_batch_miss, t_batch, num_samples=1)

        # E2 with mask
        exp_lk_observed = self.expected_log_prob(m_ss_Z, P_ss_Z, vid_batch_miss, m_batch_miss)

        # E3
        E3 = sum_log_p.sum()  # sum over v*T

        # E1 negative cross entropy: int q(Z)log N(tilde_Y|Z,tilde_var), mean/cov: [v,T,L]
        E1 = negative_gaussian_cross_entropy(m_ss_Z, P_ss_Z, qnet_mus, qnet_vars)
        E1 = E1.sum()

        KL = E1 - E3
        elbo = exp_lk_observed - beta * KL / t_batch.shape[:-2].numel()  # averaged over v
        return - elbo

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs,
            series_batch_size: int, max_norm: float = None, device='cpu', print_epochs=1
    ):
        self.to(device)
        series_dataloader = DataLoader(self.entire_train_dataset, batch_size=series_batch_size, shuffle=True)
        for epoch in range(epochs):
            for series_y_miss, series_m_miss in series_dataloader:
                miniseries_dict = generate_miniseries_hmnist_dict(series_y_miss, series_m_miss, Y_batch_full=None)
                miniseries_dict['aux_data'] = miniseries_dict['aux_data'].expand(
                    series_y_miss.size(0), self.num_frames_per_series, 1)
                self.train_dataset = NNDataset(
                    miniseries_dict, series_shape=series_y_miss.shape[:1], missing=True, data_device='cpu', H=None
                )

                # We use the whole sequence
                y_miss, t, m_miss = self.train_dataset.Y, self.train_dataset.X, self.train_dataset.masks
                y_miss, t, m_miss = y_miss.to(device), t.to(device), m_miss.to(device)   # [v,T,28,28], t: [v,T,1]
                optimizer.zero_grad(set_to_none=True)
                loss = self.average_loss(y_miss, t, m_batch_miss=m_miss, beta=beta)
                loss.backward()
                if max_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm, error_if_nonfinite=True)
                    if grad_norm > max_norm:
                        print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item(): .6f}',
                      f'global grad norm: {grad_norm}' if max_norm is not None else ' ', '\n')

    @torch.no_grad()
    def predict_gpvae(
            self, test_file_path: Union[str, Path], series_batch_size: int, device='cpu', num_samples=1
    ):
        self.to(device)

        entire_test_dataset = LazySeriesDataset(test_file_path, train=False, return_full=True, val_split=None)
        series_dataloader = DataLoader(entire_test_dataset, batch_size=series_batch_size, shuffle=False)
        # We use the evaluation procedure of AISTATS 2020 paper: sample one `z`, metrics are computed on missing values
        nll_series, mse_series, mse_series_non_round, img_rec_series = 0., 0., 0, []
        n_missings, total_pixels = 0., 0.
        for series_y_miss, m_miss, series_y_full in series_dataloader:
            n_missings = n_missings + m_miss.sum()
            total_pixels = total_pixels + series_y_full.numel()
            miniseries_dict = generate_miniseries_hmnist_dict(series_y_miss, m_miss, series_y_full)
            miniseries_dict['aux_data'] = miniseries_dict['aux_data'].expand(
                series_y_miss.size(0), self.num_frames_per_series, 1)
            test_dataset = NNDataset(
                miniseries_dict, series_shape=series_y_miss.shape[:1], missing=True, return_full=True,
                data_device='cpu', H=None
            )

            y_test_miss, t_test = test_dataset.Y.to(device), test_dataset.X.to(device)  # put the whole series onto GPU
            # Putting the following onto GPU may cause OOM, so we do it on CPU
            m_test, y_test_full = test_dataset.masks, test_dataset.Y_full

            _, _, m_ss_Z, P_ss_Z, rec_logits, _ = self.forward(y_test_miss, t_test, num_samples)  # [v,T,L]

            # NLL
            log_p = - nn.functional.binary_cross_entropy_with_logits(
                input=rec_logits.to(y_test_full.device), target=y_test_full.expand_as(rec_logits), reduction='none')
            log_p = - math.log(num_samples) + torch.logsumexp(log_p, dim=0)
            nll = torch.where(m_test, -log_p, 0.).sum()
            nll_series += nll

            # MSE
            rec_vid = nn.functional.sigmoid(self.decoder(m_ss_Z)).to(y_test_full.device)
            img_rec_series.append(rec_vid)

            se_non_round = (rec_vid - y_test_full).square().sum()
            mse_series_non_round += se_non_round
            pixels = torch.round(rec_vid)  # round to the nearest integer
            se = torch.where(m_test, (pixels - y_test_full).square(), 0.).sum()
            mse_series += se
        return (
            (nll_series / n_missings).item(), (mse_series / n_missings).item(),
            (mse_series_non_round/total_pixels).item(),
            torch.cat(img_rec_series, dim=0).cpu().numpy()
        )


if __name__ == '__main__':
    import argparse
    import datetime
    import time
    import numpy as np
    import h5py
    import pickle
    from utils.healing_mnist import plot_hmnist

    def run_short_hmnist_mgpvae(args):
        current_time = datetime.datetime.now().strftime("%m%d_%H_%M_%S")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.float64:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        jitter = 1e-8 if args.float64 else 1e-6
        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.manual_seed_all(args.seed)
            # torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.benchmark = False  # for reproducibility
            torch.backends.cudnn.deterministic = True
        else:
            device = 'cpu'
        print(f"Time: {current_time}, Markovian GPVAE for short HMNIST, Device: {device}, {torch.get_default_dtype()}, "
              f"seed: {args.seed}, lr={args.lr}, beta={args.beta}, max_norm={args.max_norm} ",
              "GP joint training" if args.GP_joint else " ")
        s = time.time()

        file_path = Path(f'../data/healing_mnist/h5/hmnist_{args.random_mechanism}.h5')

        # Training
        model = MGPVAEHealing(file_path, GP_joint=args.GP_joint, jitter=jitter)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        model.train_gpvae(
            optimizer, args.beta, args.num_epochs, args.series_batch_size,
            max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs
        )
        print(f"Training time: {time.time() - s}.")
        print(f"Total params: {sum([p.numel() for p in model.parameters() if p.requires_grad])}.\n")

        # Testing
        nll, mse, mse_non_round, y_rec = model.predict_gpvae(
            file_path, series_batch_size=args.series_batch_size, device=device, num_samples=20
        )
        print(f"\nMGPVAE NLL: {nll}, MSE at missing pixels: {mse}, MSE (not rounded) at all pixels: {mse_non_round}.\n")

        # Plot
        video_idx = [i for i in range(args.num_plots)]
        test_dict = h5py.File(file_path, 'r')
        fig, ax = plot_hmnist(
            test_dict['x_test_miss'][video_idx, :10].reshape(-1, 10, 28, 28),
            y_rec[video_idx, :10],
            test_dict['x_test_full'][video_idx, :10].reshape(-1, 10, 28, 28)
        )

        if args.save:
            GP_joint_str = "_GP_joint" if args.GP_joint else ""
            exp_dir = Path(f'exp_MGPVAE_hmnist_{args.random_mechanism}' + GP_joint_str)
            exp_dir.mkdir(exist_ok=True)
            trial_dir = exp_dir / f'{current_time}'
            trial_dir.mkdir(parents=False, exist_ok=True)

            torch.save(model.state_dict(), trial_dir / f'model_short_hmnist.pth')  # model

            img_path = trial_dir / f'img_mvgpvae_short_hmnist.pdf'  # images
            fig.savefig(img_path, bbox_inches='tight')

            everything_for_imgs = {
                'y_rec': y_rec, 'NLL': nll, 'MSE': mse, 'MSE_non_round': mse_non_round
            }
            imgs_pickle_path = trial_dir / 'everything_for_imgs.pkl'
            with imgs_pickle_path.open('wb') as f:
                pickle.dump(everything_for_imgs, f)

            print("The experiment results are saved at:")
            print(f"{trial_dir.resolve()}")  # get absolute path for moving the log file
        else:
            print("The experimental results are not saved.")


    parser = argparse.ArgumentParser(description="MGPVAE for Short HMNIST imputation")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--random_mechanism', type=str, default='random',
                        choices=['mnar', 'spatial', 'random', 'temporal_neg', 'temporal_pos'])

    parser.add_argument('--GP_joint', action='store_true', help='whether to train the kernel params jointly')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for mean loss (instead of sum loss)')
    parser.add_argument('--beta', type=float, default=0.8, help='beta parameter')
    parser.add_argument('--max_norm', type=float, default=1e4, help='max norm of gradients in grad norm clipping')

    parser.add_argument('--series_batch_size', type=int, default=40, help='batch size for series')
    parser.add_argument('--num_epochs', type=int, default=0, help='number of epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')
    parser.add_argument('--num_plots', type=int, default=10, help='number of plots')
    parser.add_argument('--save', action='store_true', help='whether to save the results')

    args_mgpvae_hmnist = parser.parse_args()
    run_short_hmnist_mgpvae(args_mgpvae_hmnist)



