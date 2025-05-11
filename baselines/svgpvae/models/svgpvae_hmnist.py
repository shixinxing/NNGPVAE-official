import math
from typing import Union
from pathlib import Path
import h5py
import pickle

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from gpytorch.kernels import ScaleKernel, RQKernel

from baselines.svgpvae.models.svgpvae_base import SVGPVAEBase
from baselines.svgpvae.models.svgp import SVGP
from baselines.svgpvae.utils.elbo_utils import negative_gaussian_cross_entropy
from models.building_blocks.enc_dec_healing_mnist import HMNISTEncoderDiagonal, HMNISTDecoder
from utils.healing_mnist import LazySeriesDataset, generate_miniseries_hmnist_dict, plot_hmnist
from utils.build_datasets import NNDataset


class SVGPVAEHealing(SVGPVAEBase):
    def __init__(
            self, M: int, file_path: Union[str, Path],
            GP_joint=True, IP_joint=True, device='cpu', jitter=1e-6
    ):
        self.entire_train_dataset = LazySeriesDataset(file_path, train=True, return_full=False, val_split=None)
        self.train_dataset = None  # iterable along frames
        self.num_videos = len(self.entire_train_dataset)
        self.num_frames_per_series = self.entire_train_dataset[0][0].size(0)

        latent_dims = 256
        encoder = HMNISTEncoderDiagonal(latent_dims)
        decoder = HMNISTDecoder(latent_dims)

        kernel = ScaleKernel(RQKernel(batch_shape=torch.Size([latent_dims])))
        kernel.outputscale, kernel.base_kernel.lengthscale, kernel.base_kernel.alpha = 1, math.sqrt(2), 1
        for p in kernel.parameters():
            p.requires_grad_(GP_joint)
        inducing_points = nn.Parameter(
            torch.linspace(0, self.num_frames_per_series-1, M).unsqueeze(-1).repeat(latent_dims, 1, 1),
            requires_grad=IP_joint   # [L, M, 1]
        )
        svgp = SVGP(latent_dims, kernel, inducing_points, N_train=self.num_frames_per_series, jitter=jitter)

        super(SVGPVAEHealing, self).__init__(encoder, decoder, svgp, device=device, jitter=jitter, geco=False)

    # override
    def build_MLP_inference_graph(self, y_batch: Tensor):
        qnet_mus, qnet_vars = super(SVGPVAEHealing, self).build_MLP_inference_graph(y_batch)
        qnet_vars = torch.clamp(qnet_vars, 1e-6, 1e3)
        return qnet_mus, qnet_vars  # [(v),n,L]

    # override, using sum loss for this experiment
    def expected_log_prob(
            self, mu_qs: Tensor, vars_qs_diag_clipped: Tensor, y_batch_miss, m_batch_miss=None
    ):   # [v,L,b], [v,b,28,28]
        if m_batch_miss is None:
            m_batch_miss = torch.zeros_like(y_batch_miss, dtype=torch.bool)
        else:
            m_batch_miss = m_batch_miss.to(torch.bool)

        latent_samles = mu_qs + torch.randn_like(mu_qs) * torch.sqrt(vars_qs_diag_clipped)
        latent_samles = latent_samles.mT
        rec_img = self.decoder(latent_samles)

        assert self.decoder.output_distribution == 'bernoulli'
        expected_lk = - nn.functional.binary_cross_entropy_with_logits(
            input=rec_img, target=y_batch_miss, reduction='none')
        expected_lk = torch.where(m_batch_miss, 0., expected_lk).sum()
        scale = y_batch_miss.shape[:len(self.train_dataset.series_shape)+1].numel()
        scale = len(self.train_dataset) / scale   # averaged over series dim
        return scale * expected_lk

    def average_loss(
            self, vid_batch: Tensor, t_batch: Tensor, m_batch=None, clip_qs_var_min=None, clip_qs_var_max=None, beta=1.
    ):
        qnet_mus, qnet_vars = self.build_MLP_inference_graph(vid_batch)  # [(v),b,L]
        # get q(Z_U) and q_s(Z): [(v),L,b]
        mu_qs, cov_qs_diag, mu, A = self.svgp.approx_posterior_params(
            t_batch, qnet_mus, qnet_vars, x_test=t_batch, diag_cov_qs=True
        )

        # term 1/3 - expected-log-likelihood
        expected_lk = self.expected_log_prob(
            mu_qs, torch.clamp(cov_qs_diag, min=clip_qs_var_min, max=clip_qs_var_max), vid_batch, m_batch
        )

        # term 2/3 - cross-entropy
        cross_entry = negative_gaussian_cross_entropy(mu_qs, cov_qs_diag, qnet_mus.mT, qnet_vars.mT)  # [(v),L,b]
        cross_entry = cross_entry.sum()

        # term 3/3 - L_H
        inside_elbo_batch_log, inside_elbo_KL = self.svgp.variational_loss(t_batch, qnet_mus, qnet_vars, mu, A)
        L_H = self.N_train / t_batch.size(-2) * inside_elbo_batch_log.sum() - inside_elbo_KL.sum()

        v, b = qnet_mus.shape[:-2].numel(), qnet_vars.shape[-2]
        kl = self.N_train / b * cross_entry - L_H
        elbo = expected_lk - beta * kl / v  # sum loss, averaged over v
        return - elbo

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs,
            series_batch_size: int, timestamp_batch_size: int, max_norm: float = None,
            device='cpu', print_epochs=1
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

                time_dataloader = DataLoader(self.train_dataset, batch_size=timestamp_batch_size, shuffle=True)
                for y_miss, t, m_miss in time_dataloader:
                    y_miss, t, m_miss = y_miss.to(device), t.to(device), m_miss.to(device)
                    y_miss, t, m_miss = y_miss.transpose(0, 1), t.transpose(0, 1), m_miss.transpose(0, 1)  # [v,f,28,28], t: [v,f,1]
                    optimizer.zero_grad(set_to_none=True)
                    # faiss needs continuity (t is usually contiguous from dataloader)
                    loss = self.average_loss(
                        y_miss, t, m_batch=m_miss, clip_qs_var_min=1e-4, clip_qs_var_max=1e3, beta=beta
                    )
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

    @torch.no_grad()
    def predict_gpvae(
            self, test_file_path: Union[str, Path], series_batch_size: int, timestamp_batch_size: int,
            device: str = "cpu", num_samples=1
    ):
        self.to(device)

        entire_test_dataset = LazySeriesDataset(test_file_path, train=False, return_full=True)
        series_dataloader = DataLoader(entire_test_dataset, batch_size=series_batch_size, shuffle=False)

        nll_series, mse_series, mse_series_non_round, img_rec_series = 0., 0., 0, []
        n_missings, total_pixels = 0., 0.
        for series_y_miss, m_miss, series_y_full in series_dataloader:
            n_missings += m_miss.sum()
            total_pixels = total_pixels + series_y_full.numel()

            miniseries_dict = generate_miniseries_hmnist_dict(series_y_miss, m_miss, series_y_full)
            miniseries_dict['aux_data'] = miniseries_dict['aux_data'].expand(
                series_y_miss.size(0), self.num_frames_per_series, 1)
            test_dataset = NNDataset(
                miniseries_dict, series_shape=series_y_miss.shape[:1], missing=True, return_full=True,
                data_device='cpu', H=None
            )
            time_loader = DataLoader(test_dataset, batch_size=timestamp_batch_size, shuffle=False)

            nll_batch, se_batch, se_non_round_batch, img_rec_batch = 0., 0., 0., []
            # As suggested by the paper, we should have used all train t for prediction,
            #  but for now, we implement a more general case by using the mini-batch timestamps
            for y_test_miss, t_test, m_test, y_test_full in time_loader:
                y_test_miss, t_test = y_test_miss.to(device).transpose(0, 1), t_test.to(device).transpose(0, 1)
                m_test, y_test_full = m_test.to(device).transpose(0, 1), y_test_full.to(device).transpose(0, 1)

                full_p_mu, full_p_var, logits = self.forward(
                    y_test_miss, t_test, cov_latent_diag=True, num_samples=num_samples)  # [v,f,L]

                # NLL
                log_p = - nn.functional.binary_cross_entropy_with_logits(
                    input=logits, target=y_test_full.expand_as(logits), reduction='none')
                log_p = - math.log(num_samples) + torch.logsumexp(log_p, dim=0)
                nll = torch.where(m_test, -log_p, 0.).sum()
                nll_batch += nll

                # MSE
                img_rec = nn.functional.sigmoid(self.decoder(full_p_mu))
                img_rec_batch.append(img_rec)

                se_non_round = (img_rec - y_test_full).square().sum()
                se_non_round_batch += se_non_round
                pixels = torch.round(img_rec)  # round to the nearest integer
                se = torch.where(m_test, (pixels - y_test_full).square(), 0.).sum()
                se_batch += se
            nll_series += nll_batch
            mse_series_non_round += se_non_round_batch
            mse_series += se_batch
            img_rec_series.append(torch.cat(img_rec_batch, dim=1))  # [v,f,28,28]
        return (
            (nll_series / n_missings).item(), (mse_series / n_missings).item(),
            (mse_series_non_round/total_pixels).item(),
            torch.cat(img_rec_series, dim=0).cpu().numpy()
        )


if __name__ == '__main__':
    import argparse
    import datetime
    import time

    def run_short_hmnist(args):
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
            # torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.benchmark = False  # for reproducibility
            torch.backends.cudnn.deterministic = True
        else:
            device = 'cpu'
        print(f"Time: {current_time}, SVGPVAE-M{args.M} for short HMNIST, {device}, {torch.get_default_dtype()},\n"
              f"seed: {args.seed}, lr: {args.lr}, beta={args.beta}, max_norm={args.max_norm}  ",
              'GP joint training ' if args.GP_joint else " ", 'IP joint training' if args.IP_joint else " ")
        s = time.time()

        file_path = Path(f'../data/healing_mnist/h5/hmnist_{args.random_mechanism}.h5')

        model = SVGPVAEHealing(
            args.M, file_path, GP_joint=args.GP_joint, IP_joint=args.IP_joint, device=device, jitter=jitter
        )
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        model.train_gpvae(
            optimizer, args.beta, args.num_epochs, args.series_batch_size, args.timestamp_batch_size, args.max_norm,
            device=device, print_epochs=args.num_print_epochs
        )
        print(f"Total training time: {time.time() - s}.")
        print(f"Total parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}.\n")

        # Testing
        nll, mse, mse_non_round, y_rec = model.predict_gpvae(
            file_path, args.series_batch_size, args.timestamp_batch_size, device=device, num_samples=20
        )
        print(f"\nSVGPVAE NLL: {nll}, MSE at missing pixels: {mse}, MSE (not rounded) at all pixels: {mse_non_round}.\n")

        # Plot
        video_idx = [i for i in range(args.num_plots)]
        test_dict = h5py.File(file_path, 'r')
        fig, ax = plot_hmnist(
            test_dict['x_test_miss'][video_idx, :10].reshape(-1, 10, 28, 28),
            y_rec[video_idx, :10].cpu().numpy(),
            test_dict['x_test_full'][video_idx, :10].reshape(-1, 10, 28, 28)
        )

        if args.save:
            GP_joint_str = "_GP_joint" if args.GP_joint else ""
            IP_joint_str = "_IP_joint" if args.IP_joint else ""
            exp_dir = Path(f'exp_hmnist_SVGPVAE_M{args.M}_{args.random_mechanism}' + GP_joint_str + IP_joint_str)
            exp_dir.mkdir(exist_ok=True)
            trial_dir = exp_dir / f'{current_time}'
            trial_dir.mkdir(parents=False, exist_ok=True)

            torch.save(model.state_dict(), trial_dir / f'model_short_hmnist.pth')  # model

            img_path = trial_dir / f'img_svgpvae_short_hmnist.pdf'  # images
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


    parser = argparse.ArgumentParser(description='SVGPVAE for Long HMNIST imputation')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--random_mechanism', type=str, default='random',
                        choices=['mnar', 'spatial', 'random', 'temporal_neg', 'temporal_pos'])

    parser.add_argument('--M', type=int, default=10, help='number of inducing points')
    parser.add_argument('--GP_joint', action='store_true', help='whether to train the kernel params jointly')
    parser.add_argument('--IP_joint', action='store_true', help='whether to train the inducing locations jointly')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for mean loss (instead of sum loss)')
    parser.add_argument('--beta', type=float, default=0.8, help='beta parameter')
    parser.add_argument('--series_batch_size', type=int, default=50, help='batch size for series')
    parser.add_argument('--timestamp_batch_size', type=int, default=20, help='batch size for timestamp')
    parser.add_argument('--max_norm', type=float, default=1e4, help='max norm of gradients in grad norm clipping')

    parser.add_argument('--num_epochs', type=int, default=0, help='number of epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')
    parser.add_argument('--num_plots', type=int, default=6, help='number of plots')
    parser.add_argument('--save', action='store_true', help='whether to save the results')

    args_svgpvae_hmnist = parser.parse_args()
    run_short_hmnist(args_svgpvae_hmnist)



