from typing import Union
from pathlib import Path
import h5py
import pickle

import numpy as np
import torch
from torch import nn
from gpytorch.kernels import ScaleKernel, RBFKernel

from baselines.svgpvae.models.svgpvae_hmnist import SVGPVAEHealing
from baselines.svgpvae.models.svgpvae_base import SVGPVAEBase
from baselines.svgpvae.models.svgp import SVGP
from models.building_blocks.enc_dec_longhmnist import LongHMNISTEncoderDiagonal, LongHMNISTDecoder
from utils.healing_mnist import LazySeriesDataset, plot_hmnist


class SVGPVAELongHealing(SVGPVAEHealing, SVGPVAEBase):
    def __init__(
            self, M: int, file_path: Union[str, Path],
            GP_joint=True, IP_joint=True, device='cpu', jitter=1e-6

    ):
        self.entire_train_dataset = LazySeriesDataset(file_path, train=True, return_full=False, val_split=None)
        self.train_dataset = None
        self.num_videos = len(self.entire_train_dataset)
        self.num_frames_per_series = self.entire_train_dataset[0][0].size(0)

        latent_dims = 16
        encoder = LongHMNISTEncoderDiagonal(latent_dims)
        decoder = LongHMNISTDecoder(latent_dims)

        kernel = ScaleKernel(RBFKernel(batch_shape=torch.Size([latent_dims])))
        kernel.outputscale, kernel.base_kernel.lengthscale = 1., 5.
        for p in kernel.parameters():
            p.requires_grad_(GP_joint)
        inducing_points = nn.Parameter(
            torch.linspace(0, self.num_frames_per_series-1, M).unsqueeze(-1).repeat(latent_dims, 1, 1),
            requires_grad=IP_joint   # [L, M, 1]
        )
        svgp = SVGP(latent_dims, kernel, inducing_points, N_train=self.num_frames_per_series, jitter=jitter)
        self.train_dataset = None  # iterable along frames

        super(SVGPVAEHealing, self).__init__(encoder, decoder, svgp, device=device, jitter=jitter, geco=False)


if __name__ == '__main__':
    import argparse
    import datetime
    import time

    def run_long_hmnist(args):
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
        print(f"Time: {current_time}, SVGPVAE-M{args.M}, Device: {device}, {torch.get_default_dtype()},\n"
              f"seed: {args.seed}, lr: {args.lr} ",
              'GP joint training ' if args.GP_joint else " ", 'IP joint training' if args.IP_joint else " ")
        s = time.time()

        file_path = Path(f'../data/healing_mnist/h5/long_hmnist_all_{args.random_mechanism}.h5')

        model = SVGPVAELongHealing(
            args.M, file_path, GP_joint=args.GP_joint, IP_joint=args.IP_joint, device=device, jitter=jitter
        )
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        model.train_gpvae(
            optimizer, args.beta, args.num_epochs, args.series_batch_size, args.timestamp_batch_size, args.max_norm,
            device=device, print_epochs=args.num_print_epochs
        )

        # Testing
        nll, mse, mse_non_round, y_rec = model.predict_gpvae(
            file_path, series_batch_size=args.series_batch_size, timestamp_batch_size=args.timestamp_batch_size,
            device=device, num_samples=20
        )
        print(f"\nSVGPVAE-M{args.M} NLL: {nll}, MSE at missing pixels: {mse}, "
              f"MSE (not rounded) at all pixels: {mse_non_round}.\n")

        # Plot
        video_idx = [10*i for i in range(args.num_plots)]
        test_dict = h5py.File(file_path, 'r')
        fig, ax = plot_hmnist(
            test_dict['x_test_miss'][video_idx, :10].reshape(-1, 10, 28, 28),
            y_rec[video_idx, :10],
            test_dict['x_test_full'][video_idx, :10].reshape(-1, 10, 28, 28)
        )

        print(f"Total time: {time.time() - s}.\n")
        print(f"Total params: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

        if args.save:
            GP_joint_str = "_GP_joint" if args.GP_joint else ""
            IP_joint_str = "_IP_joint" if args.IP_joint else ""
            exp_dir = Path(f'exp_SVGPVAE_M{args.M}_long_hmnist_{args.random_mechanism}' + GP_joint_str + IP_joint_str)
            exp_dir.mkdir(exist_ok=True)
            trial_dir = exp_dir / f'{current_time}'
            trial_dir.mkdir(parents=False, exist_ok=True)

            torch.save(model.state_dict(), trial_dir / f'model_longhmnist.pth')  # model

            img_path = trial_dir / f'img_svgpvae_longhmnist.pdf'  # images
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

    parser.add_argument('--random_mechanism', type=str, default='consecutive_normal_random_1000',
                        choices=['anticlockwise_mnar_100', 'consecutive_normal_random_1000'])

    parser.add_argument('--M', type=int, default=10, help='number of inducing points')
    parser.add_argument('--GP_joint', action='store_true', help='whether to train the kernel params jointly')
    parser.add_argument('--IP_joint', action='store_true', help='whether to train the inducing locations jointly')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for mean loss (instead of sum loss)')
    parser.add_argument('--beta', type=float, default=0.8, help='beta parameter')
    parser.add_argument('--series_batch_size', type=int, default=40, help='batch size for series')
    parser.add_argument('--timestamp_batch_size', type=int, default=20, help='batch size for timestamp')
    parser.add_argument('--max_norm', type=float, default=1e4, help='max norm of gradients in grad norm clipping')

    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')
    parser.add_argument('--num_plots', type=int, default=6, help='number of plots')
    parser.add_argument('--save', action='store_true', help='whether to save the results')

    args_svgpvae_hmnist = parser.parse_args()
    run_long_hmnist(args_svgpvae_hmnist)



