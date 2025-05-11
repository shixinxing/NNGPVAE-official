from typing import Union
from pathlib import Path

import torch
from gpytorch.kernels import MaternKernel

from baselines.mgpvae.models.mgpvae_base import MGPVAEBase
from baselines.mgpvae.models.mgpvae_hmnist import MGPVAEHealing
from baselines.mgpvae.models.ssm import SSM
from baselines.mgpvae.models.building_blocks.kernels import Matern32SSM
from models.building_blocks.enc_dec_longhmnist import LongHMNISTEncoderDiagonal, LongHMNISTDecoder
from utils.healing_mnist import LazySeriesDataset


class MGPVAELongHealing(MGPVAEHealing, MGPVAEBase):
    def __init__(
            self, file_path: Union[str, Path], GP_joint=True, jitter: float = 1e-6,
    ):
        latent_dims = 16
        encoder = LongHMNISTEncoderDiagonal(latent_dims=latent_dims)
        decoder = LongHMNISTDecoder(latent_dims=latent_dims)

        kernel = Matern32SSM(base_kernel=MaternKernel(nu=1.5, batch_shape=torch.Size([latent_dims])))
        kernel.outputscale, kernel.base_kernel.lengthscale = 1., 5.
        for p in kernel.parameters():
            p.requires_grad_(GP_joint)
        ssm = SSM(latent_dims, kernel, jitter=jitter)

        super(MGPVAEHealing, self).__init__(encoder, decoder, ssm)

        self.entire_train_dataset = LazySeriesDataset(file_path, train=True, return_full=False, val_split=None)
        self.num_videos = len(self.entire_train_dataset)
        self.num_frames_per_series = self.entire_train_dataset[0][0].size(0)
        self.train_dataset = None


if __name__ == '__main__':
    import argparse
    import datetime
    import time
    import numpy as np
    import h5py
    import pickle
    from utils.healing_mnist import plot_hmnist

    def run_long_hmnist_mgpvae(args):
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
        print(f"Time: {current_time}, Markovian GPVAE for long HMNIST, Device: {device}, {torch.get_default_dtype()},\n"
              f"seed: {args.seed}, lr={args.lr}, beta={args.beta}, max_norm={args.max_norm} ",
              "GP joint training" if args.GP_joint else " ")
        s = time.time()

        file_path = Path(f'../data/healing_mnist/h5/long_hmnist_all_{args.random_mechanism}.h5')

        # Training
        model = MGPVAELongHealing(file_path, GP_joint=args.GP_joint, jitter=jitter)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        model.train_gpvae(
            optimizer, args.beta, args.num_epochs, args.series_batch_size,
            max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs
        )
        print(f"Training time: {time.time() - s}.\n")
        print(f"Total params: {sum([p.numel() for p in model.parameters() if p.requires_grad])}.\n")

        # Testing, series batch
        nll, mse, mse_non_round, y_rec = model.predict_gpvae(
            file_path, series_batch_size=args.series_batch_size, device=device, num_samples=20
        )
        print(f"\nMGPVAE NLL: {nll}, MSE at missing pixels: {mse}, MSE (not rounded) at all pixels: {mse_non_round}.\n")

        # Plot
        video_idx = [10*i for i in range(args.num_plots)]
        test_dict = h5py.File(file_path, 'r')
        fig, ax = plot_hmnist(
            test_dict['x_test_miss'][video_idx, :10].reshape(-1, 10, 28, 28),
            y_rec[video_idx, :10],
            test_dict['x_test_full'][video_idx, :10].reshape(-1, 10, 28, 28)
        )

        if args.save:
            GP_joint_str = "_GP_joint" if args.GP_joint else ""
            exp_dir = Path(f'exp_MVGPVAE_long_hmnist_{args.random_mechanism}' + GP_joint_str)
            exp_dir.mkdir(exist_ok=True)
            trial_dir = exp_dir / f'{current_time}'
            trial_dir.mkdir(parents=False, exist_ok=True)

            torch.save(model.state_dict(), trial_dir / f'model_long_hmnist_{args.random_mechanism}.pth')  # model

            img_path = trial_dir / f'img_mvgpvae_long_hmnist.pdf'  # images
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


    parser = argparse.ArgumentParser(description="MGPVAE for Long HMNIST imputation")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--random_mechanism', type=str, default='anticlockwise_mnar_100',
                        choices=['anticlockwise_mnar_100', 'consecutive_normal_random_1000'])

    parser.add_argument('--GP_joint', action='store_true', help='whether to train the kernel params jointly')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for mean loss (instead of sum loss)')
    parser.add_argument('--beta', type=float, default=0.8, help='beta parameter')
    parser.add_argument('--max_norm', type=float, default=1e4, help='max norm of gradients in grad norm clipping')

    parser.add_argument('--series_batch_size', type=int, default=40, help='batch size for series')
    parser.add_argument('--num_epochs', type=int, default=0, help='number of epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')
    parser.add_argument('--num_plots', type=int, default=6, help='number of plots')
    parser.add_argument('--save', action='store_true', help='whether to save the results')

    args_mgpvae_hmnist = parser.parse_args()
    run_long_hmnist_mgpvae(args_mgpvae_hmnist)

