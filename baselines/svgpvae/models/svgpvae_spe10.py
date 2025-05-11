from typing import Union
from pathlib import Path
import pickle

import torch
from torch import nn

from baselines.svgpvae.models.svgpvae_base_spatial_imp import SVGPVAESpatialImp
from baselines.svgpvae.models.svgp import SVGP
from models.building_blocks.enc_dec_spe10 import SPE10Encoder, SPE10Decoder
from utils.spe10 import NNDatasetSPE10, plot_spe10, define_kernel


class SVGPVAESPE10(SVGPVAESpatialImp):
    def __init__(
            self, file_path: Union[str, Path], M: int,  GP_joint=True, IP_joint=True,
            sigma2_y=1., fix_variance=True, kernel_type='rbf', lengthscale=2., device='cpu', jitter=1e-6
    ):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        entire_train_dataset = NNDatasetSPE10(file_path, H=None)

        latent_dims, y_dims, x_dims = 3, 4, 3
        encoder = SPE10Encoder(latent_dims, input_dims=y_dims)
        decoder = SPE10Decoder(latent_dims, output_dims=y_dims, sigma2_y=sigma2_y, fix_variance=fix_variance)

        kernel = define_kernel(kernel_type, latent_dims, x_dims)
        kernel.outputscale, kernel.base_kernel.lengthscale = 1., lengthscale
        for p in kernel.parameters():
            p.requires_grad_(GP_joint)
        if kernel_type == 'cauchy':
            kernel.base_kernel.alpha = 1.
            kernel.base_kernel.raw_alpha.requires_grad_(False)
        # sample initial inducing locations evenly distributed at space
        inducing_points = nn.Parameter(
            torch.rand([latent_dims, M, 3]) * torch.tensor([30, 110, 43]), requires_grad=IP_joint
        )   # [L, M, 3]

        svgp = SVGP(latent_dims, kernel, inducing_points, N_train=len(entire_train_dataset), jitter=jitter)

        super(SVGPVAESPE10, self).__init__(encoder, decoder, svgp, device=device, jitter=jitter)
        self.train_dataset = entire_train_dataset  # override


if __name__ == "__main__":
    import argparse
    import datetime
    import time
    import numpy as np

    def run_spe10(args):
        current_time = datetime.datetime.now().strftime("%m%d_%H_%M_%S")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float64 if args.float64 else torch.float32)
        jitter = 1e-8 if args.float64 else 1e-6
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"{current_time}, SPE10, SVGPVAE-M{args.M}, {device}, {torch.get_default_dtype()},\n"
              f"seed: {args.seed}, lr: {args.lr}, beta={args.beta}, max_norm={args.max_norm}\n",
              f"sigma2_y={args.sigma2_y}, kernel_type={args.kernel_type}, l={args.kernel_lengthscale} "
              'GP joint training ' if args.GP_joint else "", 'IP joint training\n' if args.IP_joint else "\n")
        s = time.time()

        file_path = Path(f'../data/SPE10/spe10.npz')

        model = SVGPVAESPE10(
            file_path, args.M, GP_joint=args.GP_joint, IP_joint=args.IP_joint,
            sigma2_y=args.sigma2_y, kernel_type=args.kernel_type, lengthscale=args.kernel_lengthscale,
            device=device, jitter=jitter
        )
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        model.train_gpvae(
            optimizer, args.beta, args.num_epochs, args.batch_size, args.max_norm,
            device=device, print_epochs=args.num_print_epochs
        )
        print(f"Total training time: {time.time() - s}.")
        print(f"Total parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}.\n")

        # Testing
        mse, mae, nll, pred_mean, pred_std = model.predict_gpvae(
            args.batch_size, stat=None, device=device, num_samples=20, return_pred=True
        )
        print(f'\nnll: {nll}, mse: {mse}, mae: {mae}.\n')

        # Plot
        data_dict_true = np.load(file_path)
        fig, ax = plot_spe10(data_dict_true, pred_mean, layer_idx=20, nll=nll, mse=mse)

        if args.save:
            GP_joint_str = "_GP_joint" if args.GP_joint else ""
            IP_joint_str = "_IP_joint" if args.IP_joint else ""
            exp_dir = Path(f'exp_spe10_SVGPVAE_M{args.M}' + GP_joint_str + IP_joint_str)
            exp_dir.mkdir(exist_ok=True)
            trial_dir = exp_dir / f'{current_time}'
            trial_dir.mkdir(parents=False, exist_ok=True)

            torch.save(model.state_dict(), trial_dir / f'model_svgpvae_spe10.pth')  # model

            img_path = trial_dir / f'img_svgpvae_spe10.pdf'  # images
            fig.savefig(img_path, bbox_inches='tight')

            everything_for_imgs = {
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


    parser = argparse.ArgumentParser(description='SVGPVAE for Long HMNIST imputation')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--M', type=int, default=10, help='number of inducing points')
    parser.add_argument('--GP_joint', action='store_true', help='whether to train the kernel params jointly')
    parser.add_argument('--IP_joint', action='store_true', help='whether to train the inducing locations jointly')
    parser.add_argument('--sigma2_y', type=float, default=1., help='noise variance in output distribution')
    parser.add_argument('--kernel_type', type=str, default='rbf', help='kernel type')
    parser.add_argument('--kernel_lengthscale', type=float, default=2., help='kernel lengthscale')

    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for mean loss (instead of sum loss)')
    parser.add_argument('--beta', type=float, default=0.8, help='beta parameter')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size along series')
    parser.add_argument('--max_norm', type=float, default=1e4, help='max norm of gradients in grad norm clipping')

    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')
    parser.add_argument('--save', action='store_true', help='whether to save the results')

    args_svgpvae_spe10 = parser.parse_args()
    run_spe10(args_svgpvae_spe10)

