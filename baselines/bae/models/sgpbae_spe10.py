from typing import Union
from pathlib import Path
import math

import torch
from torch import nn

from models.building_blocks.enc_dec_spe10 import SPE10Decoder
from baselines.bae.models.spgbae_spatial_imp import SGPBAESpatialImp
from baselines.bae.models.nn.bae_enc_spe10 import SPE10EncoderBAE
from baselines.bae.distributions.conditional.normal import ConditionalMeanNormal
from baselines.bae.priors.fixed_priors import PriorGaussian
from baselines.bae.gp.likelihoods import Gaussian
from baselines.bae.models.bsgp import BSGP
from utils.spe10 import NNDatasetSPE10, define_kernel


class SGPBAESPE10(SGPBAESpatialImp):
    def __init__(
            self, file_path: Union[str, Path], M: int, GP_joint=False, IP_joint=False,
            sigma2_y=1., fix_variance=True, kernel_type='rbf', lengthscale=2.,
            sample_dir=Path('bae_samples/spe10'), device='cpu'
    ):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        entire_train_dataset = NNDatasetSPE10(file_path, H=None)

        # AE
        latent_dims, y_dims, x_dims = 3, 4, 3
        encoder = SPE10EncoderBAE(latent_dims, input_dims=y_dims)
        decoder_net = SPE10Decoder(latent_dims, output_dims=y_dims, sigma2_y=sigma2_y, fix_variance=fix_variance)
        decoder = ConditionalMeanNormal(decoder_net, scale=math.sqrt(sigma2_y))
        decoder_prior = PriorGaussian(sw2=1.)

        # GP
        scaled_kernel = define_kernel(kernel_type, latent_dims, x_dims)
        scaled_kernel.outputscale, scaled_kernel.base_kernel.lengthscale = 1., lengthscale
        for p in scaled_kernel.parameters():
            p.requires_grad_(GP_joint)
        if kernel_type == 'cauchy':
            scaled_kernel.base_kernel.alpha = 1.
            scaled_kernel.base_kernel.raw_alpha.requires_grad_(False)
        gp_likelihood = Gaussian(variance=0.00001)

        # IP, different from https://github.com/tranbahien/sgp-bae/blob/master/gp/models/bsgp.py#L62, which shares IP
        # across latent dims, here we make the model as powerful as possible
        inducing_points = nn.Parameter(
            torch.rand([latent_dims, M, 3]) * torch.tensor([30, 110, 43]), requires_grad=IP_joint
        )   # [L, M, 3]
        inducing_values = nn.Parameter(torch.zeros(latent_dims, M), requires_grad=True)

        bsgp = BSGP(
            latent_dims, scaled_kernel, gp_likelihood, inducing_points, inducing_values,
            N_train=len(entire_train_dataset), prior_S_type='uniform',
            prior_lengthscale=2., prior_variance=0.05, prior_lik_var=0.05
        )

        super(SGPBAESPE10, self).__init__(
            encoder, decoder, decoder_prior, bsgp, sample_dir=sample_dir, device=device
        )
        self.train_dataset = entire_train_dataset  # override


if __name__ == '__main__':
    import argparse
    import datetime
    import time
    import pickle
    import numpy as np

    from baselines.bae.adaptive_sghmc import AdaptiveSGHMC
    from utils.spe10 import plot_spe10

    def run_sgpbae_spe10(args):
        current_time = datetime.datetime.now().strftime('%m%d_%H_%M_%S')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float64 if args.float64 else torch.float32)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Time {current_time}, SGPBAE-M{args.M}, Seed: {args.seed}, '
              f'sigma2_y: {args.sigma2_y}, kernel_type={args.kernel_type}, l={args.kernel_lengthscale}\n',
              f'SGHMC_step_size: {args.SGHMC_step_size}, steps(K): {args.SGHMC_steps}, '
              f'encoder steps: {args.encoder_steps}\n'
              f'burn_in_iters: {args.total_burnin_iters}, num_samples: {args.num_samples}, thinning_interval: {args.thinning_interval},\n'
              f'adaptive_burnin_steps: {args.adaptive_burnin_steps}, Device: {device}, {torch.get_default_dtype()}\n')

        s = time.time()
        file_path = Path(f'../data/SPE10/spe10.npz')

        GP_joint_str = "_GP_joint" if args.GP_joint else ""
        IP_joint_str = "_IP_joint" if args.IP_joint else ""
        exp_dir = Path(f'exp_spe10_SGPBAE_M{args.M}' + GP_joint_str + IP_joint_str)
        exp_dir.mkdir(exist_ok=True)
        trial_dir = exp_dir / f'{current_time}'
        trial_dir.mkdir()

        model = SGPBAESPE10(
            file_path, args.M, GP_joint=args.GP_joint, IP_joint=args.IP_joint,
            sigma2_y=args.sigma2_y, kernel_type=args.kernel_type, lengthscale=args.kernel_lengthscale,
            sample_dir=trial_dir / 'samples', device=device
        )
        data_dict = np.load(file_path)
        # create `self.Z` [b,3], `self.Y_tilde`
        model.init_z(torch.as_tensor(data_dict['Y'][:args.batch_size], dtype=torch.get_default_dtype()))

        bae_sampler = AdaptiveSGHMC(
            model.get_parameters(), lr=args.SGHMC_step_size, num_burn_in_steps=args.adaptive_burnin_steps,
            mdecay=0.05, scale_grad=len(model.train_dataset)  # ⚠️ scale_grad???
        )
        encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=0.001)
        model.train_sgpbae(
            bae_sampler, encoder_optimizer,
            n_burnin_iters=args.total_burnin_iters, collect_every=args.thinning_interval, n_samples=args.num_samples,
            SGHMC_steps=args.SGHMC_steps, encoder_steps=args.encoder_steps,
            batch_size=args.batch_size, device=device,
            num_print_iters=args.num_print_iters
        )
        print(f"The samples are saved in {trial_dir.resolve()}.\n")
        print(f"Total training time: {time.time() - s}.")
        print(f"Total parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}.\n")

        # Testing
        mse, mae, nll, pred_mean, pred_std = model.predict_sgpbae(
            batch_size=args.batch_size, stat=None, device=device, return_pred=True
        )
        print(f'\nnll: {nll}, mse: {mse}, mae: {mae}.\n')

        # plot
        fig, axes = plot_spe10(data_dict, pred_mean, layer_idx=20, nll=nll, mse=mse)

        if args.save:
            torch.save(model.state_dict(), trial_dir / 'SGPBAE_spe10.pth')
            fig.savefig(trial_dir / f'SGPBAE_M{args.M}_spe.pdf', bbox_inches='tight')

            everything_for_imgs = {
                'y_rec_mean': pred_mean, 'y_rec_std': pred_std,
                'NLL': nll, 'MSE': mse, 'MAE': mae
            }
            imgs_pickle_path = trial_dir / 'everything_for_imgs.pkl'
            with imgs_pickle_path.open('wb') as f:
                pickle.dump(everything_for_imgs, f)
            print('The experiment results are saved at:')
            print(f'{trial_dir.resolve()}')
        else:
            print('The experiment results are not saved')


    parser = argparse.ArgumentParser(description="SGPBAE for SPE10 in ICML 2023")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--float64', action='store_true', help='use float64')

    parser.add_argument('--M', type=int, default=10, help='number of inducing points')
    parser.add_argument('--GP_joint', action='store_true', help='whether to sample the GP params')
    parser.add_argument('--IP_joint', action='store_true', help='whether to sample the inducing locations')
    parser.add_argument('--sigma2_y', type=float, default=1., help='sigma2_y')
    parser.add_argument('--kernel_type', type=str, default='rbf', help='kernel type')
    parser.add_argument('--kernel_lengthscale', type=float, default=2., help='kernel lengthscale')

    parser.add_argument('--SGHMC_step_size', type=float, default=0.0001, help='step size for SGHMC')
    parser.add_argument('--SGHMC_steps', type=int, default=100, help='K, number of SGHMC steps in Hamiltonian dynamics')
    parser.add_argument('--encoder_steps', type=int, default=30, help='J, number of update steps for encoder params')

    parser.add_argument('--adaptive_burnin_steps', type=int, default=5, help='number of adaptive burn-in steps')
    parser.add_argument('--total_burnin_iters', type=int, default=15, help='number of total burn-in iterations')
    parser.add_argument('--num_samples', type=int, default=2, help='number of stored samples')
    parser.add_argument('--thinning_interval', type=int, default=4,
                        help='number of iterations between thinning steps')

    parser.add_argument('--batch_size', type=int, default=1000, help='mini-batch size in SGHMC')
    parser.add_argument('--num_print_iters', type=int, default=1)
    parser.add_argument('--save', action='store_true', help='whether to save models and results')

    args_bae = parser.parse_args()
    run_sgpbae_spe10(args_bae)



