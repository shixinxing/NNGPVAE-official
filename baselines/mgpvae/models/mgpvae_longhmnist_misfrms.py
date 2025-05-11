from typing import Union
from pathlib import Path
import math

import torch
from torch import nn
from torch.utils.data import DataLoader
from gpytorch.kernels import MaternKernel

from baselines.mgpvae.models.mgpvae_base_misfrms import MGPVAEBaseMisFrms
from baselines.mgpvae.models.ssm import SSM
from baselines.mgpvae.models.building_blocks.kernels import Matern32SSM
from models.building_blocks.enc_dec_longhmnist import LongHMNISTEncoderDiagonal, LongHMNISTDecoder
from utils.lazy_datasets_misfrms import LazySeriesDatasetMisFrms


class MGPVAELongHMNISTMisFrms(MGPVAEBaseMisFrms):
    def __init__(self, file_path: Union[str, Path], GP_joint=True, jitter: float = 1e-6):

        latent_dims = 16
        encoder = LongHMNISTEncoderDiagonal(latent_dims)
        decoder = LongHMNISTDecoder(latent_dims)

        kernel = Matern32SSM(base_kernel=MaternKernel(nu=1.5, batch_shape=torch.Size([latent_dims])))
        kernel.outputscale, kernel.base_kernel.lengthscale = 1., 5.
        for p in kernel.parameters():
            p.requires_grad_(GP_joint)
        ssm = SSM(latent_dims, kernel, jitter=jitter)

        super(MGPVAELongHMNISTMisFrms, self).__init__(encoder, decoder, ssm)
        self.entire_train_dataset = LazySeriesDatasetMisFrms(file_path, train=True, H=None)

    @torch.no_grad()
    def predict_gpvae(
            self, test_file_path: Union[str, Path], series_batch_size: int,
            device: str = "cpu", num_samples=1
    ):
        self.to(device)

        entire_test_dataset = LazySeriesDatasetMisFrms(test_file_path, train=False, H=None)
        series_dataloader = DataLoader(entire_test_dataset, batch_size=series_batch_size, shuffle=False)

        nll_series, mse_series, mse_series_non_round = 0., 0., 0
        num_missing_frames = 0.
        img_rec_full = torch.as_tensor(entire_test_dataset.data_dict['seq_test_full'][:])
        img_rec_full = img_rec_full.clone().to(device)
        for sq_full, t_full, m, vid_ids in series_dataloader:
            # We put the whole sequence onto GPU
            sq_full = sq_full.to(device)                 # [v,T,28,28]
            t_full, m = t_full.to(device), m.to(device=device, dtype=torch.bool)  # [v,T,1]
            Tmax_test = (~m).sum(dim=(-1, -2)).max()

            qnet_mus, qnet_vars, _, _, _, _ = self.forward(sq_full, t_full, m)
            mean_pred_Z, cov_pred_Z = self.ssm.predict(t_full, m, qnet_mus, qnet_vars)  # [v,L,Tmax]
            assert mean_pred_Z.size(-1) == Tmax_test and cov_pred_Z.size(-1) == Tmax_test
            m = m[..., 0]
            mean_pred_Z = mean_pred_Z.mT[~m[..., -Tmax_test:]]  # [<v*Tmax,L]
            cov_pred_Z = cov_pred_Z.mT[~m[..., -Tmax_test:]]
            assert torch.isfinite(cov_pred_Z).all() and (cov_pred_Z > 0.).all(), f"post covar min: {cov_pred_Z.min()}."

            # NLL
            eps = torch.randn([num_samples, *mean_pred_Z.shape], device=device, dtype=torch.get_default_dtype())
            latent_samples = mean_pred_Z + cov_pred_Z.sqrt() * eps
            logits = self.decoder(latent_samples)
            assert torch.isfinite(logits).all()
            log_p = - nn.functional.binary_cross_entropy_with_logits(
                input=logits, target=sq_full[~m].expand_as(logits), reduction='none'
            )  # [s, <v*n, ...]
            log_p = - math.log(num_samples) + torch.logsumexp(log_p, dim=0)
            nll_series += - log_p.sum()

            # MSE
            img_rec = nn.functional.sigmoid(self.decoder(mean_pred_Z))  # [<v*T,28,28]
            mse_series_non_round += (img_rec - sq_full[~m]).square().sum()
            pixels = torch.round(img_rec)
            mse_series += (pixels - sq_full[~m]).square().sum()

            # Fill in the reconstructed
            frame_loc = (~m).to(torch.bool)  # [v,f]
            vid_coords = vid_ids.unsqueeze(-1).expand_as(frame_loc).to(device)
            frms_coords = torch.arange(t_full.size(-2)).expand_as(frame_loc).to(device)
            img_rec_full[vid_coords[frame_loc], frms_coords[frame_loc]] = img_rec

            num_missing_frames += (~m).sum()

        prediction_coll = {
            'images': img_rec_full.cpu().numpy(),                      # ndarray: [V,T,28,28]
            'masks': entire_test_dataset.data_dict['m_test_miss'][:],  # ndarray: [V,T,1]
            'aux_data': entire_test_dataset.data_dict['t_test_full'][:]
        }
        return (
            nll_series / num_missing_frames, mse_series / num_missing_frames, mse_series_non_round / num_missing_frames,
            prediction_coll
        )


if __name__ == '__main__':
    import numpy as np
    import warnings
    from datetime import datetime
    import time
    import pickle
    import argparse
    from utils.healing_mnist_misfrms import sort_frames_time_mask, plot_hmnist_misfrms

    def run_mgpvae_hmnist_misfrms(args):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.float64:
            torch.set_default_dtype(torch.float64)
            warnings.warn(f"Faiss-GPU does not support torch.float64 yet.")
        else:
            torch.set_default_dtype(torch.float32)

        jitter = 1e-8 if args.float64 else 1e-5

        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.manual_seed_all(args.seed)
            # torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.benchmark = False  # for reproducibility
            torch.backends.cudnn.deterministic = True
            print(f"cudnn benchmark: {torch.backends.cudnn.benchmark}")
        else:
            device = 'cpu'

        current_time = datetime.now().strftime("%m%d_%H_%M_%S")
        print(f"Time: {current_time}", f"MarkovianGPVAE for {args.dataset} ",
              f"Using: {device}, seed: {args.seed}, lr={args.lr}, beta={args.beta}, max_norm={args.max_norm} ",
              "GP joint training, " if args.GP_joint else "", f"{torch.get_default_dtype()}\n")

        file_path = Path(f'../data/healing_mnist/h5/{args.dataset}.h5')

        s = time.time()
        model = MGPVAELongHMNISTMisFrms(file_path, GP_joint=args.GP_joint, jitter=jitter)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        model.train_gpvae(
            optimizer, args.beta, epochs=args.num_epochs, series_batch_size=args.series_batch_size,
            max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs
        )
        e = time.time()
        print(f"Training time: {e - s}")
        print(f"Total parameters: {sum([p.shape.numel() for p in model.parameters() if p.requires_grad])}\n")

        # Prediction
        nll, mse, mse_non_round, pred_coll = model.predict_gpvae(
            file_path, series_batch_size=args.series_batch_size, device=device, num_samples=20
        )
        print(f"\nNLL: {nll}, MSE at missing frames: {mse}, MSE (not rounded) per frame: {mse_non_round}.\n")
        pred_coll = sort_frames_time_mask(pred_coll)

        # Plot
        seq_ids = [100 * i for i in range(args.num_plots)]
        fig, ax = plot_hmnist_misfrms(pred_coll, seqs=seq_ids, num_imgs=20)

        experiment_dir = Path(f'./exp_MGPVAE_{args.dataset}')
        if args.save:
            experiment_dir.mkdir(exist_ok=True)
            trial_dir = experiment_dir / f'{current_time}'
            trial_dir.mkdir(parents=False, exist_ok=True)

            torch.save(model.state_dict(), trial_dir / f'model_{args.dataset}.pth')  # model

            img_path = trial_dir / f'img_misfrms.pdf'  # images
            fig.savefig(img_path, bbox_inches='tight')

            everything_for_imgs = {
                # 'y_test_full': y_test_full, 'y_test_miss': y_test_miss, # to save storage
                'y_rec': pred_coll, 'NLL': nll, 'MSE': mse, 'MSE_non_round': mse_non_round
            }
            imgs_pickle_path = trial_dir / 'everything_for_imgs.pkl'
            with imgs_pickle_path.open('wb') as f:
                pickle.dump(everything_for_imgs, f)

            print("The experiment results are saved at:")
            print(f"{trial_dir.resolve()}")  # get absolute path for moving the log file
        else:
            print("The experimental results are not saved.")


    parser = argparse.ArgumentParser(description="Markovian GPVAE on Healing MNIST with Missing Frames")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--beta', type=float, default=0.8, help='beta in ELBO')
    parser.add_argument('--GP_joint', action='store_true', help='whether to train GP params jointly')

    parser.add_argument('--dataset', type=str, default="misfrms_hmnist_all_anticlockwise_100", help='file name')
    parser.add_argument('--series_batch_size', type=int, default=50, help='batch size for series')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate for sum loss (instead of mean loss)')
    parser.add_argument('--max_norm', type=float, default=1e10, help='max norm of gradients in grad norm clipping')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')

    parser.add_argument('--num_plots', type=int, default=10, help='number of series to plot')
    parser.add_argument('--save', action='store_true', help='store everything into a folder')

    args_misfrms_hmnist = parser.parse_args()

    run_mgpvae_hmnist_misfrms(args_misfrms_hmnist)

