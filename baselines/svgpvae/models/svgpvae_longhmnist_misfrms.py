from typing import Union
from pathlib import Path
import math

import torch
from torch import nn
from torch.utils.data import DataLoader
from gpytorch.kernels import RBFKernel, ScaleKernel

from baselines.svgpvae.models.svgpvae_base_misfrms import SVGPVAEBaseMisFrms
from baselines.svgpvae.models.svgp import SVGP
from models.building_blocks.enc_dec_longhmnist import LongHMNISTEncoderDiagonal, LongHMNISTDecoder
from utils.lazy_datasets_misfrms import LazySeriesDatasetMisFrms, generate_shorter_miniseries_dict, NNDatasetMisFrms


class SVGPVAELongHMNISTMisFrms(SVGPVAEBaseMisFrms):
    def __init__(
            self, M: int, file_path: Union[str, Path],
            GP_joint=True, IP_joint=True, device="cpu", jitter=1e-5
    ):
        self.entire_train_dataset = LazySeriesDatasetMisFrms(file_path, train=True, H=None)
        self.num_frames_before_clip = self.entire_train_dataset[0][0].size(0)
        self.train_dataset = None
        self.data_device = 'cpu'

        latent_dims = 16
        encoder = LongHMNISTEncoderDiagonal(latent_dims)
        decoder = LongHMNISTDecoder(latent_dims)

        kernel = ScaleKernel(RBFKernel(batch_shape=torch.Size([latent_dims])))
        kernel.outputscale, kernel.base_kernel.lengthscale = 1., 5.
        for p in kernel.parameters():
            p.requires_grad_(GP_joint)
        inducing_points = nn.Parameter(
            torch.linspace(0, self.num_frames_before_clip, M).unsqueeze(-1).repeat(latent_dims, 1, 1),
            requires_grad=IP_joint
        )
        svgp = SVGP(latent_dims, kernel, inducing_points, N_train=None, jitter=jitter)

        super(SVGPVAELongHMNISTMisFrms, self).__init__(encoder, decoder, svgp, device=device, jitter=jitter)

    def train_gpvae(self, *args, **kwargs):
        return super(SVGPVAELongHMNISTMisFrms, self).train_gpvae(*args, **kwargs)

    @torch.no_grad()
    def predict_gpvae(
            self, test_file_path: Union[str, Path], series_batch_size: int, timestamp_batch_size: int,
            device: str = "cpu", num_samples=1
    ):
        self.to(device)

        entire_test_dataset = LazySeriesDatasetMisFrms(test_file_path, train=False, H=None)
        series_dataloader = DataLoader(entire_test_dataset, batch_size=series_batch_size, shuffle=False)

        nll_series, mse_series, mse_series_non_round = 0., 0., 0
        num_missing_frames = 0.
        img_rec_full = torch.as_tensor(
            entire_test_dataset.data_dict['seq_test_full'][:], dtype=torch.get_default_dtype())
        img_rec_full = img_rec_full.clone().to(device)
        for miniseries_sq_full, miniseries_t_full, miniseries_mask, vid_ids in series_dataloader:

            miniseries_whole_dict = generate_shorter_miniseries_dict(
                miniseries_sq_full, miniseries_t_full, miniseries_mask.to(torch.bool), clip=False
            )
            test_whole_dataset = NNDatasetMisFrms(
                miniseries_whole_dict, series_shape=miniseries_t_full.shape[:-2], data_device=self.data_device
            )
            time_dataloader = DataLoader(test_whole_dataset, batch_size=timestamp_batch_size, shuffle=False)

            # mini-batch encoding
            tmp_store = [[] for _ in range(5)]
            for sq_short, t_short, m_short in time_dataloader:
                sq_short = sq_short.to(device).transpose(0, 1)                  # [v,b,D]
                t_short = t_short.to(device).transpose(0, 1).contiguous()       # [v,b,1]
                m_short = m_short.to(device).transpose(0, 1)                    # [v,b,1]
                qnet_mus, qnet_vars = self.build_MLP_inference_graph(sq_short)  # [v,b,L]

                for i, item in enumerate([sq_short, t_short, m_short, qnet_mus, qnet_vars]):
                    tmp_store[i].append(item)
            # concatenate -> [v,T,28,28]
            sq_long, t_long, m_long, qnet_mus, qnet_vars = [torch.cat(x, dim=1) for x in tmp_store]

            # predict using the whole series
            mu_qs, cov_qs, _, _ = self.svgp.approx_posterior_params(
                t_long, qnet_mus, qnet_vars, x_test=t_long, diag_cov_qs=True,
                m_train_batch=m_long, N_train=miniseries_mask.to(device).sum(dim=(-1, -2))
            )                            # [v,L,T], [v,L,T]
            assert torch.isfinite(cov_qs).all() and (cov_qs > 0.).all()
            m_long = m_long[..., 0]
            mu_qs = mu_qs.mT[~m_long]     # [<v*T, L]
            cov_qs = cov_qs.mT[~m_long]

            # NLL
            eps = torch.randn([num_samples, *mu_qs.shape], device=device, dtype=mu_qs.dtype)
            latent_samples = mu_qs + eps * cov_qs.sqrt()
            logits = self.decoder(latent_samples)
            log_p = - nn.functional.binary_cross_entropy_with_logits(
                input=logits, target=sq_long[~m_long].expand_as(logits), reduction='none'
            )  # [s, <v*T, ...]
            log_p = - math.log(num_samples) + torch.logsumexp(log_p, dim=0)
            nll_series += - log_p.sum()

            # MSE
            img_rec = nn.functional.sigmoid(self.decoder(mu_qs))  # [<v*T,28,28]
            mse_series_non_round += (img_rec - sq_long[~m_long]).square().sum()
            pixels = torch.round(img_rec)
            mse_series += (pixels - sq_long[~m_long]).square().sum()

            # Fill in the reconstructed
            frame_loc = (~m_long).to(torch.bool)    # [v,f]
            vid_coords = vid_ids.unsqueeze(-1).expand_as(frame_loc).to(device)
            frms_coords = torch.arange(miniseries_t_full.size(-2)).expand_as(frame_loc).to(device)
            img_rec_full[vid_coords[frame_loc], frms_coords[frame_loc]] = img_rec

            num_missing_frames += (~m_long).sum()

        prediction_coll = {
            'images': img_rec_full.cpu().numpy(),  # ndarray: [V,T,28,28]
            'masks': entire_test_dataset.data_dict['m_test_miss'][:],  # ndarray: [V,T,1]
            'aux_data': entire_test_dataset.data_dict['t_test_full'][:]
        }
        return (
            (nll_series / num_missing_frames).item(), (mse_series / num_missing_frames).item(),
            (mse_series_non_round / num_missing_frames).item(),
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

    def run_svgpvae_hmnist_misfrms(args):
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
        print(f"Time: {current_time}", f"SVGPVAE-M{args.M} for {args.dataset} ",
              f"Using: {device}, seed: {args.seed}, lr={args.lr}, beta={args.beta}, max_norm={args.max_norm} ",
              "GP joint training, " if args.GP_joint else "",
              "IP joint training, " if args.IP_joint else "", f"{torch.get_default_dtype()}\n")

        file_path = Path(f'../data/healing_mnist/h5/{args.dataset}.h5')

        s = time.time()
        model = SVGPVAELongHMNISTMisFrms(
            args.M, file_path, GP_joint=args.GP_joint, IP_joint=args.IP_joint, device=device, jitter=jitter
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        model.train_gpvae(
            optimizer, args.beta, epochs=args.num_epochs,
            series_batch_size=args.series_batch_size, timestamp_batch_size=args.timestamp_batch_size,
            max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs
        )
        e = time.time()
        print(f"Training time: {e - s}")
        print(f"Total parameters: {sum([p.shape.numel() for p in model.parameters() if p.requires_grad])}\n")

        # Prediction
        nll, mse, mse_non_round, pred_coll = model.predict_gpvae(
            file_path, series_batch_size=args.series_batch_size, timestamp_batch_size=args.timestamp_batch_size,
            device=device, num_samples=20
        )
        print(f"\nNLL: {nll}, MSE at missing frames: {mse}, MSE (not rounded) per frame: {mse_non_round}.\n")
        pred_coll = sort_frames_time_mask(pred_coll)

        # Plot
        seq_ids = [100 * i for i in range(args.num_plots)]
        fig, ax = plot_hmnist_misfrms(pred_coll, seqs=seq_ids, num_imgs=20)

        experiment_dir = Path(f'./exp_{args.dataset}_SVGPVAE_M{args.M}')
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


    parser = argparse.ArgumentParser(description="SVGPVAE on Healing MNIST with Missing Frames")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--M', type=int, default=10, help='number of inducing points')
    parser.add_argument('--beta', type=float, default=0.8, help='beta in ELBO')
    parser.add_argument('--GP_joint', action='store_true', help='whether to train GP params jointly')
    parser.add_argument('--IP_joint', action='store_true', help='whether to train inducing locations jointly')

    parser.add_argument('--dataset', type=str, default="misfrms_hmnist_all_anticlockwise_100", help='file name')
    parser.add_argument('--series_batch_size', type=int, default=50, help='batch size for series')
    parser.add_argument('--timestamp_batch_size', type=int, default=20, help='batch size for timestamp')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate for sum loss (instead of mean loss)')
    parser.add_argument('--max_norm', type=float, default=1e10, help='max norm of gradients in grad norm clipping')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')

    parser.add_argument('--num_plots', type=int, default=10, help='number of series to plot')
    parser.add_argument('--save', action='store_true', help='store everything into a folder')

    args_misfrms_hmnist = parser.parse_args()

    run_svgpvae_hmnist_misfrms(args_misfrms_hmnist)

