from typing import Union
from pathlib import Path
import warnings
import math
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from baselines.vae.cvae import CVAEBase
from baselines.vae.cvae_building_blocks.aux_nets import AuxEncoderHMNISTMisFrms
from baselines.vae.cvae_building_blocks.enc_dec_cvae_hmnist import CVAELongHMNISTEncoderDiagonal, CVAELongHMNISTDecoder
from utils.lazy_datasets_misfrms import LazySeriesDatasetMisFrms, NNDatasetMisFrms, generate_shorter_miniseries_dict


class CVAELongHMNISTMisFrms(CVAEBase):
    def __init__(self, file_path: Union[str, Path]):
        latent_dims = 16
        encoder = CVAELongHMNISTEncoderDiagonal(latent_dims)
        decoder = CVAELongHMNISTDecoder(latent_dims)
        aux_encoder = AuxEncoderHMNISTMisFrms(latent_dims)
        # aux_decoder = AuxEncoderHMNISTMisFrms(latent_dims)
        aux_decoder = aux_encoder

        super(CVAELongHMNISTMisFrms, self).__init__(encoder, decoder, aux_encoder, aux_decoder)
        self.entire_train_dataset = LazySeriesDatasetMisFrms(file_path, train=True, H=None)
        self.train_dataset = None

    def _handle_m_batch(self, y_batch, m_batch):
        if m_batch is None:
            m_batch = torch.ones(
                y_batch.shape[:len(self.train_dataset.series_shape) + 1], dtype=torch.bool, device=y_batch.device)
        else:
            m_batch = m_batch[..., 0].to(torch.bool)  # [v,t]
            if torch.all(~m_batch):
                warnings.warn("Got a batch without observed frames.")
                return torch.tensor(0., device=y_batch.device)
            assert m_batch.shape[:-1] == self.train_dataset.series_shape
        return m_batch  # [v,T]

    def expected_log_prob(self, y_batch: Tensor, t_batch: Tensor, m_batch: Tensor = None):
        m_batch = self._handle_m_batch(y_batch, m_batch)
        # mask first and then feed to save computation
        y_pick, t_pick = y_batch[m_batch], t_batch[m_batch]  # [<v*t, D]
        means, stds, y_logits = self.forward(y_pick, t_pick)

        assert self.decoder.output_distribution == 'bernoulli'
        expected_lk = - nn.functional.binary_cross_entropy_with_logits(
            input=y_logits, target=y_pick, reduction='none'  # [<v*t, D]
        )
        expected_lk = expected_lk.sum(dim=[i for i in range(1, expected_lk.ndim)])  # [<v*t]
        expected_lk_mat = torch.zeros_like(m_batch, requires_grad=True, dtype=y_batch.dtype)  # [v,t]
        expected_lk_mat = expected_lk_mat.clone()  # in-place operation not allowed by leaf variables
        expected_lk_mat[m_batch] = expected_lk     # assign

        # scaling of each video
        len_batch_inv = torch.where(m_batch.sum(dim=-1) > 0, 1 / m_batch.sum(-1), 0.)  # [v] in case of no observations
        lik = torch.sum(expected_lk_mat * len_batch_inv.unsqueeze(-1))
        return lik / self.train_dataset.series_shape.numel()  # average over v and f

    def kl_divergence(self, y_batch: Tensor, t_batch: Tensor, m_batch=None) -> Tensor:
        m_batch = self._handle_m_batch(y_batch, m_batch)
        # KL is factorized across data points
        mean_q, std_q = self.encode(y_batch, t_batch)
        mahalanobis = mean_q.square().sum(dim=-1)
        trace = std_q.square().sum(dim=-1)
        log_det_cov_q = std_q.log().sum(dim=-1)

        res = 0.5 * (mahalanobis + trace - self.latent_dims) - log_det_cov_q  # [v,t]
        len_batch_inv = torch.where(m_batch.sum(dim=-1) > 0, 1 / m_batch.sum(-1), 0.)  # [v] in case of no observations
        res = (res * m_batch).sum(dim=-1) * len_batch_inv
        return res.mean()  # average over v

    def average_loss(self, y_batch: Tensor, t_batch: Tensor, m_batch=None, beta=1.):
        exp_lik = self.expected_log_prob(y_batch, t_batch, m_batch)
        kl = self.kl_divergence(y_batch, t_batch, m_batch)
        assert torch.isfinite(exp_lik).all(), "Likelihood now is not finite."
        assert torch.isfinite(kl).all(), "KL divergence now is not finite."
        elbo = exp_lik - beta * kl
        return - elbo

    def train_cvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs: int,
            series_batch_size: int, timestamp_batch_size: int, max_norm: float = None,
            device='cpu', print_epochs=1
    ):
        self.to(device)

        series_dataloader = DataLoader(self.entire_train_dataset, batch_size=series_batch_size, shuffle=True)
        for epoch in range(epochs):
            for miniseries_sq_full, miniseries_t_full, miniseries_t_mask, vid_ids in series_dataloader:
                miniseries_shorter_dict = generate_shorter_miniseries_dict(
                    miniseries_sq_full, miniseries_t_full, miniseries_t_mask, clip=True
                )
                self.train_dataset = NNDatasetMisFrms(
                    miniseries_shorter_dict, series_shape=miniseries_t_full.shape[:-2]
                )

                time_dataloader = DataLoader(self.train_dataset, batch_size=timestamp_batch_size, shuffle=True)
                for sq_short, t_short, m_short in time_dataloader:
                    sq_short = sq_short.to(device).transpose(0, 1)  # [v, T, D]
                    t_short = t_short.to(device).transpose(0, 1).contiguous()  # [v, T, 1]
                    m_short = m_short.to(device).transpose(0, 1)

                    optimizer.zero_grad(set_to_none=True)
                    loss = self.average_loss(sq_short, t_short, m_short, beta=beta)
                    loss.backward()
                    if max_norm is not None:
                        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm, error_if_nonfinite=True)
                        if grad_norm > max_norm:
                            print(f"previous grad norm: {grad_norm} > {max_norm}, gradient clipped!!!")
                    optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item(): .6f}',
                      f'global grad norm: {grad_norm}\n' if max_norm is not None else '\n')

    @torch.no_grad()
    def predict_cvae(
            self, test_file_path: Union[str, Path], series_batch_size: int,
            device='cpu', num_samples=1
    ):
        """
        refer to Casale 2018 paper Page 6 on conditional generation, utilizing mean latent representation
        """
        self.to(device)
        entire_test_dataset = LazySeriesDatasetMisFrms(test_file_path, train=False, H=None)
        series_dataloader = DataLoader(entire_test_dataset, batch_size=series_batch_size, shuffle=False)

        img_rec_full = []
        num_missing_frames, nll, se = 0., 0., 0.
        for miniseries_se_full, miniseries_t_full, miniseries_t_mask, vid_ids in series_dataloader:
            miniseries_se_full, miniseries_t_full = miniseries_se_full.to(device), miniseries_t_full.to(device)
            miniseries_t_mask = miniseries_t_mask.to(device=device, dtype=torch.bool)
            num_missing_frames += (~miniseries_t_mask).sum()

            nll_batch, se_batch = 0., 0.
            seq_rec_coll = []
            for v in range(miniseries_se_full.shape[0]):
                seq, t = miniseries_se_full[v], miniseries_t_full[v]  # [T,28,28], [T,1]
                m = miniseries_t_mask[v].squeeze(-1)                  # [T]
                seq_obs = seq[m]                                      # [~T,28,28]

                # Step 1: get all latent representation of all observations
                encoder_out = self.encoder(seq_obs)
                # Step 2: average over one series
                latent_rep_mean = encoder_out.mean(dim=0)            # [2L]
                # Step 3: conditional generation
                t_unobs = t[~m]
                aux_enc_out = self.aux_encoder(t_unobs)
                before_z = torch.cat([latent_rep_mean.expand(t_unobs.size(0), -1), aux_enc_out], dim=-1)
                z = self.before_z(before_z)
                means, stds = z[..., :self.latent_dims], nn.functional.softplus(z[..., self.latent_dims:])

                # NLL
                eps = torch.randn([num_samples, *means.shape], device=device)
                z = means + eps * stds
                logits = self.decode(z, t_unobs.expand(num_samples, *t_unobs.shape))
                log_p = - nn.functional.binary_cross_entropy_with_logits(
                    input=logits, target=seq[~m].expand_as(logits), reduction='none'
                )
                log_p = - math.log(num_samples) + torch.logsumexp(log_p, dim=0)
                nll_batch += - log_p.sum()

                # MSE
                img_rec = nn.functional.sigmoid(self.decode(means, t_unobs))
                pixels = torch.round(img_rec)
                se_batch += (pixels - seq[~m]).square().sum()
                seq_rec = seq.clone()
                seq_rec[~m] = img_rec  # assign
                seq_rec_coll.append(seq_rec)
            nll += nll_batch
            se += se_batch
            seq_rec_coll = torch.stack(seq_rec_coll, dim=0)
            img_rec_full.append(seq_rec_coll)

        img_rec_full = torch.cat(img_rec_full, dim=0).cpu().numpy()
        prediction_coll = {
            'images': img_rec_full, 'masks': entire_test_dataset.data_dict['m_test_miss'][:],
            'aux_data': entire_test_dataset.data_dict['t_test_full'][:]
        }

        return  (nll/num_missing_frames).item(), (se/num_missing_frames).item(), prediction_coll


if __name__ == '__main__':
    import argparse
    import numpy as np
    from datetime import datetime
    import time
    import pickle
    from utils.healing_mnist_misfrms import plot_hmnist_misfrms, sort_frames_time_mask


    def run_cvae_hmnist_misfrms(args):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.float64:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False  # for reproducibility
            torch.backends.cudnn.deterministic = True
            print(f"cudnn benchmark: {torch.backends.cudnn.benchmark}")
        else:
            device = 'cpu'

        current_time = datetime.now().strftime("%m%d_%H_%M_%S")
        print(f"Time: {current_time}, CVAE for HMNIST misfrms ",
              f"Using: {device}, seed: {args.seed}, lr={args.lr}, beta={args.beta}, max_norm={args.max_norm} ",
              f"{torch.get_default_dtype()}\n")

        file_path = Path(f'../data/healing_mnist/h5/{args.dataset}.h5')

        s = time.time()
        model = CVAELongHMNISTMisFrms(file_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        model.train_cvae(
            optimizer, args.beta, epochs=args.num_epochs,
            series_batch_size=args.series_batch_size, timestamp_batch_size=args.timestamp_batch_size,
            max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs
        )
        print(f"Training time: {time.time() - s}")
        print(f"Total parameters: {sum([p.shape.numel() for p in model.parameters() if p.requires_grad])}\n")

        # Prediction
        nll, mse, pred_coll = model.predict_cvae(
            file_path, series_batch_size=args.series_batch_size, device=device, num_samples=20
        )
        print(f"\nNLL: {nll}, MSE at missing frames: {mse}.\n")

        # Plot
        pred_coll = sort_frames_time_mask(pred_coll)
        seq_ids = [100 * i for i in range(args.num_plots)]
        fig, ax = plot_hmnist_misfrms(pred_coll, seqs=seq_ids, num_imgs=20)

        experiment_dir = Path(f'./exp_{args.dataset}_CVAE')
        if args.save:
            experiment_dir.mkdir(exist_ok=True)
            trial_dir = experiment_dir / f'{current_time}'
            trial_dir.mkdir(parents=False, exist_ok=True)

            torch.save(model.state_dict(), trial_dir / f'cvae_{args.dataset}.pth')  # model

            img_path = trial_dir / f'img_cvae_misfrms.pdf'  # images
            fig.savefig(img_path, bbox_inches='tight')

            everything_for_imgs = {
                # 'y_test_full': y_test_full, 'y_test_miss': y_test_miss, # to save storage
                'y_rec': pred_coll, 'NLL': nll, 'MSE': mse
            }
            imgs_pickle_path = trial_dir / 'everything_for_imgs.pkl'
            with imgs_pickle_path.open('wb') as f:
                pickle.dump(everything_for_imgs, f)

            print("The experiment results are saved at:")
            print(f"{trial_dir.resolve()}")  # get absolute path for moving the log file
        else:
            print("The experimental results are not saved.")


    parser = argparse.ArgumentParser(description='CVAE for Long HMNIST Misfrms')
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--beta', type=float, default=1., help='beta in ELBO')
    parser.add_argument('--dataset', type=str, default="misfrms_hmnist_all_anticlockwise_100", help='file name')
    parser.add_argument('--series_batch_size', type=int, default=50, help='batch size for series')
    parser.add_argument('--timestamp_batch_size', type=int, default=20, help='batch size for timestamp')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate for sum loss (instead of mean loss)')
    parser.add_argument('--max_norm', type=float, default=1e5, help='max norm of gradients in grad norm clipping')
    parser.add_argument('--num_epochs', type=int, default=0, help='number of training epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')

    parser.add_argument('--num_plots', type=int, default=4, help='number of series to plot')
    parser.add_argument('--save', action='store_true', help='store everything into a folder')

    args_cvae = parser.parse_args()
    run_cvae_hmnist_misfrms(args_cvae)

