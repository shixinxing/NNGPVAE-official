import torch
from torch import nn
from torch.utils.data import DataLoader
from linear_operator.utils.cholesky import psd_safe_cholesky

from models.gpvae_hmnist import GPVAEHealingBase
from utils.healing_mnist import generate_miniseries_hmnist_dict, LazySeriesDataset


class GPPVAECasaleHealing(GPVAEHealingBase):
    def __init__(self, file_path, val_split=50000, search_device='cpu', jitter=1e-6):
        super(GPPVAECasaleHealing, self).__init__(10, file_path, val_split, search_device, jitter=jitter)
        self.num_frames_per_series = 10

    def expected_log_prob(self, y_batch_miss, m_batch_miss):  # noqa
        means, stds, y_rec = self.forward(y_batch_miss)
        m_batch_miss = m_batch_miss.to(torch.bool)
        assert self.decoder.output_distribution == 'bernoulli'
        expected_lk = - nn.functional.binary_cross_entropy(input=y_rec, target=y_batch_miss, reduction='none')
        expected_lk = torch.where(m_batch_miss, 0., expected_lk).sum()
        # ⚠️ we currently do not include missing rate, which is the same as the AISTATS 2020
        return expected_lk  # * self.num_frames_per_series / y_batch_miss.shape[:2].numel()

    def kl_divergence(self, y_miss: torch.Tensor, t: torch.Tensor):               # [v,10,28,28], [v,10,1]
        mean_q, std_q = self.encoder(y_miss)                                      # [v,10,L]
        mean_q, std_q = mean_q.mT, std_q.mT                                       # [v,L,10]
        mean_p, cov_p = self.gp.prior(t, are_neighbors=False)                     # [v,L,10], [v,1,10,10], params shared

        L = psd_safe_cholesky(cov_p + self.jitter * torch.eye(cov_p.size(-1), device=t.device))
        mean_diff = (mean_q - mean_p).unsqueeze(-1)
        mahalanobis = torch.linalg.solve_triangular(L, mean_diff, upper=False)
        mahalanobis = mahalanobis.square().sum(dim=(-1, -2))

        tmp = torch.linalg.solve_triangular(L, torch.diag_embed(std_q), upper=False)  # [v,L,10,10]
        trace = tmp.square().sum(dim=(-1, -2))

        log_det_cov_p = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        log_det_cov_q = std_q.log().sum(dim=-1)
        log_det = log_det_cov_p - log_det_cov_q

        res = 0.5 * (mahalanobis + trace - self.num_frames_per_series) + log_det
        res = res.sum()
        # test
        print(f"my kl divergence: {res}")
        p = torch.distributions.MultivariateNormal(loc=mean_p,
                                                   covariance_matrix=cov_p + torch.eye(cov_p.size(-1)) * self.jitter)
        q = torch.distributions.MultivariateNormal(loc=mean_q, scale_tril=torch.diag_embed(std_q))
        res_true = torch.distributions.kl.kl_divergence(q, p).sum()
        print(f"true kl divergence: {res_true}\n")
        return res

    def average_loss(self, vid_batch, t_batch, m_batch, beta=1.):
        elbo = self.expected_log_prob(vid_batch, m_batch) - beta * self.kl_divergence(vid_batch, t_batch)
        return - elbo / (self.num_frames_per_series * vid_batch.size(0))

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer, beta, epochs, series_batch_size: int,
            device="cpu", print_epochs=1
    ):
        self.to(device)
        series_dataloader = DataLoader(self.entire_train_dataset, batch_size=series_batch_size, shuffle=True)
        for epoch in range(epochs):
            for series_y_miss, series_m_miss in series_dataloader:
                # We feed forward the whole series
                miniseries_dict = generate_miniseries_hmnist_dict(series_y_miss, series_m_miss, Y_batch_full=None)
                vid_batch, m_batch = miniseries_dict['images'].to(device), miniseries_dict['masks'].to(device)
                t_batch = miniseries_dict['aux_data'].to(device).expand(vid_batch.size(0), -1, -1)  # [v,10,1]

                optimizer.zero_grad(set_to_none=True)
                loss = self.average_loss(vid_batch, t_batch, m_batch, beta=beta)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}')

    @torch.no_grad()
    def predict_gpvae(self, test_file_path, series_batch_size, device="cpu"):
        self.module_device = device
        self.to(device)

        entire_test_dataset = LazySeriesDataset(test_file_path, train=False, return_full=True)
        series_dataloader = DataLoader(entire_test_dataset, batch_size=series_batch_size, shuffle=False)
        # We use the evaluation procedure of AISTATS 2020 paper: sample one `z`, metrics are computed on missing values
        nll_series, mse_series, img_rec_series, n_missings = 0., 0., [], 0.
        for series_y_miss, m_miss, series_y_full in series_dataloader:
            n_missings += m_miss.sum()
            miniseries_dict = generate_miniseries_hmnist_dict(series_y_miss, m_miss, series_y_full)
            y_test_miss= miniseries_dict['images'].to(device)
            m_test = miniseries_dict['masks'].to(device=device, dtype=torch.bool)
            y_test_full = miniseries_dict['images_full'].to(device)   # [v, 10, 28, 28]

            enc_means, enc_stds = self.encoder(y_test_miss)
            latent_samples = enc_means + enc_stds * torch.randn_like(enc_means)
            probs = self.decoder(latent_samples)
            nll = nn.functional.binary_cross_entropy(input=probs, target=y_test_full, reduction='none')
            nll = torch.where(m_test, nll, 0.).sum()
            nll_series += nll

            img_rec = self.decoder(enc_means)
            img_rec_series.append(img_rec)
            pixels = torch.round(img_rec)
            se = torch.where(m_test, (pixels - y_test_full).square(), 0.).sum()
            mse_series += se
        return nll_series / n_missings, mse_series / n_missings, torch.cat(img_rec_series, dim=0)


if __name__ == '__main__':
    from datetime import datetime
    import argparse
    import numpy as np
    import time
    from pathlib import Path
    import h5py
    import pickle

    from utils.healing_mnist import plot_hmnist

    def run_base_healing(args):
        current_time = datetime.now().strftime("%m%d_%H_%M_%S")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.benchmark = True
        print(f"Casale GPPVAE, using device: {device}, {torch.get_default_dtype()}, seed: {args.seed}")

        file_path = Path(f'../data/healing_mnist/h5/hmnist_{args.random_mechanism}.h5')

        start_time = time.time()
        model = GPPVAECasaleHealing(file_path, val_split=50000, search_device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        model.train_gpvae(
            optimizer, beta=args.beta, epochs=args.num_epochs, series_batch_size=args.series_batch_size,
            device=device, print_epochs=args.num_print_epochs
        )

        nll, mse, y_rec = model.predict_gpvae(file_path, series_batch_size=args.series_batch_size, device=device)
        print(f"\nNLL: {nll}, MSE: {mse}.\n")

        # Plot reconstructed images
        video_idx, time_idx = [0 for _ in range(6)], [i for i in range(6)]  # v,f
        test_dict = h5py.File(file_path, 'r')
        y_test_full = test_dict['x_test_full'][:1][video_idx, time_idx]  # [num_plots, 28, 28]
        y_test_miss = test_dict['x_test_miss'][:1][video_idx, time_idx]
        fig, ax = plot_hmnist(
            y_test_miss.reshape(*y_test_full.shape[:-1], 28, 28), y_rec[video_idx, time_idx].cpu().numpy(),
            y_test_full.reshape(*y_test_miss.shape[:-1], 28, 28)
        )

        end_time = time.time()
        print(f"Total time: {end_time - start_time}.\n")

        if args.save:
            experiment_dir = Path(f'./exp_base_hmnist_{args.random_mechanism}')
            experiment_dir.mkdir(exist_ok=True)
            trial_dir = experiment_dir / f'{current_time}'
            trial_dir.mkdir()

            torch.save(model.state_dict(), trial_dir/'model_hmnist.pth')  # model

            fig.savefig(trial_dir/'img_hmnist.pdf', bbox_inches='tight')  # imgs

            everything_for_imgs = {
                'y_test_full': y_test_full, 'y_test_miss': y_test_miss, 'y_rec': y_rec, 'NLL': nll, 'mse': mse
            }
            imgs_pickle_path = trial_dir / 'everything_for_imgs.pkl'
            with imgs_pickle_path.open('wb') as f:
                pickle.dump(everything_for_imgs, f)

            print("The experiment results are saved at:")
            print(f"{trial_dir.resolve()}")  # get absolute path for moving the log file


    parser = argparse.ArgumentParser(description="Full NN-GPVAE from Casale's paper for Healing MNIST")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--random_mechanism', type=str, default='random',
                        choices=['mnar', 'spatial', 'random', 'temporal_neg', 'temporal_pos'], help='data used')
    parser.add_argument('--beta', type=float, default=0.8, help='beta')

    parser.add_argument('--series_batch_size', type=int, default=32, help='batch size for series')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of training epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')
    parser.add_argument('--save', action='store_true', help='whether to save results')

    args_hmnist = parser.parse_args()

    run_base_healing(args_hmnist)









