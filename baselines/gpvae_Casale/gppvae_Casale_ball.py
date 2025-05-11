import numpy as np

import torch
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor

from models.gpvae_ball import GPVAEBallBase


class GPPVAECasaleBall(GPVAEBallBase):
    """
    GPP-VAE with a mean-field encoder (NIPS 2018) for moving ball data
    using RBF kernels and full batch training in order to test if this encoder is suitable for this dataset
    Unlike NIPS 2018, this class doesn't share kernel params in latent dims.
    """
    def __init__(
            self, num_videos=35, num_frames=30, img_size=32, lengthscale=2., r=3, device='cpu',
    ):
        super(GPPVAECasaleBall, self).__init__(
            H=num_frames,   # choose all frames as nearest neighbors
            num_videos=num_videos, num_frames=num_frames, img_size=img_size, lengthscale=lengthscale, r=r,
            search_device=device
        )

    @torch.no_grad()   # override
    def predict_gpvae(self, test_dict_collection: list, device='cpu', num_samples=1):
        path_coll, target_path_coll, rec_img_coll, target_img_coll, se_coll = super().predict_gpvae(
            test_dict_collection, batch_size=None,    # full-batch, i.e. feed all frames
            device=device, num_samples=num_samples
        )
        return path_coll, target_path_coll, rec_img_coll, target_img_coll, se_coll

    def expected_log_prob(self, y_batch: Tensor, **kwargs) -> Tensor:
        means, stds, y_rec = self.forward(y_batch)
        assert self.decoder.output_distribution == 'bernoulli'
        expected_lk = - torch.nn.functional.binary_cross_entropy(input=y_rec, target=y_batch, reduction='sum')
        return expected_lk  # sum over v * b

    def kl_divergence(self, y_full, x_full):
        """
        When H covers all data, we don't need to compute the same KL divergence terms n_B times.
        y_full: [v,f,32,32], x_full: [v,f,1]
        """
        mean_q, std_q = self.encoder(y_full)                        # [v,f,2]
        mean_q, std_q = mean_q.mT, std_q.mT                         # [v,2,f]
        mean_p, cov_p = self.gp.prior(x_full, are_neighbors=False)  # [v,2,f], [v,2,f,f]

        L = psd_safe_cholesky(cov_p + self.jitter * torch.eye(cov_p.size(-1), device=y_full.device))
        mean_diff = (mean_q - mean_p).unsqueeze(-1)
        mahalanobis = torch.linalg.solve_triangular(L, mean_diff, upper=False)
        mahalanobis = mahalanobis.square().sum(dim=(-1, -2))        # [(v),2]

        # L_inv = torch.linalg.solve_triangular(L, torch.eye(L.size(-1)).to(self.device), upper=False)
        # tmp = L_inv * std_q.unsqueeze(-2)  # [v,2,f,f]
        tmp = torch.linalg.solve_triangular(L, torch.diag_embed(std_q), upper=False)  # [v,2,f,f]
        trace = tmp.square().sum(dim=(-1, -2))

        log_det_cov_p = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        log_det_cov_q = std_q.log().sum(dim=-1)
        log_det = log_det_cov_p - log_det_cov_q

        res = 0.5 * (mahalanobis + trace - self.num_frames_per_epoch) + log_det
        # test
        print(f"my kl divergence (shape: {res.shape}): {res.sum()}")
        p = torch.distributions.MultivariateNormal(loc=mean_p, covariance_matrix=cov_p + torch.eye(cov_p.size(-1)) * self.jitter)
        q = torch.distributions.MultivariateNormal(loc=mean_q, scale_tril=torch.diag_embed(std_q))
        res_true = torch.distributions.kl.kl_divergence(q, p).sum()
        print(f"true kl divergence: {res_true}\n")
        return res.sum()

    def average_loss(self, vid_batch, t_batch):
        exp_lk = self.expected_log_prob(vid_batch) * self.num_frames_per_epoch / t_batch.size(-2)
        kl = self.kl_divergence(vid_batch, t_batch)
        return - (exp_lk - kl) / (self.num_frames_per_epoch * self.num_videos_per_epoch)

    def train_gpvae(self, optimizer, epochs, device='cpu', print_epochs=1):
        self.to(device)
        self.module_device = device
        local_seeds = np.random.choice(epochs * 2, size=epochs, replace=False)
        for epoch in range(epochs):
            train_data_dict = self.build_dataset_per_epoch(local_seed=local_seeds[epoch])
            train_data_dict['aux_data'] = np.repeat(
                train_data_dict['aux_data'][np.newaxis, :], self.num_videos_per_epoch, axis=0)  # [v,f,1]
            # we don't need dataloader here due to full batch of frames
            vid_batch = torch.as_tensor(
                train_data_dict['images'], dtype=torch.get_default_dtype(), device=device
            )    # [v, f, 32, 32]
            t_batch = torch.as_tensor(
                train_data_dict['aux_data'], dtype=torch.get_default_dtype(), device=device
            )    # [v, f, 1]
            optimizer.zero_grad(set_to_none=True)
            loss = self.average_loss(vid_batch, t_batch)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f"Epoch: {epoch + 1}, Loss: {loss.item():.6f}")


if __name__ == "__main__":
    from datetime import datetime
    import time
    from pathlib import Path
    import pickle
    import argparse
    from utils.moving_ball import generate_moving_ball, generate_video_dict, plot_balls

    def run_base(args):
        current_time = datetime.now().strftime("%m%d_%H_%M_%S")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float32)  # faiss seems not to accept float64
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        print(f"Casale GPVAE, Using device: {device}, {torch.get_default_dtype()}, seed: {args.seed}")

        start_time = time.time()

        l = 2.
        model = GPPVAECasaleBall(lengthscale=l, device=device)
        # model.load_state_dict(torch.load("model.pth"))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

        model.train_gpvae(optimizer, epochs=args.num_epochs, print_epochs=args.num_print_epochs, device=device)

        # generate test video collection
        test_dict_coll = []
        for seed in range(args.num_epochs * 3, args.num_epochs * 3 + 10):  # generate the same test dataset
            p, v = generate_moving_ball(lengthscale=l, seed=seed)
            data_dict = generate_video_dict(p, v)
            test_dict_coll.append(data_dict)

        path_coll, target_path_coll, rec_img_coll, target_img_coll, se_coll = model.predict_gpvae(test_dict_coll,
                                                                                                  device=device)
        print(f"\nMean Test SE: {se_coll.mean()}, Test SE: {se_coll}")

        # plot trajectories
        fig, axes = plot_balls(target_img_coll, target_path_coll, rec_img_coll, path_coll, nplots=10)

        end_time = time.time()
        print(f"Total time: {end_time - start_time}")

        if args.save:
            exp_dir = Path('./exp_moving_ball_Casale')
            exp_dir.mkdir(exist_ok=True)
            trial_dir = exp_dir/f'{current_time}'
            trial_dir.mkdir()

            torch.save(model.state_dict(), trial_dir/"model_ball_base.pth")
            fig.savefig(trial_dir/"moving_ball_Casale.pdf", bbox_inches='tight')
            everything_for_imgs = {
                'path_coll': path_coll, 'target_path_coll': target_path_coll,  # [10*v,f,2]
                'rec_img_coll': rec_img_coll, 'target_img_coll': target_img_coll,  # [10*v,f,32,32]
                'se_coll': se_coll
            }
            imgs_pickle_path = trial_dir / 'everything_for_imgs.pkl'  # save everything for future plotting
            with imgs_pickle_path.open('wb') as f:
                pickle.dump(everything_for_imgs, f)

            print("The experiment results are saved at:")
            print(f"{trial_dir.resolve()}")  # absolute path for saving log


    parser = argparse.ArgumentParser(description='Full Nearest Neighbor GPVAE for Moving Ball')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')
    parser.add_argument('--save', action='store_true', help='save results')

    args_base = parser.parse_args()
    run_base(args_base)
