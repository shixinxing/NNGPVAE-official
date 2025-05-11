# Pearce's GPVAE for moving ball dataset using PyTorch,
# adapted from https://github.com/ratschlab/SVGP-VAE/blob/main/GPVAE_Pearce_model.py

import numpy as np
import torch
from torch import Tensor
from torch import nn
from linear_operator.utils.cholesky import psd_safe_cholesky

from models.gpvae_ball import GPVAEBallBase
from utils.moving_ball import path_rotation
from baselines.svgpvae.utils.elbo_utils import negative_gaussian_cross_entropy


class GPVAEPearceBall(GPVAEBallBase):
    """
    GPVAE proposed in Pearce's paper (AABI 2019), kernel params are not trained in this model.
    Experiments showed if training kernel params jointly ...
    """
    def __init__(
            self, num_videos=35, num_frames=30, img_size=32, lengthscale=2., r=3, device='cpu'
    ):
        super(GPVAEPearceBall, self).__init__(
            H=num_frames, num_videos=num_videos, num_frames=num_frames, img_size=img_size, lengthscale=lengthscale, r=r,
            GP_joint=False, search_device=device, jitter=1e-6
        )

    def build_MLP_inference_graph(self, y_batch: Tensor):
        """return the distribution \tilde q with shape [v,f,2] produced by recognition network，"""
        qnet_mus, qnet_stds = self.encoder(y_batch)
        qnet_vars = qnet_stds.square()
        return qnet_mus, qnet_vars

    def build_gp(self, X, qnet_mu, qnet_var, X_test):
        """
        X: [v,f,1], qnet_mu/qnet_var: [v,f,2], X_test: [v,f*,1].
        compute the terms of the variational distribution q(Z) ~ p(Z) * \tilde q(Z|Y): [v,2,f]
        from a GP regression with target (`X`,`qnet_mu`) and noise `q_var`;
        return posterior mean and posterior variance at `X_test` and logZ.
        refer to https://github.com/scrambledpie/GPVAE/blob/master/GPVAEmodel.py#L145 or our report for details
        """
        # logZ term 1/3 - constant
        lhood_pi_term = - 0.5 * X.size(-2) * torch.tensor(2 * torch.pi).log()

        # logZ term 2/3 - determinant
        mean_gp, K = self.gp.prior(X, are_neighbors=False)
        K = K + torch.diag_embed(qnet_var.mT)                     # [v,2,f,f]
        chol_K = psd_safe_cholesky(K)
        lhood_logdet_term = - torch.diagonal(chol_K, dim1=-2, dim2=-1).log().sum(dim=-1)  # [v,2]

        # logZ term 3/3 - exp
        delta_mean = qnet_mu.mT - mean_gp                                          # [v,2,f]
        iKY = torch.cholesky_solve(delta_mean.unsqueeze(-1), chol_K, upper=False)  # [v,2,f,1]
        lh_quad_term = - 0.5 * torch.sum(delta_mean * iKY.squeeze(-1), dim=-1)     # [v,2]

        # log P(Y|X) = -1/2 * (n log(2 pi) + Y inv(K+noise) Y + log det(K+noise))
        gp_lhood = lhood_pi_term + lhood_logdet_term + lh_quad_term                # [v,2]

        # compute posterior
        X_test, X = X_test.unsqueeze(-3), X.unsqueeze(-3)         # [v,1,f,1]
        Ks = self.gp.kernel(X_test, X).to_dense()                 # [v,2,f*,f]
        # posterior mean & variance
        p_m = (Ks @ iKY).squeeze(-1)                              # [v,2,f*]
        iK_Ks = torch.cholesky_solve(Ks.mT, chol_K, upper=False)  # [v,2,f,f*]
        Ks_iK_Ks = torch.sum(Ks * iK_Ks.mT, dim=-1)               # [v,2,f*]
        p_v = self.gp.kernel(X_test, diag=True).to_dense() - Ks_iK_Ks

        return p_m, p_v, gp_lhood

    def forward(self, y_batch: Tensor, x_batch: Tensor):  # noqa
        qnet_mus, qnet_vars = self.build_MLP_inference_graph(y_batch)
        p_m, p_v, _ = self.build_gp(x_batch, qnet_mus, qnet_vars, X_test=x_batch)
        full_p_mu, full_p_var = p_m.mT, p_v.mT    # [v,f,2]
        latent_samples = full_p_mu + torch.sqrt(full_p_var) * torch.randn_like(full_p_mu)
        recon_imgs = self.decoder(latent_samples)
        return full_p_mu, full_p_var, recon_imgs

    def average_loss(self, vid_batch: Tensor, t_batch: Tensor, beta=1) -> Tensor:
        qnet_mus, qnet_vars = self.build_MLP_inference_graph(vid_batch)
        p_m, p_v, gp_lhood = self.build_gp(t_batch, qnet_mus, qnet_vars, X_test=t_batch)
        full_p_mu, full_p_var = p_m.mT, p_v.mT   # [v,f,2]

        # KL = cross_entropy - logZ
        full_lhood = gp_lhood.sum(dim=-1)        # [v]
        cross_entry = negative_gaussian_cross_entropy(full_p_mu, full_p_var, qnet_mus, qnet_vars).sum(dim=(-1, -2))  # [v]
        kl = cross_entry - full_lhood

        # Expected-log-likelihood
        eps = torch.randn_like(full_p_mu)
        latent_samples = full_p_mu + eps * torch.sqrt(full_p_var)
        recon_imgs = self.decoder(latent_samples)  # [v,f,32,32]
        expected_lk = - nn.functional.binary_cross_entropy(input=recon_imgs, target=vid_batch, reduction='sum')

        v, f = self.num_videos_per_epoch, self.num_frames_per_epoch
        elbo = expected_lk / (v * f) - beta * kl.sum() / (v * f)
        return - elbo

    def train_gpvae(self, optimizer: torch.optim.Optimizer, epochs: int, device='cpu', print_epochs=1):
        self.module_device = device
        self.to(device)

        local_seeds = np.random.choice(epochs * 2, size=epochs, replace=False)
        for epoch in range(epochs):
            train_data_dict = self.build_dataset_per_epoch(local_seed=local_seeds[epoch])
            t_batch = torch.as_tensor(train_data_dict['aux_data'], device=device, dtype=torch.get_default_dtype())
            t_batch = t_batch.expand(self.num_videos_per_epoch, self.num_frames_per_epoch, 1)
            vid_batch = torch.as_tensor(train_data_dict['images'], device=device, dtype=torch.get_default_dtype())

            optimizer.zero_grad(set_to_none=True)
            loss = self.average_loss(vid_batch, t_batch)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.detach().item():.6f}')

    @torch.no_grad()
    def predict_gpvae(self, test_dict_collection: list, device='cpu'):
        """ test on collection of 10 vid batches """
        self.module_device = device
        self.to(device)

        path_coll, target_path_coll, rec_img_coll, target_img_coll, se_coll = [], [], [], [], []
        for test_data_dict in test_dict_collection:
            t_batch = torch.as_tensor(test_data_dict['aux_data'], device=device, dtype=torch.get_default_dtype())
            t_batch = t_batch.expand(self.num_videos_per_epoch, self.num_frames_per_epoch, 1)
            vid_batch = torch.as_tensor(test_data_dict['images'], device=device, dtype=torch.get_default_dtype())

            paths, _, rec_imgs = self.forward(vid_batch, t_batch)  # [v,f,2], [v,f,32,32]
            paths, rec_imgs = paths.to('cpu').numpy(), rec_imgs.to('cpu').numpy()
            target_paths, target_imgs = test_data_dict['paths'], test_data_dict['images']
            ro_paths, se = path_rotation(paths, target_paths)

            for coll, obj in ((path_coll, ro_paths), (target_path_coll, target_paths),
                              (rec_img_coll, rec_imgs), (target_img_coll, target_imgs), (se_coll, se)):
                coll.append(obj)
        path_coll, target_path_coll = np.concatenate(path_coll, axis=-3), np.concatenate(target_path_coll, axis=-3)
        rec_img_coll, target_img_coll = np.concatenate(rec_img_coll, axis=-4), np.concatenate(target_img_coll, axis=-4)
        se_coll = np.array(se_coll)
        return path_coll, target_path_coll, rec_img_coll, target_img_coll, se_coll


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    import time
    from pathlib import Path
    import pickle

    from utils.moving_ball import generate_moving_ball, generate_video_dict, plot_balls

    def run_Pearce_ball(args):
        current_time = datetime.now().strftime('%m%d_%H_%M_%S')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float64 if args.float64 else torch.float32)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Pearce GPVAE for moving ball, Device: {device}, {torch.get_default_dtype()}, seed: {args.seed}')

        start_time = time.time()
        l = 2.
        model = GPVAEPearceBall(lengthscale=l, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        model.train_gpvae(optimizer, args.num_epochs, device=device, print_epochs=args.num_print_epochs)

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
            exp_dir = Path('exp_moving_ball_Pearce')  # in `baselines`
            exp_dir.mkdir(exist_ok=True)
            trial_dir = exp_dir / f'{current_time}'
            trial_dir.mkdir()

            torch.save(model.state_dict(), trial_dir / "Pearce_moving_ball.pth")
            fig.savefig(trial_dir / "moving_ball_Pearce.pdf", bbox_inches='tight')
            everything_for_imgs = {
                'path_coll': path_coll, 'target_path_coll': target_path_coll,      # [10*v,f,2]
                'rec_img_coll': rec_img_coll, 'target_img_coll': target_img_coll,  # [10*v,f,32,32]
                'se_coll': se_coll
            }
            imgs_pickle_path = trial_dir / 'everything_for_imgs.pkl'  # save everything for future plotting
            with imgs_pickle_path.open('wb') as f:
                pickle.dump(everything_for_imgs, f)

            print("The experiment results are saved at:")
            print(f"{trial_dir.resolve()}")  # absolute path for saving log


    parser = argparse.ArgumentParser(description="Pearce's GPVAE for Moving Ball")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--num_epochs', type=int, default=25000, help='number of training epochs')
    parser.add_argument('--num_print_epochs', type=int, default=100, help='number of printing epochs')
    parser.add_argument('--save', action='store_true', help='save everything into a folder')

    args_Pearce = parser.parse_args()
    run_Pearce_ball(args_Pearce)

