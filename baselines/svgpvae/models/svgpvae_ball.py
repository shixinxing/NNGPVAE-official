import numpy as np
import torch
from torch import Tensor
from torch import nn
from gpytorch.kernels import RBFKernel

from baselines.svgpvae.models.svgpvae_base import SVGPVAEBase
from baselines.svgpvae.models.svgp import SVGP
from models.building_blocks.enc_dec_ball import BallEncoder, BallDecoder
from utils.moving_ball import generate_moving_ball, generate_video_dict, path_rotation


class SVGPVAEBall(SVGPVAEBase):
    """
    SVGPVAE for moving ball, note that both the output variances of inference network and variational distribution
    are clipped by some values.
    """
    def __init__(
            self, M,
            num_videos=35, num_frames=30, img_size=32, lengthscale=2., r=3,
            GP_joint=False, IP_joint=False, device: str = 'cpu', jitter=1e-6
    ):
        encoder = BallEncoder(img_size=img_size, hidden_dims=(500,))
        decoder = BallDecoder(img_size=img_size, hidden_dims=(500,))

        # Each latent dim in this experiment has separate kernel params.
        kernel = RBFKernel(batch_shape=torch.Size([2]))
        kernel.lengthscale = lengthscale
        kernel.raw_lengthscale.requires_grad_(GP_joint)
        # each latent dim has a set of inducing points
        inducing_points = nn.Parameter(
            torch.linspace(0, num_frames-1, M).unsqueeze(-1).repeat(2, 1, 1), requires_grad=IP_joint  # [2,M,1]
        )
        svgp = SVGP(2, kernel, inducing_points, N_train=num_frames, jitter=jitter)

        self.num_videos_per_epoch = num_videos
        self.num_frames_per_epoch = num_frames
        self.img_size, self.lengthscale, self.r = img_size, lengthscale, r

        super(SVGPVAEBall, self).__init__(
            encoder, decoder, svgp, device=device, jitter=jitter, geco=False
        )

    def build_dataset_per_epoch(self, local_seed):
        # Pearce and we have time range between 0-29, SVGPVAE is 1-30
        paths, vid_batch = generate_moving_ball(
            self.num_videos_per_epoch, self.num_frames_per_epoch, self.img_size, self.lengthscale, self.r,
            seed=local_seed
        )
        train_data_dict = generate_video_dict(paths, vid_batch)
        return train_data_dict

    # override
    def build_MLP_inference_graph(self, y_batch: Tensor):
        qnet_mus, qnet_vars = super(SVGPVAEBall, self).build_MLP_inference_graph(y_batch)
        qnet_vars = torch.clamp(qnet_vars, 1e-6, 1e3)
        return qnet_mus, qnet_vars

    def train_gpvae(self, optimizer: torch.optim.Optimizer, epochs: int, device='cpu', print_epochs=1):
        assert not self.geco, "The current training does not support GECO."
        self.to(device)

        local_seeds = np.random.choice(epochs * 2, size=epochs, replace=False)
        for epoch in range(epochs):
            train_data_dict = self.build_dataset_per_epoch(local_seed=local_seeds[epoch])
            t_batch = torch.as_tensor(train_data_dict['aux_data'], device=device, dtype=torch.get_default_dtype())
            t_batch = t_batch.expand(self.num_videos_per_epoch, self.num_frames_per_epoch, 1)
            vid_batch = torch.as_tensor(train_data_dict['images'], device=device, dtype=torch.get_default_dtype())

            optimizer.zero_grad(set_to_none=True)
            loss = self.average_loss(vid_batch, t_batch, clip_qs_var_min=1e-4, clip_qs_var_max=1e3, beta=1.)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.detach().item():.6f}')

    @torch.no_grad()
    def predict_gpvae(self, test_dict_collection: list, device='cpu'):
        """ test on collection of 10 vid batches """
        self.to(device)

        path_coll, target_path_coll, rec_img_coll, target_img_coll, se_coll = [], [], [], [], []
        for test_data_dict in test_dict_collection:
            t_batch = torch.as_tensor(test_data_dict['aux_data'], device=device, dtype=torch.get_default_dtype())
            t_batch = t_batch.expand(self.num_videos_per_epoch, self.num_frames_per_epoch, 1)
            vid_batch = torch.as_tensor(test_data_dict['images'], device=device, dtype=torch.get_default_dtype())

            paths, _, rec_imgs = self.forward(vid_batch, t_batch, cov_latent_diag=True)  # [v,f,2], [v,f,32,32]
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

    from utils.moving_ball import plot_balls

    def run_svgpvae_ball(args):
        current_time = datetime.now().strftime('%m%d_%H_%M_%S')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float64 if args.float64 else torch.float32)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Time: {current_time}, SVGPVAE-M{args.M}, Device: {device}, {torch.get_default_dtype()}, '
              f'seed: {args.seed}\n')

        start_time = time.time()
        l = 2.
        model = SVGPVAEBall(
            args.M, GP_joint=args.GP_joint, IP_joint=args.IP_joint, lengthscale=l, device=device, jitter=1e-6
        )
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
        print(f"\nMean Test SE: {se_coll.mean()}, Std: {se_coll.std()};\n"
              f"Test SE for each video: {se_coll}")

        # plot trajectories
        fig, axes = plot_balls(target_img_coll, target_path_coll, rec_img_coll, path_coll, nplots=10)

        end_time = time.time()
        print(f"Total time: {end_time - start_time}")

        if args.save:
            GP_joint_str = "_GP_joint" if args.GP_joint else ""
            IP_joint_str = "_IP_joint" if args.IP_joint else ""
            exp_dir = Path(f'exp_moving_ball_SVGPVAE_M{args.M}' + GP_joint_str + IP_joint_str)
            exp_dir.mkdir(exist_ok=True)
            trial_dir = exp_dir / f'{current_time}'
            trial_dir.mkdir()

            torch.save(model.state_dict(), trial_dir / "SVGPVAE_moving_ball.pth")
            fig.savefig(trial_dir / f"moving_ball_SVGPVAE_M{args.M}.pdf", bbox_inches='tight')
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


    parser = argparse.ArgumentParser(description="SVGPVAE for Moving Ball in AISTATS 2021")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--M', type=int, default=10, help='number of inducing points')
    parser.add_argument('--GP_joint', action='store_true', help='whether to train the GP params')
    parser.add_argument('--IP_joint', action='store_true', help='whether to train the inducing locations')

    parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')
    parser.add_argument('--save', action='store_true', help='save everything into a folder')

    args_svgpvae = parser.parse_args()
    run_svgpvae_ball(args_svgpvae)







