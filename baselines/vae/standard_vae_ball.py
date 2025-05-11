from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.vae.standard_vae import VAEBase
from models.building_blocks.enc_dec_ball import BallEncoder, BallDecoder
from utils.moving_ball import generate_moving_ball, generate_video_dict, path_rotation
from utils.build_datasets import NNDataset


class VAEBall(VAEBase):
    def __init__(
            self, num_videos=35, num_frames=30, img_size=32, lengthscale=2., r=3
    ):
        encoder = BallEncoder(img_size=img_size, hidden_dims=(500,))
        decoder = BallDecoder(img_size=img_size, hidden_dims=(500,))

        self.num_videos_per_epoch = num_videos
        self.num_frames_per_epoch = num_frames
        self.img_size, self.lengthscale, self.r = img_size, lengthscale, r

        super(VAEBall, self).__init__(encoder, decoder, latent_dims=2, extra_data_batch_shape=torch.Size([num_videos]))
        self.train_dataset = None

    def build_dataset_per_epoch(self, local_seed):
        paths, vid_batch = generate_moving_ball(
            self.num_videos_per_epoch, self.num_frames_per_epoch, self.img_size, self.lengthscale, self.r,
            seed=local_seed
        )
        train_data_dict = generate_video_dict(paths, vid_batch)
        return train_data_dict

    def train_vae(
            self, optimizer: torch.optim.Optimizer,
            epochs: int, batch_size: Optional[int] = None,
            device='cpu', print_epochs=1
    ):
        if batch_size is None:
            batch_size = self.num_frames_per_epoch

        local_seeds = np.random.choice(epochs * 2, size=epochs, replace=False)
        self.to(device)
        for epoch in range(epochs):
            train_data_dict = self.build_dataset_per_epoch(local_seeds[epoch])
            self.train_dataset = NNDataset(train_data_dict, series_shape=self.extra_data_batch_shape, data_device='cpu')
            dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
            for vid_batch, _ in dataloader:
                vid_batch = vid_batch.to(device).transpose(0, 1)  # [v,f,32,32]
                optimizer.zero_grad(set_to_none=True)
                loss = self.average_loss(vid_batch)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}')

    @torch.no_grad()
    def predict_vae(
            self, test_dict_collection: list,  # collection of 10 vid batches
            batch_size: int = None, device='cpu', num_samples=1
    ):
        """Same as the GPVAE"""
        if batch_size is None:
            batch_size = len(self.train_dataset)
        self.to(device)

        path_coll, target_path_coll, rec_img_coll, target_img_coll, se_coll = [], [], [], [], []  # for each vid batch
        for test_data_dict in test_dict_collection:
            test_dataset = NNDataset(test_data_dict, series_shape=self.extra_data_batch_shape, data_device='cpu')
            dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            paths, rec_imgs = [], []
            for vid_batch, _ in dataloader:
                vid_batch = vid_batch.to(device).transpose(0, 1)                     # [v,f,32,32]
                enc_means, enc_stds = self.encoder(vid_batch)                        # [v,f,2]
                paths.append(enc_means)
                # reconstruction
                eps = torch.randn(num_samples, *enc_stds.shape).to(device)
                latent_samples = enc_means + enc_stds * eps
                img = self.decoder(latent_samples).mean(dim=0)                       # [v,f,32,32]
                rec_imgs.append(img)
            paths, rec_imgs = torch.cat(paths, dim=-2), torch.cat(rec_imgs, dim=-3)  # [v,f,2], [v,f,32,32]
            paths, rec_imgs = paths.to(torch.device('cpu')).numpy(), rec_imgs.to(torch.device('cpu')).numpy()
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
    from datetime import datetime
    import time
    from pathlib import Path
    import pickle
    import argparse
    from utils.moving_ball import plot_balls


    def run_vae(args):
        current_time = datetime.now().strftime("%m%d_%H_%M_%S")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float32)  # faiss seems not to accept float64
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        print(f"Standard VAE on moving ball, Using device: {device}.")

        start_time = time.time()

        l = 2.
        model = VAEBall(lengthscale=l)
        # model.load_state_dict(torch.load("model.pth"))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

        model.train_vae(optimizer, epochs=args.num_epochs, print_epochs=args.num_print_epochs, device=device)

        # generate test video collection
        test_dict_coll = []
        for seed in range(args.num_epochs * 3, args.num_epochs * 3 + 10):
            p, v = generate_moving_ball(lengthscale=l, seed=seed)
            data_dict = generate_video_dict(p, v)
            test_dict_coll.append(data_dict)

        path_coll, target_path_coll, rec_img_coll, target_img_coll, se_coll = model.predict_vae(test_dict_coll,
                                                                                                device=device)
        print(f"\nMean Test SE: {se_coll.mean()}, Test SE: {se_coll}")

        # plot trajectories
        fig, axes = plot_balls(target_img_coll, target_path_coll, rec_img_coll, path_coll, nplots=10)

        end_time = time.time()
        print(f"Total time: {end_time - start_time}")

        if args.save:
            exp_dir = Path('./exp_moving_ball_VAE')
            exp_dir.mkdir(exist_ok=True)
            trial_dir = exp_dir / f'{current_time}'
            trial_dir.mkdir()

            torch.save(model.state_dict(), trial_dir / "model_ball_vae.pth")
            fig.savefig(trial_dir / "moving_ball_standard_vae.pdf", bbox_inches='tight')
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
        else:
            print("The experiment results are not saved.")


    parser = argparse.ArgumentParser(description='Standard VAE for Moving Ball as a baseline')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--num_print_epochs', type=int, default=1)
    parser.add_argument('--save', action='store_true')
    args_base = parser.parse_args()

    run_vae(args_base)




