import numpy as np
import torch
from torch import Tensor, LongTensor

from gpytorch.kernels import RBFKernel
from gpytorch.means import ZeroMean
from torch.utils.data import DataLoader

from models.gpvae_base import GPVAEBase
from models.building_blocks.enc_dec_ball import BallEncoder, BallDecoder
from models.building_blocks.gp import GP
from utils.moving_ball import generate_moving_ball, generate_video_dict, path_rotation
from utils.build_datasets import NNDataset


class GPVAEBallBase(GPVAEBase):
    def __init__(
            self, H,
            num_videos=35, num_frames=30, img_size=32, lengthscale=2., r=3,   # for data generation at each epoch
            GP_joint=False,      # whether train GP params (i.e., kernel lengthscale) jointly
            search_device: str = 'cpu', jitter=1e-6
    ):
        encoder = BallEncoder(img_size=img_size, hidden_dims=(500,))
        decoder = BallDecoder(img_size=img_size, hidden_dims=(500,))

        # Each latent dim in this experiment has separate kernel params.
        kernel = RBFKernel(batch_shape=torch.Size([2]))  # The original paper didn't use outputscale
        mean = ZeroMean(batch_shape=torch.Size([2]))
        gp = GP(output_dims=2, kernel=kernel, mean=mean)
        # initialize lengthscale
        gp.kernel.lengthscale = lengthscale
        for param in gp.parameters():
            param.requires_grad_(GP_joint)

        self.num_videos_per_epoch = num_videos
        self.num_frames_per_epoch = num_frames
        self.img_size, self.lengthscale, self.r = img_size, lengthscale, r

        super(GPVAEBallBase, self).__init__(
            encoder, decoder, gp, H, search_device=search_device, data_device='cpu', module_device=search_device,
            jitter=jitter, geco=False  # We don't bother with GECO in moving-ball models
        )

    def build_dataset_per_epoch(self, local_seed):
        """ Following Pearce, new video batch is generated per epoch. """
        paths, vid_batch = generate_moving_ball(
            self.num_videos_per_epoch, self.num_frames_per_epoch, self.img_size, self.lengthscale, self.r,
            seed=local_seed
        )
        train_data_dict = generate_video_dict(paths, vid_batch)
        return train_data_dict

    @torch.no_grad()
    def predict_gpvae(
            self, test_dict_collection: list,  # collection of 10 vid batches
            batch_size: int = None, device='cpu', num_samples=1
    ):
        batch_size = self.num_frames_per_epoch if batch_size is None else batch_size
        self.module_device = device
        self.to(device)

        path_coll, target_path_coll, rec_img_coll, target_img_coll, se_coll = [], [], [], [], []  # for each vid batch
        for test_data_dict in test_dict_collection:
            test_dataset = NNDataset(
                test_data_dict, series_shape=torch.Size([self.num_videos_per_epoch]), data_device=self.data_device,
                H=None, search_device=None, build_sequential_first=False
            )   # we don't need nnutils
            dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            paths, rec_imgs = [], []
            for vid_batch, t_batch in dataloader:
                vid_batch, t_batch = vid_batch.to(device).transpose(0, 1), t_batch.to(device)  # [v,f,32,32], [f,1]
                enc_means, enc_stds = self.encoder(vid_batch)   # [v,f,2]
                paths.append(enc_means)
                # reconstruction
                eps = torch.randn(num_samples, *enc_stds.shape, device=device)
                latent_samples = enc_means + enc_stds * eps
                img = self.decoder(latent_samples).mean(dim=0)  # [v,f,32,32]
                rec_imgs.append(img)
            paths = torch.cat(paths, dim=-2)                    # [v,f,2]
            rec_imgs = torch.cat(rec_imgs, dim=-3)              # [v,f,32,32]
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


class GPVAEBall_SWS(GPVAEBallBase):
    def average_loss(self, vid_batch: Tensor, t_batch: Tensor) -> Tensor:
        return - (self.expected_log_prob(vid_batch) - self.kl_divergence_sws(t_batch)) / len(self.train_dataset)

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer,
            epochs: int, batch_size: int = None,  # `batch size` refers to video length
            device='cpu', print_epochs=1
    ):
        self.module_device = device
        self.to(device)
        batch_size = self.num_frames_per_epoch if batch_size is None else batch_size

        local_seeds = np.random.choice(epochs * 2, size=epochs, replace=False)
        # print(f"seed: {local_seeds}")
        nn_util_cached = None
        for epoch in range(epochs):
            # generate training data per epoch
            train_data_dict = self.build_dataset_per_epoch(local_seed=local_seeds[epoch])
            # update data dict
            self.train_dataset = NNDataset(
                train_data_dict, series_shape=torch.Size([self.num_videos_per_epoch]), data_device=self.data_device,
                H=None if nn_util_cached else self.H, search_device=self.search_device, build_sequential_first=False
            )
            if nn_util_cached is None:
                nn_util_cached = self.train_dataset.nn_util
            else:
                self.train_dataset.nn_util = nn_util_cached

            dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)  # ⚠️ shuffle data or not?
            for vid_batch, t_batch in dataloader:
                vid_batch, t_batch = vid_batch.to(device), t_batch.to(device)  # [f,1]
                vid_batch = vid_batch.transpose(0, 1)  # [v,f,32,32] since mini-batch is along the time dim
                optimizer.zero_grad(set_to_none=True)
                loss = self.average_loss(vid_batch, t_batch)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f"Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")


class GPVAEBall_VNN(GPVAEBallBase):
    def average_loss(self, vid_batch: Tensor, current_kl_idx: LongTensor, seq_nn_structure: LongTensor) -> Tensor:
        expected_lk = self.expected_log_prob(vid_batch)
        kl = self.kl_divergence_vnn(current_kl_idx, seq_nn_structure)
        return - (expected_lk - kl) / len(self.train_dataset)

    def train_gpvae(
            self, optimizer: torch.optim.Optimizer,
            epochs: int, batch_size_kl: int = None,
            batch_size_expected_log: int = None, device='cpu', print_epochs=1
    ):
        self.module_device = device
        self.to(device)
        batch_size_kl = self.num_frames_per_epoch if batch_size_kl is None else batch_size_kl
        self._set_training_iterator(batch_size_kl, N=self.num_frames_per_epoch)

        # The NN structure doesn't change with the dataset generated per epoch, so we compute it only once
        seq_nn_idx_cached, nn_util_cached = None, None
        local_seeds = np.random.choice(epochs * 2, size=epochs, replace=False)
        for epoch in range(epochs):
            # data generation per epoch; only compute the structure once
            train_data_dict = self.build_dataset_per_epoch(local_seed=local_seeds[epoch])
            self.train_dataset = NNDataset(
                train_data_dict, series_shape=torch.Size([self.num_videos_per_epoch]), data_device=self.data_device,
                H=None if nn_util_cached is not None else self.H, search_device=self.search_device,
                build_sequential_first=False if seq_nn_idx_cached is not None else True
            )
            if seq_nn_idx_cached is None:
                seq_nn_idx_cached = self.train_dataset.seq_nn_idx
            # else:  # for completion
            #     self.train_dataset.seq_nn_idx = seq_nn_idx_cached
            if nn_util_cached is None:
                nn_util_cached = self.train_dataset.nn_util
            else:
                self.train_dataset.nn_util = nn_util_cached

            if batch_size_expected_log is None:  # use the same KL indices
                for _ in range(self._total_training_batches):
                    current_kl_indices = self._get_training_indices(batch_size_kl)
                    vid_batch, t_batch = self.train_dataset[current_kl_indices]  # [v,b,32,32], [b,1]
                    vid_batch, t_batch = vid_batch.to(device), t_batch.to(device)

                    optimizer.zero_grad(set_to_none=True)
                    loss = self.average_loss(vid_batch, current_kl_indices, seq_nn_idx_cached)
                    loss.backward()
                    optimizer.step()
            else:
                dataloader_lk = DataLoader(self.train_dataset, batch_size=batch_size_expected_log, shuffle=False)  # ⚠️ shuffle data or not?
                for vid_batch, t_batch in dataloader_lk:
                    current_kl_indices = self._get_training_indices(batch_size_kl)
                    vid_batch = vid_batch.to(device).transpose(0, 1)  # [v,b,32,32]

                    optimizer.zero_grad(set_to_none=True)
                    loss = self.average_loss(vid_batch, current_kl_indices, seq_nn_idx_cached)
                    loss.backward()
                    optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f"Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")


