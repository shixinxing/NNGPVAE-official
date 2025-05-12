###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Authors: Yulia Rubanova and Ricky Chen
###########################

import os
import numpy as np
import torch
# from latent_ode.lib.utils import get_dict_template
# import latent_ode.lib.utils as utils
from tqdm import tqdm
from torchvision.datasets.utils import download_url


def normalize_data(data):
    # copy from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py#L328
    reshaped = data.reshape(-1, data.size(-1))

    att_min = torch.min(reshaped, 0)[0]
    att_max = torch.max(reshaped, 0)[0]

    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    return data_norm, att_min, att_max


class MujocoPhysics(object):
    # For Hopper:
    # T = 1000
    # D = 14
    # n_training_samples = 500  # number of series
    # training_file = 'training.pt'

    def __init__(
        self, root, T=1000, D=14, n_training_samples=500, domain_name="hopper", task_name="stand",
        download=False, generate=True, device=torch.device("cpu"), local_seed=123
    ):
        self.root = root
        self.T = T
        self.D = D
        self.n_training_samples = n_training_samples
        self.domain_name = domain_name
        self.task_name = task_name
        self.training_file = f'training_{T}.pt'
        self.local_seed = local_seed
        if download:
            self._download()

        if generate:
            print("GENERATE")
            self._generate_dataset()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        data_file = os.path.join(self.data_folder, self.training_file)

        self.data = torch.Tensor(torch.load(data_file)).to(device)
        self.data, self.data_min, self.data_max = normalize_data(self.data)

        self.device = device

    def visualize(self, traj, plot_name='traj', dirname='hopper_imgs', video_name=None):
        raise NotImplementedError

    def _generate_dataset(self):
        if self._check_exists():
            print("Don't generate dataset, it already exists.")
            return
        os.makedirs(self.data_folder, exist_ok=True)
        print('Generating dataset...')
        train_data = self._generate_random_trajectories(self.n_training_samples)
        torch.save(train_data, os.path.join(self.data_folder, self.training_file))

    def _download(self):
        raise NotImplementedError

    def _generate_random_trajectories(self, n_samples, seed=123):
        raise NotImplementedError

    def _check_exists(self):
        return os.path.exists(os.path.join(self.data_folder, self.training_file))

    @property
    def data_folder(self):
        return str(os.path.join(self.root))

    def get_dataset(self):
        return self.data  # [v,T,D]

    def __len__(self):
        return len(self.data)

    def size(self, ind=None):
        if ind is not None:
            return self.data.shape[ind]
        return self.data.shape

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


class HopperPhysics(MujocoPhysics):

    def __init__(
        self, root, T=1000, D=14, n_training_samples=500, task_name="stand",
        download=False, generate=True, device=torch.device("cpu"), local_seed=123
    ):
        super(HopperPhysics, self).__init__(root, T=T, D=D, n_training_samples=n_training_samples,
                                            domain_name="hopper", task_name=task_name, download=download,
                                            generate=generate, device=device, local_seed=local_seed)

    def visualize(self, traj, plot_name='traj', dirname='hopper_imgs', video_name=None):
        r"""Generates images of the trajectory and stores them as <dirname>/traj<index>-<t>.jpg"""

        T, D = traj.size()

        traj = traj.cpu() * self.data_max.cpu() + self.data_min.cpu()

        try:
            from dm_control import suite  # noqa: F401
        except ImportError as e:
            raise Exception('Deepmind Control Suite is required to visualize the dataset.') from e

        try:
            from PIL import Image  # noqa: F401
        except ImportError as e:
            raise Exception('PIL is required to visualize the dataset.') from e

        def save_image(data, filename):
            im = Image.fromarray(data)
            im.save(filename)

        os.makedirs(dirname, exist_ok=True)

        env = suite.load('hopper', 'stand')
        physics = env.physics

        for t in range(T):
            with physics.reset_context():
                physics.data.qpos[:] = traj[t, :D // 2]
                physics.data.qvel[:] = traj[t, D // 2:]
            save_image(
                physics.render(height=480, width=640, camera_id=0),
                os.path.join(dirname, plot_name + '-{:03d}.jpg'.format(t))
            )

    def _download(self):
        if self._check_exists():
            return

        print("Downloading the dataset [325MB] ...")
        os.makedirs(self.data_folder, exist_ok=True)
        url = "http://www.cs.toronto.edu/~rtqichen/datasets/HopperPhysics/training.pt"
        download_url(url, self.data_folder, "training.pt", None)

    def _generate_random_trajectories(self, n_samples, seed=123):
        try:
            from dm_control import suite  # noqa: F401
        except ImportError as e:
            raise Exception('Deepmind Control Suite is required to generate the dataset.') from e

        env = suite.load(self.domain_name, self.task_name)
        physics = env.physics

        # Store the state of the RNG to restore later.
        if self.local_seed is not None:
            np_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()
            np.random.seed(self.local_seed)
            torch.manual_seed(self.local_seed)

        data = np.zeros((n_samples, self.T, self.D))
        for i in tqdm(range(n_samples)):
            with physics.reset_context():
                # x and z positions of the hopper. We want z > 0 for the hopper to stay above ground.
                physics.data.qpos[:2] = np.random.uniform(0, 10, size=2)
                # physics.data.qpos[:2] = np.random.uniform(0, 0.5, size=2)
                physics.data.qpos[2:] = np.random.uniform(-2, 2, size=physics.data.qpos[2:].shape)
                physics.data.qvel[:] = np.random.uniform(-5, 5, size=physics.data.qvel.shape) * 0.1  # 0.1
            for t in range(self.T):
                data[i, t, :self.D // 2] = physics.data.qpos
                data[i, t, self.D // 2:] = physics.data.qvel
                physics.step()

        # Restore RNG.
        if self.local_seed is not None:
            np.random.set_state(np_random_state)
            torch.set_rng_state(torch_random_state)
        return data


class HumanoidCMUPhysics(MujocoPhysics):

    def __init__(
        self, root, T=10_000, D=20, n_training_samples=200, task_name="walk",  # stand, walk, run
        download=False, generate=True, device=torch.device("cpu"), local_seed=123
    ):
        super(HumanoidCMUPhysics, self).__init__(root, T=T, D=D, n_training_samples=n_training_samples,
                                            domain_name="humanoid_CMU", task_name=task_name, download=download,
                                            generate=generate, device=device, local_seed=local_seed)
    def _generate_random_trajectories(self, n_samples, seed=123):
        try:
            from dm_control import suite  # noqa: F401
        except ImportError as e:
            raise Exception('Deepmind Control Suite is required to generate the dataset.') from e

        env = suite.load(self.domain_name, self.task_name)
        physics = env.physics

        # Store the state of the RNG to restore later.
        if self.local_seed is not None:
            np_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()
            np.random.seed(self.local_seed)
            torch.manual_seed(self.local_seed)

        data = np.zeros((n_samples, self.T, self.D))
        for i in tqdm(range(n_samples)):
            with physics.reset_context():  # qpos: [63]; qvel: [62]
                physics.data.qpos[:] = np.random.uniform(-2, 2, size=physics.data.qpos[:].shape)
                physics.data.qvel[:] = np.random.uniform(-5, 5, size=physics.data.qvel.shape) * 0.1
            for t in range(self.T):
                data[i, t, :self.D] = physics.data.qpos[:self.D]
                # data[i, t, (self.D // 2 + 1):] = physics.data.qvel
                physics.step()

        # Restore RNG.
        if self.local_seed is not None:
            np.random.set_state(np_random_state)
            torch.set_rng_state(torch_random_state)
        return data

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    root = '.'
    device = torch.device('cpu')

    # hopper
    """
    hopper = HopperPhysics(root=root, download=False, generate=True, device=device, local_seed=123)  # use local seed
    data = hopper.get_dataset()  # [500, 1000, 14]

    print("Generated data shape:", data.shape)
    print(f"Generated data[0, 0]:\n{data[0, 0]}")

    # plot in every sub-figure
    id = 1
    fig, axes = plt.subplots(7, 2, figsize=(14, 18))
    axes = axes.ravel()
    for i in range(14):
        ax = axes[i]
        ax.plot(data[0, :, i], label=f'Mean', color='blue', alpha=0.9)
        ax.set_title(f'Feature {i + 1}')
        ax.legend()

    plt.tight_layout()
    fig.savefig(f'./data_{id}.pdf', dpi=3000)
    # plt.scatter([i for i in range(data.size(-2))], data[0, :, 7], s=2)
    # plt.show()
    """

    # HumanoidCMU
    HumanoidCMU = HumanoidCMUPhysics(root=root, T=10_000, D=20, n_training_samples=200,
                                    task_name="walk", download=False, generate=True, device=device, local_seed=123)
    data = HumanoidCMU.get_dataset()  # [200, 10_000, 20]

    print("Generated data shape:", data.shape)
    print(f"Generated data[0, 0]:\n{data[0, 0]}")

    output_folder = "."

    num_features = data.shape[-1]
    for feature_idx in range(num_features):
        feature_data = data[0, :, feature_idx].numpy()
        plt.figure()
        plt.plot(feature_data)
        plt.title(f"Feature {feature_idx}")
        plt.xlabel("Time")
        plt.ylabel("Value")

        output_path = os.path.join(output_folder, f"feature_{feature_idx}.pdf")
        plt.savefig(output_path)
        plt.close()
