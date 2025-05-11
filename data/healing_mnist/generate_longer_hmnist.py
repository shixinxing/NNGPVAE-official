from typing import Union
import random
import scipy
from pathlib import Path
import warnings

import numpy as np
import torch

from torchvision import datasets
from torchvision.transforms import transforms
from gpytorch.kernels import RBFKernel


def get_rotations(img, angles):
    for rot in angles:
        yield scipy.ndimage.rotate(img, rot, reshape=False)


def generate_one_video(img, angles):
    rotations = []
    for rot_img in get_rotations(img, angles):
        rotations.append(rot_img)
    return np.stack(rotations, axis=0)


def select_MNIST(
        save_path: Union[str, Path] = '.', seed=0, digit: Union[str, int] = 'all',
        num_train_digits=300, num_test_digits=100
):
    """Select images from MNIST, return a list as a collection."""
    # Load MNIST data
    mnist_train = datasets.MNIST(
        root=str(save_path / 'mnist'), train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )  # pixel range [0,1]

    # Get the indices
    local_random = random.Random(seed)
    if digit != 'all':  # Filter out images with correct digit
        indices = [i for i, data in enumerate(mnist_train) if data[1] == int(digit)]
        assert len(indices) >= (num_train_digits + num_test_digits), "Too many MNIST digits required"
        indices = local_random.sample(indices, num_train_digits + num_test_digits)
    else:  # Evenly collect the digits
        assert (num_train_digits % 10 == 0) and (num_test_digits % 10 == 0)
        num_samples_per_class = (num_train_digits + num_test_digits) // 10
        indices_per_class = {i: [] for i in range(10)}
        for idx, (img, label) in enumerate(mnist_train):
            if len(indices_per_class[label]) < num_samples_per_class:
                indices_per_class[label].append(idx)
            if all(len(indices_per_class[i]) >= num_samples_per_class for i in range(10)):
                break
        indices_train, indices_test = [], []
        for i in range(10):
            indices_train.extend(indices_per_class[i][:num_train_digits // 10])
            indices_test.extend(indices_per_class[i][num_train_digits // 10:])
        indices = indices_train + indices_test  # training idx first

    # Sample images according to indices
    train_data = []
    for idx in indices:
        img = mnist_train[idx][0]
        train_data.append(img.numpy().squeeze().astype(np.float32))
    print(f"====== Sampled {len(train_data)} digit '{digit}' from {len(indices)} images.======\n")
    return train_data


def rotate_MNIST(
        train_data: list, seed, num_train_digits=300, seq_len: int = 100,
        mode='consecutive_normal', periods=2, lenthscale=5., outputscale=3600.,  # angle args, (3\sigma=180°)
):
    """Rotate the selected MNIST images"""
    local_random = random.Random(seed)
    rng = np.random.default_rng(seed)

    # Rotate images
    x_train_full = []
    for img in train_data:
        if mode == 'random90':
            angles = [local_random.uniform(-90., 90) for _ in range(seq_len)]
        elif mode == 'anticlockwise':
            angles = np.linspace(0, 360 * periods, seq_len)
        elif mode == 'consecutive_normal':
            timestamps = torch.arange(0, seq_len, dtype=torch.float64)  # to improve numerical stability in MVN later
            with torch.no_grad():
                k = RBFKernel()
                k.lengthscale = lenthscale
                covar = k(timestamps).to_dense().numpy() * outputscale + np.eye(seq_len) * 1e-5
            angles = rng.multivariate_normal(mean=np.zeros(covar.shape[-1]), cov=covar)
        else:
            raise NotImplementedError(f"wrong mode: {mode}.")
        x_train_full.append(generate_one_video(img, angles))

    x_train_full = np.stack(x_train_full, axis=0)  # [v, f, 28, 28]
    x_train_full = np.array(x_train_full > 0.5).astype(np.float32)  # binarize

    rotated_train = x_train_full[:num_train_digits]
    rotated_test = x_train_full[num_train_digits:]

    print(f"====== Rotated {rotated_train.shape[0]} training series, {rotated_test.shape[0]} test series ======\n")
    return rotated_train, rotated_test


def generate_Healing_MNIST_miss_pixels(
        rotated_train, rotated_test, seed=0, num_train_digits=300, seq_len: int = 100,
        random_mechanism='random', missing_ratio=0.6,  # if`mnar`, this ratio is for black pixels
):
    """
    Missing pixels. Saves train/test sets as npz files: {'x_train_full', 'x_test_full',
    'x_train_miss', 'm_train_miss', 'x_test_miss', 'm_test_miss'} with shape like [60000 / 10000, 10, 784]
    """
    data_dict = {  # noqa
        'x_train_full': None, 'x_train_miss': None, 'm_train_miss': None,
        'x_test_full': None, 'x_test_miss': None, 'm_test_miss': None
    }
    data_dict['x_train_full'] = rotated_train
    data_dict['x_test_full'] = rotated_test

    # Mask some pixels Randomly
    rng = np.random.default_rng(seed)
    x_full = np.concatenate([rotated_train, rotated_test], axis=0)
    if random_mechanism == 'random':
        mask = rng.uniform(size=x_full.shape) > missing_ratio  # observed
        mask = ~ mask      # missed
        x_train_miss = x_full.copy()
        x_train_miss[mask] = 0.
    elif random_mechanism == 'mnar':  # white pixels are twice likely to be missing than black pixels
        pixel_probs = missing_ratio + x_full * missing_ratio  # this ratio is for black pixels
        mask = rng.uniform(size=x_full.shape) > pixel_probs
        mask = ~ mask
        x_train_miss = x_full.copy()
        x_train_miss[mask] = 0.
    else:
        raise NotImplementedError(f"wrong random_mechanism: {random_mechanism}.")

    data_dict['x_train_miss'] = x_train_miss[:num_train_digits]
    data_dict['m_train_miss'] = mask[:num_train_digits].astype(np.float32)
    data_dict['x_test_miss'] = x_train_miss[num_train_digits:]
    data_dict['m_test_miss'] = mask[num_train_digits:].astype(np.float32)

    # Flatten images to remain consistent to the AISTATS baseline, but we don't do so in missing-frame exp.
    for key in data_dict.keys():
        data_dict[key] = data_dict[key].reshape(-1, seq_len, 28 * 28)

    print(f"====== Generated Healing MNIST videos with missing pixels. ======\n")
    return data_dict


def save_Healing_MNIST_miss_pixels(
        save_path: Union[str, Path] = '.', seed=0, digit: Union[str, int] = 'all',
        num_train_digits=300, num_test_digits=100, seq_len: int = 100,
        mode='consecutive_normal', random_mechanism='random', missing_ratio=0.6,  # if`mnar`, this ratio is for black pixels
        periods=2, lenthscale=5., outputscale=60**2.  # angle args, (3\sigma=180°)
):
    save_path = Path(save_path) if not isinstance(save_path, Path) else save_path
    save_path.mkdir(parents=False, exist_ok=True)
    dict_name = save_path / f'long_hmnist_{digit}_{mode}_{random_mechanism}_{seq_len}.npz'
    if dict_name.exists():
        warnings.warn(f"The file {dict_name} already exists! We don't change anything!")
        data_dict = np.load(dict_name)
        return data_dict

    imgs = select_MNIST(save_path, seed, digit, num_train_digits, num_test_digits)
    seq_train, seq_test = rotate_MNIST(imgs, seed, num_train_digits, seq_len, mode, periods, lenthscale, outputscale)
    data_dict = generate_Healing_MNIST_miss_pixels(
        seq_train, seq_test, seed, num_train_digits, seq_len, random_mechanism, missing_ratio
    )

    # Save data dict
    np.savez_compressed(
        dict_name,
        x_train_full=data_dict['x_train_full'],
        x_train_miss=data_dict['x_train_miss'],
        m_train_miss=data_dict['m_train_miss'],
        x_test_full=data_dict['x_test_full'],
        x_test_miss=data_dict['x_test_miss'],
        m_test_miss=data_dict['m_test_miss']
    )
    print(f"====== Successfully saved {dict_name}. ======\n")
    return data_dict


def save_Healing_MNIST_miss_frames(
        save_path: Union[str, Path] = '.', seed=0, digit: Union[str, int] = 'all',
        num_train_digits=300, num_test_digits=100, seq_len: int = 100,
        mode='consecutive_normal', missing_ratio=0.6,
        periods=2, lenthscale=5., outputscale=60**2.  # angle args, (3\sigma=180°)
):
    save_path = Path(save_path) if not isinstance(save_path, Path) else save_path
    save_path.mkdir(parents=False, exist_ok=True)
    dict_name = save_path / f'misfrms_hmnist_{digit}_{mode}_{seq_len}.npz'
    if dict_name.exists():
        warnings.warn(f"The file {dict_name} already exists! We don't change anything!")
        data_dict = np.load(dict_name)
        return data_dict

    # Generate sequences
    imgs = select_MNIST(save_path, seed, digit, num_train_digits, num_test_digits)
    seq_train, seq_test = rotate_MNIST(
        imgs, seed, num_train_digits, seq_len, mode, periods, lenthscale, outputscale
    )   # [v,f,28,28]

    # Mask frames
    rng = np.random.default_rng(seed)
    seqs = np.concatenate([seq_train, seq_test], axis=0)
    seq_full = np.empty(seqs.shape, dtype=np.float32)   # [v,f,28,28]
    t_full = np.empty(seqs.shape[:2], dtype=np.float32)[:, :, None]
    t_mask = np.empty(seqs.shape[:2], dtype=np.float32)[:, :, None]
    for i in range(seqs.shape[0]):
        mask_frames = rng.binomial(1, 1-missing_ratio, size=(seqs.shape[1] - 2))
        mask_frames = np.array([1] + list(mask_frames) + [1]).astype(bool)  # must select the start and the end

        # move observation to the left, only the first part contains the selected data, second part are zeros.
        sequence = seqs[i]
        seq_selected = sequence[mask_frames]
        seq_full[i, :seq_selected.shape[0]] = seq_selected
        seq_full[i, seq_selected.shape[0]:] = sequence[~mask_frames]

        t_input = np.linspace(0, seqs.shape[1] - 1, seqs.shape[1])[:, None]  # [f,1]
        t_selected = t_input[mask_frames]
        t_full[i, :t_selected.shape[-2]] = t_selected
        t_full[i, t_selected.shape[-2]:] = t_input[~mask_frames]
        t_mask[i, :t_selected.shape[-2]] = 1.
        t_mask[i, t_selected.shape[-2]:] = 0.

    data_dict = {
        "seq_train_full": seq_full[:num_train_digits],
        "t_train_full": t_full[:num_train_digits],
        "m_train_miss": t_mask[:num_train_digits],
        "seq_test_full": seq_full[num_train_digits:],
        "t_test_full": t_full[num_train_digits:],
        "m_test_miss": t_mask[num_train_digits:]
    }
    np.savez_compressed(dict_name, **data_dict)
    print(f"====== Successfully saved {dict_name}. ======\n")
    return data_dict


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use('MacOSX')

    FLAG = 0

    if FLAG == 0:
        # mode = 'anticlockwise'
        mode = 'consecutive_normal'
        # random_mech, mis_rate = 'mnar', 0.4
        random_mech, mis_rate = 'random', 0.6

        dic = save_Healing_MNIST_miss_pixels(
            mode=mode, random_mechanism=random_mech, seed=1024,
            num_train_digits=4000, num_test_digits=1000, seq_len=100, missing_ratio=mis_rate,
            periods=2, lenthscale=5., outputscale=120**2.
        )
        for key in dic.keys():
            print(f"{key}: {dic[key].shape}, {dic[key].dtype}")

        num_seqs, num_imgs = 3, 20
        fig, axs = plt.subplots(nrows=3*num_seqs, ncols=num_imgs, figsize=(2 * num_imgs, 2*3 * num_seqs), constrained_layout=True)
        for i in range(num_seqs):
            for j in range(num_imgs):
                im = axs[3*i, j].imshow(dic['x_train_miss'][i, j].reshape(28, 28), cmap='gray')
                axs[3*i+1, j].imshow(dic['m_train_miss'][i, j].reshape(28, 28), cmap='gray')
                # to check if we can get corrupted imgs from original ones
                # axs[3*i+2, j].imshow((dic['x_train_full'][i, j]*(1-dic['m_train_miss'][i, j])).reshape(28, 28), cmap='gray')
                axs[3*i+2, j].imshow(dic['x_train_full'][i, j].reshape(28, 28), cmap='gray')
                axs[3*i, j].axis('off')
                axs[3*i+1, j].axis('off')
                axs[3*i+2, j].axis('off')

        cbar = fig.colorbar(im, orientation='vertical', shrink=0.6)
        cbar.set_label('pixel intensity')
        plt.show()

    elif FLAG == 1:
        mode = 'anticlockwise'

        dic = save_Healing_MNIST_miss_frames(
            mode=mode, seed=1024,
            num_train_digits=4000, num_test_digits=1000, seq_len=100, missing_ratio=0.6,
            periods=2, lenthscale=5., outputscale=120 ** 2.
        )
        for key in dic.keys():
            print(f"{key}: {dic[key].shape}, {dic[key].dtype}")

        num_seqs, num_imgs = 3, 20
        fig, axs = plt.subplots(nrows=2 * num_seqs, ncols=num_imgs, figsize=(2 * num_imgs, 2.1 * 2 * num_seqs),
                                constrained_layout=True)

        for i in range(num_seqs):
            for j in range(num_imgs):
                im = axs[2*i, j].imshow(dic['seq_train_full'][i, j], cmap='gray')
                axs[2*i, j].axis('off')
                axs[2*i, j].text(
                    0.5, -0.01, f"Train-{dic['t_train_full'][i, j, 0]}-m{dic['m_train_miss'][i, j, 0]}",
                    ha='center', va='top', transform=axs[2*i, j].transAxes, fontsize=6, color='blue')

                axs[2*i+1, j].imshow(dic['seq_test_full'][i, -j], cmap='gray')
                axs[2*i+1, j].axis('off')
                axs[2*i+1, j].text(
                    0.5, -0.01, f"Test-{dic['t_test_full'][i, -j, 0]}-m{dic['m_test_miss'][i, -j, 0]}",
                    ha='center', va='top', transform=axs[2*i+1, j].transAxes, fontsize=6, color='green')

        cbar = fig.colorbar(im, orientation='vertical', shrink=0.6)
        cbar.set_label('pixel intensity')
        plt.show()


