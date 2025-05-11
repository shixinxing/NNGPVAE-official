import numpy as np
import h5py
import os
from pathlib import Path
from matplotlib import pyplot as plt


def transform_npz2h5py(file_name='hmnist_random', h5_file_path='h5'):
    print(f"====== loading {file_name} ======")
    npz_data = np.load(f'{file_name}.npz')
    h5_file_path = Path(h5_file_path) / f"{file_name}.h5"
    h5_file_path.parent.mkdir(exist_ok=True, parents=False)

    print(f"====== saving {file_name} to h5py ======")
    if h5_file_path.exists():
        print(f"====== {h5_file_path} already exists !!! ======")
    else:
        with h5py.File(h5_file_path, 'w') as h5_file:
            for key in npz_data.keys():
                h5_file.create_dataset(key, data=npz_data[key])

        npz_size = os.path.getsize(f'{file_name}.npz')
        h5_size = os.path.getsize(h5_file_path)
        print(f"Successfully converted npz (size: {npz_size}) to h5py (size: {h5_size}).\n")


def plot_seq(dic, num_seqs=5, num_imgs=25):
    fig, axs = plt.subplots(
        nrows=3 * num_seqs, ncols=num_imgs, figsize=(2 * num_imgs, 2 * 3 * num_seqs), constrained_layout=True
    )
    for i in range(num_seqs):
        for j in range(num_imgs):
            im = axs[3 * i, j].imshow(dic['x_test_miss'][i, j].reshape(28, 28), cmap='gray')
            axs[3 * i + 1, j].imshow(dic['m_test_miss'][i, j].reshape(28, 28), cmap='gray')
            # to check if we can get corrupted imgs from original ones
            axs[3 * i + 2, j].imshow(dic['x_test_full'][i, j].reshape(28, 28), cmap='gray')
            axs[3 * i, j].axis('off')
            axs[3 * i + 1, j].axis('off')
            axs[3 * i + 2, j].axis('off')

    cbar = fig.colorbar(im, orientation='vertical', shrink=0.6)
    cbar.set_label('pixel intensity')
    plt.show()


def plot_seq_misfrms(dic, num_seqs=5, num_imgs=50):
    fig, axs = plt.subplots(
        nrows=2 * num_seqs, ncols=num_imgs, figsize=(4 * num_imgs, 4 * 2 * num_seqs), constrained_layout=True
    )

    for i in range(num_seqs):
        for j in range(num_imgs):
            im = axs[2 * i, j].imshow(dic['seq_train_full'][i, j], cmap='gray')
            axs[2 * i, j].axis('off')
            axs[2 * i, j].text(
                0.5, -0.01, f"{int(dic['t_train_full'][i, j, 0])}-m{int(dic['m_train_miss'][i, j, 0])}",
                ha='center', va='top', transform=axs[2 * i, j].transAxes, fontsize=4, color='blue')

            axs[2 * i + 1, j].imshow(dic['seq_test_full'][i, -(j+1)], cmap='gray')
            axs[2 * i + 1, j].axis('off')
            axs[2 * i + 1, j].text(
                0.5, -0.01, f"{int(dic['t_test_full'][i, -(j+1), 0])}-m{int(dic['m_test_miss'][i, -(j+1), 0])}",
                ha='center', va='top', transform=axs[2 * i + 1, j].transAxes, fontsize=4, color='green')

    cbar = fig.colorbar(im, orientation='vertical', shrink=0.6)
    cbar.set_label('pixel intensity')
    plt.show()


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')

    flag = 'transform_long_hmnist'
    # flag = 'transform_hmnist'
    # flag = 'transform_hmnist_misfrms'

    if flag == 'transform_hmnist':
        random_mech = 'temporal_pos'
        transform_npz2h5py(file_name=f'hmnist_{random_mech}')

        with h5py.File(f"h5/hmnist_{random_mech}.h5", 'r') as h5_file:
            print("Datasets in the file:")
            for key in h5_file.keys():
                print(f"\t{key}: type:{type(h5_file[key])}, shape: {h5_file[key].shape}, "
                      f"take a fragment: {type(h5_file[key][:])}")

    elif flag == 'transform_long_hmnist':
        random_mech = 'consecutive_normal_random'
        # random_mech = 'anticlockwise_random'
        digit, seq_len = 'all', 100
        transform_npz2h5py(file_name=f'long_hmnist_{digit}_{random_mech}_{seq_len}', h5_file_path='h5')
        with h5py.File(f"h5/long_hmnist_{digit}_{random_mech}_{seq_len}.h5", 'r') as h5_file:
            print("Datasets in the file:")
            for key in h5_file.keys():
                print(f"\t{key}: type:{type(h5_file[key])}, shape: {h5_file[key].shape}, "
                      f"take a fragment: {type(h5_file[key][:])}")

            plot_seq(h5_file, num_seqs=5, num_imgs=25)

    elif flag == 'transform_hmnist_misfrms':
        file = 'misfrms_hmnist_all_anticlockwise_100'
        transform_npz2h5py(file_name=file, h5_file_path='h5')

        with h5py.File(f"h5/{file}.h5", 'r') as h5_file:
            print("Datasets in the file:")
            for key in h5_file.keys():
                print(f"\t{key}: type:{type(h5_file[key])}, shape: {h5_file[key].shape}, "
                      f"take a fragment: {type(h5_file[key][:])}")

            plot_seq_misfrms(h5_file, num_seqs=5, num_imgs=25)

    else:
        raise ValueError(f'Unknown flag: {flag}')


