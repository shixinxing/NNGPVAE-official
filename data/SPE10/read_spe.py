import numpy as np
from pathlib import Path


def read_poro_file(file_path='spe10_phi.data', Nx=60, Ny=220, Nz=85):
    arr_poro = np.loadtxt(file_path, skiprows=1)
    arr_poro = arr_poro.reshape(Nz, Ny, Nx)
    return arr_poro  # [Nz, Ny, Nx]


def read_perm_file(file_path='spe10_perm.data', Nx=60, Ny=220, Nz=85):
    perm_data = {}
    total_cells = Nx * Ny * Nz

    with open(file_path, 'r') as f:
        current_keyword = None
        temp_values = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line in ['PERMX', 'PERMY', 'PERMZ']:
                if current_keyword is not None and len(temp_values) > 0:
                    if len(temp_values) != total_cells:
                        raise ValueError(f"Got {len(temp_values)}.")

                    arr = np.array(temp_values).reshape(Nz, Ny, Nx)
                    perm_data[current_keyword] = arr
                    temp_values = []
                current_keyword = line
            else:  # data row
                parts = line.split()
                temp_values.extend([float(x) for x in parts])

        if current_keyword is not None and len(temp_values) > 0:
            if len(temp_values) != total_cells:
                raise ValueError(f"Got {len(temp_values)}.")

            arr = np.array(temp_values).reshape(Nz, Ny, Nx)
            perm_data[current_keyword] = arr

    # get array [Nz, Ny, Nx, 3]
    permx_arr = perm_data['PERMX']
    permy_arr = perm_data['PERMY']
    permz_arr = perm_data['PERMZ']
    perm_4d = np.stack([permx_arr, permy_arr, permz_arr], axis=-1)

    return perm_4d


def merge_data(data_poro, data_perm):
    # permeability requires a logarithmic scale
    data_perm = np.log(data_perm)
    data = np.concatenate([data_poro[..., np.newaxis], data_perm], axis=-1)  # [Nz, Ny, Nx, 4]
    data = data.transpose((2, 1, 0, 3))   # [Nx, Ny, Nz, 4]
    return data


def construct_dataset(data, factor: int, save=True):
    Nx, Ny, Nz = data.shape[:-1]
    loc = np.array(np.meshgrid(
        np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij'
    ))                                          # [3, 60, 220, 85]
    loc = loc.transpose(1, 2, 3, 0)             # [60, 220, 85, 3]

    # down-sampling along x-y
    loc = loc[::factor, ::factor, ::factor]     # [30, 110, 43, 3]
    data = data[::factor, ::factor, ::factor]   # [30, 110, 43, 4]

    # mask
    rng = np.random.default_rng(1024)
    mask = rng.uniform(size=data.shape) > 0.5
    # mask = np.ones_like(data, dtype=bool)
    # mask[15-5:15+5, 55-18:55+18, 20-6:20+6, 0] = False
    data_masked = data * mask

    # normalize; compute mean and std using observed pixels
    mean = np.mean(data_masked, axis=(0, 1, 2))
    std = np.std(data_masked, axis=(0, 1, 2))
    assert np.isfinite(std).all() and (std > 1e-5).all()
    data = (data - mean) / std
    data_masked = ((data_masked - mean) / std) * mask

    data_dict = {
        'X': loc.reshape(-1, 3),
        'Y': data_masked.reshape(-1, 4),
        'masks': mask.astype(np.float32).reshape(-1, 4),
        'Y_full': data.reshape(-1, 4),
        'Y_mean': mean, 'Y_std': std
    }

    if save:
        if Path('spe10.npz').exists():
            print(f"==== spe10.npz already exists! ====\n")
            return data_dict
        np.savez_compressed('spe10.npz', **data_dict)
    return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('MacOSX')

    data_poro = read_poro_file()
    data_perm = read_perm_file()
    data = merge_data(data_poro, data_perm)
    print(f"final data shape: {data.shape}\n")

    # refer to Chp 2.5, page 43 of An Introduction to Reservoir Simulation Using MATLAB/GNU Octave
    # https://www.cambridge.org/core/services/aop-cambridge-core/content/view/D54C1FD08E64FDD810436E0651E40650/9781108492430c2_21-54.pdf/modeling_reservoir_rocks.pdf
    fig, axs = plt.subplots(3, 1, figsize=(5, 8))
    im = axs[0].imshow(data[..., 0, 0], cmap='coolwarm')
    fig.colorbar(im, orientation='horizontal', shrink=0.5)
    axs[0].set_title("Poro, first layer")
    axs[0].axis('off')

    im = axs[1].imshow(data[..., 0, 1], cmap='coolwarm')
    fig.colorbar(im, orientation='horizontal', shrink=0.5)
    axs[1].set_title("Permeability, along x")
    axs[1].axis('off')

    im = axs[2].imshow(data[..., 0, 3], cmap='coolwarm')
    fig.colorbar(im, orientation='horizontal', shrink=0.5)
    axs[2].set_title("Permeability, along z")
    axs[2].axis('off')
    plt.show()

    construct_dataset(data, factor=2, save=True)
    dic = np.load('spe10.npz')
    print(f"Y_mean: {dic['Y_mean']}, Y_std: {dic['Y_std']}")
    for k, v in dic.items():
        print(k, v.shape)

    # plot missing
    last_layer = dic['Y'].reshape(30, 110, 43, 4)[..., 20, :]
    vmin, vmax = last_layer.min(), last_layer.max()

    fig, axs = plt.subplots(3, 1, figsize=(5, 8))
    im = axs[0].imshow(last_layer[..., 0], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[0].set_title("20th layer's poro")

    axs[1].imshow(last_layer[..., 1], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[1].set_title("20th permeability along x")

    axs[2].imshow(last_layer[..., 3], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[2].set_title("20th permeability along z")

    cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.02, pad=0.05)
    plt.show()



