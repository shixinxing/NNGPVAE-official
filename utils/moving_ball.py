# adapt from https://github.com/ratschlab/SVGP-VAE/blob/4775cfb83a576606736e944ac0ef4c84b358c4af/utils.py

import numpy as np
from matplotlib import pyplot as plt


def generate_moving_ball(
        num_videos=35, num_frames=30, img_size=32,
        lengthscale=2., r=3,
        seed=None, constraint=True
):
    """
    Generate moving-ball videos from Pearce's paper;
    use numpy and adapt the original code as closely as possible for fair comparison

    return:
    (non-scaled) trajectory [num_videos, num_frames, 2]
    videos [num_videos, num_frames， img_size, img_size]
    """
    K = np.arange(num_frames)
    K = - (K.reshape(num_frames, 1) - K.reshape(1, num_frames)) ** 2 / (2 * lengthscale ** 2)
    K = np.exp(K) + 1e-5 * np.eye(num_frames)
    L = np.linalg.cholesky(K)

    # sample latent coordinates (x,y)
    rng = np.random.RandomState(seed)  # local random seed, for fair comparison
    eps = rng.normal(loc=0, scale=1, size=(num_frames, 2 * num_videos))
    paths = (L @ eps).reshape(num_frames, num_videos, 2).transpose(1, 0, 2)  # [videos, frames, 2]
    if constraint:  # the x/y's 3sigma = 3
        paths_rescale = img_size / 2 + np.clip(
            paths / 3 * (img_size / 2 - r), a_min=-(img_size / 2 - r), a_max=(img_size / 2 - r))
    else:           # as in the original paper
        paths_rescale = img_size / 2 + paths * img_size / 5

    # draw pictures
    rr = r * r
    x, y = np.arange(img_size), np.arange(img_size)
    vid_batch = []
    for v in range(num_videos):
        frames = []
        for f in range(num_frames):
            lx = ((x - paths_rescale[v, f, 0]) ** 2).reshape(1, img_size)
            ly = ((y - paths_rescale[v, f, 1]) ** 2).reshape(img_size, 1)
            frame = lx + ly < rr  # [32, 32]
            frames.append(frame.reshape(1, 1, img_size, img_size))
        frames = np.concatenate(frames, axis=1)  # [1, f, 32, 32]
        vid_batch.append(frames)
    vid_batch = np.concatenate(vid_batch, axis=0)  # [v, f, 32, 32]

    return paths_rescale, vid_batch


def generate_video_dict(paths, vid_batch):
    """
    Generate a dict {'images': [v,f,32,32], 'aux_data': [f,1], 'paths':[v,f,2]}
    """
    data_dict = {
        'images': vid_batch,
        'aux_data': None,
        'paths': paths
    }
    num_frames = paths.shape[-2]
    t = np.arange(0, num_frames, dtype=paths.dtype)
    # num_videos = paths.shape[-3]
    # t = np.repeat(t[np.newaxis, :], num_videos, axis=0)[..., np.newaxis]
    data_dict['aux_data'] = t[:, np.newaxis]
    return data_dict


def path_rotation(paths: np.ndarray, target_paths: np.ndarray):
    """
    rotate and translate predictive paths to target_paths: XW + B = Y
    paths / target_paths: [v, f, 2]
    return rotated paths [v, f, 2] and squared error per video
    """
    num_videos, num_frames = paths.shape[:-1]
    paths = paths.reshape(num_videos * num_frames, 2)
    paths = np.hstack([paths, np.ones((num_videos * num_frames, 1))])
    target_paths = target_paths.reshape(num_videos * num_frames, 2)

    W, residuals, _, _ = np.linalg.lstsq(paths, target_paths, rcond=None)

    rotate_paths = (paths @ W).reshape(num_videos, num_frames, 2)
    squared_error = residuals.sum() / num_videos

    return rotate_paths, squared_error


def plot_balls(
        true_vids, true_path, recon_vids=None, recon_path=None,
        nplots=4
):
    """
    Plots an array of input videos and reconstructions.
    args:
        truevids: (batch, tmax, px, py) np array of videos
        truepath: (batch, tmax, 2) np array of latent positions
        reconvids: (batch, tmax, px, py) np array of videos
        reconpath: (batch, tmax, 2) np array of latent positions
        reconvar: (batch, tmax, 2, 2) np array, cov mat
        ax: (optional) list of lists of axes objects for plotting
        nplots: int, number of rows of plot, each row is one video
        paths: (batch, tmax, 2) np array optional extra array to plot

    returns:
        fig: figure object with all plots
    """
    def make_heatmap(vid):
        """
        :param vid: tmax, px, py
        :returns: flat_vid: px, py
        """
        vid = np.array([(t + 4) * v for t, v in enumerate(vid)])
        tmax = vid.shape[-3]
        flat_vid = np.max(vid, axis=0) / (4 + tmax)  # [px, py]
        return flat_vid

    def plot_rows(ax, i):
        # i is batch element = plot column

        # true data heatmap
        tv = make_heatmap(true_vids[i, :, :, :])
        ax[i, 0].imshow(1 - tv, origin='lower', cmap='Greys')
        ax[i, 0].axis('off')

        # middle row is trajectories
        ax[i, 1].plot(true_path[i, :, 0], true_path[i, :, 1])
        ax[i, 1].set_xlim([xmin, xmax])
        ax[i, 1].set_ylim([ymin, ymax])
        ax[i, 1].scatter(true_path[i, -1, 0], true_path[i, -1, 1])

        if recon_path is not None:
            ax[i, 1].plot(recon_path[i, :, 0], recon_path[i, :, 1])
            ax[i, 1].scatter(recon_path[i, -1, 0], recon_path[i, -1, 1])

        # reconstructed video
        if recon_vids is not None:
            rv = make_heatmap(recon_vids[i, :, :, :])
            ax[i, 2].imshow(1 - rv, origin='lower', cmap='Greys')
            ax[i, 2].axis('off')

    fig, axes = plt.subplots(nplots, 3, figsize=(3 * 1.5, nplots * 1.5))

    # get axis limits for the latent space
    xmin = np.min([true_path[:nplots, :, 0].min(), recon_path[:nplots, :, 0].min()]) - 0.1
    xmin = np.min([xmin, -2.5])
    xmax = np.max([true_path[:nplots, :, 0].max(), recon_path[:nplots, :, 0].max()]) + 0.1
    xmax = np.max([xmax, 2.5])

    ymin = np.min([true_path[:nplots, :, 1].min(), recon_path[:nplots, :, 1].min()]) - 0.1
    ymin = np.min([ymin, -2.5])
    ymax = np.max([true_path[:nplots, :, 1].max(), recon_path[:nplots, :, 1].max()]) + 0.1
    ymax = np.max([ymax, 2.5])

    for n in range(nplots):
        plot_rows(axes, n)

    return fig, axes


