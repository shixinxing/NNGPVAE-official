import numpy as np
from matplotlib import pyplot as plt


def sort_frames_time_mask(pred_dict: dict):
    """
    given prediction dict with ndarray: {'images': [V,T,28,28], 'aux_data': [V,T,1]'masks': [V,T,1]},
    sort them to the original order
    """
    videos = pred_dict['images']
    timestamps = pred_dict['aux_data']
    masks = pred_dict['masks']

    sorted_indices = np.argsort(timestamps.squeeze(-1), axis=-1)   # [V,T]

    # sort time and masks
    sorted_timestamps = np.take_along_axis(timestamps, sorted_indices[:, :, np.newaxis], axis=1)  # [V, T, 1]
    sorted_time_mask = np.take_along_axis(masks, sorted_indices[:, :, np.newaxis], axis=1)        # [V, T, 1]

    # sort videos
    img_size = videos.shape[-1]
    sorted_indices_video = sorted_indices[:, :, np.newaxis, np.newaxis]
    sorted_indices_video = np.repeat(sorted_indices_video, img_size, axis=-2)
    sorted_indices_video = np.repeat(sorted_indices_video, img_size, axis=-1)
    sorted_videos = np.take_along_axis(videos, sorted_indices_video, axis=1)

    return {
        'images': sorted_videos, 'aux_data': sorted_timestamps, 'masks': sorted_time_mask
    }


def plot_hmnist_misfrms(pred_dict, seqs, num_imgs=20):
    num_seqs = len(seqs)
    fig, axs = plt.subplots(
        nrows=num_seqs, ncols=num_imgs, figsize=(4 * num_imgs, 4 * num_seqs), constrained_layout=True
    )

    for i in range(num_seqs):
        for j in range(num_imgs):
            im = axs[i, j].imshow(pred_dict['images'][seqs[i], -(j+1)], cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].text(
                0.5, -0.01, f"{int(pred_dict['aux_data'][seqs[i], -(j+1), 0])}",
                ha='center', va='top', transform=axs[i, j].transAxes, fontsize=15,
                color='green' if pred_dict['masks'][seqs[i], -(j+1), 0] else 'red')

    cbar = fig.colorbar(im, orientation='vertical', shrink=0.6)
    cbar.set_label('pixel intensity')

    return fig, axs



