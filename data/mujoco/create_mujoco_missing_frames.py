import numpy as np
from latent_ode.mujoco_physics import HopperPhysics, HumanoidCMUPhysics
from pathlib import Path
from tqdm import tqdm
import random

random.seed(0)
np.random.seed(0)

mode = "missing"
selected_prob = 0.4
extrap = False
timepoints = 10_000  # 1000
D, n_training_samples = 20, 200  # 500

if timepoints == 1000:
    dataset_obj = HopperPhysics(root='./data/mujoco/latent_ode', T=timepoints, D=D, n_training_samples=n_training_samples,
                                download=False, generate=True, local_seed=123)
elif timepoints == 10_000:
    dataset_obj = HumanoidCMUPhysics(root='./data/mujoco/latent_ode', T=timepoints, D=D, n_training_samples=n_training_samples,
                              task_name="walk", download=False, generate=True, local_seed=123)  # walk, run, stand

dataset = dataset_obj.get_dataset()  # torch
dataset = dataset.numpy()
n_tp_data = dataset[:].shape[1]
dataset = dataset[:, :timepoints, :]

if mode == "missing":
    # Creating dataset for interpolation
    # sample time points from different parts of the timeline,
    # so that the model learns from different parts of hopper trajectory
    n_traj = len(dataset)
    n_tp_data = dataset.shape[1]
    n_reduced_tp = timepoints

    # sample time points from different parts of the timeline,
    # so that the model learns from different parts of hopper trajectory
    start_ind = np.random.randint(0, high=n_tp_data - n_reduced_tp + 1, size=n_traj)
    end_ind = start_ind + n_reduced_tp

    seq_full = np.empty(dataset.shape, dtype=np.float32)   # [500,1000,14]
    t_full = np.empty(dataset.shape[:-1], dtype=np.float32)[:, :, None]
    t_mask = np.empty(dataset.shape[:-1], dtype=np.float32)[:, :,  None]
    for i in tqdm(range(dataset.shape[0])):
        mask_frames = np.random.binomial(1, selected_prob, size=(timepoints - 2))
        mask_frames = np.array([1] + list(mask_frames) + [1]).astype(bool)  # must select the start and the end
        seq_length = int(mask_frames.sum())  # number of selected (non-missing) frames

        # move observation to the left, only the first part contains the selected data, second part are zeros.
        sequence = dataset[i]
        seq_selected = sequence[mask_frames]
        seq_full[i, :seq_selected.shape[-2]] = seq_selected
        seq_full[i, seq_selected.shape[-2]:] = sequence[~mask_frames]

        t_input = np.linspace(0, timepoints - 1, timepoints)[:, None]  # [1000,1] or [100, 1]
        t_selected = t_input[mask_frames]
        t_full[i, :t_selected.shape[-2]] = t_selected
        t_full[i, t_selected.shape[-2]:] = t_input[~mask_frames]
        t_mask[i, :t_selected.shape[-2]] = 1.
        t_mask[i, t_selected.shape[-2]:] = 0.

    if timepoints == 1000:
        assert D == 14 and n_training_samples == 500
        data_dict = {
            "seq_train_full": seq_full[:320], "t_train_full": t_full[:320], "m_train_miss": t_mask[:320],
            "seq_val_full": seq_full[320:-100], "t_val_full": t_full[320:-100], "m_val_miss": t_mask[320:-100],
            "seq_test_full": seq_full[-100:], "t_test_full": t_full[-100:], "m_test_miss": t_mask[-100:]
        }
        dict_name = f"./data/mujoco/mujoco_{mode}_1000.npz"

    elif timepoints == 100:
        assert D == 14 and n_training_samples == 2000
        data_dict = {
            "seq_train_full": seq_full[:1280], "t_train_full": t_full[:1280], "m_train_miss": t_mask[:1280],
            "seq_val_full": seq_full[1280:-400], "t_val_full": t_full[1280:-400], "m_val_miss": t_mask[1280:-400],
            "seq_test_full": seq_full[-400:], "t_test_full": t_full[-400:], "m_test_miss": t_mask[-400:]
        }
        dict_name = f"./data/mujoco/mujoco_{mode}_100.npz"

    elif timepoints == 10_000:
        assert D == 20 and n_training_samples == 200
        data_dict = {
            "seq_train_full": seq_full[:100], "t_train_full": t_full[:100], "m_train_miss": t_mask[:100],
            "seq_val_full": seq_full[100:140], "t_val_full": t_full[100:140], "m_val_miss": t_mask[100:140],
            "seq_test_full": seq_full[140:], "t_test_full": t_full[140:], "m_test_miss": t_mask[140:]
        }
        dict_name = f"./data/mujoco/mujoco_{mode}_10000.npz"

    else:
        raise NotImplementedError

    if Path(dict_name).exists():
        print(f"{dict_name} already exists! ")
    else:
        np.savez_compressed(dict_name, **data_dict)
    print(f"Mujoco data dict Saved!")

    data_dict = np.load(dict_name)
    for key in data_dict.keys():
        print(f"{key}: {data_dict[key].shape}")

