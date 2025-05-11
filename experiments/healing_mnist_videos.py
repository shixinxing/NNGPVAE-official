import argparse
from datetime import datetime
from pathlib import Path
import pickle
import h5py
import warnings

import numpy as np
import torch

from models.gpvae_hmnist import GPVAEHealing_SWS, GPVAEHealing_VNN
from models.gpvae_longhmnist import GPVAELongHealing_SWS, GPVAELongHealing_VNN
from utils.healing_mnist import plot_hmnist
from utils.exp_utils.memo_analysis import (set_up_logger, start_record_memory_history,
                                           export_memory_snapshot, stop_record_memory_history, count_params)
import time


def run_healing_mnist(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.float64:
        torch.set_default_dtype(torch.float64)
        warnings.warn(f"Faiss-GPU does not support torch.float64 yet.")
    else:
        torch.set_default_dtype(torch.float32)

    jitter = 1e-8 if args.float64 else 1e-6

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False   # for reproducibility
        torch.backends.cudnn.deterministic = True
        print(f"cudnn benchmark: {torch.backends.cudnn.benchmark}")
    else:
        device = 'cpu'

    current_time = datetime.now().strftime("%m%d_%H_%M_%S")
    print(f"Time: {current_time}", f"{args.model_type}-H{args.H} for {args.dataset}, \n",
          f"Using: {device}, seed: {args.seed}, "
          f"lr={args.lr}, beta={args.beta}, max_norm={args.max_norm} ", "GP joint training, " if args.GP_joint else "",
          f"{torch.get_default_dtype()}\n")

    # Load data from h5 files
    file_path = Path(f'../data/healing_mnist/h5/{args.dataset}.h5')

    # logger = set_up_logger()
    # logger = start_record_memory_history(logger)
    # Training
    s = time.time()
    if args.model_type == 'SWS':
        if args.dataset.startswith('hmnist'):
            model = GPVAEHealing_SWS(
                args.H, file_path, args.GP_joint, val_split=args.val_split, search_device=device, jitter=jitter
            )
        elif args.dataset.startswith('long_hmnist'):
            model = GPVAELongHealing_SWS(args.H, file_path, GP_joint=args.GP_joint, search_device=device, jitter=jitter)
        else:
            raise ValueError(f"Unknown dataset {args.dataset}.")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        model.train_gpvae(
            optimizer, args.beta, epochs=args.num_epochs,
            series_batch_size=args.series_batch_size, timestamp_batch_size=args.timestamp_batch_size,
            max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs
        )
    elif args.model_type == 'VNN':
        if args.dataset.startswith('hmnist'):
            model = GPVAEHealing_VNN(
                args.H, file_path, args.GP_joint, val_split=args.val_split, search_device=device, jitter=jitter
            )
        elif args.dataset.startswith('long_hmnist'):
            model = GPVAELongHealing_VNN(args.H, file_path, args.GP_joint, search_device=device, jitter=jitter)
        else:
            raise ValueError(f"Unknown dataset {args.dataset}.")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        model.train_gpvae(
            optimizer, args.beta, epochs=args.num_epochs,
            series_batch_size=args.series_batch_size, timestamp_kl_batch_size=args.timestamp_batch_size,
            timestamp_expected_lk_batch_size=None,
            max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    # logger = export_memory_snapshot(logger)
    # stop_record_memory_history(logger)
    # print(f"Total params: {count_params(model)}\n")  # 802065; conv1D baselines: 1203473 (diag) / 1269265 (tridiag)

    e = time.time()
    print(f"Training time: {e - s}")
    # To give a sense of model size
    print(f"Total parameters: {sum([p.shape.numel() for p in model.parameters() if p.requires_grad])}\n")

    nll, mse, mse_non_round, y_rec = model.predict_gpvae(
        file_path, series_batch_size=args.series_batch_size, timestamp_batch_size=args.timestamp_batch_size,
        device=device, num_samples=20
    )
    print(f"\nNLL: {nll}, MSE at missing pixels: {mse}, MSE (not rounded) at all pixels: {mse_non_round}.\n")

    # Plot reconstructed images
    video_idx = [10*i for i in range(args.num_plots)]
    test_dict = h5py.File(file_path, 'r')
    fig, ax = plot_hmnist(
        test_dict['x_test_miss'][video_idx, :10].reshape(-1, 10, 28, 28),
        y_rec[video_idx, :10],
        test_dict['x_test_full'][video_idx, :10].reshape(-1, 10, 28, 28)
    )

    experiment_dir = Path(f'./exp_{args.model_type}_H{args.H}_{args.dataset}')
    if args.save:
        experiment_dir.mkdir(exist_ok=True)
        trial_dir = experiment_dir / f'{current_time}'
        trial_dir.mkdir(parents=False, exist_ok=True)

        torch.save(model.state_dict(), trial_dir / f'model_{args.dataset}.pth')   # model

        img_path = trial_dir / f'img_{args.dataset}.pdf'                          # images
        fig.savefig(img_path, bbox_inches='tight')

        everything_for_imgs = {
            # 'y_test_full': y_test_full, 'y_test_miss': y_test_miss, # to save storage
            'y_rec': y_rec, 'NLL': nll, 'MSE': mse, 'MSE_non_round': mse_non_round
        }
        imgs_pickle_path = trial_dir / 'everything_for_imgs.pkl'
        with imgs_pickle_path.open('wb') as f:
            pickle.dump(everything_for_imgs, f)

        print("The experiment results are saved at:")
        print(f"{trial_dir.resolve()}")  # get absolute path for moving the log file
    else:
        print("The experimental results are not saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Healing MNIST Imputation Experiment')

    parser.add_argument('--seed', type=int, default=5, help='random seed')
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--model_type', type=str, default='SWS', choices=['SWS', 'VNN'], help='model used')
    parser.add_argument('--H', type=int, default=10, help='number of nearest neighbors')
    parser.add_argument('--beta', type=float, default=0.8, help='beta in ELBO')

    parser.add_argument(
        '--dataset', type=str, default='hmnist_random',
        help="e.g.,'hmnist_random', 'long_hmnist_all_anticlockwise_100', 'long_hmnist_5_consecutive_normal_1000'"
    )
    parser.add_argument('--val_split', type=int, default=50000,   # special for HMNIST
                        help='validation split < 60000 for hmnist, i.e., num of training data')

    parser.add_argument('--GP_joint', action='store_true', help='whether to train GP params jointly')

    parser.add_argument('--series_batch_size', type=int, default=50, help='batch size for series')
    parser.add_argument('--timestamp_batch_size', type=int, default=10, help='batch size for timestamp')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for sum loss (instead of mean loss)')
    parser.add_argument('--max_norm', type=float, default=1e4, help='max norm of gradients in grad norm clipping')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')

    parser.add_argument('--num_plots', type=int, default=10, help='number of series to plot')
    parser.add_argument('--save', action='store_true', help='store everything into a folder')

    args_hmnist = parser.parse_args()

    from utils.exp_utils.memo_analysis import print_RAM_usage

    print_RAM_usage()

    run_healing_mnist(args_hmnist)



