import argparse
import warnings
from datetime import datetime
from pathlib import Path
import time
import pickle

import numpy as np
import torch

from models.gpvae_longhmnist_misfrms import GPVAELongHMNISTMisFrms_SWS, GPVAELongHMNISTMisFrms_VNN
from utils.healing_mnist_misfrms import sort_frames_time_mask, plot_hmnist_misfrms


def run_hmnist_missing_frames(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.float64:
        torch.set_default_dtype(torch.float64)
        warnings.warn(f"Faiss-GPU does not support torch.float64 yet.")
    else:
        torch.set_default_dtype(torch.float32)

    jitter = 1e-8 if args.float64 else 1e-5   # 1e-6 may encounter numerical warnings

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False  # for reproducibility
        torch.backends.cudnn.deterministic = True
        print(f"cudnn benchmark: {torch.backends.cudnn.benchmark}")
    else:
        device = 'cpu'

    current_time = datetime.now().strftime("%m%d_%H_%M_%S")
    print(f"Time: {current_time}", f"{args.model_type}-H{args.H} for {args.dataset} ",
          f"Using: {device}, seed: {args.seed}, "
          f"lr={args.lr}, beta={args.beta}, max_norm={args.max_norm} ", "GP joint training, " if args.GP_joint else "",
          f"{torch.get_default_dtype()}\n")

    file_path = Path(f'../data/healing_mnist/h5/{args.dataset}.h5')

    s = time.time()
    if args.model_type == 'SWS':
        model = GPVAELongHMNISTMisFrms_SWS(
            args.H, file_path, GP_joint=args.GP_joint, search_device=device, jitter=jitter)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        # with torch.autograd.detect_anomaly():
        model.train_gpvae(
            optimizer, args.beta, epochs=args.num_epochs,
            series_batch_size=args.series_batch_size, timestamp_batch_size=args.timestamp_batch_size,
            max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs
        )
    elif args.model_type == 'VNN':
        model = GPVAELongHMNISTMisFrms_VNN(
            args.H, file_path, GP_joint=args.GP_joint, search_device=device, jitter=jitter)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        model.train_gpvae(
            optimizer, args.beta, epochs=args.num_epochs,
            series_batch_size=args.series_batch_size, timestamp_kl_batch_size=args.timestamp_batch_size,
            max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs
        )
    else:
        raise NotImplementedError(f"Wrong model type: {args.model_type}")

    e = time.time()
    print(f"Training time: {e - s}")
    print(f"Total parameters: {sum([p.shape.numel() for p in model.parameters() if p.requires_grad])}\n")

    # Prediction
    nll, mse, mse_non_round, pred_coll = model.predict_gpvae(
        file_path, series_batch_size=args.series_batch_size, timestamp_batch_size=args.timestamp_batch_size,
        device=device, num_samples=20
    )
    print(f"\nNLL: {nll}, MSE at missing frames: {mse}, MSE (not rounded) per frame: {mse_non_round}.\n")
    pred_coll = sort_frames_time_mask(pred_coll)

    # Plot
    seq_ids = [100*i for i in range(args.num_plots)]
    fig, ax = plot_hmnist_misfrms(pred_coll, seqs=seq_ids, num_imgs=20)

    experiment_dir = Path(f'./exp_{args.dataset}_{args.model_type}_H{args.H}')
    if args.save:
        experiment_dir.mkdir(exist_ok=True)
        trial_dir = experiment_dir / f'{current_time}'
        trial_dir.mkdir(parents=False, exist_ok=True)

        torch.save(model.state_dict(), trial_dir / f'model_{args.dataset}.pth')  # model

        img_path = trial_dir / f'img_misfrms.pdf'  # images
        fig.savefig(img_path, bbox_inches='tight')

        everything_for_imgs = {
            # 'y_test_full': y_test_full, 'y_test_miss': y_test_miss, # to save storage
            'y_rec': pred_coll, 'NLL': nll, 'MSE': mse, 'MSE_non_round': mse_non_round
        }
        imgs_pickle_path = trial_dir / 'everything_for_imgs.pkl'
        with imgs_pickle_path.open('wb') as f:
            pickle.dump(everything_for_imgs, f)

        print("The experiment results are saved at:")
        print(f"{trial_dir.resolve()}")  # get absolute path for moving the log file
    else:
        print("The experimental results are not saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Healing MNIST with Missing Frames")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--model_type', type=str, default='SWS', choices=['SWS', 'VNN'], help='model used')
    parser.add_argument('--H', type=int, default=20, help='number of nearest neighbors')
    parser.add_argument('--beta', type=float, default=0.8, help='beta in ELBO')
    parser.add_argument('--GP_joint', action='store_true', help='whether to train GP params jointly')

    parser.add_argument('--dataset', type=str, default="misfrms_hmnist_all_anticlockwise_100", help='file name')
    parser.add_argument('--series_batch_size', type=int, default=50, help='batch size for series')
    parser.add_argument('--timestamp_batch_size', type=int, default=20, help='batch size for timestamp')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for sum loss (instead of mean loss)')
    parser.add_argument('--max_norm', type=float, default=1e10, help='max norm of gradients in grad norm clipping')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of training epochs')

    parser.add_argument('--num_plots', type=int, default=4, help='number of series to plot')
    parser.add_argument('--save', action='store_true', help='store everything into a folder')

    args_misfrms_hmnist = parser.parse_args()

    run_hmnist_missing_frames(args_misfrms_hmnist)

