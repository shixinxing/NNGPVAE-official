import os
import time
import h5py
import argparse
import warnings
from datetime import datetime
import numpy as np
from pathlib import Path

import torch

from models.gpvae_mujoco import GPVAEMujoco_SWS, GPVAEMujoco_VNN

def run_mujoco_missing_frames(args):
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
        search_device = 'cuda'
    else:
        device = 'cpu'
        search_device = 'cpu'

    current_time = datetime.now().strftime("%m%d_%H_%M_%S")
    print(f"Time: {current_time}, T: {args.T}, {args.model_type}-H{args.H}, Using: {device}, seed: {args.seed}, ",
          f"lr={args.lr}, beta={args.beta}, ", "GP joint training, " if args.GP_joint else "",
          f"Fix decoder variance, " if args.fix_decoder_variance else "Train decoder variance",
          f"{torch.get_default_dtype()}\n")

    exp_result_folder = f"./exp/mujoco/T={args.T}/{args.model_type}/H={args.H}/{current_time}"
    os.makedirs(exp_result_folder, exist_ok=True)

    # Save configs
    with open(f'{exp_result_folder}/configs.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}={value};\n')

    file_path = Path(f'./data/mujoco/mujoco_missing_{args.T}.npz')
    assert file_path.exists(), f"{file_path} does not exist, do you choose a wrong T?"

    if args.T == 1000:
        init_lengthscale = 50.
        y_dim = 14
    elif args.T == 100:
        init_lengthscale = 5.
        y_dim = y_dim = 14
    elif args.T == 10_000:
        init_lengthscale = 50.
        y_dim = 125
    else:
        raise NotImplementedError

    s = time.time()
    if args.model_type == 'SWS':
        model = GPVAEMujoco_SWS(args.H,
                                file_path,
                                GP_joint=args.GP_joint,
                                fix_decoder_variance=args.fix_decoder_variance,
                                y_dim=y_dim,
                                init_lengthscale=init_lengthscale,
                                search_device=search_device,
                                jitter=jitter)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

        best_val_rmse, best_val_nll = model.train_gpvae(
            optimizer, args.beta, epochs=args.num_epochs,
            series_batch_size=args.series_batch_size, timestamp_batch_size=args.timestamp_batch_size,
            max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs,
            num_samples=args.num_samples, save_folder=exp_result_folder
        )

    elif args.model_type == 'VNN':
        model = GPVAEMujoco_VNN(args.H,
                                file_path,
                                GP_joint=args.GP_joint,
                                fix_decoder_variance=args.fix_decoder_variance,
                                y_dim=y_dim,
                                init_lengthscale=init_lengthscale,
                                search_device=search_device,
                                jitter=jitter)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

        best_val_rmse, best_val_nll = model.train_gpvae(
            optimizer, args.beta, epochs=args.num_epochs,
            series_batch_size=args.series_batch_size, timestamp_kl_batch_size=args.timestamp_batch_size,
            max_norm=args.max_norm, device=device, print_epochs=args.num_print_epochs,
            num_samples=args.num_samples, save_folder=exp_result_folder
        )

    else:
        raise NotImplementedError(f"Wrong model type: {args.model_type}")
    e = time.time()

    total_training_time = e - s
    print(f"Total Training time: {total_training_time}\n")

    # Testing
    model.load_state_dict(torch.load(f"{exp_result_folder}/best_val_rmse.pt", weights_only=True))
    mean_rmse, mean_nll, all_pred_mean, all_pred_std = model.predict_gpvae(file_path,
                                                                           data_split="test",
                                                                           series_batch_size=args.series_batch_size,
                                                                           timestamp_batch_size=args.timestamp_batch_size,
                                                                           device=device,
                                                                           num_samples=args.num_samples,
                                                                           return_pred=True)

    print(f'Best val rmse: {best_val_rmse}')
    print(f'Test rmse: {mean_rmse}')
    print(f'Test nll: {mean_nll}')
    
    # Save test & model info
    with open(f'{exp_result_folder}/metrices.txt', 'w') as f:
        f.write(f'Test rmse is: {mean_rmse}\n')
        f.write(f'Test nll is: {mean_nll}\n')
        f.write(f'\n')
        f.write(f'Best val rmse is: {best_val_rmse}\n')
        f.write(f'Best val nll is: {best_val_nll}\n')
        f.write(f'Total time is: {total_training_time}\n')
        f.write(f'GP kernel parameters:\n')
        for n, p in model.gp.kernel.named_parameters():
            f.write(f'{n}:{p}\n')
        f.write(f'Model decoder var: {model.decoder.sigma2_y}\n')

    # Save prediction results
    with h5py.File(f'{exp_result_folder}/predictions.h5', 'w') as f:
        f.create_dataset('means', data=all_pred_mean.cpu().numpy())
        f.create_dataset('stds', data=all_pred_std.cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mujoco with Missing Frames")

    parser.add_argument('--T', type=int, default=1000, help="frame length")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--model_type', type=str, default='SWS', choices=['SWS', 'VNN'], help='model used')
    parser.add_argument('--H', type=int, default=10, help='number of nearest neighbors')
    parser.add_argument('--beta', type=float, default=1.0, help='beta in ELBO')
    parser.add_argument('--GP_joint', action='store_true', help='whether to train GP params jointly')
    parser.add_argument('--fix_decoder_variance', action='store_true', help='whether to train decoder var')

    parser.add_argument('--series_batch_size', type=int, default=20, help='batch size for series')
    parser.add_argument('--timestamp_batch_size', type=int, default=64, help='batch size for timestamp')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for sum loss (instead of mean loss)')
    parser.add_argument('--max_norm', type=float, default=None, help='max norm of gradients in grad norm clipping')
    parser.add_argument('--num_epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--num_samples', type=int, default=20, help='number of samples for prediction')
    parser.add_argument('--num_print_epochs', type=int, default=10, help='number of printing epochs')

    args_mujoco = parser.parse_args()

    run_mujoco_missing_frames(args_mujoco)

