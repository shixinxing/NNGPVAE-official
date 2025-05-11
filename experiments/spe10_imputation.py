import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np
import torch

from models.gpvae_spe10 import GPVAESPE10_SWS, GPVAESPE10_VNN
from utils.spe10 import plot_spe10


def run_spe10(args):
    # big_tensor = torch.empty((500, 1000), device='cuda')
    # print(f"Allocated GPU memory: {torch.cuda.memory_allocated()} bytes\n")

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
    else:
        device = 'cpu'

    current_time = datetime.now().strftime("%m%d_%H_%M_%S")
    print(f"Time: {current_time}, {args.model_type}-H{args.H}, Using: {device}, seed: {args.seed}, lr={args.lr}, ",
          f"beta={args.beta}, sigma2_y={args.sigma2_y}, kernel_type={args.kernel_type}, l={args.kernel_lengthscale}",
          " GP joint training, " if args.GP_joint else "",
          f"{torch.get_default_dtype()}\n")

    file_path = Path("../data/SPE10/spe10.npz")

    s = time.time()
    if args.model_type == 'SWS':
        model = GPVAESPE10_SWS(
            file_path, H=args.H, GP_joint=args.GP_joint, search_device=device,
            sigma2_y=args.sigma2_y, fix_variance=True,
            kernel_type=args.kernel_type, lengthscale=args.kernel_lengthscale, jitter=jitter
        )
        # del big_tensor
        # torch.cuda.empty_cache()

        print(f"GPU memory after release: {torch.cuda.memory_allocated()} bytes")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        model.train_gpvae(
            optimizer, args.beta, args.num_epochs, args.batch_size, max_norm=args.max_norm,
            device=device, print_epochs=args.num_print_epochs
        )
    elif args.model_type == 'VNN':
        model = GPVAESPE10_VNN(
            file_path, H=args.H, GP_joint=args.GP_joint, search_device=device,
            sigma2_y=args.sigma2_y, fix_variance=True,
            kernel_type=args.kernel_type, lengthscale=args.kernel_lengthscale, jitter=jitter
        )
        # del big_tensor
        # torch.cuda.empty_cache()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        model.train_gpvae(
            optimizer, args.beta, args.num_epochs, args.batch_size, expected_lk_batch_size=None, max_norm=args.max_norm,
            device=device, print_epochs=args.num_print_epochs
        )
    else:
        raise NotImplementedError(f"Wrong model type: {args.model_type}")

    print(f"\nTotal Training time: {time.time() - s}")
    print(f"Total parameters: {sum([p.shape.numel() for p in model.parameters() if p.requires_grad])}\n")

    # Testing
    mse, mae, nll, pred_mean, pred_std = model.predict_gpvae(
        args.batch_size,
        stat=None,  # {'mean': np.load(file_path)['Y_mean'], 'std': np.load(file_path)['Y_std']},
        device=device, num_samples=20, return_pred=True
    )
    print(f'\nnll: {nll}, mse: {mse}, mae: {mae}.\n')

    # Plot
    data_dict_true = np.load(file_path)
    fig, ax = plot_spe10(data_dict_true, pred_mean, layer_idx=20, nll=nll, mse=mse, y_dim_idx=0)

    experiment_dir = Path(f'./exp_spe10_{args.model_type}_H{args.H}')
    if args.save:
        experiment_dir.mkdir(exist_ok=True)
        trial_dir = experiment_dir / f'{current_time}'
        trial_dir.mkdir(parents=False, exist_ok=True)

        torch.save(model.state_dict(), trial_dir / f'model_spe10.pth')  # model

        img_path = trial_dir / f'img_spe10.pdf'  # images
        fig.savefig(img_path, bbox_inches='tight')

        everything_for_imgs = {
            # 'y_test_full': y_test_full, 'y_test_miss': y_test_miss, # to save storage
            'y_rec_mean': pred_mean, 'y_rec_std': pred_std,
            'NLL': nll, 'MSE': mse, 'MAE': mae
        }
        imgs_pickle_path = trial_dir / 'everything_for_imgs.pkl'
        with imgs_pickle_path.open('wb') as f:
            pickle.dump(everything_for_imgs, f)

        print("The experiment results are saved at:")
        print(f"{trial_dir.resolve()}")  # get absolute path for moving the log file
    else:
        print("The experimental results are not saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPE10 imputation")

    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--model_type', type=str, default='VNN', choices=['SWS', 'VNN'], help='model used')
    parser.add_argument('--H', type=int, default=20, help='number of nearest neighbors')
    parser.add_argument('--beta', type=float, default=1., help='beta in ELBO')
    parser.add_argument('--GP_joint', action='store_true', help='whether to train GP params jointly')
    # ⚠️
    parser.add_argument('--sigma2_y', type=float, default=1., help='the scale of the decoder likelihood')
    parser.add_argument('--kernel_type', type=str, default='rbf', help='kernel type')
    parser.add_argument('--kernel_lengthscale', type=float, default=2., help='kernel lengthscale')
    parser.add_argument('--max_norm', type=float, default=1e4, help='max norm of gradients in grad norm clipping')

    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for sum loss (instead of mean loss)')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')

    parser.add_argument('--save', action='store_true', help='store everything into a folder')

    args_spe10 = parser.parse_args()

    run_spe10(args_spe10)
