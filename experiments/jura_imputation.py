import os
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path
import h5py
import math
import numpy as np
import torch

from models.gpvae_jura import GPVAEJuraImp_SWS, GPVAEJuraImp_VNN


def run_jura(args):
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
    print(f"Time: {current_time}, {args.model_type}-H{args.H}, Using: {device}, seed: {args.seed}, ",
          f"lr={args.lr}, beta={args.beta}, ", "GP joint training, " if args.GP_joint else "",
          f"{torch.get_default_dtype()}\n")

    exp_result_folder = f"./exp/jura/{args.model_type}/H={args.H}/beta={args.beta}/epoch={args.num_epochs}/{current_time}"
    os.makedirs(exp_result_folder, exist_ok=True)

    # Save configs
    with open(f'{exp_result_folder}/configs.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}={value};\n')

    file_path = Path(f'./data/jura')
    assert file_path.exists()

    s = time.time()
    if args.model_type == 'SWS':
        model = GPVAEJuraImp_SWS(H=args.H,
                                GP_joint=args.GP_joint,
                                file_path=file_path,
                                search_device=search_device,
                                sigma2_y=args.sigma2_y,
                                fix_variance=args.fix_variance,
                                jitter=jitter)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

        model.train_gpvae(
            optimizer=optimizer,
            batch_size=args.batch_size,
            epochs=args.num_epochs,
            device=device,
            print_epochs=args.num_print_epochs,
            beta=args.beta,
        )

    elif args.model_type == 'VNN':
        model = GPVAEJuraImp_VNN(H=args.H,
                                GP_joint=args.GP_joint,
                                file_path=file_path,
                                search_device=search_device,
                                sigma2_y=args.sigma2_y,
                                fix_variance=args.fix_variance,
                                jitter=jitter)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

        model.train_gpvae(
            optimizer=optimizer,
            kl_batch_size=args.batch_size,
            epochs=args.num_epochs,
            expected_lk_batch_size=None,
            device=device,
            print_epochs=args.num_print_epochs,
            beta=args.beta,
        )

    else:
        raise NotImplementedError(f"Wrong model type: {args.model_type}")
    e = time.time()

    total_training_time = e - s
    print(f"Total Training time: {total_training_time}\n")

    # Testing
    mse, mae, mean_nll, all_pred_mean, all_pred_std = model.predict_gpvae(batch_size=args.batch_size,
                                                                     stat=model.Ystat,
                                                                     device=device,
                                                                     num_samples=args.num_samples,
                                                                     return_pred=True)

    print(f'Test mse: {mse}')
    print(f'Test rmse: {math.sqrt(mse)}')
    print(f'Test mae: {mae}')
    print(f'Test nll: {mean_nll}')

    # Save test & model info
    with open(f'{exp_result_folder}/metrices.txt', 'w') as f:
        f.write(f'Test mse is: {mse}\n')
        f.write(f'Test rmse is: {math.sqrt(mse)}\n')
        f.write(f'Test mae is: {mae}\n')
        f.write(f'Test nll is: {mean_nll}\n')
        f.write(f'\n')
        f.write(f'Total time is: {total_training_time}\n')
        f.write(f'GP kernel parameters:\n')
        for n, p in model.gp.kernel.named_parameters():
            f.write(f'{n}:{p}\n')

    # Save prediction results
    with h5py.File(f'{exp_result_folder}/predictions.h5', 'w') as f:
        f.create_dataset('means', data=all_pred_mean.cpu().numpy())  # [359, 3]
        f.create_dataset('stds', data=all_pred_std.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jura imputation")

    parser.add_argument("--seed", type=int, default=2, help="Random seed")
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--model_type', type=str, default='VNN', choices=['SWS', 'VNN'], help='model used')
    parser.add_argument('--H', type=int, default=10, help='number of nearest neighbors')
    parser.add_argument('--beta', type=float, default=1., help='beta in ELBO')
    parser.add_argument('--GP_joint', action='store_true', help='whether to train GP params jointly')
    parser.add_argument('--sigma2_y', type=float, default=0.25, help='the scale of the decoder likelihood')
    parser.add_argument('--fix_variance', action='store_true', help='whether to train decoder variance')

    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for sum loss (instead of mean loss)')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--num_samples', type=int, default=50, help='number of samples for prediction')
    parser.add_argument('--num_print_epochs', type=int, default=10, help='number of printing epochs')

    args_jura = parser.parse_args()

    # args_jura.float64 = False
    # args_jura.GP_joint = True

    run_jura(args_jura)

