# Stochastic Variational GPR as a baseline on SPE10

from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ZeroMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO

from utils.spe10 import define_kernel


class SubSVGP(ApproximateGP):
    def __init__(
            self, M, train_X, train_Y, test_X, test_Y,
            kernel_type='rbf', lengthscale=2., noise_var=0.01
    ):
        inducing_points = torch.rand([M, train_X.size(-1)])
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=M)
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SubSVGP, self).__init__(variational_strategy)
        self.mean_module = ZeroMean()
        self.covar_module = define_kernel(kernel_type, latent_dims=None, x_dims=train_X.size(-1))
        self.covar_module.outputscale, self.covar_module.base_kernel.lengthscale = 1., lengthscale
        if kernel_type == 'cauchy':
            self.covar_module.base_kernel.alpha = 1.
            self.covar_module.base_kernel.raw_alpha.requires_grad_(False)

        self.likelihood = GaussianLikelihood()
        self.likelihood.noise = noise_var

        self.train_dataset = TensorDataset(train_X, train_Y)
        self.test_dataset = TensorDataset(test_X, test_Y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def train_gp(
            self, optimizer: torch.optim.Optimizer, epochs: int, batch_size: int,
            device='cpu', print_epochs=1
    ):
        self.to(device)
        self.train()
        mll = VariationalELBO(self.likelihood, self, num_data=len(self.train_dataset))

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = self(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item(): .6f}\n')

    @torch.no_grad()
    def predict_gp(self, batch_size: int, masks: Tensor, device='cpu'):  # masks:[N]
        self.to(device)
        self.eval()

        # prediction on training points for plotting
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
        pred_mean_train, pred_std_train = [], []
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            preds = self(x_train)
            pred_mean_train.append(preds.mean)
            pred_std_train.append(preds.variance.sqrt())
        pred_mean_train = torch.cat(pred_mean_train, dim=-1)
        pred_std_train = torch.cat(pred_std_train, dim=-1)

        # prediction on test points
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        pred_mean_test, pred_std_test, nlls, ses, aes = [], [], [], [], []
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            preds = self(x_test)
            pred_mean_test.append(preds.mean)
            pred_std_test.append(preds.variance.sqrt())

            # nll
            normal_diag = torch.distributions.Normal(loc=preds.mean, scale=preds.variance.sqrt())
            log_p = normal_diag.log_prob(y_test)
            nlls.append(-log_p)

            # se/ae
            se, ae = (preds.mean - y_test).square(), (preds.mean - y_test).abs()
            ses.append(se)
            aes.append(ae)
        pred_mean_test = torch.cat(pred_mean_test, dim=-1)
        pred_std_test = torch.cat(pred_std_test, dim=-1)
        nll_sum = torch.cat(nlls, dim=-1).sum()
        se_sum = torch.cat(ses, dim=-1).sum()
        ae_sum = torch.cat(aes, dim=-1).sum()

        # re-organize prediction for future plotting
        pred_mean = torch.empty([len(self.train_dataset) + len(self.test_dataset)], device=device)
        pred_std = torch.empty_like(pred_mean)
        masks = masks.to(device=device, dtype=torch.bool)
        pred_mean[masks], pred_std[masks] = pred_mean_train, pred_std_train
        pred_mean[~masks], pred_std[~masks] = pred_mean_test, pred_std_test

        return se_sum, ae_sum, nll_sum, pred_mean, pred_std


class SVGPIndeptSPE10(nn.Module):
    def __init__(self, file_path: Path, M: int, kernel_type='rbf', lengthscale=2., noise_var=1.):
        super(SVGPIndeptSPE10, self).__init__()
        self.M = M

        data_dict = np.load(file_path)
        m = torch.as_tensor(data_dict['masks'], dtype=torch.bool)                       # [N,4]
        X = torch.as_tensor(data_dict["X"], dtype=torch.get_default_dtype())            # [N,3]
        Y_full = torch.as_tensor(data_dict["Y_full"], dtype=torch.get_default_dtype())  # [N,4]

        self.sub_models = nn.ModuleList([])
        num_sub_models = Y_full.size(-1)
        for i in range(num_sub_models):
            m_sub, Y_sub = m[:, i], Y_full[:, i]
            train_X, train_Y, test_X, test_Y = X[m_sub], Y_sub[m_sub], X[~m_sub], Y_sub[~m_sub]
            sub_model = SubSVGP(M, train_X, train_Y, test_X, test_Y, kernel_type, lengthscale, noise_var=noise_var)
            self.sub_models.append(sub_model)

    def train_gp(self, lr, epochs, batch_size, device='cpu', print_epochs=1):
        for (i, sub_model) in enumerate(self.sub_models):
            print(f'\nTraining sub_model {i} on {device}...')
            optimizer = torch.optim.Adam(sub_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
            sub_model.train_gp(
                optimizer, epochs, batch_size, device=device, print_epochs=print_epochs
            )

    @torch.no_grad()
    def predict_gp(self, batch_size: int, masks: Tensor, device='cpu'):  # masks: [N,4]
        se_models, ae_models, nll_models = 0., 0., 0.
        pred_means, pred_stds = [], []  # collect 4 models
        for (i, sub_model) in enumerate(self.sub_models):
            print(f'\nPredicting sub_model {i} on {device}...')
            se, ae, nll, pred_mean, pred_std = sub_model.predict_gp(batch_size, masks[..., i], device=device)
            se_models += se
            ae_models += ae
            nll_models += nll
            pred_means.append(pred_mean)
            pred_stds.append(pred_std)
        se_mean, ae_mean, nll_mean = se_models / (~masks).sum(), ae_models / (~masks).sum(), nll_models / (~masks).sum()
        pred_means = torch.stack(pred_means, dim=-1)   # [N,4]
        pred_stds = torch.stack(pred_stds, dim=-1)

        return se_mean.item(), ae_mean.item(), nll_mean.item(), pred_means.cpu().numpy(), pred_stds.cpu().numpy()


if __name__ == '__main__':
    from datetime import datetime
    import time
    import argparse
    import pickle
    from utils.spe10 import plot_spe10

    def run_svgp(args):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float64) if args.float64 else torch.set_default_dtype(torch.float32)
        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.manual_seed(args.seed)
        else:
            device = 'cpu'

        current_time = datetime.now().strftime("%m%d_%H_%M_%S")
        print(f"{current_time}, Independent SVGP-M{args.M}, {device}, seed: {args.seed}, lr={args.lr}, "
              f"likelihood noise={args.sigma2_y}, kernel_type:{args.kernel_type}, l={args.kernel_lengthscale} "
              f"{torch.get_default_dtype()}\n")

        file_path = Path("../data/SPE10/spe10.npz")

        s = time.time()
        model = SVGPIndeptSPE10(
            file_path, M=args.M,
            kernel_type=args.kernel_type, lengthscale=args.kernel_lengthscale, noise_var=args.sigma2_y
        )
        # Training
        model.train_gp(
            lr=args.lr, epochs=args.num_epochs, batch_size=args.batch_size,
            device=device, print_epochs=args.num_print_epochs
        )
        print(f"\nTotal training time: {time.time() - s}")
        print(f'Total params: {sum([p.numel() for p in model.parameters() if p.requires_grad])}\n')

        # Testing
        masks = torch.as_tensor(np.load(file_path)['masks'], dtype=torch.bool)
        se_mean, ae_mean, nll_mean, pred_means, pred_stds = model.predict_gp(args.batch_size, masks, device=device)
        print(f'\nnll: {nll_mean}, mse: {se_mean}, mae: {ae_mean}.\n')

        # Plot
        fig, ax = plot_spe10(np.load(file_path), pred_means, layer_idx=20, nll=nll_mean, mse=se_mean)

        experiment_dir = Path(f'./exp_spe10_IndptSVGP_M{args.M}')
        if args.save:
            experiment_dir.mkdir(exist_ok=True)
            trial_dir = experiment_dir / f'{current_time}'
            trial_dir.mkdir(exist_ok=True, parents=False)

            torch.save(model.state_dict(), trial_dir / 'svgp_spe10.pth')

            img_path = trial_dir / f'img_gp_spe10.pdf'
            fig.savefig(img_path, bbox_inches='tight')
            everything_for_imgs = {
                'y_rec_mean': pred_means, 'y_rec_std': pred_stds,
                'NLL': nll_mean, 'MSE': se_mean, 'MAE': ae_mean
            }
            imgs_pickle_path = trial_dir / f'everything_for_imgs.pkl'
            with imgs_pickle_path.open('wb') as f:
                pickle.dump(everything_for_imgs, f)
            print("The experiment results are saved at:")
            print(f"{trial_dir.resolve()}")
        else:
            print("The experimental results are not saved.")

    parser = argparse.ArgumentParser(description="Independent SVGP for SPE10 Imputation")

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--M', type=int, default=50, help='number of inducing points each sub model')
    parser.add_argument('--sigma2_y', type=float, default=0.01, help='likelihood noise variance')
    parser.add_argument('--kernel_type', type=str, default='cauchy', help='kernel type')
    parser.add_argument('--kernel_lengthscale', type=float, default=2., help='kernel lengthscale')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='epochs for printing')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')

    parser.add_argument('--save', action='store_true', help='whether to save model')

    args_spe10 = parser.parse_args()
    run_svgp(args_spe10)










