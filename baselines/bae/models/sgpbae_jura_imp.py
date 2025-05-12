import torch
from torch import nn
from torch.utils.data import DataLoader
from gpytorch.kernels import RBFKernel, ScaleKernel

from models.building_blocks.enc_dec_jura import MLP
from utils.build_datasets import NNDataset
from baselines.bae.models.bsgp import BSGP
from baselines.bae.gp.likelihoods import Gaussian
from baselines.bae.priors.fixed_priors import PriorGaussian
from baselines.bae.distributions.conditional.normal import ConditionalMeanNormal
from baselines.bae.models.sgpbae_spatial_imp import SGPBAESpatialImp


class SGPBAEJuraImp(SGPBAESpatialImp):
    """
    Jura, follows https://github.com/tranbahien/sgp-bae/blob/master/dsgpbae_experiment.py
    """
    def __init__(self, train_dataset: NNDataset, init_inducing_points, GP_joint, IP_joint,
                 sample_dir='./exp/bae', M = 128, sigma_y = 0.5, decoder_var = 1.
    ):
        # fixed hyper-parameters for Jura dataset
        self.train_dataset = train_dataset

        x_dim, y_dim, latent_dim = 2, 3, 2
        N_train = 359
        encoder = MLP(input_size=y_dim * 2, output_size=latent_dim, hidden_units=[20])
        decoder_net = MLP(input_size=latent_dim, output_size=y_dim, hidden_units=[5, 5])
        decoder = ConditionalMeanNormal(decoder_net, scale=sigma_y)
        decoder_prior = PriorGaussian(decoder_var)

        scaled_kernel = ScaleKernel(RBFKernel(ard_num_dims=x_dim, batch_shape=torch.Size([1])))  # batch_shape=torch.Size([latent_dim])
        scaled_kernel.base_kernel.lengthscale = 0.1
        scaled_kernel.base_kernel.raw_lengthscale.requires_grad_(GP_joint)
        scaled_kernel.outputscale = 1.
        scaled_kernel.raw_outputscale.requires_grad_(GP_joint)

        gp_likelihood = Gaussian(variance=0.00001)
        # for param in gp_likelihood.parameters():
        #     param.requires_grad_(False)

        # shared inducing locations  [M, 2]
        inducing_points = nn.Parameter(
            init_inducing_points, requires_grad=IP_joint
        )
        inducing_values = nn.Parameter(torch.zeros(latent_dim, M), requires_grad=True)

        bsgp = BSGP(
            latent_dim, scaled_kernel, gp_likelihood, inducing_points, inducing_values,
            N_train=N_train, prior_S_type='uniform',
            prior_lengthscale=2., prior_variance=0.05, prior_lik_var=0.05
        )

        super(SGPBAEJuraImp, self).__init__(encoder=encoder, decoder=decoder, decoder_prior=decoder_prior,
                                            bsgp=bsgp, sample_dir=sample_dir)


    @torch.no_grad()
    def predict_sgpbae(
            self, stat, batch_size=30, device='cpu', return_pred=False,
    ):
        """
        Don't provide test_dataset, as in Jura, the test samples are the missing elements in train_dataset.
        """
        self.to(device)
        test_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=False
        )

        all_ae, all_se, all_nll, all_pred_mean, all_pred_std = [], [], [], [], []

        for data in test_dataloader:
            y_miss_b, x_b, m_b, y_full_b = data  # NOTE: y_miss_b are normalized, but y_full_b are not.
            r_b = self.predict_y(y_miss_b, m_b, y_full_b, stat=stat)

            all_ae.append(r_b['ae'])
            all_se.append(r_b['se'])  # 1d
            all_nll.append(r_b['nll'])
            all_pred_mean.append(r_b['pred_mean'])  # [b,...]
            all_pred_std.append(r_b['pred_std'])

        all_ae = torch.cat(all_ae, dim=0)
        all_se = torch.cat(all_se, dim=0)  # 1d
        all_nll = torch.cat(all_nll, dim=0)  # 1d

        assert all_ae.size(0) == all_se.size(0) == all_nll.size(0) == 100, "Please follows the same setting as FBAESGP."

        if not return_pred:
            return all_se.mean().item(), all_ae.mean().item(), all_nll.mean().item()
        else:
            all_pred_mean = torch.cat(all_pred_mean, dim=0)  # [N, ...]
            all_pred_std = torch.cat(all_pred_std, dim=0)
            return all_se.mean().item(), all_ae.mean().item(), all_nll.mean().item(), all_pred_mean, all_pred_std


