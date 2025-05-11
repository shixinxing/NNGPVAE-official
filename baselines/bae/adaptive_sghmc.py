# copy from https://github.com/automl/pybnn/blob/master/pybnn/sampler/adaptive_sghmc.py#L6

import torch
from torch.optim import Optimizer


class AdaptiveSGHMC(Optimizer):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
        procedure to adapt its own hyperparameters during the initial stages
        of sampling.

        See [1] for more details on this burn-in procedure.
        See [2] for more details on Stochastic Gradient Hamiltonian Monte-Carlo.

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            In Advances in Neural Information Processing Systems 29 (2016).
            `Bayesian Optimization with Robust Bayesian Neural Networks.`
        [2] T. Chen, E. B. Fox, C. Guestrin
            In Proceedings of Machine Learning Research 32 (2014).\n
            `Stochastic Gradient Hamiltonian Monte Carlo`
        """

    def __init__(
            self, params,
            lr: float = 1e-2, num_burn_in_steps: int = 3000, mdecay: float = 0.05, scale_grad: float = 1., epsilon=1e-16
    ):
        """ Set up an Adaptive SGHMC Optimizer, as in eq(17) of the paper.

        Args:
            params: iterable, parameters serving as optimization variable.
            lr: float, base learning rate for this optimizer.
                Must be tuned to the specific function being minimized.
            num_burn_in_steps: int, number of burn-in steps to perform.
                In each burn-in step, this sampler will adapt its own internal parameters to decrease its error.
                Set to `0` to turn scale adaption off.
            mdecay:constant, momentum decay per time-step. (i.e., \alpha in the paper)
            scale_grad: float, optional
                Value that is used to scale the magnitude of the epsilon used during sampling. In a typical
                batches-of-data setting this usually corresponds to the number of examples in the entire dataset.
            epsilon: float, per-parameter epsilon level.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(
                num_burn_in_steps))

        defaults = dict(
            lr=lr, scale_grad=float(scale_grad),
            num_burn_in_steps=num_burn_in_steps, mdecay=mdecay, epsilon=epsilon
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:  # for params with requires_grads=False
                    continue

                state = self.state[parameter]

                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(parameter)
                    state["g"] = torch.ones_like(parameter)
                    state["v_hat"] = torch.ones_like(parameter)
                    state["momentum"] = torch.zeros_like(parameter)
                state["iteration"] += 1

                mdecay, epsilon, lr = group["mdecay"], group["epsilon"], group["lr"]
                scale_grad = torch.tensor(group["scale_grad"], dtype=parameter.dtype)
                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]

                momentum = state["momentum"]
                # if log likelihood divided by N_data
                # as stated in https://github.com/automl/pybnn/blob/master/pybnn/bohamiann.py#L267
                gradient = parameter.grad.data * scale_grad

                tau_inv = 1. / (tau + 1.)

                # Update parameters during burn-in
                if state["iteration"] <= group["num_burn_in_steps"]:
                    # Specifies the moving average window, see eq(15) left of the paper
                    tau.add_(- (g ** 2) / (v_hat + epsilon) * tau + 1)

                    # Average gradient, see eq(15) right of the paper
                    g.add_(- tau_inv * g + tau_inv * gradient)

                    # Gradient variance see eq(14) of the paper
                    v_hat.add_(- tau_inv * v_hat + tau_inv * (gradient ** 2))

                if state["iteration"] <= group["num_burn_in_steps"]:
                    # specifies the moving average window, see Eq 9 in [1] left
                    tau.add_(- tau * (g * g / (v_hat + epsilon)) + 1)

                    # average gradient see Eq 9 in [1] right
                    g.add_(- g * tau_inv + tau_inv * gradient)

                    # gradient variance see Eq 8 in [1]
                    v_hat.add_(- v_hat * tau_inv + tau_inv * (gradient ** 2))

                # Pre-conditioner
                minv_t = 1. / (torch.sqrt(v_hat) + epsilon)

                epsilon_var = (2. * (lr ** 2) * mdecay * minv_t - (lr ** 4))

                # Sample random epsilon
                sigma = torch.sqrt(torch.clamp(epsilon_var, min=1e-16))
                sample_t = torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient) * sigma)

                # Update momentum (Eq 10 right in [1])
                momentum.add_(
                    - (lr ** 2) * minv_t * gradient - mdecay * momentum + sample_t
                )

                # Update theta (Eq 10 left in [1])
                parameter.data.add_(momentum)

        return loss

