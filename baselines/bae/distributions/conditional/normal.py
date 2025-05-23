import torch
import torch.nn as nn
from torch.distributions import Normal

from baselines.bae.distributions.conditional.base import ConditionalDistribution
from baselines.bae.utils.tensors import sum_except_batch


class ConditionalMeanNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and fixed std."""

    def __init__(self, net, scale=1.0):
        super(ConditionalMeanNormal, self).__init__(net)
        self.scale = scale
        
    def forward(self, context, **kwargs):
        return self.net(context)

    def cond_dist(self, context):
        mean = self.net(context)
        return Normal(loc=mean, scale=self.scale)

    def log_prob(self, x, context=None, return_sum=True):
        dist = self.cond_dist(context)
        if return_sum:
            return sum_except_batch(dist.log_prob(x))
        else:
            return dist.log_prob(x)
    
    def log_prob_wihout_context(self, x, mean):
        dist = Normal(loc=mean, scale=self.scale)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob, dist

    def mean(self, context):
        return self.cond_dist(context).mean


class ConditionalMeanStdNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and learned std."""

    def __init__(self, net, scale_shape):
        super(ConditionalMeanStdNormal, self).__init__(net)
        self.net = net
        self.log_scale = nn.Parameter(torch.zeros(scale_shape))

    def cond_dist(self, context):
        mean = self.net(context)
        return Normal(loc=mean, scale=self.log_scale.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob, dist

    def mean(self, context):
        return self.cond_dist(context).mean


class ConditionalNormal(ConditionalDistribution):
    """A multivariate Normal with conditional mean and log_std."""

    def __init__(self, net, split_dim=-1):
        super(ConditionalNormal, self).__init__(net)
        self.net = net
        self.split_dim = split_dim

    def forward(self, context):
        return self.net(context)

    def cond_dist(self, context):
        params = self.net(context)
        mean, log_std = torch.chunk(params, chunks=2, dim=self.split_dim)
        return Normal(loc=mean, scale=log_std.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)        
        return z, log_prob, dist

    def mean(self, context):
        return self.cond_dist(context).mean

    def mean_stddev(self, context):
        dist = self.cond_dist(context)
        return dist.mean, dist.stddev


