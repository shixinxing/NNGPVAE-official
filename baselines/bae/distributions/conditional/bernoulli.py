from torch.distributions import Bernoulli
from baselines.bae.distributions.conditional.base import ConditionalDistribution
from baselines.bae.utils.tensors import sum_except_batch


class ConditionalBernoulli(ConditionalDistribution):
    """A Bernoulli distribution with conditional logits."""

    def __init__(self, net):
        super(ConditionalBernoulli, self).__init__(net)

    def forward(self, context):
        probs = self.net(context)
        return probs

    def cond_dist(self, context=None):
        probs = self.net(context)
        return Bernoulli(probs=probs)
        # return ContinuousBernoulli(probs=probs)

    def log_prob(self, x, context):
        dist = self.cond_dist(context=context)
        return sum_except_batch(dist.log_prob(x))

    def log_prob_wihout_context(self, x, probs):
        dist = Bernoulli(probs=probs)
        # dist = ContinuousBernoulli(probs=probs)
        return sum_except_batch(dist.log_prob(x))

    def logits(self, context):
        return self.cond_dist(context=context).logits
    
    def probs(self, context):
        return self.cond_dist(context=context).probs

    def mean(self, context):
        return self.cond_dist(context=context).mean
