import torch
import torch.nn as nn
import torch.nn.functional as fn
import re
from numpy import mean, std
from math import *
import numpy as np
import gymnasium as gym

from typing import Any, Tuple

import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using Device: {device}')

tol = 1e-7  # Tolerance.

def count_parameters(model: torch.nn.Module)->int:
    """
    Returns:
        int: the number of trainable parameters present in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




# ------------------------- Gaussian Sampling ---------------------------------

def sample_from_gaussian(mean: torch.Tensor, variance: torch.Tensor)->torch.Tensor:
    """ Currently Untested, Unused, and Undocumented"""
    
    # untested and undocumented, for proposal 3, maybe torch.distributions already has something

    # reparametrization trick for sampling operation to remain differentiable
    std = torch.sqrt(variance)
    epsilon = torch.randn_like(std)
    return mean + variance*epsilon

def sample_from_gaussian_mixture(weights: torch.Tensor, means: torch.Tensor, variances: torch.Tensor)->torch.Tensor:
    """ Currently Untested, Unused, and Undocumented"""

    # untested and undocumented, for proposal 3, maybe torch.distributions already has something

    # reparametrization trick for sampling operation to remain differentiable 
    stds = torch.sqrt(variances)
    epsilons = torch.randn_like(stds)
    weighted_samples = weights.unsqueeze(-1)*(means + variances*epsilons)    # ensure weights is broadcastable
    return weighted_samples.sum(dim=0)



# ------------------------- Gaussian Integrals --------------------------------

# Gaussian Cumulative Distribution Function (Gaussian integral from -∞ to x) using error function (torch.erf)
def gaussian_cdf(x: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """Returns the Gaussian Cumulative Probability P[X <= x | X~N(mean, variance)] using torch.erf

    Args:
        x (torch.Tensor): Upper bound of the integration interval.
        mean (torch.Tensor): Mean of the normal distribution.
        variance (torch.Tensor): Variance of the normal distribution.
    """
    
    std = torch.sqrt(variance[:,:])
    return 0.5 * (1 + torch.erf((x[:,:] - mean[:,:]) / (torch.sqrt(torch.tensor(2.0, dtype=torch.float32, device=device)) * std)))

# Gaussian Integral expressed in terms of Gaussian cdf subtraction
def gaussian_integral(mean: torch.Tensor, variance: torch.Tensor, x_from: torch.Tensor = None, x_to: torch.Tensor = None) -> torch.Tensor:
    """Returns the probability of X being in the interval [x_from, x_to] using torch.erf
    for a closed-form solution that remains differentiable.

    If both x_from and x_to are None, returns 1 (i.e., total probability = 1).\\
    If x_from is None, returns P[X ≤ x_to].\\
    If x_to is None, returns P[X ≥ x_from].
    
    Args:
        mean (torch.Tensor): Mean of the normal distribution.
        variance (torch.Tensor): Variance of the normal distribution.
        x_from (torch.Tensor, optional): Lower bound of the integration interval.
        x_to (torch.Tensor, optional): Upper bound of the integration interval.
    
    Returns:
        torch.Tensor: Probability P[x_from ≤ x ≤ x_to | x ~ N(x | mean, variance)]
    """
    if x_from is None and x_to is None:
        return torch.ones_like(mean)
    
    if x_from is None and x_to is not None:
        return gaussian_cdf(x_to, mean, variance)
    
    if x_from is not None and x_to is None:
        return 1 - gaussian_cdf(x_from, mean, variance)
    
    return (gaussian_cdf(x_to, mean, variance) - gaussian_cdf(x_from, mean, variance))

def gaussian_mixture_integral(means: torch.Tensor, variances: torch.Tensor, weights: torch.Tensor, x_from: torch.Tensor = None, x_to: torch.Tensor = None) -> torch.Tensor:
    
    """Returns the probability of X being in the interval [x_from, x_to] using torch.erf
    for a closed-form solution that remains differentiable.

    If both x_from and x_to are None, returns 1 (i.e., total probability = 1).
    If x_from is None, returns P[X ≤ x_to].
    If x_to is None, returns P[X ≥ x_from].
    
    Args:
        means (torch.Tensor): Means of the normal distribution components.
        variances (torch.Tensor): Variances of the normal distribution components.
        weights (torch.Tensor): Weights of the normal distribution components.
        x_from (torch.Tensor, optional): Lower bound of the integration interval.
        x_to (torch.Tensor, optional): Upper bound of the integration interval.
    
    Returns:
        torch.Tensor: Probability P[x_from ≤ X ≤ x_to | X ~ GMM(means, variances, weights)]
    """

    if x_from is None and x_to is None:
        return torch.ones_like(means)       # [B, E, 1, K]
    
    # x_from and x_to [B, E, N, 1]
    
    gaussian_integrals = None

    if x_from is None and x_to is not None:
        gaussian_integrals = gaussian_cdf(x_to, means, variances)
    
    elif x_from is not None and x_to is None:
        gaussian_integrals =  1 - gaussian_cdf(x_from, means, variances)
    
    else:
        gaussian_integrals = gaussian_cdf(x_to, means, variances) - gaussian_cdf(x_from, means, variances)
        
    return torch.sum(weights * gaussian_integrals, dim=3)   # aggregate by summing along the K dimension 



# see shapes when implementing proposal 3
def intents(means):
    return torch.exp(means)

def confidences(variances):
    return 1/(1+variances)



# ---------------------------------- Profiling ---------------------------------------

def timeit(fn):
    """wrapper for timing functions and logging on console
    """
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"[TIME] {fn.__name__} took {t1-t0:.4f}s")
        return out
    return wrapped


# -------------------------------- Interval Functions --------------------------------
def intervals_trivial(xmin=None, xmax=None, action_space_size=None):
    """tuples from linspace defined by xmin, xmax, steps=action_space_size+1
    """

    rnge = np.linspace(xmin, xmax, action_space_size+1)
    return [(x,y) for x,y in zip(rnge[:-1], rnge[1:])]


resolve_interval_fn = {
    'trivial' : intervals_trivial,
    # currently only trivial function. When I try other experiments I will try other functions
}

def tanh_squash_to_interval(x, x_from, x_to):
    """Similar to torch.clamp but remains differentiable.
    """
    return 0.5*(x_from-x_to)*torch.tanh(x) + (x_from+x_to)*0.5

#-------------------------------- Mapping Functions ----------------------------------

def mapping_trivial(action_space_size=None):
    """returns [0,...,N]
    """
    return list(range(action_space_size))

resolve_mapping_fn = {
    'trivial' : mapping_trivial,
    # currently only trivial mapping, no need for re-ordering
}


class Categorical(torch.distributions.Categorical):
    """Subclass of torch.distributions.Categorical to support noisy sampling
    """
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    def noisy_sample(self, noise_std: float = 0.0) -> torch.Tensor:
        """Sample from a categorical distribution with additive gaussian noise with std = noise_std
        """

        batch_shape = self._batch_shape
        device = self.probs.device                                      # ensure we sample on the same device


        u = torch.rand(batch_shape, device=device)                      # draw sample u ~ Uniform(0,1)

        noise = torch.randn_like(u) * noise_std                         # sample from gaussian with variance=std^2 (torch.normal_like(u, std=std) but it doesnt exist. Same effect)
        
        u_noisy = torch.clamp(u + noise, min=0.0, max=1.0)              # Add noise to the uniform draw and clamp to [0, 1]:
        cdf = torch.cumsum(self.probs, dim=-1)                          # compute cdf for each action
        u_noisy_expanded = u_noisy.unsqueeze(-1)                        # reshape for sampling
        sample = torch.searchsorted(cdf, u_noisy_expanded).squeeze(-1).clamp(max=self.probs.size(-1)-1)  # determine bin corresponding to action (= sample)

        return sample
    

# Wrapper class to discretize the environment's action space
class BoxToDiscreteWrapper(gym.ActionWrapper):

    def __init__(self, env, num_bins):
        super().__init__(env)

        self.n_bins = num_bins
        self.low = env.action_space.low[0]
        self.high = env.action_space.high[0]

        # bin values from linspace (ensures low and high are included) 
        self.actions = np.linspace(self.low, self.high, num_bins).reshape(-1, 1)

        self.action_space = gym.spaces.Discrete(num_bins)   # box -> discrete

    def action(self, action_idx):
        return self.actions[action_idx]
