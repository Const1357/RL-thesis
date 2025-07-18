import torch
import torch.nn as nn
import torch.nn.functional as fn
import re
from numpy import mean, std
from math import *
import numpy as np
import ale_py  # this registers ALE environments internally, DO NOT DELETE.
import gymnasium as gym

from typing import Any, Tuple

import time

import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using Device: {device}')

tol = 1e-6  # Tolerance.

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
def gaussian_cdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Returns the Gaussian Cumulative Probability P[X <= x | X~N(mean, variance)] using torch.erf

    Args:
        x (torch.Tensor): Upper bound of the integration interval.
        mean (torch.Tensor): Mean of the normal distribution.
        std (torch.Tensor): Standard Deviation of the normal distribution.
    """
    # x has shape [B, N, 1]
    # mean and std have shapes [B, 1, K]

    return 0.5 * (1 + torch.erf((x[:] - mean[:]) / (1.4142135623730951 * (std.clamp_min_(tol))))) # 1.4142135623730951 = sqrt(2)

# Gaussian Integral expressed in terms of Gaussian cdf subtraction
def gaussian_integral(mean: torch.Tensor, std: torch.Tensor, x_from: torch.Tensor = None, x_to: torch.Tensor = None) -> torch.Tensor:
    """Returns the probability of X being in the interval [x_from, x_to] using torch.erf
    for a closed-form solution that remains differentiable.

    If both x_from and x_to are None, returns 1 (i.e., total probability = 1).\\
    If x_from is None, returns P[X ≤ x_to].\\
    If x_to is None, returns P[X ≥ x_from].
    
    Args:
        mean (torch.Tensor): Mean of the normal distribution.
        std (torch.Tensor): Standard Deviation of the normal distribution.
        x_from (torch.Tensor, optional): Lower bound of the integration interval.
        x_to (torch.Tensor, optional): Upper bound of the integration interval.
    
    Returns:
        torch.Tensor: Probability P[x_from ≤ x ≤ x_to | x ~ N(x | mean, variance)]
    """

    B, _, _ = mean.shape


    if x_from is None and x_to is None:
        return torch.ones((B, 1, 1), device=device)
    
    elif x_from is None and x_to is not None:
        ret =  gaussian_cdf(x_to, mean, std)
    
    elif x_from is not None and x_to is None:
        ret =  1 - gaussian_cdf(x_from, mean, std)
        
    else:
        ret = (gaussian_cdf(x_to, mean, std) - gaussian_cdf(x_from, mean, std))

    ret.clamp_min_(tol)
    return ret

def gaussian_mixture_integral(means: torch.Tensor, stds: torch.Tensor, weights: torch.Tensor, x_from: torch.Tensor = None, x_to: torch.Tensor = None) -> torch.Tensor:
    
    """Returns the probability of X being in the interval [x_from, x_to] using torch.erf
    for a closed-form solution that remains differentiable.

    If both x_from and x_to are None, returns 1 (i.e., total probability = 1).
    If x_from is None, returns P[X ≤ x_to].
    If x_to is None, returns P[X ≥ x_from].
    
    Args:
        means (torch.Tensor): Means of the normal distribution components.
        stds (torch.Tensor): Standard Deviations of the normal distribution components.
        weights (torch.Tensor): Weights of the normal distribution components.
        x_from (torch.Tensor, optional): Lower bound of the integration interval.
        x_to (torch.Tensor, optional): Upper bound of the integration interval.
    
    Returns:
        torch.Tensor: Probability P[x_from ≤ X ≤ x_to | X ~ GMM(means, variances, weights)]
    """

    B, _, K = means.shape

    if x_from is None and x_to is None:
        return torch.ones((B,1,1), device=device)       # [B, 1, 1] is broadcastable to [B, N, 1]. N is unknown if x_from and x_to are None
    
    # x_from and x_to [B, N, 1]
    
    elif x_from is None and x_to is not None:
        gaussian_integrals = gaussian_cdf(x_to, means, stds)
    
    elif x_from is not None and x_to is None:
        gaussian_integrals =  1 - gaussian_cdf(x_from, means, stds)
    
    else:
        gaussian_integrals = gaussian_cdf(x_to, means, stds) - gaussian_cdf(x_from, means, stds)
        
    ret = torch.sum(weights * gaussian_integrals, dim=-1, keepdim=True)   # weighted sum along the K dimension. Keep 1 as K-th dimension for consistency after. 
    ret.clamp_min_(tol)
    return ret



# see shapes when implementing proposal 3
def intents(means: torch.Tensor) -> torch.Tensor:
    return torch.exp(means.clamp(min=-30.0, max=30.0))

def confidences(variances: torch.Tensor) -> torch.Tensor:
    return 1/(1+variances.clamp_min(tol))

def spread(cluster: torch.Tensor, target: torch.Tensor):

    mask = (cluster != -1.0)                    # [B, N]
    diffs = (cluster - target.unsqueeze(-1))    # [B, N]
    sq_diffs = (diffs ** 2) * mask              # zero out padded values

    spread = sq_diffs.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)           # [B]
    norm = sq_diffs.masked_fill(~mask, float('-inf')).max(dim=-1).values    # [B]
    norm = torch.clamp(norm, min=tol)
    return spread / norm

def ic_penalty(I: torch.Tensor, C: torch.Tensor, a: float, b: float) -> torch.Tensor:

    x = (a * I).clamp(min=-30.0, max=30.0)              # Safe input for expm1
    scale = (1 - C)                                     # C in [0,1]

    penalty = torch.expm1(x) * scale * torch.exp(b * scale)
    return penalty


def sigmoid_bound(x: torch.Tensor, M: float) -> torch.Tensor:
    """
    Bounding transform T(x) = M * (1 - exp(-x / M)), preserves monotonicity and curvature.  
    """
    return M * (1 - torch.exp(-x / M))


def loss_penalty(I: torch.Tensor, C: torch.Tensor, a: float, b: float, M: float) -> torch.Tensor:
    penalty = ic_penalty(I, C, a, b)            # shape [B, N]
    bounded = sigmoid_bound(penalty, M)         # shape [B, N]
    # print('[IC PENALTY]', bounded.mean(dim=-1).mean().item())
    return bounded.mean(dim=-1)                 # [B]

def margin_loss(I:torch.Tensor) -> Tuple[torch.Tensor]:

    B, N = I.shape

    top2, _ = I.topk(2, dim=-1)                                                     # [B, 2]   
    I_max = top2[:, 0:1]                                                            # [B, 1]
    I_2nd = top2[:, 1:2]                                                            # [B, 1]
    I_min = I.min(dim=-1, keepdim=True).values                                      # [B, 1]

    norm_factor = (I_max - I_min).pow(2) + tol                                      # [B, 1]

    # include I_max in the distances because it contributes 0 to the sum, but divide with N-1 for mean => ignoring it.
    norm_dists = (I_max - I).pow(2).sum(dim=-1) / ((N-1)*norm_factor.squeeze(-1))   # [B] in [0,1]

    margin_separation_bonus = ((I_max-I_2nd).pow(2) / norm_factor).squeeze(-1)      # [B] in [0,1]

    loss = -(norm_dists + margin_separation_bonus)                                  # [B] in [-2, 0]

    # print('[MARGIN LOSS]', loss.mean().item())

    return -torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)
    # objective is the negative L2 distance (normalized) from the highest intent to each other intent
    # encouraging clear separation of highest intent.
    # a bonus is added if the second highest point is farther away to encourage even more separation of highest and rest.
    



    # old implementation (TODO: should probably delete if ^ works)

    # prefer clusters that are far apart to each other
    # penalize scattered clusters (among each cluster). Spread across high cluster is more important to be minimized so it takes significance 0.8 over 0.2(L)

    I_max = I.max(dim=-1, keepdim=True).values
    I_min = I.min(dim=-1, keepdim=True).values
    radius = (I_max - I_min) / 2

    high_mask = (I_max - I) < radius
    low_mask = ~high_mask

    H = I.masked_fill(low_mask, -1.0)      # [B, N] with padded nans for indices not belonging to the cluster
    L = I.masked_fill(high_mask, -1.0)     # [B, N] with padded nans for indices not belonging to the cluster

    Hmin = H.masked_fill(low_mask, float('inf')).min(dim=-1).values       # [B] at least one non-nan in each cluster at N-dim
    Lmax = L.masked_fill(high_mask, float('-inf')).max(dim=-1).values     # [B] at least one non-nan in each cluster at N-dim

    Hmin = torch.clamp(Hmin, min=-1e6, max=1e6)
    Lmax = torch.clamp(Lmax, min=-1e6, max=1e6)

    # Spread of Clusters
    spread_H = spread(H, Hmin)                  # [B]
    spread_L = spread(L, Lmax)                  # [B]

    # Margin term: maximize separation between margin points in clusters
    denom = Hmin + Lmax + tol
    denom = torch.where(denom.abs() < tol, torch.ones_like(denom) * tol, denom) # distance from 0 is at least tol

    Lmargin = - (Hmin - Lmax)/denom                 # [B] in [-1, 0]
    Lspread = (0.8*spread_H + 0.2*spread_L)         # [B] in [0, 1]
    L_margin_spread = (Lmargin + Lspread)           # [B] in [-1, 1]
    L_margin_spread = (L_margin_spread + 1) / 2     # [B] in [0, 1]

    # sanitization
    return torch.nan_to_num(L_margin_spread, nan=0.0, posinf=1.0, neginf=-1.0)

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


# -------------------- Wrappers -----------------------------

class Categorical(torch.distributions.Categorical):
    """Subclass of torch.distributions.Categorical to support noisy sampling (assumming ordered actions)
    """
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    def noisy_sample(self, noise_std: float = 0.0) -> torch.Tensor:
        """Sample from a categorical distribution with additive gaussian noise with std = noise_std
        """

        batch_shape = self._batch_shape
        device = self.probs.device                          # ensure we sample on the same device


        u = torch.rand(batch_shape, device=device)          # draw sample u ~ Uniform(0,1)

        noise = torch.randn_like(u) * noise_std             # sample from gaussian with variance=std^2 (torch.normal_like(u, std=std) but it doesnt exist. Same effect)
        
        u_noisy = torch.clamp(u + noise, min=0.0, max=1.0)  # Add noise to the uniform draw and clamp to [0, 1]:
        cdf = torch.cumsum(self.probs, dim=-1)              # compute cdf for each action
        u_noisy_expanded = u_noisy.unsqueeze(-1)            # reshape for sampling
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
    

class ClipRewardEnv(gym.RewardWrapper):
    # Cleanrl implementation: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/atari_wrappers.py
    """
    Clip the reward to {+1, 0, -1} by its sign.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(float(reward))


# Atari environments always start from a fixed loaded-from-ROM state. Diversity in initial state is given by performing an amount of no-op actions.
# To begin the game, a 'FIRE' action must be performed. It is done so, after the noops.

class NoopFireResetEnv(gym.Wrapper):

    # Code inspiration: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/atari_wrappers.py

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        self.needs_fire = "FIRE" in env.unwrapped.get_action_meanings()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Perform random number of NOOPs first
        noops = self.env.np_random.integers(0, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

        # Then FIRE if required
        if self.needs_fire:
            obs, _, terminated, truncated, _ = self.env.step(1)  # FIRE
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
            obs, _, terminated, truncated, _ = self.env.step(2)  # usually UP or second FIRE
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

        return obs, info
