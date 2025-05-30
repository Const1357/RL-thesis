from Utilities import *


class TrajectoryBuffer:
    """
    A pre-allocated buffer for vectorized rollouts over multiple parallel environments.
    Buffers transitions in fixed-size GPU tensors and computes
    advantages (GAE) and returns for PPO updates.
    """
    def __init__(self, rollout_steps: int, num_envs: int, obs_dim: int):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = device
        self.reset()

    def reset(self):
        """Clears buffer and allocates fresh tensors on device."""
        self.ptr = 0  # insertion pointer
        # pre-allocate buffers
        self.observations  = torch.empty((self.rollout_steps, self.num_envs, self.obs_dim), device=self.device)
        self.actions  = torch.empty((self.rollout_steps, self.num_envs), dtype=torch.long, device=self.device)
        self.log_probs = torch.empty((self.rollout_steps, self.num_envs), device=self.device)
        self.values  = torch.empty((self.rollout_steps, self.num_envs), device=self.device)
        self.rewards  = torch.empty((self.rollout_steps, self.num_envs), device=self.device)
        self.dones = torch.empty((self.rollout_steps, self.num_envs), device=self.device)
        # filled after calling compute_returns_and_GAE
        self.advantages = None
        self.returns = None

    def add_batch(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
    ):
        """
        Insert one step of data for all environments at the current pointer.
        Each tensor must have shape [E(num_envs), ...].
        """
        t = self.ptr
        self.observations[t].copy_(observations)
        self.actions[t].copy_(actions)
        self.log_probs[t].copy_(log_probs)
        self.values[t].copy_(values)
        self.rewards[t].copy_(rewards)
        self.dones[t].copy_(dones)
        self.ptr += 1

    def compute_returns_and_GAE(self, last_values, gamma=0.99, lam=0.97):
        """
        Compute GAE advantages and discounted returns.

        Args:
            last_values (torch.Tensor): Bootstrap Value V(s_T) for each env, shape [E] or [E, 1].
            gamma (float): Discount factor.
            lam (float): GAE λ parameter (controls bias-variance tradeoff).

        After calling this, the following attributes are set:
            self.advantages (torch.Tensor): GAE advantages, shape [T, E].
            self.returns    (torch.Tensor): Discounted returns, shape [T, E].
        """

        last_vals = last_values.reshape(-1)                                 # [1, E, 1] -> [E]

        T, E = self.rollout_steps, self.num_envs                            # T = rollout_steps, E = num_envs

        # building a [T+1, E] tensor of values
        vals = self.values                                                  # [T, E]
        tail = last_vals.unsqueeze(0)                                       # [1, E]
        values = torch.cat([vals, tail], dim=0)                             # [T+1, E]

        # computing deltas: δ_t = r_t + γ*V_{t+1}*(1−done_t) − V_t
        mask = 1.0 - self.dones                                             # [T, E]
        deltas = self.rewards + gamma * values[1:] * mask - values[:-1]     # [T, E]

        # allocating advantage and return buffers
        self.advantages = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)

        # reversed pass for GAE
        gae = torch.zeros(E, device=self.device)
        for t in reversed(range(T)):
            gae = deltas[t] + gamma * lam * mask[t] * gae
            self.advantages[t] = gae

        # computing returns: R_t = adv_t + V_t (GAE style instead of true monte carlo => reduces variance)
        self.returns = self.advantages + vals                   # [T, E]

        # monte carlo style returns
        # R = last_vals.clone()                                 # start from V(s_T)
        # for t in reversed(range(T)):
        #     R = self.rewards[t] + gamma * mask[t] * R
        #     self.returns[t] = R
