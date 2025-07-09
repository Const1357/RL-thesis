
from Utilities import *
from Network import *

import copy
from Schedulers import NoiseSTDScheduler


class Agent():

    def __init__(self, value_net: ValueNetwork, policy_net: PolicyNetwork, config: dict[str, Any]):
        
        self.hyperparameters = {
            'value_clip' : config['value_clip'],
            'policy_clip' : config['policy_clip'],
            'value_lr' : config['value_lr'],
            'policy_lr' : config['policy_lr'],
            'gamma' : config['gamma'],
            'gae_lambda' : config['gae_lambda'],
            'max_KL' : config['max_KL'],
            'entropy_coef' : config['entropy_coef'],
            'noise_scheduler_kwargs' : config['noise_scheduler_kwargs'],
            'batch_size' : config['batch_size'],
            'max_epochs' : config['max_epochs'],
        }

        self.mode = 'train'
        
        self.value_net = value_net
        self.policy_net = policy_net

        self.value_optimizer = torch.optim.AdamW(self.value_net.parameters(), lr=config['value_lr'])
        self.policy_optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=config['policy_lr'])

        self.policy_noise_scheduler = NoiseSTDScheduler(**config['noise_scheduler_kwargs'])

        self.old_policy_net = copy.deepcopy(self.policy_net)

        self.config = config

        self.aux_loss_kwargs = self.config['aux_loss_kwargs'] if 'aux_loss_kwargs' in self.config.keys() else None
        print(self.aux_loss_kwargs)

        self.ALE = config['policy_net_size'] == 'ALE'


    def train_mode(self):
        """
        Sets the Agent on Train Mode. Policy and Value networks are set to train mode and schedulers are unfrozen if they were initialized as unfrozen
        """
        self.value_net.train()
        self.policy_net.train()
        if 'frozen' in self.hyperparameters['noise_scheduler_kwargs'].keys() and not self.hyperparameters['noise_scheduler_kwargs']['frozen']:
            self.policy_noise_scheduler.unfreeze()
        self.mode = 'train'

    def eval_mode(self):
        """Sets the Agent on Eval Mode. Policy and Value networks are set to eval mode, and schedulers are frozen"""
        self.value_net.eval()
        self.policy_net.eval()
        self.policy_temp_scheduler.freeze()
        self.mode = 'eval'

    def sync_policy_nets(self):
        """Synchronizes the old policy network with the current one. Performed after each call of update_policy.
        """
        with torch.no_grad():
            self.old_policy_net.load_state_dict(self.policy_net.state_dict())

    def get_params(self)->dict[str, Any]:
        """Returns:
            dict[str, Any]: Dictionary containing all hyperparameters of the current instance.
        """
        return self.hyperparameters
        
    def actBatched(self, observations: torch.Tensor)->Tuple[torch.Tensor]:
        """Performs an Actor Critic forward pass given an observation

        Args:
            observations (torch.Tensor): Shape of [batch_dim, num_envs, observation_dim]

        Returns:
            Tuples of (action, log_prob, value, entropy), batched for num_envs environments (1D, shape [E])
        """
      
        # observations [B, E, O]
        
        probs, raw = self.policy_net(observations)          # [B, E, N]
        values = self.value_net(observations).squeeze(-1)   # [B, E, 1] -> [B, E]

        actions   = probs.noisy_sample(self.policy_noise_scheduler()).squeeze(0)               # [1, E] -> [E]
        log_probs = probs.log_prob(actions).squeeze(0)      # [1, E] -> [E]
        entropies = probs.entropy().squeeze(0)              # [1, E] -> [E]

        return actions, log_probs, values, entropies
    
    @timeit
    def update_value(self, observations: torch.Tensor, returns: torch.Tensor)->float:
        """Computes the ValueLoss (MSE) and Performs backward pass through the Critic (Value) Network.

        Args:
            observations (torch.Tensor)
            returns (torch.Tensor)

        Returns:
            float: average value loss across all batches
        """
        # returns = (returns - returns.mean()) / (returns.std() + tol)    # normalizing returns

        eps = self.hyperparameters['value_clip']

        total_loss = 0.0

        # frozen snapshot of values before optimizing, for clipping (detched)
        with torch.no_grad():
            old_values = self.value_net(observations).squeeze(-1).detach()   # [T, E, 1] -> [T, E]

        for _ in range(self.hyperparameters['max_epochs']):

            # T = rollout_length // E(num_envs)
            values = self.value_net(observations).squeeze(-1)  # [T, E, 1] -> [T, E]
            clipped_values = old_values + torch.clamp(values - old_values, -eps, eps)   # clipping

            # MSE Loss
            unclipped_loss = 0.5 * (values - returns).pow(2)
            clipped_loss   = 0.5 * (clipped_values - returns).pow(2)

            loss = torch.max(unclipped_loss, clipped_loss).mean()

            # optimizer step
            self.value_optimizer.zero_grad()
            loss.backward()
            if self.config['clip_grad'] != 0:
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=self.config['clip_grad']) # Gradient Clipping (if specified in config)
            self.value_optimizer.step()

            total_loss += loss.item()

        return total_loss / self.hyperparameters['max_epochs']


    # Policy with Flattened T,E = T*E
    @timeit
    def update_policy(self, observations: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor)->float:
        """Computes the PPOLoss and Performs backward pass through the Actor (Policy) Network.\\
        Repeats for max_epochs epochs, but terminates early if max_KL divergence is surpassed.

        Args:
            observations (torch.Tensor)
            actions (torch.Tensor)
            old_log_probs (torch.Tensor)
            advantages (torch.Tensor)

        Returns:
            float: average policy loss across all batches
        """
        torch.autograd.set_detect_anomaly(True)

        # T = rollout_length // E(num_envs)

        assert not old_log_probs.requires_grad
        

        # observations  [T, E, O(obs_dim)]
        # actions       [T, E]
        # old_log_probs [T, E]
        # advantages    [T, E]

        # Constructing a dataset of [B, T, E] is problematic. The point of running multiple environments is to 
        # merge the results together, therefore we flatten the T,E dimensions to a single T*E dimension.

        
        if self.ALE:
            T,E,C,H,W = observations.shape
        else:
            T,E,O = observations.shape
        if self.ALE:
            observations_flat = observations.reshape(T*E, C,H,W)    # [T*E] = rollout_length
        else:
            observations_flat = observations.reshape(T*E, O)        # [T*E] = rollout_length

        actions_flat = actions.reshape(T*E)                 # [T*E] = rollout_length
        old_log_probs_flat = old_log_probs.reshape(T*E)     # [T*E] = rollout_length
        advantages_flat = advantages.reshape(T*E)           # [T*E] = rollout_length

        print(f"[Advantage] mean: {advantages_flat.mean():.6f}, std: {advantages_flat.std():.6f}, min: {advantages_flat.min():.6f}, max: {advantages_flat.max():.6f}")

        # normalizing advantages after flattening (merging all environment transitions) => ensures global mean=0, std=1
        advantages_flat_norm = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + tol)    # [T*E] = rollout_length

        advantages_flat_norm = advantages_flat_norm.clamp(min=-10.0, max=10.0)

        dataset = torch.utils.data.TensorDataset(observations_flat, actions_flat, old_log_probs_flat, advantages_flat_norm)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.hyperparameters['batch_size'], shuffle=True)

        total_policy_loss = 0
        total_steps = 0

        for n_epochs in range(self.hyperparameters['max_epochs']):

            stopped_early = False

            for batch_observations, batch_actions, batch_old_log_probs, batch_advantages in loader:

                # batch_observations    [B, O]    
                # batch_actions         [B]
                # batch_old_log_probs   [B]    
                # batch_advantages      [B]

                batch_old_log_probs = batch_old_log_probs.detach()      # ensuring no gradient flow from old predictions

                # new forward pass for new log probs. Unsqueezing dim 1 because network expects [B, E, O] shape -> [B, 1, O]
                new_probs, raw = self.policy_net(batch_observations.unsqueeze(1))                           # [B, 1, N] new FORWARD PASS
                # should squeeze dim 1 of new_probs
                new_probs = torch.distributions.categorical.Categorical(probs=new_probs.probs.squeeze(1))   # [B, N]
        
                new_log_probs = new_probs.log_prob(batch_actions)       # new log probs                       [B]
                entropy = new_probs.entropy()                           # entropy for regularization          [B]

                epsilon = self.hyperparameters['policy_clip']           

                # PPO Loss:
                ratio = torch.exp(new_log_probs - batch_old_log_probs)                                  # [B]
                clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)                                # [B]
                surrogate = torch.min(ratio*batch_advantages, clipped_ratio*batch_advantages)           # [B]
                policy_loss = - surrogate - self.hyperparameters['entropy_coef']*entropy                # [B] reduction later

                # aux loss here PER ACTION GAUSSIAN SCORES
                if self.aux_loss_kwargs is not None:
                    
                    # extract args
                    a = self.aux_loss_kwargs.get('a', 0.1)
                    b = self.aux_loss_kwargs.get('b', 0.1)
                    M = self.aux_loss_kwargs.get('M', 1)

                    aux_coeff = self.aux_loss_kwargs.get('aux_coeff', 0.02)
                    aux_mix = self.aux_loss_kwargs.get('aux_mix', 0.5)


                    means, stds = raw                                               # [B, E, N], squeeze E
                    means = means.squeeze(1)                                        # [B, N]
                    stds = stds.squeeze(1)                                          # [B, N]


                    I = intents(means)                                              # [B, N]
                    C = confidences(stds**2)                                        # [B, N]

                    L_penalty = loss_penalty(I, C, a, b, M).clamp(min=0.0, max=M)   # [B] in [0,1] (differentiable bounding transformation)
                    L_margin_spread = margin_loss(I).clamp(min=-1.0, max=1.0)       # [B] in [-1, 1] (margin in [-1, 0], spread in [0, 1])

                    # print('\n----------------------------')
                    # print("Penalty Loss = ", L_penalty.mean().item())
                    # print("Margin-Spread Loss", L_margin_spread.mean().item())
                    # print('----------------------------\n')
                    
                    # Mixing into Existing Loss
                    mixed_aux_loss = aux_mix*L_penalty + (1-aux_mix)*L_margin_spread
                    # print("Policy loss before:", policy_loss.mean().item())
                    # policy_loss = (1- aux_coeff * mixed_aux_loss)*policy_loss       # comment if additive aux loss
                    policy_loss = policy_loss + aux_coeff*mixed_aux_loss          # uncomment if additive aux loss
                    # print("Policy loss after:", policy_loss.mean().item())

                policy_loss = policy_loss.mean()                                    # [] scalar, batch-wise averaging
                total_policy_loss += policy_loss.item()
                total_steps += 1

                # optimizer step
                self.policy_optimizer.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    policy_loss = policy_loss.mean()
                    policy_loss = torch.nan_to_num(policy_loss, nan=0.0, posinf=10.0, neginf=-10.0)

                    policy_loss.backward()
                if self.config['clip_grad'] != 0:
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.config['clip_grad']) # Gradient Clipping (if specified in config)
                self.policy_optimizer.step()

                # if KL divergence exceeds a limit -> early stopping (TRPO-like)
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean().item()   # Rough estimate of D_{KL}, [] = scalar

                    if kl_div >= self.hyperparameters['max_KL']:
                        print(f"    - Early stopping at epoch {n_epochs+1}, batch {total_steps}, KL={kl_div:.4f}")
                        stopped_early = True
                        break
            if stopped_early:
                break
        
        self.sync_policy_nets()

        print(f"Policy optimized in {n_epochs+1} epochs in {total_steps} batch updates.")   # to see if KL condition was met

        return total_policy_loss/total_steps
   
    
    def optimize(self, observations: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor)->Tuple[float, float]:
        """Performs optimization routines for Actor (Policy) and Critic (Value) networks.

        Args:
            observations (torch.Tensor)
            actions (torch.Tensor)
            old_log_probs (torch.Tensor)
            advantages (torch.Tensor)
            returns (torch.Tensor)

        Returns:
            Tuple[float, float]: policy_loss, value_loss
        """

        policy_loss = self.update_policy(observations, actions, old_log_probs, advantages)
        value_loss = self.update_value(observations, returns)

        return policy_loss, value_loss