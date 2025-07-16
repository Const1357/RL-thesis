
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
            'batch_size' : config['batch_size'],
            'max_epochs' : config['max_epochs'],
        }

        self.config = config

        self.mode = 'train'
        
        self.value_net = value_net
        self.policy_net = policy_net

        self.value_optimizer = torch.optim.AdamW(self.value_net.parameters(), lr=config['value_lr'])
        self.policy_optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=config['policy_lr'])

        self.use_noise = 'noise_scheduler_kwargs' in self.config.keys()
        if self.use_noise:
            self.policy_noise_scheduler = NoiseSTDScheduler(**config['noise_scheduler_kwargs'])
            self.hyperparameters['noise_scheduler_kwargs'] = config['noise_scheduler_kwargs']
        else:
            # HACK
            # frozen with start, min, max std = 0.
            self.policy_noise_scheduler = NoiseSTDScheduler(num_steps=1, start_std=0, min_std=0, max_std=0, frozen=True)
            self.hyperparameters['noise_scheduler_kwargs'] = {'num_steps' : 1, 'start_std' : 0, 'min_std' : 0, 'max_std' : 0, 'frozen' : True}

        self.old_policy_net = copy.deepcopy(self.policy_net)

        self.aux_loss = 'aux_loss_kwargs' in self.config.keys()
        if self.aux_loss:
            self.aux_loss_kwargs = self.config['aux_loss_kwargs']

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
        self.policy_noise_scheduler.freeze()
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
        
    def act_batched(self, observations: torch.Tensor)->Tuple[torch.Tensor]:
        """Performs an Actor Critic forward pass given an observation.

        Args:
            observations (torch.Tensor): Shape of [batch_dim, observation_dim]

        Returns:
            Tuples of batched (action, log_prob, value, entropy), shape [B] each
        """
      
        # observations [B, O]   B is either batch dim or env dim
        
        probs, raw = self.policy_net(observations)          # [B, O] -> [B, N]
        values = self.value_net(observations).squeeze(-1)   # [B, O] -> [B]

        actions   = probs.noisy_sample(self.policy_noise_scheduler())                           # [B]
        log_probs = probs.log_prob(actions)         # [B]
        entropies = probs.entropy()                 # [B] - verified.

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


        if self.ALE:
            T,E,C,H,W = observations.shape
            observations_flat = observations.reshape(T*E, C,H,W)    # [T*E] = rollout_length
        else:
            T,E,O = observations.shape
            observations_flat = observations.reshape(T*E, O)        # [T*E]
        returns_flat = returns.view(T*E)


        eps = self.hyperparameters['value_clip']

        # frozen snapshot of values before optimizing, for clipping (detched)
        with torch.no_grad():
            old_values_flat = self.value_net(observations_flat).squeeze(-1).detach()   # [T*E, 1] -> [T*E]
            
        dataset = torch.utils.data.TensorDataset(observations_flat, returns_flat, old_values_flat)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.hyperparameters['batch_size'], shuffle=True)

        total_loss = 0.0
        total_steps = 0

        for _ in range(self.hyperparameters['max_epochs']):

            for batch_observations, batch_returns, batch_old_values in loader:

                # B = rollout_length // E(num_envs)
                batch_values = self.value_net(batch_observations).squeeze(-1)  # [B, 1] -> [B]
                batch_clipped_values = batch_old_values + torch.clamp(batch_values - batch_old_values, -eps, eps)   # clipping

                # MSE Loss
                unclipped_loss = 0.5 * (batch_values - batch_returns).pow(2)
                clipped_loss   = 0.5 * (batch_clipped_values - batch_returns).pow(2)

                loss = torch.max(unclipped_loss, clipped_loss).mean()

                # optimizer step
                self.value_optimizer.zero_grad()
                loss.backward()
                if self.config['clip_grad'] != 0:
                        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=self.config['clip_grad']) # Gradient Clipping (if specified in config)
                self.value_optimizer.step()

                total_loss += loss.item()
                total_steps += 1

        return total_loss / total_steps


    # Policy with Flattened T,E = T*E
    @timeit
    def update_policy(self, observations: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor)->Tuple[float, dict|None]:
        """Computes the PPOLoss and Performs backward pass through the Actor (Policy) Network.\\
        Repeats for max_epochs epochs, but terminates early if max_KL divergence is surpassed.

        Args:
            observations (torch.Tensor)
            actions (torch.Tensor)
            old_log_probs (torch.Tensor)
            advantages (torch.Tensor)

        Returns:
            float: average policy loss across all batches
            dict: miscellaneous stats for auxiliary loss logging, or None if aux loss is not used
        """
        # torch.autograd.set_detect_anomaly(True)

        # B = T = rollout_length // E(num_envs)

        assert not old_log_probs.requires_grad
        

        # observations  [T, O(obs_dim)]          O can also be C, H, W if image based.
        # actions       [T]
        # old_log_probs [T]
        # advantages    [T]

        # Flattening Trajectory from shape [T, E, ...] to [TxE, ...]
        if self.ALE:
            T,E,C,H,W = observations.shape
            observations_flat = observations.reshape(T*E, C,H,W)    # [T*E] = rollout_length
        else:
            T,E,O = observations.shape
            observations_flat = observations.reshape(T*E, O)        # [T*E]

        actions_flat = actions.reshape(T*E)                         # [T*E]
        old_log_probs_flat = old_log_probs.reshape(T*E)             # [T*E]
        advantages_flat = advantages.reshape(T*E)                   # [T*E]

        # print(f"[Advantage] mean: {advantages_flat.mean():.6f}, std: {advantages_flat.std():.6f}, min: {advantages_flat.min():.6f}, max: {advantages_flat.max():.6f}")

        # normalizing advantages after flattening (merging all environment transitions) => ensures global mean=0, std=1
        advantages_flat_norm = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + tol)    # [T*E]

        advantages_flat_norm = advantages_flat_norm.clamp(min=-10.0, max=10.0)

        dataset = torch.utils.data.TensorDataset(observations_flat, actions_flat, old_log_probs_flat, advantages_flat_norm)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.hyperparameters['batch_size'], shuffle=True)

        total_policy_loss = 0
        if self.aux_loss:
            with torch.no_grad():       # simply for logging
                total_ppo_loss = 0
                total_penalty_loss = 0
                total_margin_loss = 0

        total_steps = 0

        for n_epochs in range(self.hyperparameters['max_epochs']):

            stopped_early = False

            for batch_observations, batch_actions, batch_old_log_probs, batch_advantages in loader:

                # batch_observations    [B, O]    
                # batch_actions         [B]
                # batch_old_log_probs   [B]    
                # batch_advantages      [B]

                batch_old_log_probs = batch_old_log_probs.detach()      # ensuring no gradient flow from old predictions

                # new forward pass for new log probs.

                new_probs, raw = self.policy_net(batch_observations)                                        # [B, N] new FORWARD PASS
                new_probs = torch.distributions.categorical.Categorical(probs=new_probs.probs)              # [B, N]
        
                new_log_probs = new_probs.log_prob(batch_actions)       # new log probs                       [B]
                entropy = new_probs.entropy()                           # entropy for regularization          [B]

                epsilon = self.hyperparameters['policy_clip']           

                # PPO Loss:
                ratio = torch.exp(new_log_probs - batch_old_log_probs)                                  # [B]
                clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)                                # [B]
                surrogate = torch.min(ratio*batch_advantages, clipped_ratio*batch_advantages)           # [B]
                policy_loss = - surrogate - self.hyperparameters['entropy_coef']*entropy                # [B] mean reduction later

                # Auxiliarry Loss: (Only for GNN_N)
                if self.aux_loss:
                    
                    # extract args
                    a = self.aux_loss_kwargs.get('a', 0.1)
                    b = self.aux_loss_kwargs.get('b', 0.1)
                    M = self.aux_loss_kwargs.get('M', 1)

                    aux_coeff = self.aux_loss_kwargs.get('aux_coeff', 0.02)
                    aux_mix = self.aux_loss_kwargs.get('aux_mix', 0.5)

                    # raw contains the raw output of the network, shaped into mean and std tensors
                    means, stds = raw                                               # [B, E, N], squeeze E
                    means = means.squeeze(1)                                        # [B, N]
                    stds = stds.squeeze(1)                                          # [B, N]

                    I = intents(means)                                              # [B, N]
                    C = confidences(stds**2)                                        # [B, N]

                    L_penalty = loss_penalty(I, C, a, b, M).clamp(min=0.0, max=M)   # [B] in [0,1] (differentiable bounding transformation)
                    L_margin_spread = margin_loss(I).clamp(min=-1.0, max=1.0)       # [B] in [-1, 1] (margin in [-1, 0], spread in [0, 1])

                    with torch.no_grad():
                        total_ppo_loss += policy_loss.mean().item()
                        total_penalty_loss += L_penalty.mean().item()
                        total_margin_loss += L_margin_spread.mean().item()
                    
                    # Mixing into Existing Loss
                    mixed_aux_loss = aux_mix*L_penalty + (1-aux_mix)*L_margin_spread
                    policy_loss = (1- aux_coeff * mixed_aux_loss)*policy_loss       # comment if additive aux loss
                    # policy_loss = policy_loss + aux_coeff*mixed_aux_loss          # uncomment if additive aux loss

                policy_loss = policy_loss.mean()                                    # [] scalar, batch-wise averaging
                total_policy_loss += policy_loss.item()
                total_steps += 1

                # optimizer step
                self.policy_optimizer.zero_grad()

                policy_loss = policy_loss.mean()
                policy_loss = torch.nan_to_num(policy_loss, nan=0.0, posinf=10.0, neginf=-10.0)

                policy_loss.backward()

                if self.config['clip_grad'] != 0:
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.config['clip_grad']) # Gradient Clipping (if specified in config)
                self.policy_optimizer.step()

                # if KL divergence exceeds a limit -> early stopping (TRPO-like), not used in Atari environments (see configs)
                
                with torch.no_grad():
                    if self.hyperparameters['max_KL'] <= 0:
                        kl_div = -1 # illegal value so that the condition will never be met
                    else:
                        kl_div = (batch_old_log_probs - new_log_probs).mean().item()   # Rough estimate of D_{KL}, [] = scalar

                    if kl_div >= self.hyperparameters['max_KL']:
                        print(f"    - Early stopping at epoch {n_epochs+1}, batch {total_steps}, KL={kl_div:.4f}")
                        stopped_early = True
                        break
            if stopped_early:
                break
        
        self.sync_policy_nets()

        # print(f"Policy optimized in {n_epochs+1} epochs in {total_steps} batch updates.")   # to see if KL condition was met

        if self.aux_loss:
            return total_policy_loss/total_steps, {
                'ppo_loss': total_ppo_loss/total_steps,
                'penalty_loss': total_penalty_loss/total_steps,
                'margin_loss': total_margin_loss/total_steps,
                }
        else:
            return total_policy_loss/total_steps, None
   
    
    def optimize(self, observations: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor)->Tuple[float, float]:
        """Performs optimization routines for Actor (Policy) and Critic (Value) networks. \\
        For more details see Agent.update_policy and Agent.update_value.
        """

        policy_loss, policy_misc = self.update_policy(observations, actions, old_log_probs, advantages)
        value_loss = self.update_value(observations, returns)

        return policy_loss, value_loss, policy_misc