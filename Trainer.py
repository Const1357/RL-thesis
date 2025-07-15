from Utilities import *
from Network import *
import gymnasium as gym
from Agent import Agent
from TrajectoryBuffer import TrajectoryBuffer

import pygame

class Trainer():
    def __init__(self,
                 env: gym.Env,
                 log_env: gym.Env,
                 env_name: str,
                 experiment_name,
                 experiment_tag,
                 agent: Agent,
                 obs_dim: int,
                 config: dict[str, Any],
                 ):
        
        self.config = config
        
        self._render = config['render']
        
        self.env = env
        self.log_env = log_env
        self.env_name = env_name
        self.agent = agent
        self.num_envs = config['num_envs']
        self.obs_dim = obs_dim
        self.num_episodes = config['num_episodes']
        self.max_episode_length = config['max_episode_length']
        self.rollout_length = config['rollout_length']

        self.gamma = agent.hyperparameters['gamma']
        self.gae_lambda = agent.hyperparameters['gae_lambda']

        self.episode_rewards = []

        value_size = count_parameters(self.agent.value_net)
        policy_size = count_parameters(self.agent.policy_net)

        self.aux_loss = 'aux_loss_kwargs' in self.config.keys()

        # Logging: Graphs = f(episode) NOTE: logging happens only for completed episodes
        self.data = {
            'env_name' : env_name,
            'experiment_name' : experiment_name,
            'update_steps' : config['rollout_length'],
            'log_steps' : config['rollout_length'] * config['log_frequency'],
            'total_steps' : config['rollout_length'] * config['num_episodes'],
            'use_aux_loss' : self.aux_loss,

            'reward_curve' : [],        # stores tuples of (min, avg, max, std) -> main focus is avg
            'entropy_curve' : [],       # stores tuples of (min, avg, max, std) -> main focus is avg

            'policy_loss_curve' : [],   
            'value_loss_curve' : [],
            'noise_stds_curve' : [],

            'value_size' : value_size,
            'policy_size' : policy_size,
            'total_size' : (value_size + policy_size)
        }

        if self.aux_loss:
            self.data['ppo_loss_curve'] = []
            self.data['margin_loss_curve'] = []
            self.data['penalty_loss_curve'] = []

    @timeit
    def rollout(self, num_steps: int)->TrajectoryBuffer:
        """
        Collects a rollout of experience from vectorized environments using the current policy.\\
        Terminated environments are reset, with the appropriate values added in the trajectory for correct GAE computation.\\
        Environments are not being reset at the end of the rollout. A trajectory obtained from a rollout can contain partial episodes.\\
        On subsequent calls of rollout, the environments continue exactly where they left off.

        Args:
            num_steps (int): Total number of environment steps to collect across all environments.
                            The number of steps per environment (horizon) is calculated as num_steps // num_envs.

        Returns:
            TrajectoryBuffer: A buffer containing observations, actions, log probabilities, values,
                            rewards, and masks for the collected rollout. The buffer also computes
                            advantages and returns using Generalized Advantage Estimation (GAE).
        """
        
        with torch.no_grad():

            rollout_steps = num_steps // self.num_envs
            trajectory = TrajectoryBuffer(rollout_steps, self.num_envs, self.obs_dim)

            # Rollout loop
            while trajectory.ptr < rollout_steps:

                obs = self.obs  # [E, O]

                actions, logps, values, entropies = self.agent.act_batched(obs)  # forward pass through actor critic networks

                # env step
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions.detach().cpu().numpy())

                # Manual reset of individual environments that have terminated
                done_mask_np = np.logical_or(terminated, truncated)
                if np.any(done_mask_np):
                    reset_obs, _ = self.env.reset(options={'reset_mask' : done_mask_np})
                    next_obs[done_mask_np] = reset_obs[done_mask_np]

                # Clone next_obs to preserve true post-reset obs (initial state)
                true_next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)  # [E, O]
                bootstrap_obs = true_next_obs.clone()  # this will be patched with final obs

                # Patch bootstrap_obs for environments that terminated
                for i, info in enumerate(infos):
                    if "final_observation" in info:
                        f_obs = torch.tensor(info["final_observation"], dtype=torch.float32, device=device)
                        bootstrap_obs[i] = f_obs  # use final observation for GAE

                # done mask (total batch consists of multiple episodes, compute gae correctly without episodes bleeding into each other)
                done_mask = torch.tensor(done_mask_np, dtype=torch.float32, device=device)

                # entry is added to buffer
                trajectory.add_batch(
                    logps,                                      # [E]
                    torch.tensor(rewards, device=device),       # [E]
                    values,                                     # [E]
                    obs,                                        # [E, O]
                    actions,                                    # [E]
                    done_mask                                   # [E]
                )

                self.obs = true_next_obs

            # Bootstrapping last value
            last_values = self.agent.value_net(bootstrap_obs).detach()  # shape: [E]

            # returns and GAE computation (inside the buffer)
            trajectory.compute_returns_and_GAE(last_values, gamma=self.gamma, lam=self.gae_lambda)


            return trajectory

    @timeit
    def rollout_for_logging(self)->Tuple[list, list, list]:
        """Similar to Trainer.rollout.\\
        Collects evaluation stats: reward, episode_length, entropy per actor/sub-environment.\\
        Performs one complete episode froms start to finish for each sub-environment.
        """

        with torch.no_grad():
            # Reset envs
            obs, infos = self.log_env.reset()                                   # [E, O]
            obs = torch.tensor(obs, dtype=torch.float32, device=device)     # to tensor

            # Logging
            ep_rewards   = [0.0] * self.num_envs
            ep_lengths   = [0]   * self.num_envs
            ep_entropies = [[]  for _ in range(self.num_envs)]

            # holds stats for completed episodes (1 per env)
            all_rewards   = []      # completed-episode rewards
            all_lengths   = []      # completed-episode lengths    
            all_entropies = []      # completed-episode entropies

            done_envs = [False] * self.num_envs

            # actual rollout here
            while not all(done_envs):

                actions, logps, values, entropies = self.agent.act_batched(obs)  # forward pass through actor critic networks

                # env step
                next_obs, rewards, terminated, truncated, infos = self.log_env.step(actions.detach().cpu().numpy())

                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

                # done mask (total batch consists of multiple episodes, compute gae correctly without episodes bleeding into each other)
                done_mask = torch.tensor([ter or tru for ter, tru in zip(terminated, truncated)],dtype=torch.float32, device=device)

                # accumulation of logging stats per env
                for i in range(self.num_envs):
                    if done_envs[i]:
                        continue

                    ep_rewards[i] += rewards[i]
                    ep_lengths[i] += 1
                    ep_entropies[i].append(entropies[i].cpu().item())
                    if done_mask[i].item() == 1.0:

                        done_envs[i] = True

                        # record completed episode i
                        all_rewards.append(ep_rewards[i])
                        all_lengths.append(ep_lengths[i])
                        all_entropies.append(float(mean(ep_entropies[i])))

                obs = next_obs

            return all_rewards, all_lengths, all_entropies

    def train(self)->dict[str, Any]:
        """Trains the Agent in the Environment for a number of episodes specified in the configuration file.

        Returns:
            dict[str, Any]: Dictionary of Training Stats
        """

        # Reset envs (setup for initial rollout which expects initialized self.obs)
        obs, infos = self.env.reset()                                       # [E, O]
        self.obs = torch.tensor(obs, dtype=torch.float32, device=device)    # to tensor

        for ep in range(self.num_episodes):

            # rollout -> trajectory of multiple episodes accross multiple environments
            trajectory = self.rollout(self.rollout_length)

            # Actor Critic Optimization
            policy_loss, value_loss, policy_misc = self.agent.optimize(
                trajectory.observations, 
                trajectory.actions, 
                trajectory.log_probs, 
                trajectory.advantages, 
                trajectory.returns)
            
            self.data['policy_loss_curve'].append(policy_loss)
            self.data['value_loss_curve'].append(value_loss)
            if self.aux_loss:
                self.data['ppo_loss_curve'].append(policy_misc['ppo_loss'])
                self.data['penalty_loss_curve'].append(policy_misc['penalty_loss'])
                self.data['margin_loss_curve'].append(policy_misc['margin_loss'])

            if ep % self.config['log_frequency'] == 0:

                rewards, lengths, entropies = self.rollout_for_logging()

                # aggregate reward stats
                min_r, avg_r, max_r, std_r = (
                    min(rewards), 
                    sum(rewards)/len(rewards), 
                    max(rewards), 
                    float(torch.std(torch.tensor(rewards)))
                )
                avg_len = sum(lengths)/len(lengths)

                # aggregate entropy stats
                min_e, avg_e, max_e, std_e = (
                    min(entropies),
                    sum(entropies)/len(entropies),
                    max(entropies),
                    float(std(entropies))
                )

                # log noise std
                noise_std = self.agent.policy_noise_scheduler()

                # store for plotting later
                self.data['reward_curve'].append((min_r, avg_r, max_r, std_r))
                self.data['entropy_curve'].append((min_e, avg_e, max_e, std_e))

                # console logging
                print(f"Episode {ep+1:0{len(str(self.num_episodes))}d}/{self.num_episodes}: ---------------------------------------------------------------- LOGGING")
                print(f" - Reward  min: {min_r:.1f}  avg: <{avg_r:.1f}>  max: {max_r:.1f}  std: <{std_r:.2f}>")
                print(f" - Entropy min: {min_e:.3f} avg: <{avg_e:.3f}> max: {max_e:.3f} std: <{std_e:.3f}>")
                print(f" - EpisodeLength avg: {avg_len:.1f}")
                print(f" - PolicyLoss: {policy_loss:.6f}  ValueLoss: {value_loss:.6f}")
                print(f" - Noise std: {noise_std:.4f}")
                print(f"----------------------------------------------------------------------------------------")

            else:
                print(f"Episode {ep+1:0{len(str(self.num_episodes))}d}/{self.num_episodes}:")
                print(f" - PolicyLoss: {policy_loss:.6f}  ValueLoss: {value_loss:.6f}")

            # step noise scheduler
            self.agent.policy_noise_scheduler.step()

        return self.data    # dictionary of collected data
        

    def render(self):
        """Renders a single environment for visualization."""
        # NOTE: will probably not work (as is) for different O shapes, I will fix it when trying other environment types

        if not self._render:
            return
        
        with torch.inference_mode():

            pygame.init()
            screen = pygame.display.set_mode((800, 600))

            vis_env = gym.make(self.env_name, render_mode="human")   # separate environment for visualization (not vectorized)

            # [O] -> [1] simulating expected [B, O]
            observation, info = vis_env.reset()
            observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


            running = True
            while running:
                for event in pygame.event.get():    # poll window events
                    if event.type == pygame.QUIT:   # detect window close button
                        running = False

                action, *_ = self.agent.act_batched(observation)
                action = int(action.item())

                observation, _reward, terminated, truncated, _info = vis_env.step(action)
                vis_env.render()

                # [O] -> [1, O] simulating expected [B, O]
                observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                if terminated or truncated:
                    observation, _info = vis_env.reset()
                    observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            pygame.quit()