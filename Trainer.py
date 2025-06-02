from Utilities import *
from Network import *
import gymnasium as gym
from Agent import Agent
from TrajectoryBuffer import TrajectoryBuffer

import pygame

from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self,
                 env: gym.Env,
                 env_name: str,
                 experiment_name,
                 experiment_tag,
                 agent: Agent,
                 obs_dim: int,
                 config: dict[str, Any],
                 ):
        
        self.writer = SummaryWriter(log_dir=f'runs/{env_name}/tensorboard/{experiment_name}/{experiment_name+experiment_tag}')  # tensorboard

        self._render = config['render']
        
        self.env = env
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

        # Logging: Graphs = f(episode) NOTE: logging happens only for completed episodes
        self.data = {
            'env_name' : env_name,
            'experiment_name' : experiment_name,
            'reward_curve' : [],        # stores tuples of (min, avg, max, std) -> main focus is avg
            'entropy_curve' : [],       # stores tuples of (min, avg, max, std) -> main focus is avg
            'policy_loss_curve' : [],   
            'value_loss_curve' : [],    
            'noise_stds_curve' : [],
            'value_size' : value_size,
            'policy_size' : policy_size,
            'total_size' : (value_size + policy_size)
        }

    @timeit
    def rollout(self, num_steps: int)->Tuple[TrajectoryBuffer, list, list, list]:
        """
        Vectorized rollout using the preallocated TrajectoryBuffer.\\
        Collects exactly num_steps total env steps across self.num_envs envs.
        
        Returns:
        - TrajectoryBuffer: contains observations, actions, log_probs, values, rewwards, dones, advantages, returns
        - Lists of aggregated (completed) episode stats: episode_rewards, episode_lengths, episode_entropies: 

        Workload is evenly split among all environments so each environment performs exactly\\
        num_steps // num_envs steps. It is critical that num_steps // num_envs > max_reward\\
        to ensure that at least one episode has been completed per rollout. Only complete episodes\\
        per rollout and logged. At least one episode must be logged per rollout. 
        """
        
        with torch.no_grad():

            rollout_steps = num_steps // self.num_envs
            trajectory = TrajectoryBuffer(rollout_steps, self.num_envs, self.obs_dim)

            # Reset envs
            obs, infos = self.env.reset()                                # [E, O]
            obs = torch.tensor(obs, dtype=torch.float32, device=device)  # to tensor
            if obs.dim() == 2:
                obs = obs.unsqueeze(0)                                      # [1, E, O] adding batch dim for forward pass through network

            # for logging (only for completed episodes)
            ep_rewards   = [0.0] * self.num_envs
            ep_lengths   = [0]   * self.num_envs
            ep_entropies = [[]  for _ in range(self.num_envs)]

            all_rewards   = []      # completed-episode rewards
            all_lengths   = []      # completed-episode lengths    
            all_entropies = []      # completed-episode entropies


            # actual rollout here
            while trajectory.ptr < rollout_steps:
                
                actions, logps, values, entropies = self.agent.actBatched(obs)  # forward pass through actor critic networks

                # env step
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions.detach().cpu().numpy())

                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
                if next_obs.dim() == 2:
                    next_obs = next_obs.unsqueeze(0)    # adding batch dim

                # done mask (total batch consists of multiple episodes, compute gae correctly without episodes bleeding into each other)
                done_mask = torch.tensor([ter or tru for ter, tru in zip(terminated, truncated)],dtype=torch.float32, device=device)

                if obs.dim() == 3:
                    obs = obs.squeeze(0)    # prepare to add to buffer, remove batch_dim

                if values.dim() == 2:
                    values = values.squeeze(0)  # prepare to add to buffer, remove batch_dim

                # entry is added to buffer
                trajectory.add_batch(
                    logps,                                      # [E]
                    torch.tensor(rewards, device=device),       # [E]
                    values,                                     # [E]
                    obs,                                        # [E, O]
                    actions,                                    # [E]
                    done_mask                                   # [E]
                )

                # accumulation of logging stats per env
                for i in range(self.num_envs):
                    ep_rewards[i]   += rewards[i]
                    ep_lengths[i]   += 1
                    ep_entropies[i].append(entropies[i].cpu().item())
                    if done_mask[i].item() == 1.0:
                        # record completed episode i
                        all_rewards.append(ep_rewards[i])
                        all_lengths.append(ep_lengths[i])
                        all_entropies.append(float(mean(ep_entropies[i])))

                        # reset trackers for env i
                        ep_rewards[i]   = 0.0
                        ep_lengths[i]   = 0
                        ep_entropies[i].clear()

                obs = next_obs

            # Bootstrapping last value
            last_values = self.agent.value_net(obs).detach()  # shape: [E]

            # returns and GAE inside buffer
            trajectory.compute_returns_and_GAE(last_values, gamma=self.gamma, lam=self.gae_lambda)

            return trajectory, all_rewards, all_lengths, all_entropies



    def train(self)->dict[str, Any]:
        """Trains the Agent in the Environment for a number of episodes specified in the configuration file.

        Returns:
            dict[str, Any]: Dictionary of Training Stats
        """
        self.agent.train_mode()

        for ep in range(self.num_episodes):

            # rollout -> trajectory of multiple episodes accross multiple environments
            trajectory, rewards, lengths, entropies = self.rollout(self.rollout_length)

            # Actor Critic Optimization
            policy_loss, value_loss = self.agent.optimize(
                trajectory.observations, 
                trajectory.actions, 
                trajectory.log_probs, 
                trajectory.advantages, 
                trajectory.returns)

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
            self.data['policy_loss_curve'].append(policy_loss)
            self.data['value_loss_curve'].append(value_loss)
            self.data['noise_stds_curve'].append(noise_std)

            # console logging
            print(f"Episode {ep+1}/{self.num_episodes}:")
            print(f" - Reward  min: {min_r:.1f}  avg: <{avg_r:.1f}>  max: {max_r:.1f}  std: <{std_r:.2f}>")
            print(f" - Entropy min: {min_e:.3f} avg: <{avg_e:.3f}> max: {max_e:.3f} std: <{std_e:.3f}>")
            print(f" - EpisodeLength avg: {avg_len:.1f}")
            print(f" - PolicyLoss: {policy_loss:.6f}  ValueLoss: {value_loss:.6f}")
            print(f" - Noise std: {noise_std:.4f}")

            # TensorBoard logging
            step = ep
            self.writer.add_scalar("Reward/avg", avg_r, step)
            self.writer.add_scalar("Reward/min", min_r, step)
            self.writer.add_scalar("Reward/max", max_r, step)
            self.writer.add_scalar("Reward/std", std_r, step)

            self.writer.add_scalar("Entropy/avg", avg_e, step)
            self.writer.add_scalar("Entropy/min", min_e, step)
            self.writer.add_scalar("Entropy/max", max_e, step)
            self.writer.add_scalar("Entropy/std", std_e, step)

            self.writer.add_scalar("EpisodeLength/avg", avg_len, step)
            self.writer.add_scalar("PolicyLoss", policy_loss, step)
            self.writer.add_scalar("ValueLoss", value_loss, step)
            self.writer.add_scalar("NoiseSTD", noise_std, step)

            # step noise scheduler
            self.agent.policy_noise_scheduler.step(avg_r)

        self.writer.close() # tensorboard writer
        return self.data    # dictionary of collected data
        

    def render(self):
        """Renders a single environment for visualization."""
        # NOTE: will probably not work (as is) for different O shapes, I will fix it when trying other environment types

        if not self._render:
            return
        
        with torch.inference_mode():

            pygame.init()
            screen = pygame.display.set_mode((800, 600))

            self.agent.eval_mode()

            vis_env = gym.make(self.env_name, render_mode="human")   # separate environment for visualization (not vectorized)

            # [O] -> [1, 1, O] simulating expected [B, E, O]
            observation, info = vis_env.reset()
            observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)


            running = True
            while running:
                for event in pygame.event.get():    # poll window events
                    if event.type == pygame.QUIT:   # detect window close button
                        running = False

                action, *_ = self.agent.actBatched(observation)
                action = int(action.item())

                observation, _reward, terminated, truncated, _info = vis_env.step(action)
                vis_env.render()

                # [O] -> [1, 1, O] simulating expected [B, E, O]
                observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

                if terminated or truncated:
                    observation, _info = vis_env.reset()
                    observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            pygame.quit()