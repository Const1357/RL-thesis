from Utilities import *
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import FrameStackObservation, TimeLimit, RescaleObservation, ResizeObservation

from Agent import Agent
from Trainer import Trainer
from NetworkFactory import create_value_network, create_policy_network

from config_loader import load_config

from datetime import datetime

import os
import pickle

import random

########################################################################################################################################################

# NOTE: It is critical that when doing a configuration for an environment, that num_envs * max_episode_length >= rollout_length
#       This guarantees that per rollout, we will have at least one completed episode for logging.
#       The optimization can utilize truncated episodes with bootstrapping, but logging is done only for complete episodes.

########################################################################################################################################################

# Currently my configurations are only supporting only CartPole-v1.
# This implementation can currently support any cartpole-v1-like gym environment. Other environments are yet untesed.
# I will add configuration files for other environments after completion of all methods. (GNN_N left + continuous)

    
def make_env(name, config, seeds):
    # used for multiple environments
    def _thunk():
        print(name)
        
        if 'ALE' in name:
            env = gym.make(name, frameskip=4, obs_type='grayscale', full_action_space=False, repeat_action_probability=0.0)
        else:
            env = gym.make(name)

        if config['quantize']:
            env = BoxToDiscreteWrapper(env, config['num_bins'])

        env = TimeLimit(env, max_episode_steps=config['max_episode_length'])
        if 'atari' in config:

            env = ResizeObservation(env, shape=(84, 84))           # [84, 84]
            env = RescaleObservation(env, min_obs=np.zeros((84, 84), dtype=np.float32), max_obs=np.ones((84, 84), dtype=np.float32))    # uint8: [0-255]-> float32: [0-1]

            # env = RewardClippingWrapper(env, min=-1, max=1)
            env = FrameStackObservation(env, stack_size=config['atari']['stack_size'])
        
        return env
    return _thunk

def main():

    config = load_config()
    print(config)

    EXPERIMENT_NAME = config['experiment_name']         # experiment name is descriptive to the experiment: usually {environment}_{policy_type}_{network_type}
    TAG = "_" + datetime.now().strftime("%d%m_%H%M")    # datetime tag to differentiate each experiment

    ENV_NAME = config['env']                            # environment name exactly as is required for instantiation using gym.make

    print(f"EXPERIMENT NAME = {EXPERIMENT_NAME}")

    seed_offset = random.randint(0, 10000)  # ensure subsequent runs are different
    log_seed_offset = random.randint(10000, 20000)  # ensure subsequent runs are different

    
    seeds = [seed_offset + 127*i for i in range(config['num_envs'])]
    log_seeds = [log_seed_offset + 127*i for i in range(config['num_envs'])]
    print(seeds)
    print(log_seeds)

    env_fns = [make_env(ENV_NAME, config, seeds) for _ in range(config['num_envs'])]   # num_envs:      E
    env = SyncVectorEnv(env_fns)

    log_env_fns = [make_env(ENV_NAME, config, log_seeds) for _ in range(config['num_envs'])]   # num_envs:      E
    log_env = SyncVectorEnv(log_env_fns)

    for i in range(config['num_envs']):
        env.envs[i].reset(seed=seeds[i])
        log_env.envs[i].reset(seed=log_seeds[i])

    print(env.np_random_seed)
    
    observation_space_size = env.single_observation_space.shape[0]      # observation_space dim: O (ignored for ALE. hardcoded C x H x W = 4 x 84 x 84)
    action_space_size = env.single_action_space.n                       # action_space dim:      A

    # Actor Critic networks instantiation based on configuration
    policy_net = create_policy_network(input_size=observation_space_size, output_size=action_space_size, config=config).to(device)
    value_net = create_value_network(input_size=observation_space_size, config=config).to(device)
    # input_size is ignored for ALE configurations
    
    # Agent Instantiation based on configuration
    agent = Agent(
        value_net=value_net,
        policy_net=policy_net,
        config=config,
    )

    # Trainer Instantiation based on configuration
    trainer = Trainer(
        env=env,
        log_env=log_env,
        env_name=ENV_NAME,
        experiment_name=EXPERIMENT_NAME,
        experiment_tag=TAG,
        agent= agent,
        obs_dim=(4, 84, 84) if config['policy_net_size'] == 'ALE' else observation_space_size,
        config=config
    )

    data = trainer.train()  # Trains the Agent in the Environment using PPO routine with specified configuration 

    # store data collected during training for further visualization and analysis
    pkl_file = f'runs/{ENV_NAME}/pickle/{EXPERIMENT_NAME}/{EXPERIMENT_NAME+TAG}.pkl'
    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)

    # renders the learned policy (if specified on configuration, else is skipped)
    trainer.render()



if __name__ == "__main__":
    main()



# https://docs.cleanrl.dev/rl-algorithms/ppo/#implemented-variants

# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py
# def make_env(env_id, idx, capture_video, run_name):
#     def thunk():
#         if capture_video and idx == 0:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env = NoopResetEnv(env, noop_max=30)
#         env = MaxAndSkipEnv(env, skip=4)
#         env = EpisodicLifeEnv(env)
#         if "FIRE" in env.unwrapped.get_action_meanings():
#             env = FireResetEnv(env)
#         env = ClipRewardEnv(env)
#         env = gym.wrappers.ResizeObservation(env, (84, 84))
#         env = gym.wrappers.GrayScaleObservation(env)
#         env = gym.wrappers.FrameStack(env, 4)
#         return env

#     return thunk