from Utilities import *
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import FrameStackObservation, TimeLimit, ResizeObservation, NormalizeObservation

from Agent import Agent
from Trainer import Trainer
from NetworkFactory import create_value_network, create_policy_network

from config_loader import load_config

from datetime import datetime

import os
import pickle

import random

def make_env(name, config, is_log=False):

    # ClearnRL implementation was used as inspiration for the make_env function:
    # https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py

    render_mode = 'rgb_array' if is_log and config['save_frames'] else None

    def _thunk():

        if 'ALE' in name:
            # ALE v5 environments: args in make not env_name.
            # See more: https://ale.farama.org/environments/
            env = gym.make(name, render_mode=render_mode, frameskip=4, obs_type='grayscale', full_action_space=False, repeat_action_probability=0.0) 
        else:
            env = gym.make(name, render_mode=render_mode)

        if config['quantize']:
            env = BoxToDiscreteWrapper(env, config['num_bins'])

        env = TimeLimit(env, max_episode_steps=config['max_episode_length'])    # override max episode length
        if 'atari' in config:

            env = NoopFireResetEnv(env) # adds diversity in initial states (seeding produces same initial state, ATARI ROM loading problem, fixed by performing noops)
            # Also performs a 'FIRE' action to begin the game.
            env = ClipRewardEnv(env)                        # cleanrl implementation: clips by sign of reward: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/atari_wrappers.py
            env = ResizeObservation(env, shape=(84, 84))    # resizes observation to [84, 84] (already grayscale)
            env = NormalizeObservation(env)                 # Normalizes observation using running mean and std stats
            env = FrameStackObservation(env, stack_size=config['atari']['stack_size'], padding_type="zero") # stacks previous frames in the channel dimension
            # FrameStackObservation is used to learn time-dependent features (e.g. ball velocity and direction in pong), even with feedforward nets.
        
        return env
    return _thunk

def main():

    config = load_config()
    
    print(config)

    ENV_NAME = config['env']                            # environment name exactly as is required for instantiation using gym.make

    EXPERIMENT_NAME = config['experiment_name']         # experiment name is descriptive to the experiment: usually {environment}_{policy_type}_{network_type}
    TAG = "_" + datetime.now().strftime("%d%m_%H%M")    # datetime tag to differentiate each experiment

    print(f"EXPERIMENT NAME = {EXPERIMENT_NAME}")

    seed_offset = random.randint(0, 10000)              # ensure subsequent runs are different
    log_seed_offset = random.randint(10000, 20000)      # ensure subsequent runs are different
    seeds = [seed_offset + 127*i for i in range(config['num_envs'])]
    log_seeds = [log_seed_offset + 127*i for i in range(config['num_envs'])]

    # Rollout Vector Environments
    env_fns = [make_env(ENV_NAME, config) for _ in range(config['num_envs'])]       # num_envs:      E
    env = SyncVectorEnv(env_fns)
    env.reset(seed=seeds)                                                           # Initialization

    # Evaluation (Logging) Vector Environments - evaluation happens once every config['log_frequency'] steps
    log_env_fns = [make_env(ENV_NAME, config, is_log=True) for _ in range(config['num_envs'])]   # num_envs:      E
    log_env = SyncVectorEnv(log_env_fns)
    log_env.reset(seed=log_seeds)                                                   # Initialization


    observation_space_size = env.single_observation_space.shape[0]      # observation_space dim: O (ignored for ALE. hardcoded C x H x W = 4 x 84 x 84)
    action_space_size = env.single_action_space.n                       # action_space dim:      A

    # Actor Critic networks instantiation based on configuration
    policy_net = create_policy_network(input_size=observation_space_size, output_size=action_space_size, config=config).to(device)
    value_net = create_value_network(input_size=observation_space_size, config=config).to(device)
    # input_size is ignored for ALE configurations. It is hardcoded to 4x84x84, where 4 = config['atari']['stack_size']. 
    # It can easily be made modular, but it will not be needed in this project.
    
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
    pkl_file = f'runs_final/{ENV_NAME}/pickle/{EXPERIMENT_NAME}/{EXPERIMENT_NAME+TAG}.pkl'
    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)

    # renders the learned policy (if specified on configuration, else is skipped)
    trainer.render()    # An earlier version of Trainer.render worked. Since moving to (headless) WSL environment, it has not yet been tested.


if __name__ == "__main__":
    main()