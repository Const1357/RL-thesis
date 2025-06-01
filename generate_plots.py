import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('darkgrid')

environments = ['CartPole-v1']

# per environment constants:
CONSTANTS = {
    'CartPole-v1': {
        'num_episodes' : 150,
        'max_reward' : 500,
        'xtick_interval': 10,
        'ytick_interval': 50,
        'top_offset' : 20,
        'left_offset' : -3,
        'right_offset' : 3,
    }
}

for env in environments:
    experiments = os.listdir(f"runs/{env}/pickle/")

    C = CONSTANTS[env]

    all_reward_curves = []
    policy_model_sizes = []
    # colors = ['blue', 'red', 'green', 'darkgreen', 'purple', 'orange', 'black']
    colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#7b3ab8",  # purple
    "#00b899",  # brown
    "#ff00b3"   # pink
    ]
    markers = ['o', 's', 'D', '^', 'v', 'P', '*']

    for experiment in experiments:
        runs = os.listdir(f"runs/{env}/pickle/{experiment}/")

        data = []
        for run in runs:
            with open(f"runs/{env}/pickle/{experiment}/{run}", 'rb') as f:
                data.append(pickle.load(f))

        x = np.arange(C['num_episodes'])      # NOTE: FIXED, CHANGE FOR EACH ENVIRONMENT. Cartpole is currently 150
        
        # data[0] is the data dictionary for the first run
        # extract data that I will plot, combine it accross runs using numpy (stack and mean across dim)

        # Plot 1: Reward = f(Episode)
            # could be good to show the temperature in the same plot

        reward_curves = [[s[1] for s in d['reward_curve'] ] for d in data]

        temp_curves = [d['temperatures_curve'] for d in data]
        stacked_temp_curves = np.stack(temp_curves).mean(axis=0)
        norm_stacked_temp_curves = (stacked_temp_curves-stacked_temp_curves.min())/(stacked_temp_curves.max()-stacked_temp_curves.min())*500

        entropy_curves = [[s[1] for s in d['entropy_curve'] ] for d in data]
        stacked_entropy_curves = np.stack(entropy_curves).mean(axis=0)
        norm_stacked_entropy_curves = (stacked_entropy_curves-stacked_entropy_curves.min())/(stacked_entropy_curves.max()-stacked_entropy_curves.min())*500

        stacked_reward_curves = np.stack(reward_curves)
        mean_reward_curve = np.mean(stacked_reward_curves, axis=0)
        std_reward_curve = np.std(stacked_reward_curves, axis=0)

        all_reward_curves.append(mean_reward_curve)
        policy_model_sizes.append(data[0]['policy_size'])   # is same for all runs so pick the first = 0

        fig = plt.figure(figsize=(8,6))

        plt.plot(mean_reward_curve, color='blue', linewidth=1.5, label='Mean Reward')
        plt.ylim(top=C['max_reward'] + C['top_offset'])
        plt.xlim(left=C['left_offset'], right=C['num_episodes']+C['right_offset'])
        plt.xticks(range(0, C['num_episodes']+1, C['xtick_interval']))
        plt.yticks(range(0,C['max_reward']+1, C['ytick_interval']))

        for i,rc in enumerate(reward_curves):
            if i == 0:
                plt.plot(rc, color='green', alpha=0.2, linewidth=1, label=f"Reward (individual runs)")
            else:
                plt.plot(rc, color='green', alpha=0.2, linewidth=1)

        plt.plot(norm_stacked_temp_curves, color = 'red', linewidth=1, label='Temperature (normalized)')
        plt.plot(norm_stacked_entropy_curves, color = 'purple', linewidth=1, label='Entropy (normalized)')


        
        plt.fill_between(
            range(len(mean_reward_curve)), mean_reward_curve-std_reward_curve, mean_reward_curve+std_reward_curve, color='blue', alpha=0.1, label='Reward Standard Deviation', edgecolor='none')
                
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend(loc='lower left')
        plt.title(experiment)
        plt.tight_layout()
        savedir_svg = f"runs/{env}/plots/reward_curves/svg/"
        savedir_png = f"runs/{env}/plots/reward_curves/png/"
        # ensure dirs exist
        os.makedirs(savedir_svg, exist_ok=True)
        os.makedirs(savedir_png, exist_ok=True)
        fig.savefig(f"{savedir_svg}{experiment}_reward_curve.svg", format='svg')
        fig.savefig(f"{savedir_png}{experiment}_reward_curve.png", format='png')
        # plt.show()
        

        # Plot 2: Entropy, Temperature = f(Episode) - only mean runs, and std around entropy and temperature

        # Plot 3: Policy Loss & Value loss? useless?

    
    # For ALL experiments together:
    plt.show()
    # Plot 1: Reward = f(Episode) - only mean runs and their stds, for each experiment (in the same plot)
    fig = plt.figure(figsize=(8,6))

    for i,curve in enumerate(all_reward_curves):
        plt.plot(curve, color=colors[i], label=experiments[i])#, marker=markers[i], markevery=15)

    plt.ylim(top=C['max_reward'] + C['top_offset'])
    plt.xlim(left=C['left_offset'], right=C['num_episodes']+C['right_offset'])
    plt.xticks(range(0, C['num_episodes']+1, C['xtick_interval']))
    plt.yticks(range(0,C['max_reward']+1, C['ytick_interval']))

    plt.title(f'Rewards over Episodes for {env}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc='lower right')
    plt.tight_layout()
    savedir_svg = f"runs/{env}/plots/reward_curves/svg/all_rewards.svg"
    savedir_png = f"runs/{env}/plots/reward_curves/png/all_rewards.png"
    fig.savefig(savedir_svg, format='svg')
    fig.savefig(savedir_png, format='png')
    plt.show()

    # Plot 2: Scatter Plot of: Model Size(with tag=experiment_name)  Vs Earliest hit max reward on mean run (model size = x, earliest max reward = y)
    # fig = plt.figure(figsize=(8,6))

    # x_points = list(range(len(all_reward_curves)))
    # earliest_max = [np.argmax(curve) for curve in all_reward_curves]

    # labels = [f"{exp}_{sz}" for exp,sz in zip(experiments, policy_model_sizes)]

    # plt.scatter(x_points, earliest_max, s=100)
    # plt.xticks(ticks=x_points, labels=labels, rotation=45, ha='right')
    # plt.xlabel('Model Type and Size')
    # plt.ylabel('Earliest Max Reward Episode')
    # plt.title('Model Size vs Earliest Max')
    # plt.tight_layout()
    # plt.show()

    # ^ CURRENTLY USELESS. Should experiment with the same policy type and different model size. Unimportant for this study.


    # Plot 3: Table with rollout times in seconds for fixed amount of episodes, or table with forward pass times (compare complexity with logits-model)
    # LATER.


        
        