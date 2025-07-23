import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import re

from scipy.ndimage import gaussian_filter1d  # curve smoothing
from scipy.interpolate import interp1d

import random

sns.set_style('darkgrid')

environments = ['CartPole-v1', 'Pendulum-v1', 'ALE/Pong-v5']
# environments = ['CartPole-v1']
# environments = ['Pendulum-v1']

standard_environments = {
    'CartPole-v1' : 'CartPole-v1',
    'Pendulum-v1' : 'Pendulum-v1 (Discretized)',
    'ALE/Pong-v5' : 'Pong-v5',
}

# per environment constants:
CONSTANTS = {
    'CartPole-v1': {
        'num_episodes' : 150,
        'max_reward' : 500,
        'min_reward' : 0,
        'xtick_interval': 100000,   # tick every 100k steps
        'ytick_interval': 50,
        'top_offset' : 20,
        'bot_offset' : 1,
        'left_offset' : -3,
        'right_offset' : 3,
    },

    'Pendulum-v1': {
        'num_episodes' : 500,
        'max_reward' : 0,
        'min_reward' : -1600,
        'xtick_interval': 250000,   # tick every 250k steps
        'ytick_interval': 200,
        'top_offset' : 0,
        'bot_offset' : 0,
        'left_offset' : -3,
        'right_offset' : 3,
    },

    'ALE/Pong-v5': {
        'num_episodes' : 800,
        'max_reward' : 21,
        'min_reward' : -21,
        'xtick_interval': 250000,   # tick every 250k steps
        'ytick_interval': 3,
        'top_offset' : 0.5,
        'bot_offset' : 0.2,
        'left_offset' : -3,
        'right_offset' : 3,
    }
}

def key_from_name(experiment: str) -> str:
    if 'noise_entropy' in experiment:
        return 'noise_entropy'
    elif 'noise' in experiment:
        return 'noise'
    elif 'entropy' in experiment:
        return 'entropy'
    else: 
        return 'no_mod'
    

def type_from_name(label: str) -> str:
    if 'CMU' in label:
        return 'CMU'
    elif 'GNN_K' in label:
        return 'GNN_K'
    elif 'logits' in label:
        return 'logits'
    elif 'GNN' in label:
        return 'GNN'

    return ''

def standardize_mod(mod: str) -> str:
    if mod == 'no_mod': return 'No modification' 
    return mod.replace('_', ' + ').title()

def standardize_label(label: str) -> str:
    type = type_from_name(label)
    if type == 'logits': type = 'Logits (Baseline)'
    if type == 'CMU': type = 'CMU-Net'
    mod = key_from_name(label)
    mod = standardize_mod(mod)
    # return f"{type} - {mod}"
    return f"{type}"
    
def standardize_individual_label(label: str, env: str) -> str:
    type = type_from_name(label)
    if type == 'logits': type = 'Logits (Baseline)'
    if type == 'CMU': type = 'CMU-Net'
    mod = key_from_name(label)
    mod = standardize_mod(mod)
    env = standard_environments[env]
    return f"{env} - {type} ({mod})"

def color_from_name(label: str) -> str:
    return colors_dict[type_from_name(label)]

for env in environments:
    experiments = os.listdir(f"runs_final/{env}/pickle/")

    C = CONSTANTS[env]

    all_reward_curves = {
        'no_mod' : [],
        'noise' : [],
        'entropy' : [],
        'noise_entropy' : [],
    }

    all_std_ys = {
        'no_mod' : [],
        'noise' : [],
        'entropy' : [],
        'noise_entropy' : [],
    }

    policy_model_sizes = []
    colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#7b3ab8",  # purple
    "#00b899",  # teal
    "#ff00b3",  # pink
    "#003968",  # dark blue
    "#572102",  # brown
    "#234111",  # dark green
    "#000000",  # black
    "#915E01",  # light brown
    ]

    colors_dict = {
        'logits' : colors[0],
        'GNN' : colors[1],
        'GNN_K' : colors[2],
        'CMU' : colors[3],
   }

    for experiment in experiments:
        runs = os.listdir(f"runs_final/{env}/pickle/{experiment}/")

        if not 'pendulum' in experiment:
            continue
        print(experiment)

        data = []
        for run in runs:
            with open(f"runs_final/{env}/pickle/{experiment}/{run}", 'rb') as f:
                # print(experiment)
                data.append(pickle.load(f))

        x = np.arange(0, data[0]['total_steps'], data[0]['log_steps'])

        
        # data[0] is the data dictionary for the first run
        # extract data that I will plot, combine it accross runs using numpy (stack and mean across dim)

        # Plot 1: Reward = f(Episode)

        reward_curves = [[s[1] for s in d['reward_curve'] ] for d in data]

        # Noise follows a linear schedule so it does not need to be logged.
        # noise_std_curves = [d['noise_stds_curve'] for d in data]
        # stacked_noise_std_curves = np.stack(noise_std_curves).mean(axis=0)
        # norm_stacked_noise_std_curves = (stacked_noise_std_curves-stacked_noise_std_curves.min())/(stacked_noise_std_curves.max()-stacked_noise_std_curves.min())*(C['max_reward'] - C['min_reward']) + C['min_reward']

        entropy_curves = [[s[1] for s in d['entropy_curve'] ] for d in data]
        stacked_entropy_curves = np.stack(entropy_curves).mean(axis=0)
        norm_stacked_entropy_curves = (stacked_entropy_curves-stacked_entropy_curves.min())/(stacked_entropy_curves.max()-stacked_entropy_curves.min())*(C['max_reward'] - C['min_reward']) + C['min_reward']

        stacked_reward_curves = np.stack(reward_curves)
        mean_reward_curve = np.mean(stacked_reward_curves, axis=0)
        std_reward_curve = np.std(stacked_reward_curves, axis=0)

        all_reward_curves[key_from_name(experiment)].append((mean_reward_curve, experiment))
        policy_model_sizes.append(data[0]['policy_size'])   # is same for all runs so pick the first = 0

        fig = plt.figure(figsize=(9,6))

        plt.plot(x, mean_reward_curve, color='blue', linewidth=1.5, label='Mean Reward')
        plt.ylim(top=C['max_reward'] + C['top_offset'], bottom=C['min_reward'] + C['bot_offset'])
        plt.xlim(left=C['left_offset'], right=data[0]['total_steps']+C['right_offset'])
        plt.xticks(range(0, data[0]['total_steps']+1, C['xtick_interval']))
        plt.yticks(range(C['min_reward'],C['max_reward']+1, C['ytick_interval']))

        for i,rc in enumerate(reward_curves):
            if i == 0:
                plt.plot(x, rc, color='green', alpha=0.2, linewidth=1, label=f"Reward (individual runs)")
            else:
                plt.plot(x, rc, color='green', alpha=0.2, linewidth=1)

        # if not np.isnan(norm_stacked_noise_std_curves).any():
        #     plt.plot(norm_stacked_noise_std_curves, color = 'red', linewidth=1, label='Noise std (normalized)')

        plt.plot(x, norm_stacked_entropy_curves, color = 'purple', linewidth=1, label='Entropy (normalized)')


        std_fill_y1 = mean_reward_curve-std_reward_curve
        std_fill_y2 = mean_reward_curve+std_reward_curve
        plt.fill_between(
            x, std_fill_y1, std_fill_y2, color='blue', alpha=0.1, label='Reward Standard Deviation', edgecolor='none')
        all_std_ys[key_from_name(experiment)].append((std_fill_y1, std_fill_y2, experiment))
        
                
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.suptitle(f"Rewards: {standardize_individual_label(experiment, env)}", fontweight='bold')
        plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.08),
           ncol=4, frameon=False, fontsize='small')
        plt.tight_layout()
        savedir_svg = f"runs_final/{env}/plots/reward_curves/svg/"
        savedir_png = f"runs_final/{env}/plots/reward_curves/png/"
        # ensure dirs exist
        os.makedirs(savedir_svg, exist_ok=True)
        os.makedirs(savedir_png, exist_ok=True)
        fig.savefig(f"{savedir_svg}{experiment}_reward_curve.svg", format='svg')
        fig.savefig(f"{savedir_png}{experiment}_reward_curve.png", format='png')
        # plt.show()
        plt.close()
        

        # Plot 2: Average Policy losses (PPO, auxiliarry losses) for CMU
        if type_from_name(experiment) == 'CMU':
            fig = plt.figure(figsize=(11,6))
            plt.ylim(top=1.1, bottom=0)
            plt.yticks(ticks=np.arange(0, 40+1, 5)/40, labels=[])
            plt.xlim(left=C['left_offset'], right=data[0]['total_steps']+C['right_offset'])
            plt.xticks(range(0, data[0]['total_steps']+1, C['xtick_interval']))

            num_episodes = data[0]['total_steps']//data[0]['update_steps']
            log_frequency = data[0]['log_steps']//data[0]['update_steps']

            selected_indices = np.arange(0, data[0]['total_steps'], log_frequency)  # log_frequency
            selected_indices = selected_indices[selected_indices < num_episodes] # num_episodes

            # policy_loss_curves = [np.array(d['policy_loss_curve'])[selected_indices] for d in data]
            # ppo_loss_curves = [np.array(d['ppo_loss_curve'])[selected_indices] for d in data]
            # penalty_loss_curves = [np.array(d['penalty_loss_curve'])[selected_indices] for d in data]
            # margin_loss_curves = [np.array(d['margin_loss_curve'])[selected_indices] for d in data]

            policy_loss_curves = [np.array(d['policy_loss_curve']) for d in data]
            ppo_loss_curves = [np.array(d['ppo_loss_curve']) for d in data]
            penalty_loss_curves = [np.array(d['penalty_loss_curve']) for d in data]
            margin_loss_curves = [np.array(d['margin_loss_curve']) for d in data]

            def normalize(curves):
                curves = np.stack(curves)  # shape: [R, T]
                min_vals = curves.min(axis=1, keepdims=True)
                max_vals = curves.max(axis=1, keepdims=True)
                return (curves - min_vals) / (max_vals - min_vals + 1e-8)
            
            def smooth(curve):
                return gaussian_filter1d(curve, sigma=2)

            stacked_norm_policy_loss_curve = normalize(policy_loss_curves)
            stacked_norm_ppo_loss_curve = normalize(ppo_loss_curves)
            stacked_norm_penalty_loss_curve = normalize(penalty_loss_curves)
            stacked_norm_margin_loss_curve = normalize(margin_loss_curves)


            mean_policy_loss_curve = gaussian_filter1d(stacked_norm_policy_loss_curve.mean(axis=0), sigma=1)
            mean_ppo_loss_curve = gaussian_filter1d(stacked_norm_ppo_loss_curve.mean(axis=0), sigma=1)
            # mean_policy_loss_curve = smooth(stacked_norm_policy_loss_curve.mean(axis=0))
            # mean_ppo_loss_curve = smooth(stacked_norm_ppo_loss_curve.mean(axis=0))
            mean_penalty_loss_curve = smooth(stacked_norm_penalty_loss_curve.mean(axis=0))
            mean_margin_loss_curve = smooth(stacked_norm_margin_loss_curve.mean(axis=0))

            std_policy = smooth(stacked_norm_policy_loss_curve.std(axis=0))
            std_ppo = smooth(stacked_norm_ppo_loss_curve.std(axis=0))
            std_penalty = smooth(stacked_norm_penalty_loss_curve.std(axis=0))
            std_margin = smooth(stacked_norm_margin_loss_curve.std(axis=0))

            normalized_mean_reward_curve = (mean_reward_curve - mean_reward_curve.min())/(mean_reward_curve.max()-mean_reward_curve.min())
            normalized_entropy_curve = (norm_stacked_entropy_curves-C['min_reward'])/(C['max_reward'] - C['min_reward'])

            _x = np.arange(0, data[0]['total_steps'], data[0]['update_steps'])

            plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)

            plt.fill_between(_x, mean_penalty_loss_curve-std_penalty, mean_penalty_loss_curve+std_penalty, alpha=0.2, color=colors[2], edgecolor='none')
            plt.fill_between(_x, mean_margin_loss_curve-std_margin, mean_margin_loss_curve+std_margin, alpha=0.2, color=colors[3], edgecolor='none')

            plt.plot(_x, mean_penalty_loss_curve, label='Penalty Loss (smoothed)',color=colors[2])
            plt.plot(_x, mean_margin_loss_curve, label='Margin Loss (smoothed)',color=colors[3])
            plt.plot(_x, 0.5 + (mean_ppo_loss_curve - mean_ppo_loss_curve.mean())*5, label='PPO Loss (centered x5)',color=colors[0], alpha=1, linewidth='0.7')
            plt.plot(_x, 0.5 + (mean_policy_loss_curve - mean_policy_loss_curve.mean())*5, label='Mixed Loss (centered x5)',color=colors[1], alpha=1, linewidth='0.7')
            plt.plot(x, normalized_mean_reward_curve, color='black', label='reward')
            plt.plot(x, normalized_entropy_curve, color=colors[4], label='entropy')

            # plt.fill_between(x, mean_policy_loss_curve-std_policy, mean_policy_loss_curve+std_policy, alpha=0.2, color=colors[0], edgecolor='none')
            # plt.fill_between(x, mean_ppo_loss_curve-std_ppo, mean_ppo_loss_curve+std_ppo, alpha=0.2, color=colors[1], edgecolor='none')

            plt.xlabel('Steps')
            plt.ylabel('Normalized Axis')
            plt.suptitle(f"Objectives: {standardize_individual_label(experiment, env)}", fontweight='bold')
            plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.08),
            ncol=6, frameon=False, fontsize='small')

            savedir_svg = f"runs_final/{env}/plots/policy_loss_curves/svg/"
            savedir_png = f"runs_final/{env}/plots/policy_loss_curves/png/"
            os.makedirs(savedir_svg, exist_ok=True)
            os.makedirs(savedir_png, exist_ok=True)
            plt.tight_layout()
            fig.savefig(f"{savedir_svg}{experiment}_policy_losses.svg", format='svg')
            fig.savefig(f"{savedir_png}{experiment}_policy_losses.png", format='png')
            plt.close()


            # Plot 2.2 - Scatter plot between Margin Loss and Entropy
            fig = plt.figure(figsize=(6, 6))
            f_margin_loss = interp1d(_x, mean_margin_loss_curve, kind='linear')
            downsampled_margin_loss = f_margin_loss(x)
            plt.scatter(normalized_entropy_curve, downsampled_margin_loss, alpha=0.6)
            plt.xlabel("Entropy (Normalized)")
            plt.ylabel("Margin Loss (Normalized)")
            plt.suptitle("Correlation between Entropy and Margin Loss", fontweight='bold')
            plt.title(f"{standardize_individual_label(experiment, env)}")
            plt.grid(True)
            plt.ylim(top=1.0, bottom=0.0)
            plt.tight_layout()
            savedir_svg = f"runs_final/{env}/plots/policy_loss_curves/svg/"
            savedir_png = f"runs_final/{env}/plots/policy_loss_curves/png/"
            os.makedirs(savedir_svg, exist_ok=True)
            os.makedirs(savedir_png, exist_ok=True)
            fig.savefig(f"{savedir_svg}{experiment}_scatter_entropy_margin.svg", format='svg')
            fig.savefig(f"{savedir_png}{experiment}_scatter_entropy_margin.png", format='png')
            plt.close()

            # Plot 2.3 - Intent-Confidence evolution (averaged per-action for selected timesteps of a specific run (run 0)

            milestone_idx = [int(p * len(x)) for p in [0.0, 0.02, 0.05, 0.4, 0.60, 0.90]]
            fig, axes = plt.subplots(
                nrows=len(milestone_idx),
                figsize=(7, 1.27 * len(milestone_idx)),
                sharex=True
            )

            sampled_episode_steps = []
            all_intents = []
            all_confidences = []
            all_probs = []
            all_selected_actions = []

            for idx in milestone_idx:
                intents_trajectory = data[0]['intent_evolution']
                confidences_trajectory = data[0]['confidence_evolution']
                probs_trajectory = data[0]['action_probabilities']
                selected_actions_trajectory = data[0]['selected_actions']

                episode_length = len(intents_trajectory[idx])
                sample_range = (int(0.25 * episode_length), int(0.75 * episode_length))
                sampled_episode_step = random.randint(*sample_range)
                sampled_episode_steps.append(sampled_episode_step)

                intents = np.array(intents_trajectory[idx][sampled_episode_step])
                confidences = np.array(confidences_trajectory[idx][sampled_episode_step])
                probs = np.array(probs_trajectory[idx][sampled_episode_step])
                selected_action = selected_actions_trajectory[idx][sampled_episode_step]

                all_intents.append(intents)
                all_confidences.append(confidences)
                all_probs.append(probs)
                all_selected_actions.append(selected_action)

            # Global log normalization bounds
            max_intent = np.log(np.max([np.max(i) for i in all_intents]) + 1e-9)
            min_intent = np.log(np.min([np.min(i) for i in all_intents]) + 1e-9)

            from matplotlib.colors import Normalize
            from matplotlib import cm

            norm = Normalize(vmin=0, vmax=1)
            cmap = cm.viridis

            for i, idx in enumerate(milestone_idx):
                ax = axes[i]
                
                intents = all_intents[i]
                confidences = all_confidences[i]
                probs = all_probs[i]
                selected_action = all_selected_actions[i]

                norm_intents = (np.log(intents + 1e-9) - min_intent) / (max_intent - min_intent + 1e-7)
                action_indices = np.arange(len(intents))


                # Map confidence to colors via colormap
                color_values = [cmap(norm(c)) for c in confidences]

                # Plot all non-selected actions
                for j in range(len(intents)):
                    if j == selected_action:
                        continue
                    ax.scatter(
                        norm_intents[j], action_indices[j],
                        s=40,
                        c=[color_values[j]],
                        alpha=0.9,
                        marker='o',
                        edgecolors='black',
                        linewidths=0.5
                    )

                    # TODO: FIXME if not working, I dont have data to test yet
                    ax.scatter(
                        probs[j], action_indices[j],
                        s=15,
                        c=colors[3],
                        alpha=0.9,
                        marker='X',
                        edgecolors='black',
                        linewidths=0.3
                    )

                # Plot selected action with a distinct marker
                if 0 <= selected_action < len(intents):
                    ax.scatter(
                        norm_intents[selected_action], action_indices[selected_action],
                        s=50,
                        c=[color_values[selected_action]],
                        alpha=1.0,
                        marker='D',  # distinct marker
                        edgecolors='black',
                        linewidths=0.6
                    )

                    # TODO: FIXME if not working, I dont have data to test yet.
                    ax.scatter(
                        probs[selected_action], action_indices[selected_action],
                        s=50,
                        c=colors[3],
                        alpha=1.0,
                        marker='P',  # distinct marker
                        edgecolors='black',
                        linewidths=0.6
                    )
        
                ax.set_yticks(action_indices)
                ax.set_yticklabels([f"Action {j}" for j in action_indices], fontsize=7)
                ax.set_xlim(-0.02, 1.02)
                # ax.set_ylim(-1, len(intents))
                ax.set_ylim(len(intents), -1)
                ax.yaxis.grid(True, alpha=0.75)
                ax.set_ylabel(f"{idx}%", rotation=0, labelpad=30,
                            va='center', ha='right',
                            fontsize=10, fontweight='bold')
                ax.text(
                0.1,                         # X-position (center in axis coordinates)
                1.02,                        # Y-position slightly above the top
                f"{idx}% Training Steps",    # Your label
                transform=ax.transAxes,      # Use axis coordinate system
                ha='center', va='bottom',
                fontsize=8,
                )
                
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], marker='D', color='w', label='Selected Action',
                    markerfacecolor='gray', markersize=6, markeredgecolor='black'),
                Line2D([0], [0], marker='o', color='w', label='Other Actions',
                    markerfacecolor='gray', markersize=6, markeredgecolor='black'),
                Line2D([0], [0], marker='x', color=colors[3], label='Action Probabilities',         # TODO: FIXME if not working, no test yet
                    markerfacecolor=colors[3], markersize=3, markeredgecolor='black'),
                Line2D([0], [0], marker='+', color=colors[3], label='Selected Action Probability',  # TODO: FIXME if not working, no test yet
                    markerfacecolor=colors[3], markersize=3, markeredgecolor='black')
            ]

            # Add legend to the topmost axis
            axes[0].legend(handles=legend_elements, loc='upper left', fontsize=8, frameon=False)
                

            # Shared X-axis
            axes[-1].set_xlabel('Intent (Normalized Logarithmic Scale)', fontsize=12, fontweight='bold')

            # Title (inside plotting area)
            fig.subplots_adjust(top=0.87, right=1.0, hspace=0.5)


            fig.suptitle(
                "Intent–Confidence Distribution at Selected Milestones",
                fontsize=13, fontweight='bold'
            )
            fig.text(
                0.5, 0.92,
                f"{standardize_individual_label(experiment, env)}",
                fontsize=10,
                ha='center'
            )

            # Create a mappable object for the colorbar (needed even if scatter handles colormap)
            from matplotlib.cm import ScalarMappable

            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # required dummy array

            # Add a vertical colorbar to the right of all subplots
            cbar = fig.colorbar(
                sm,
                ax=axes,              # span all subplots
                orientation='vertical',
                pad=0.02,             # space between last subplot and colorbar
                shrink=1.0,           # adjust if your plots are very tall
                aspect=25             # controls thickness
            )
            cbar.set_label("Confidence", fontsize=10, fontweight='bold')
            cbar.ax.invert_yaxis()

            savedir_svg = f"runs_final/{env}/plots/policy_loss_curves/svg/"
            savedir_png = f"runs_final/{env}/plots/policy_loss_curves/png/"
            os.makedirs(savedir_svg, exist_ok=True)
            os.makedirs(savedir_png, exist_ok=True)
            fig.savefig(f"{savedir_svg}{experiment}_IC_distribution.svg", format='svg')
            fig.savefig(f"{savedir_png}{experiment}_IC_distribution.png", format='png', dpi=300)
            plt.close()

    
    # For ALL experiments together:

    # Group per modification
    mods = ['no_mod', 'noise', 'entropy', 'noise_entropy']
    types = ['logits', 'GNN', 'GNN_K', 'CMU']

    for mod in mods:

        if not 'pendulum' in experiment:
            continue
        print(experiment)

        # Plot 1: Reward = f(Episode) - only mean runs and their stds, for each experiment (in the same plot)
        fig = plt.figure(figsize=(9,6))

        to_plot = {
            _type : None for _type in types
        }
        to_plot_std = {
            _type : None for _type in types
        }

        for i, (std_y1, std_y2,label) in enumerate(all_std_ys[mod]):
            to_plot_std[type_from_name(label)] = (std_y1, std_y2, color_from_name(label), standardize_label(label))

        for i,(curve,label) in enumerate(all_reward_curves[mod]):
            to_plot[type_from_name(label)] = (curve, color_from_name(label), standardize_label(label))

        for t in types:
            curve = to_plot[t]
            std_info = to_plot_std[t]
            if curve is not None:
                plt.plot(x, curve[0], color=curve[1], label=curve[2])
                plt.fill_between(x, std_info[0], std_info[1], color = std_info[2], alpha = 0.2, edgecolor = 'none')

        plt.ylim(top=C['max_reward'] + C['top_offset'], bottom=C['min_reward'])
        plt.xlim(left=C['left_offset'], right=data[0]['total_steps']+C['right_offset'])
        plt.xticks(range(0, data[0]['total_steps']+1, C['xtick_interval']))
        plt.yticks(range(C['min_reward'],C['max_reward']+1, C['ytick_interval']))

        plt.suptitle(f'Rewards: {standard_environments[env]} ({standardize_mod(mod)})', fontweight='bold')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.08),
           ncol=4, frameon=False, fontsize='small')
        plt.tight_layout()
        savedir_svg = f"runs_final/{env}/plots/reward_curves/svg/all_rewards_{mod}.svg"
        savedir_png = f"runs_final/{env}/plots/reward_curves/png/all_rewards_{mod}.png"
        fig.savefig(savedir_svg, format='svg')
        fig.savefig(savedir_png, format='png')
        # plt.show()
        plt.close()

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


        
        