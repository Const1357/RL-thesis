import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

import re

from scipy.ndimage import gaussian_filter1d  # curve smoothing
from scipy.interpolate import interp1d

import random

sns.set_style('darkgrid')

DIR = 'runs_final'
DIR_ABLATION = 'runs_ablation'

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

def save_and_close(fig, savedir, name):
    savedir_svg = f"{savedir}/svg/"
    savedir_png = f"{savedir}/png/"
    os.makedirs(savedir_svg, exist_ok=True)
    os.makedirs(savedir_png, exist_ok=True)
    fig.savefig(f"{savedir_svg}{name}.svg", format='svg')
    fig.savefig(f"{savedir_png}{name}.png", format='png', dpi=300)
    plt.close()

# as seen in filenames (exception: "no_mod" = "")
envs = ['cartpole', 'pendulum', 'pong']
policy_types = ['logits', 'GNN', 'GNN_K', 'CMU']
mods = ['no_mod', 'entropy', 'noise', 'noise_entropy']
ablation_mods = [
    'alignment',
    'penalty',
    'margin',
    'alignment_penalty',
    'alignment_margin',
    'penalty_margin',
    'alignment_penalty_margin',           
    'no_mod',
]

# as seen in directories
environments = ['CartPole-v1', 'Pendulum-v1', 'ALE/Pong-v5']


def load_runs(dirname):
    runs = os.listdir(dirname)

    data = []

    for run in runs:
        with open(f"{dirname}/{run}", 'rb') as f:
            data.append(pickle.load(f))

    return data

def get_env(filename):
    if 'cartpole' in filename:
        return 'CartPole-v1'
    elif 'pendulum' in filename:
        return 'Pendulum-v1'
    elif 'pong' in filename:
        return 'ALE/Pong-v5'

def get_policy_type(filename):

    if 'CMU' in filename:
        return 'CMU'
    elif 'GNN_K' in filename:
        return 'GNN_K'
    elif 'logits' in filename:
        return 'logits'
    elif 'GNN' in filename:
        return 'GNN'
    
    raise RuntimeError("Unable to infer policy type from filename")

def get_mod(filename):
    if 'noise_entropy' in filename:
        return 'noise_entropy'
    elif 'noise' in filename:
        return 'noise'
    elif 'entropy' in filename:
        return 'entropy'
    else: 
        return 'no_mod'
    
def get_ablation_mod(filename):

    match = re.search(r'^(?:[^_]*_){2}(.*?)(?=_mlp)', filename)
    return match.group(1) if match else 'no_mod'

env_title = {
    'CartPole-v1' : 'CartPole-v1',
    'Pendulum-v1' : 'Pendulum-v1 (Discretized)',
    'ALE/Pong-v5' : 'Pong-v5',
}

def clean_mod(mod):
    if mod == 'no_mod': return 'No modification' 
    return mod.replace('_', ' + ').title()

def clean_policy_type(ptype):
    if ptype == 'logits': return 'Logits (Baseline)'
    else:
        return f"{ptype}-Net"

def get_individual_title(filename, ablation=False):
    if ablation:
        env = 'Pendulum-v1'
        ptype = 'CMU'
        mod = get_ablation_mod(filename)
    else:
        env = get_env(filename)
        ptype = get_policy_type(filename)
        mod = get_mod(filename)

    return f"{env_title[env]} | {clean_policy_type(ptype)} ({clean_mod(mod)})"

# -------------------------- Plots --------------------------------
def plot_individual_reward_curves(dirname, load_from, ablation_plot=False):

    if ablation_plot:
        env = 'Pendulum-v1'
        ptype = 'CMU'
        mod = get_ablation_mod(dirname)
    else:
        env = get_env(dirname)
        ptype = get_policy_type(dirname)
        mod = get_mod(dirname)

    data = load_runs(f"{load_from}{dirname}")
    C = CONSTANTS[env]

    print(len(data))
    print(dirname)

    x = np.arange(0, data[0]['total_steps'], data[0]['log_steps'])


    reward_curves = [[s[1] for s in d['reward_curve'] ] for d in data]
    stacked_reward_curves = np.stack(reward_curves)
    mean_reward_curve = np.mean(stacked_reward_curves, axis=0)


    entropy_curves = [[s[1] for s in d['entropy_curve'] ] for d in data]
    stacked_entropy_curves = np.stack(entropy_curves).mean(axis=0)
    norm_stacked_entropy_curves = (stacked_entropy_curves-stacked_entropy_curves.min())/(stacked_entropy_curves.max()-stacked_entropy_curves.min())*(C['max_reward'] - C['min_reward']) + C['min_reward']

    std_reward_curve = np.std(stacked_reward_curves, axis=0)

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

    plt.plot(x, norm_stacked_entropy_curves, color = colors[4], linewidth=1, label='Entropy (normalized)')


    std_fill_y1 = mean_reward_curve-std_reward_curve
    std_fill_y2 = mean_reward_curve+std_reward_curve
    plt.fill_between(
        x, std_fill_y1, std_fill_y2, color='blue', alpha=0.1, label='Reward Standard Deviation', edgecolor='none')    
            
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.suptitle(f"Rewards: {get_individual_title(dirname, ablation=ablation_plot)}", fontweight='bold')
    plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.08),
        ncol=4, frameon=False, fontsize='small')
    plt.tight_layout()

    savedir = f"{DIR_ABLATION if ablation_plot else DIR}{'' if ablation_plot else '/'+env}/plots/reward_curves" 
    save_and_close(fig, savedir, f"{dirname}_reward_curve")


    # savedir_svg = f"runs_final/{env}/plots/reward_curves/svg/"
    # savedir_png = f"runs_final/{env}/plots/reward_curves/png/"
    # # ensure dirs exist
    # os.makedirs(savedir_svg, exist_ok=True)
    # os.makedirs(savedir_png, exist_ok=True)
    # fig.savefig(f"{savedir_svg}{experiment}_reward_curve.svg", format='svg')
    # fig.savefig(f"{savedir_png}{experiment}_reward_curve.png", format='png')
    # # plt.show()
    # plt.close()

    return env, ptype, mod, x, data[0]['total_steps'], mean_reward_curve, std_fill_y1, std_fill_y2
    

def plot_CMU_policy_loss_curves(dirname, load_from, ablation_plot=False):

    if ablation_plot:
        env = 'Pendulum-v1'
        ptype = 'CMU'
        mod = get_ablation_mod(dirname)
    else:
        env = get_env(dirname)
        ptype = get_policy_type(dirname)
        mod = get_mod(dirname)

    data = load_runs(f"{load_from}{dirname}")
    C = CONSTANTS[env]


    fig = plt.figure(figsize=(9,6))
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
    alignment_loss_curves = [np.array(d['alignment_loss_curve']) for d in data]
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
    stacked_norm_alignment_loss_curve = normalize(alignment_loss_curves)
    stacked_norm_penalty_loss_curve = normalize(penalty_loss_curves)
    stacked_norm_margin_loss_curve = normalize(margin_loss_curves)


    mean_policy_loss_curve = gaussian_filter1d(stacked_norm_policy_loss_curve.mean(axis=0), sigma=1)
    mean_ppo_loss_curve = gaussian_filter1d(stacked_norm_ppo_loss_curve.mean(axis=0), sigma=1)
    # mean_policy_loss_curve = smooth(stacked_norm_policy_loss_curve.mean(axis=0))
    # mean_ppo_loss_curve = smooth(stacked_norm_ppo_loss_curve.mean(axis=0))
    mean_alignment_loss_curve = smooth(stacked_norm_alignment_loss_curve.mean(axis=0))
    mean_penalty_loss_curve = smooth(stacked_norm_penalty_loss_curve.mean(axis=0))
    mean_margin_loss_curve = smooth(stacked_norm_margin_loss_curve.mean(axis=0))

    # std_policy = smooth(stacked_norm_policy_loss_curve.std(axis=0))
    # std_ppo = smooth(stacked_norm_ppo_loss_curve.std(axis=0))
    std_alignment = smooth(stacked_norm_alignment_loss_curve.std(axis=0))
    std_penalty = smooth(stacked_norm_penalty_loss_curve.std(axis=0))
    std_margin = smooth(stacked_norm_margin_loss_curve.std(axis=0))

    reward_curves = [[s[1] for s in d['reward_curve'] ] for d in data]
    stacked_reward_curves = np.stack(reward_curves)
    mean_reward_curve = np.mean(stacked_reward_curves, axis=0)


    entropy_curves = [[s[1] for s in d['entropy_curve'] ] for d in data]
    stacked_entropy_curves = np.stack(entropy_curves).mean(axis=0)
    norm_stacked_entropy_curves = (stacked_entropy_curves-stacked_entropy_curves.min())/(stacked_entropy_curves.max()-stacked_entropy_curves.min())*(C['max_reward'] - C['min_reward']) + C['min_reward']


    normalized_mean_reward_curve = (mean_reward_curve - mean_reward_curve.min())/(mean_reward_curve.max()-mean_reward_curve.min())
    normalized_entropy_curve = (norm_stacked_entropy_curves-C['min_reward'])/(C['max_reward'] - C['min_reward'])

    x = np.arange(0, data[0]['total_steps'], data[0]['log_steps'])
    _x = np.arange(0, data[0]['total_steps'], data[0]['update_steps'])

    plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)

    plt.fill_between(_x, mean_alignment_loss_curve-std_alignment, mean_alignment_loss_curve+std_alignment, alpha=0.2, color=colors[5], edgecolor='none')
    plt.fill_between(_x, mean_penalty_loss_curve-std_penalty, mean_penalty_loss_curve+std_penalty, alpha=0.2, color=colors[2], edgecolor='none')
    plt.fill_between(_x, mean_margin_loss_curve-std_margin, mean_margin_loss_curve+std_margin, alpha=0.2, color=colors[3], edgecolor='none')

    plt.plot(_x, mean_alignment_loss_curve, label='Alignment Loss (smoothed)',color=colors[5])
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
    plt.suptitle(f"Objectives: {get_individual_title(dirname, ablation=ablation_plot)}", fontweight='bold')
    # plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.08),
    # ncol=5, frameon=False, fontsize='small')
    
    # Create individual handles
    handles = [
        Line2D([0], [0], color=colors[5], label='Alignment Loss (smoothed)'),
        Line2D([0], [0], color=colors[2], label='Penalty Loss (smoothed)'),
        Line2D([0], [0], color=colors[3], label='Margin Loss (smoothed)'),
        Line2D([0], [0], color=colors[0], label='PPO Loss (centered x5)', linewidth=0.7),
        Line2D([0], [0], color=colors[1], label='Mixed Loss (centered x5)', linewidth=0.7),
        Line2D([0], [0], color='black', label='reward'),
        Line2D([0], [0], color=colors[4], label='entropy'),
    ]

    # Now split into 2 rows:
    row1 = handles[:3]
    row2 = handles[3:]

    # Create first row of legend
    fig.legend(
        handles=row1,
        loc='upper center',
        bbox_to_anchor=(0.505, 0.95),
        ncol=3,
        frameon=False,
        fontsize='small'
    )

    # Create second row of legend (centered manually)
    fig.legend(
        handles=row2,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.91),  # adjust horizontal offset here
        ncol=4,
        frameon=False,
        fontsize='small'
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.86)  # move plot area down to make room for fig.legend


    savedir = f"{DIR_ABLATION if ablation_plot else DIR}{'' if ablation_plot else '/'+env}/plots/policy_loss_curves" 
    save_and_close(fig, savedir, f"{dirname}_policy_losses")

    # savedir_svg = f"runs_final/{env}/plots/policy_loss_curves/svg/"
    # savedir_png = f"runs_final/{env}/plots/policy_loss_curves/png/"
    # os.makedirs(savedir_svg, exist_ok=True)
    # os.makedirs(savedir_png, exist_ok=True)
    # fig.savefig(f"{savedir_svg}{experiment}_policy_losses.svg", format='svg')
    # fig.savefig(f"{savedir_png}{experiment}_policy_losses.png", format='png')
    # plt.close()

def plot_CMU_IC_evolution(dirname, load_from, ablation_plot=False):

    if ablation_plot:
        env = 'Pendulum-v1'
        ptype = 'CMU'
        mod = get_ablation_mod(dirname)
    else:
        env = get_env(dirname)
        ptype = get_policy_type(dirname)
        mod = get_mod(dirname)

    data = load_runs(f"{load_from}{dirname}")
    C = CONSTANTS[env]

    x = np.arange(0, data[0]['total_steps'], data[0]['log_steps'])

    milestone_idx = [int(p * len(x)) for p in [0.0, 0.02, 0.05, 0.2, 0.4, 0.60, 0.90]]
    fig, axes = plt.subplots(
        nrows=len(milestone_idx),
        figsize=(7, 1.27 * len(milestone_idx)),
        sharex=True,
        gridspec_kw={'height_ratios': [0] + [1]*(len(milestone_idx)-1)}
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

        if i == 0:

            axes[0].remove()

            axes[0].axis('off')


            legend_elements = [
                Line2D([0], [0], marker='D', color='w', label='Selected Action',
                    markerfacecolor='gray', markersize=6, markeredgecolor='black'),
                Line2D([0], [0], marker='o', color='w', label='Other Actions',
                    markerfacecolor='gray', markersize=6, markeredgecolor='black'),
                Line2D([0], [0], marker='+', color='w', label='Action Probabilities',
                    markerfacecolor=colors[3], markersize=5, markeredgecolor=colors[3], linewidth=0.5),
                Line2D([0], [0], marker='x', color='w', label='Selected Action Probability',
                    markerfacecolor=colors[3], markersize=4, markeredgecolor=colors[3], linewidth=0.5)
            ]

            fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.90))
            # Add legend to the topmost axis
            # axes[0].legend(handles=legend_elements, loc='center', ncol=4, fontsize=8, frameon=False)
            # axes[0].set_position([0.05, 0.94, 0.90, 0.05])

            # now re–layout so subplots slide up a bit
            continue

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

            x1 = norm_intents[j]
            x2 = probs[j]
            y  = action_indices[j]
            ax.hlines(
                y=y,
                xmin=min(x1, x2),
                xmax=max(x1, x2),
                colors=colors[3],
                alpha= 0.5,
                linewidth=0.3
            )

            ax.scatter(
                norm_intents[j], action_indices[j],
                s=40,
                c=[color_values[j]],
                alpha=1.0,
                marker='o',
                edgecolors='black',
                linewidths=0.5
            )

            ax.scatter(
                probs[j], action_indices[j],
                s=25,
                c=colors[3],
                alpha=1,
                marker='+',
                linewidths = 0.5
            )


        # Plot selected action with a distinct marker
        if 0 <= selected_action < len(intents):

            x1 = norm_intents[selected_action]
            x2 = probs[selected_action]
            y  = action_indices[selected_action]
            ax.hlines(
                y=y,
                xmin=min(x1, x2),
                xmax=max(x1, x2),
                colors=colors[3],
                alpha= 0.5,
                linewidth=0.3
            )

            ax.scatter(
                norm_intents[selected_action], action_indices[selected_action],
                s=50,
                c=[color_values[selected_action]],
                alpha=1.0,
                marker='D',  # distinct marker
                edgecolors='black',
                linewidths=0.6
            )

            ax.scatter(
                probs[selected_action], action_indices[selected_action],
                s=40,
                c=colors[3],
                alpha=1.0,
                marker='x',  # distinct marker
                linewidths = 0.5
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
        
    


    # Shared X-axis
    axes[-1].set_xlabel('Intent (Normalized Logarithmic Scale)', fontsize=12, fontweight='bold')

    # Title (inside plotting area)
    fig.subplots_adjust(top=0.87, right=1.0, bottom=0.06, hspace=0.5)


    fig.suptitle(
        "Intent–Confidence Evolution over Selected Milestones",
        fontsize=13, fontweight='bold'
    )
    fig.text(
        0.5, 0.92,
        f"{get_individual_title(dirname, ablation=ablation_plot)}",
        fontsize=9,
        ha='center',
        fontweight='bold'
    )

    # Create a mappable object for the colorbar (needed even if scatter handles colormap)
    from matplotlib.cm import ScalarMappable

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required dummy array

    # Add a vertical colorbar to the right of all subplots
    cbar = fig.colorbar(
        sm,
        ax=axes[1:],          # span all subplots apart from dedicated legend subplot
        orientation='vertical',
        pad=0.02,             # space between last subplot and colorbar
        shrink=1.0,           # adjust if your plots are very tall
        aspect=25             # controls thickness
    )
    cbar.set_label("Confidence", fontsize=10, fontweight='bold')
    cbar.ax.invert_yaxis()

    savedir = f"{DIR_ABLATION if ablation_plot else DIR}{'' if ablation_plot else '/'+env}/plots/IC_evolution" 
    save_and_close(fig, savedir, f"{dirname}_IC_evolution")


def plot_reward_comparison(data):

    env = data['env']
    C = CONSTANTS[env]
    x = data['x']
    total_steps = data['total_steps']

    for mod in mods:

        fig = plt.figure(figsize=(9,6))
        to_plot = data[mod]

        for ptype in policy_types:
            curve = to_plot[ptype]['reward_curve'] if ptype in to_plot.keys() else None
            std_low, std_high = to_plot[ptype]['std_curve'] if ptype in to_plot.keys() else None
            if curve is not None:
                plt.plot(x, curve, color=colors_dict[ptype], label=clean_policy_type(ptype))
                plt.fill_between(x, std_low, std_high, color = colors_dict[ptype], alpha = 0.2, edgecolor = 'none')

        plt.ylim(top=C['max_reward'] + C['top_offset'], bottom=C['min_reward'])
        plt.xlim(left=C['left_offset'], right=total_steps+C['right_offset'])
        plt.xticks(range(0, total_steps+1, C['xtick_interval']))
        plt.yticks(range(C['min_reward'],C['max_reward']+1, C['ytick_interval']))

        plt.suptitle(f'Rewards: {env_title[env]} ({clean_mod(mod)})', fontweight='bold')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.08),
            ncol=4, frameon=False, fontsize='small')
        plt.tight_layout()

        savedir = f"{DIR}/{env}/plots/reward_curves"
        name = f"all_rewards_{mod}"
        save_and_close(fig, savedir, name)


def plot_aux_reward_comparison(data):

    env = data['env']
    C = CONSTANTS[env]
    x = data['x']
    total_steps = data['total_steps']


    # load baseline
    dirname = f"{DIR}/{env}/pickle/pendulum_logits_mlp"
    baseline_data = load_runs(dirname)

    baseline_reward_curves = [[s[1] for s in d['reward_curve'] ] for d in baseline_data]
    baseline_stacked_reward_curves = np.stack(baseline_reward_curves)
    baseline_mean_reward_curve = np.mean(baseline_stacked_reward_curves, axis=0)






    fig = plt.figure(figsize=(9,6))

    for i,mod in enumerate(ablation_mods):

        print(mod)
        curve = data[mod]['reward_curve']
        std_low, std_high = data[mod]['std_curve']
        if curve is not None:
            plt.plot(x, curve, color=colors[i], label=clean_mod(mod))
            plt.fill_between(x, std_low, std_high, color = colors[i], alpha = 0.2, edgecolor = 'none')  # NOTE: stds visually overwhelm the plot
            # TODO: decide to either limit curves, remove stds, or increase number of plots to contain combinations of mods

    plt.plot(x, baseline_mean_reward_curve, color='black', label='Logits (Baseline)', linestyle='--')

    plt.ylim(top=C['max_reward'] + C['top_offset'], bottom=C['min_reward'])
    plt.xlim(left=C['left_offset'], right=total_steps+C['right_offset'])
    plt.xticks(range(0, total_steps+1, C['xtick_interval']))
    plt.yticks(range(C['min_reward'],C['max_reward']+1, C['ytick_interval']))

    plt.suptitle(f'Rewards: {env_title[env]} | CMU-Net (Auxiliary Loss Ablation)', fontweight='bold')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    fig.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 0.95),
        ncol=3, frameon=False, fontsize='small')
    plt.tight_layout()
    fig.subplots_adjust(top=0.84)

    savedir = f"{DIR_ABLATION}/plots/reward_curves"
    name = f"all_rewards_ablation"
    save_and_close(fig, savedir, name)



def main_plots():
    for environment in environments:

        # Normal Plots:
        load_from = f"{DIR}/{environment}/pickle/"
        dirnames = os.listdir(load_from)

        data = {
            'env' : environment,
            'x' : None,
            'no_mod' : {
                'logits' : {},
                'GNN' : {},
                'GNN_K' : {},
                'CMU' : {},
            },
            'entropy' : {
                'logits' : {},
                'GNN' : {},
                'GNN_K' : {},
                'CMU' : {},
            },
            'noise' : {
                'logits' : {},
                'GNN' : {},
                'GNN_K' : {},
                'CMU' : {},
            },
            'noise_entropy' : {
                'logits' : {},
                'GNN' : {},
                'GNN_K' : {},
                'CMU' : {},
            },
        }

        for dirname in dirnames:

            env, ptype, mod, x, total_steps, mean_reward_curve, std_fill_y1, std_fill_y2 = plot_individual_reward_curves(dirname, load_from)
            
            data['x'] = x
            data['total_steps'] = total_steps
            data[mod][ptype]['reward_curve'] = mean_reward_curve
            data[mod][ptype]['std_curve'] = (std_fill_y1, std_fill_y2)

            if ptype == 'CMU':
                plot_CMU_IC_evolution(dirname, load_from)
                plot_CMU_policy_loss_curves(dirname, load_from)
            
        plot_reward_comparison(data)


def ablation_plots():
    # Ablation Plots:
    load_from = f"{DIR_ABLATION}/pickle/"
    dirnames = os.listdir(load_from)

    data = {
            'env' : 'Pendulum-v1',
            'no_mod' : {},
            'alignment' : {},
            'alignment_penalty' : {},
            'alignment_margin' : {},
            'alignment_penalty_margin' : {},
            'penalty' : {},
            'penalty_margin' : {},
            'margin' : {},
        }

    for dirname in dirnames:
        env, ptype, mod, x, total_steps, mean_reward_curve, std_fill_y1, std_fill_y2 = plot_individual_reward_curves(dirname, load_from, ablation_plot=True)
        

        data['x'] = x
        data['total_steps'] = total_steps
        data[mod]['reward_curve'] = mean_reward_curve
        data[mod]['std_curve'] = (std_fill_y1, std_fill_y2)

        plot_CMU_IC_evolution(dirname, load_from, ablation_plot=True)
        plot_CMU_policy_loss_curves(dirname, load_from, ablation_plot=True)

    plot_aux_reward_comparison(data)


def main():
    
    # main_plots()
    ablation_plots()







if __name__ == '__main__':
    main()