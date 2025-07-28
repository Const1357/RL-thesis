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

import pandas as pd


sns.set_style('darkgrid')

DIR = 'runs_seeded'
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

    'Acrobot-v1': {
        'num_episodes' : 300,
        'max_reward' : 0,
        'min_reward' : -500,
        'xtick_interval': 100000,   # tick every 100k steps
        'ytick_interval': 100,
        'top_offset' : 0,
        'bot_offset' : -2,
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
    elif 'acrobot' in filename:
        return 'Acrobot-v1'
    elif 'pong' in filename:
        return 'ALE/Pong-v5'
    
def get_flat_env(filename):
    if 'cartpole' in filename:
        return 'cartpole'
    elif 'pendulum' in filename:
        return 'pendulum'
    elif 'acrobot' in filename:
        return 'acrobot'
    elif 'pong' in filename:
        return 'pong'
    
def flatten_env(envname):
    if 'CartPole-v1' in envname:
        return 'cartpole'
    elif 'Pendulum-v1' in envname:
        return 'pendulum'
    elif 'Acrobot-v1' in envname:
        return 'acrobot'
    elif 'ALE/Pong-v5'in envname:
        return 'pong'

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

    match = re.search(r'^(?:[^_]*_){2}(.*?)(?=_(?:mlp|cnn))', filename)
    return match.group(1) if match else 'no_mod'

def ablation_mod_short(mod: str):
    if mod == 'no_mod':
        return 'No Mod'
    mod = mod.split('_')
    mod = [m[0] for m in mod]
    mod = ''.join(mod).upper()
    mod.upper()
    return mod

env_title = {
    'CartPole-v1' : 'CartPole-v1',
    'Pendulum-v1' : 'Pendulum-v1 (Discretized)',
    'Acrobot-v1'  : 'Acrobot-v1',
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

    env = get_env(filename)
    if ablation:
        ptype = 'CMU'
        mod = get_ablation_mod(filename)
    else:
        ptype = get_policy_type(filename)
        mod = get_mod(filename)

    return f"{env_title[env]} | {clean_policy_type(ptype)} ({clean_mod(mod)})"


def smooth(curve, s=2):
        return gaussian_filter1d(curve, sigma=s)

# -------------------------- Plots --------------------------------
def plot_individual_reward_curves(dirname, load_from, ablation_plot=False):

    env = get_env(dirname)
    if ablation_plot:
        ptype = 'CMU'
        mod = get_ablation_mod(dirname)
        print('HERE', dirname, mod)
    else:
        ptype = get_policy_type(dirname)
        mod = get_mod(dirname)

    data = load_runs(f"{load_from}{dirname}")
    C = CONSTANTS[env]

    # print(dirname, len(data))

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

    dir_ablation = f"{DIR_ABLATION}_{flatten_env(env)}"
    savedir = f"{dir_ablation if ablation_plot else DIR}{'' if ablation_plot else '/'+env}/plots/reward_curves" 
    save_and_close(fig, savedir, f"{dirname}_reward_curve")

    return env, ptype, mod, x, data[0]['total_steps'], mean_reward_curve, std_fill_y1, std_fill_y2, stacked_reward_curves
    

def plot_CMU_policy_loss_curves(dirname, load_from, ablation_plot=False):

    env = get_env(dirname)
    if ablation_plot:
        ptype = 'CMU'
        mod = get_ablation_mod(dirname)
    else:
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
    

    stacked_norm_policy_loss_curve = normalize(policy_loss_curves)
    stacked_norm_ppo_loss_curve = normalize(ppo_loss_curves)
    stacked_norm_alignment_loss_curve = normalize(alignment_loss_curves)
    stacked_norm_penalty_loss_curve = normalize(penalty_loss_curves)
    stacked_norm_margin_loss_curve = normalize(margin_loss_curves)


    mean_policy_loss_curve = smooth(stacked_norm_policy_loss_curve.mean(axis=0), s=1)
    mean_ppo_loss_curve = smooth(stacked_norm_ppo_loss_curve.mean(axis=0), s=1)


    mean_alignment_loss_curve = smooth(stacked_norm_alignment_loss_curve.mean(axis=0))
    mean_penalty_loss_curve = smooth(stacked_norm_penalty_loss_curve.mean(axis=0))
    mean_margin_loss_curve = smooth(stacked_norm_margin_loss_curve.mean(axis=0))


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


    plt.xlabel('Steps')
    plt.ylabel('Normalized Axis')
    plt.suptitle(f"Objectives: {get_individual_title(dirname, ablation=ablation_plot)}", fontweight='bold')
    
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

    dir_ablation = f"{DIR_ABLATION}_{flatten_env(env)}"
    savedir = f"{dir_ablation if ablation_plot else DIR}{'' if ablation_plot else '/'+env}/plots/policy_loss_curves" 
    save_and_close(fig, savedir, f"{dirname}_policy_losses")


def plot_CMU_IC_evolution(dirname, load_from, ablation_plot=False):

    env = get_env(dirname)
    if ablation_plot:
        ptype = 'CMU'
        mod = get_ablation_mod(dirname)
    else:
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

    dir_ablation = f"{DIR_ABLATION}_{flatten_env(env)}"

    savedir = f"{dir_ablation if ablation_plot else DIR}{'' if ablation_plot else '/'+env}/plots/IC_evolution" 
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

            curve = to_plot[ptype]['reward_curve'] if ptype in to_plot.keys() and 'reward_curve' in to_plot[ptype].keys() else None
            if curve is not None:
                std_low, std_high = to_plot[ptype]['std_curve'] 
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
    C = CONSTANTS[get_env(env)]
    x = data['x']
    total_steps = data['total_steps']


    # load baseline
    dirname = f"{DIR}/{get_env(env)}/pickle/{env}_logits_{'cnn' if env == 'pong' else 'mlp'}"
    baseline_data = load_runs(dirname)

    baseline_reward_curves = [[s[1] for s in d['reward_curve'] ] for d in baseline_data]
    baseline_stacked_reward_curves = np.stack(baseline_reward_curves)
    baseline_mean_reward_curve = np.mean(baseline_stacked_reward_curves, axis=0)


    fig = plt.figure(figsize=(9,6))

    for i,mod in enumerate(ablation_mods):

        if 'reward_curve' not in data[mod]:
            continue

        curve = data[mod]['reward_curve']
        std_low, std_high = data[mod]['std_curve']
        if curve is not None:
            plt.plot(x, smooth(curve, 1), color=colors[i], label=clean_mod(mod))
            plt.fill_between(x, smooth(std_low, 1), smooth(std_high, 1), color = colors[i], alpha = 0.2, edgecolor = 'none')

    plt.plot(x, smooth(baseline_mean_reward_curve, 1), color='black', label='Logits (Baseline)', linestyle='--')

    plt.ylim(top=C['max_reward'] + C['top_offset'], bottom=C['min_reward'])
    plt.xlim(left=C['left_offset'], right=total_steps+C['right_offset'])
    plt.xticks(range(0, total_steps+1, C['xtick_interval']))
    plt.yticks(range(C['min_reward'],C['max_reward']+1, C['ytick_interval']))

    plt.suptitle(f'Rewards: {env_title[get_env(env)]} | CMU-Net (Auxiliary Loss Ablation)', fontweight='bold')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    fig.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 0.95),
        ncol=3, frameon=False, fontsize='small')
    plt.tight_layout()
    fig.subplots_adjust(top=0.84)

    dir_ablation = f"{DIR_ABLATION}_{env}"
    savedir = f"{dir_ablation}/plots/reward_curves"
    name = f"all_rewards_ablation_{env}"
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

            env, ptype, mod, x, total_steps, mean_reward_curve, std_fill_y1, std_fill_y2, _ = plot_individual_reward_curves(dirname, load_from)
            
            data['x'] = x
            data['total_steps'] = total_steps
            data[mod][ptype]['reward_curve'] = mean_reward_curve
            data[mod][ptype]['std_curve'] = (std_fill_y1, std_fill_y2)

            if ptype == 'CMU':
                plot_CMU_IC_evolution(dirname, load_from)
                plot_CMU_policy_loss_curves(dirname, load_from)
            
        plot_reward_comparison(data)


def ablation_plots(env):
    # Ablation Plots:
    load_from = f"{DIR_ABLATION}_{env}/pickle/"
    dirnames = os.listdir(load_from)

    data = {
            'env' : get_flat_env(env),
            'no_mod' : {},
            'alignment' : {},
            'alignment_penalty' : {},
            'alignment_margin' : {},
            'alignment_penalty_margin' : {},
            'penalty' : {},
            'penalty_margin' : {},
            'margin' : {},
        }
    
    metrics = {}

    for dirname in dirnames:
        env, ptype, mod, x, total_steps, mean_reward_curve, std_fill_y1, std_fill_y2, stacked_reward_curves = plot_individual_reward_curves(dirname, load_from, ablation_plot=True)
        
        # print(mod)

        data['x'] = x
        data['total_steps'] = total_steps
        data[mod]['reward_curve'] = mean_reward_curve
        data[mod]['std_curve'] = (std_fill_y1, std_fill_y2)

        plot_CMU_IC_evolution(dirname, load_from, ablation_plot=True)
        plot_CMU_policy_loss_curves(dirname, load_from, ablation_plot=True)

        if flatten_env(env) == 'pong':
            threshold = 19.5
        elif flatten_env(env) == 'pendulum':
            threshold = -200.0
        elif flatten_env(env) == 'acrobot':
            threshold = -100

        metrics[dirname] = compute_metrics(stacked_reward_curves, threshold=threshold)

    plot_aux_reward_comparison(data)
    return metrics





# ------------- Quantitative Analysis -----------------
def compute_metrics(rewards: np.ndarray, threshold: float):

    # rewards = np.array with shape num_seeds, num_episodes = [S, E]
    
    # Metric 1: Convergence speed
    mean_reward = rewards.mean(axis=0)  # [S]

    convergence_step = np.argmax(mean_reward > threshold)
    if convergence_step == 0:
        convergence_step = len(mean_reward) + 1 # signal outlier, compute score as if worst (=0)

    # Metric 2: Variance
    std_reward = rewards.std(axis=0)    # [S]
    std = std_reward.mean()

    # Metric 3: Stability (instability. Prefer lower values)
    # per run differences in reward(t) - reward(t-1)
    diffs = np.abs(np.diff(rewards, axis=1))  # [N, E-1]
    instability = diffs.mean()

    # Metric 4: AUC:
    auc = np.trapezoid(mean_reward, np.arange(len(mean_reward)))

    return {
        'Convergence' : convergence_step,
        'STD (5 runs)' : round(std, 2),
        'Instability' : round(instability, 2),
        'AUC' : int(auc),
    }

def bold_best(df, column, best='max', steps = -2):
    vals = df[column]
    if best == 'max':
        idx = vals.idxmax()
    elif best == 'min':
        idx = vals.idxmin()
    df[column] = df[column].apply(lambda v: f"\\textbf{{{v}}}" if v == vals[idx] else 'failed' if column == 'Convergence' and v == steps + 1 else f"{v}")

    return df

def add_score_column(df, columns, steps=0):
    """
    Adds a 'Score' column to the DataFrame based on normalized metrics.

    Parameters:
        df (pd.DataFrame): your input DataFrame
        columns (dict): {column_name: 'max' or 'min'}

    Returns:
        pd.DataFrame: same DataFrame with added 'Score' column
    """
    df = df.copy()
    normalized = []

    for col, direction in columns.items():
        col_values = df[col].astype(float)


        if direction == 'max':
            norm = (col_values - col_values.min()) / (col_values.max() - col_values.min())
        elif direction == 'min':
            norm = (col_values.max() - col_values) / (col_values.max() - col_values.min())
        else:
            raise ValueError(f"Invalid direction '{direction}' for column '{col}'")

        normalized.append(norm)

    df["Score"] = sum(normalized).round(2)  # optionally: / len(columns) for avg
    return df

def quantitative_analysis(metrics, env):

    formatted_metrics = {'CMU-Net (' + ablation_mod_short(get_ablation_mod(dirname)) + ')': results for dirname,results in metrics.items()}

    baseline_dirname = f"{DIR}/{get_env(env)}/pickle/{env}_logits_{'cnn' if env == 'pong' else 'mlp'}"
    baseline_data = load_runs(baseline_dirname)
    baseline_reward_curves = [[s[1] for s in d['reward_curve'] ] for d in baseline_data]
    baseline_stacked_reward_curves = np.stack(baseline_reward_curves)

    steps = len(baseline_stacked_reward_curves[0])

    if env == 'pong':
        threshold = 19.5
    elif env == 'pendulum':
        threshold = -200.0
    elif env == 'acrobot':
        threshold = -100

    baseline_metrics = compute_metrics(baseline_stacked_reward_curves, threshold=threshold)

    formatted_metrics['Logits (Baseline)'] = baseline_metrics


    df = pd.DataFrame.from_dict(data=formatted_metrics, orient='index')
    df.index.name = 'Method'
    df.reset_index(inplace=True)

    column_directions = {
        'Convergence': 'min',
        'STD (5 runs)': 'min',
        'Instability': 'min',
        'AUC': 'max',
    }

    df = add_score_column(df, column_directions, steps)

    column_directions['Score'] = 'max'

    [bold_best(df, k, v, steps) for k,v in column_directions.items()]


    # Generate LaTeX
    latex_code = df.to_latex(index=False, escape=False)  # escape=False to allow LaTeX in cells

    # Insert \midrule before Baseline
    lines = latex_code.splitlines()
    for i, line in enumerate(lines):
        if "Baseline" in line:
            lines.insert(i, "\\midrule")
            break

    latex_code = "\n".join(lines)

    print(env)
    print(latex_code)
    print(df)


def main():
    
    # main_plots()
    pendulum_metrics = ablation_plots('pendulum')
    acrobot_metrics = ablation_plots('acrobot')
    pong_metrics = ablation_plots('pong')

    quantitative_analysis(pendulum_metrics, 'pendulum')
    quantitative_analysis(acrobot_metrics, 'acrobot')
    quantitative_analysis(pong_metrics, 'pong')




if __name__ == '__main__':
    main()