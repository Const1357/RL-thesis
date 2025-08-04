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

from matplotlib.ticker import FuncFormatter

from matplotlib.colors import Normalize
from matplotlib import cm
import imageio


_original_plot = plt.plot  # backup the original

def patched_plot(*args, **kwargs):
    lines = _original_plot(*args, **kwargs)
    ax = plt.gca()

    def format_ticks(x, pos):
        x_rounded = round(x / 1000.0) * 1000  # round to nearest 1k
        if x_rounded >= 1_000_000:
            val = x_rounded / 1_000_000
            return f"{val:.1f}M" if val % 1 else f"{int(val)}M"
        elif x_rounded >= 1000:
            return f"{int(x_rounded / 1000)}k"
        else:
            return f"{int(x_rounded)}"

    ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
    return lines

plt.plot = patched_plot




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

colors = sns.color_palette('bright', 9).as_hex()

colors_dict = {
    'logits' : colors[0],
    'GNN' : colors[1],
    'GNN_K' : colors[2],
    'CMU' : colors[3],

    'no_mod' : colors[0],
    'entropy' : colors[1],
    'noise' : colors[2],
    'noise_entropy' : colors[3],
}

# per environment constants:
CONSTANTS = {
    'CartPole-v1': {
        'num_episodes' : 150,
        'max_reward' : 500,
        'min_reward' : 0,
        'xtick_interval': 200000,   # tick every 200k steps
        'ytick_interval': 100,
        'top_offset' : 15,
        'bot_offset' : -15,
        'left_offset' : -12000,
        'right_offset' : 2000,
        'threshold' : 475,
    },

    'Pendulum-v1': {
        'num_episodes' : 500,
        'max_reward' : 0,
        'min_reward' : -1500,
        'xtick_interval': 500000,   # tick every 500k steps
        'ytick_interval': 300,
        'top_offset' : 40,
        'bot_offset' : -40,
        'left_offset' : -40000,
        'right_offset' : 10000,
        'threshold' : -200,
    },

    'Acrobot-v1': {
        'num_episodes' : 300,
        'max_reward' : 0,
        'min_reward' : -500,
        'xtick_interval': 200000,   # tick every 200k steps
        'ytick_interval': 100,
        'top_offset' : 15,
        'bot_offset' : -15,
        'left_offset' : -12000,
        'right_offset' : 2000,
        'threshold' : -100,
    },

    'ALE/Pong-v5': {
        'num_episodes' : 1000,
        'max_reward' : 21,
        'min_reward' : -21,
        'xtick_interval': 250000,   # tick every 250k steps
        'ytick_interval': 3,
        'top_offset' : 1.26,
        'bot_offset' : -1.26,
        'left_offset' : -20000,
        'right_offset' : 10000,
        'threshold' : 19.5,
    }
}


def round_to_leading_digit(x):
    if x == 0:
        return 0
    magnitude = int(np.floor(np.log10(abs(x))))
    return round(x, -magnitude)

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
environments = ['CartPole-v1', 'Pendulum-v1', 'ALE/Pong-v5', 'Acrobot-v1']


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

def mod_short(mod: str):
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
    elif ptype == 'CMU':
        return 'CMU-Net'
    else:
        return f"{ptype}"

def get_individual_title(filename, ablation=False):

    env = get_env(filename)
    if ablation:
        ptype = 'CMU'
        mod = mod_short(get_ablation_mod(filename))
    else:
        ptype = get_policy_type(filename)
        mod = get_mod(filename)

    return f"{env_title[env]} | {clean_policy_type(ptype)} ({mod if ablation else clean_mod(mod)})"

def get_individual_title_no_env(filename, ablation=False):

    if ablation:
        ptype = 'CMU'
        mod = mod_short(get_ablation_mod(filename))
    else:
        ptype = get_policy_type(filename)
        mod = get_mod(filename)

    return f"{clean_policy_type(ptype)} | {mod if ablation else clean_mod(mod)}"

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

    fig = plt.figure(figsize=(7.5,5))

    plt.ylim(top=C['max_reward'] + C['top_offset'], bottom=C['min_reward'] + C['bot_offset'])
    plt.xlim(left=C['left_offset'], right=data[0]['total_steps']+C['right_offset'])
    plt.xticks(range(0, data[0]['total_steps']+1, round_to_leading_digit(data[0]['total_steps'])), fontsize=14)
    plt.yticks(range(C['min_reward'],C['max_reward']+1, C['ytick_interval']), fontsize=14)

    for i,rc in enumerate(reward_curves):
        if i == 0:
            plt.plot(x, rc, color=colors[2], alpha=0.27, linewidth=1.25, label=f"Reward (individual runs)")
        else:
            plt.plot(x, rc, color=colors[2], alpha=0.27, linewidth=1.25)

    plt.plot(x, norm_stacked_entropy_curves, color = colors[4], linewidth=1.5, label='Entropy (normalized)')

    plt.plot(x, mean_reward_curve, color=colors[0], linewidth=2, label='Mean Reward')
    plt.axhline(C['threshold'], linewidth=1.25, color='black', alpha=0.6, linestyle='dotted')


    std_fill_y1 = mean_reward_curve-std_reward_curve
    std_fill_y2 = mean_reward_curve+std_reward_curve
    plt.fill_between(
        x, std_fill_y1, std_fill_y2, color=colors[0], alpha=0.1, label='Reward Standard Deviation', edgecolor='none')    
            
    plt.xlabel('')
    plt.ylabel('')

    title = get_individual_title_no_env(dirname, ablation=ablation_plot)
    # title = f"{get_individual_title(dirname, ablation=ablation_plot)}"
    if (env == 'ALE/Pong-v5' or env == 'Acrobot-v1') and 'Logits (Baseline)' in title:
        title = re.sub(r'\s*\([^()]*\)\s*$', '', title)
      
    plt.suptitle(title, fontweight='bold', fontsize=24)

    # fig.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 0.95),
    #     ncol=2, frameon=False, fontsize=14)
    plt.tight_layout()
    # plt.subplots_adjust(top=1.0)

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


    fig = plt.figure(figsize=(7.5,5))
    plt.ylim(top=1.1, bottom=-0.02)
    plt.yticks(ticks=np.arange(0, 40+1, 5)/40, labels=[])
    plt.xlim(left=C['left_offset'], right=data[0]['total_steps']+C['right_offset'])
    plt.xticks(range(0, data[0]['total_steps']+1, round_to_leading_digit(data[0]['total_steps'])), fontsize=14)

    num_episodes = data[0]['total_steps']//data[0]['update_steps']
    log_frequency = data[0]['log_steps']//data[0]['update_steps']

    selected_indices = np.arange(0, data[0]['total_steps'], log_frequency)  # log_frequency
    selected_indices = selected_indices[selected_indices < num_episodes] # num_episodes

    policy_loss_curves = [np.array(d['policy_loss_curve']) for d in data]
    ppo_loss_curves = [np.array(d['ppo_loss_curve']) for d in data]
    alignment_loss_curves = [np.array(d['alignment_loss_curve']) for d in data]
    penalty_loss_curves = [np.array(d['penalty_loss_curve']) for d in data]
    margin_loss_curves = [np.array(d['margin_loss_curve']) for d in data]


    def normalize_mean_std(curves):

        curves = np.array(curves)  # shape: (N, T)

        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)


        # normalize based on the mean curve's min and max
        min_val = mean_curve.min()
        max_val = mean_curve.max()
        scale = max_val - min_val if max_val > min_val else 1.0

        normalized_mean = (mean_curve - min_val) / scale
        normalized_std  = std_curve / scale  # scale std consistently

        return normalized_mean, normalized_std


    mean_policy_loss_curve, _ =  normalize_mean_std(policy_loss_curves)
    mean_ppo_loss_curve, _ =  normalize_mean_std(ppo_loss_curves)
    mean_alignment_loss_curve, std_alignment =  normalize_mean_std(alignment_loss_curves)
    mean_penalty_loss_curve, std_penalty =  normalize_mean_std(penalty_loss_curves)
    mean_margin_loss_curve, std_margin =  normalize_mean_std(margin_loss_curves)

    mean_policy_loss_curve = smooth(mean_policy_loss_curve, 1.25)
    mean_ppo_loss_curve = smooth(mean_ppo_loss_curve, 1.25)
    
    smooth_factor=2

    mean_alignment_loss_curve = smooth(mean_alignment_loss_curve, smooth_factor)
    mean_penalty_loss_curve = smooth(mean_penalty_loss_curve, smooth_factor)
    mean_margin_loss_curve = smooth(mean_margin_loss_curve, smooth_factor)

    std_alignment = smooth(std_alignment, smooth_factor)
    std_penalty = smooth(std_penalty, smooth_factor)
    std_margin = smooth(std_margin, smooth_factor)
    


    reward_curves = [[s[1] for s in d['reward_curve'] ] for d in data]
    stacked_reward_curves = np.stack(reward_curves)
    mean_reward_curve = smooth(np.mean(stacked_reward_curves, axis=0), 0.75)


    entropy_curves = [[s[1] for s in d['entropy_curve'] ] for d in data]
    stacked_entropy_curves = np.stack(entropy_curves).mean(axis=0)
    norm_stacked_entropy_curves = (stacked_entropy_curves-stacked_entropy_curves.min())/(stacked_entropy_curves.max()-stacked_entropy_curves.min())*(C['max_reward'] - C['min_reward']) + C['min_reward']


    normalized_mean_reward_curve = (mean_reward_curve - mean_reward_curve.min())/(mean_reward_curve.max()-mean_reward_curve.min())
    normalized_entropy_curve = smooth((norm_stacked_entropy_curves-C['min_reward'])/(C['max_reward'] - C['min_reward']), 0.75)

    x = np.arange(0, data[0]['total_steps'], data[0]['log_steps'])
    _x = np.arange(0, data[0]['total_steps'], data[0]['update_steps'])

    plt.plot(_x, mean_ppo_loss_curve, label='PPO Loss',color=colors[0], alpha=1, linewidth=1.0)
    plt.plot(_x, mean_policy_loss_curve, label='Mixed Loss',color=colors[1], alpha=1, linewidth=1.0)

    if not np.allclose(mean_alignment_loss_curve, 0):
        plt.fill_between(_x, mean_alignment_loss_curve-std_alignment, mean_alignment_loss_curve+std_alignment, alpha=0.1, color=colors[2], edgecolor='none')
        plt.plot(_x, mean_alignment_loss_curve, label='Alignment Loss (smoothed)',color=colors[2], linewidth=2.0)

    if not np.allclose(mean_penalty_loss_curve, 0):
        plt.fill_between(_x, mean_penalty_loss_curve-std_penalty, mean_penalty_loss_curve+std_penalty, alpha=0.1, color=colors[3], edgecolor='none')
        plt.plot(_x, mean_penalty_loss_curve, label='Penalty Loss (smoothed)',color=colors[3], linewidth=2.0)

    if not np.allclose(mean_margin_loss_curve, 0):
        plt.fill_between(_x, mean_margin_loss_curve-std_margin, mean_margin_loss_curve+std_margin, alpha=0.1, color=colors[5], edgecolor='none')
        plt.plot(_x, mean_margin_loss_curve, label='Margin Loss (smoothed)',color=colors[5], linewidth=2.0)




    plt.plot(x, normalized_mean_reward_curve, color='black', label='reward', linewidth=1.5)
    plt.plot(x, normalized_entropy_curve, color=colors[4], label='entropy', linewidth=1.5)

    

    plt.xlabel('')
    # plt.ylabel('Normalized Axis', fontsize=16)
    plt.ylabel('')

    title = get_individual_title_no_env(dirname, ablation=ablation_plot)
    plt.suptitle(title, fontweight='bold', fontsize=24)
    # plt.suptitle(f"{env_title[env]} | {clean_policy_type(ptype)} ({mod_short(mod)})", fontweight='bold', fontsize=18)
    # plt.suptitle(f"{get_individual_title(dirname, ablation=ablation_plot)}", fontweight='bold', fontsize=18)
    
    # Create individual handles
    # handles = [
    #     Line2D([0], [0], color=colors[0], label='PPO Loss', linewidth=1),
    #     Line2D([0], [0], color=colors[1], label='Mixed Loss', linewidth=1),
    #     Line2D([0], [0], color=colors[2], label='Alignment Loss (smoothed)', linewidth=2.0),
    #     Line2D([0], [0], color=colors[3], label='Penalty Loss (smoothed)', linewidth=2.0),
    #     Line2D([0], [0], color=colors[5], label='Margin Loss (smoothed)', linewidth=2.0),
    #     Line2D([0], [0], color='black', label='reward', linewidth=1.5),
    #     Line2D([0], [0], color=colors[4], label='entropy', linewidth=1.5),
    # ]

    # Now split into 2 rows:
    # row1 = handles[:3]
    # row2 = handles[3:]

    # # Create first row of legend
    # fig.legend(
    #     handles=row2,
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, 0.95),  # adjust horizontal offset here
    #     ncol=2,
    #     frameon=False,
    #     fontsize=14
    # )

    # # Create second row of legend (centered manually)
    # fig.legend(
    #     handles=row1,
    #     loc='upper center',
    #     bbox_to_anchor=(0.505, 0.87),
    #     ncol=3,
    #     frameon=False,
    #     fontsize=14
    # )


    plt.tight_layout()
    # fig.subplots_adjust(top=0.8)  # move plot area down to make room for fig.legend

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







def gif_CMU_IC_evolution(dirname, load_from, ablation_plot=False):
    env = get_env(dirname)
    if ablation_plot:
        ptype = 'CMU'
        mod = get_ablation_mod(dirname)
    else:
        ptype = get_policy_type(dirname)
        mod = get_mod(dirname)

    data = load_runs(f"{load_from}{dirname}")
    C = CONSTANTS[env]

    dir_ablation = f"{DIR_ABLATION}_{flatten_env(env)}"
    savedir = f"{dir_ablation if ablation_plot else DIR}{'' if ablation_plot else '/'+env}/plots/IC_evolution"
    os.makedirs(savedir, exist_ok=True)
    gif_path = os.path.join(savedir, f"{dirname}_IC_evolution.gif")

    x = np.arange(0, data[0]['total_steps'], data[0]['log_steps'])
    milestone_idx = [int(p * len(x)) for p in [0.0, 0.02, 0.05, 0.2, 0.4, 0.60, 0.90]]

    intents_trajectory = data[0]['intent_evolution']
    confidences_trajectory = data[0]['confidence_evolution']
    probs_trajectory = data[0]['action_probabilities']
    selected_actions_trajectory = data[0]['selected_actions']


    # Find max episode length across milestones to pad or restrict
    max_episode_length = max(len(intents_trajectory[idx]) for idx in milestone_idx)

    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=1)

    # Precompute global min/max for intent normalization
    all_intents_raw = []
    for idx in milestone_idx:
        for t in range(len(intents_trajectory[idx])):
            all_intents_raw.append(np.array(intents_trajectory[idx][t]))
    max_intent = np.log(np.max([np.max(i) for i in all_intents_raw]) + 1e-9)
    min_intent = np.log(np.min([np.min(i) for i in all_intents_raw]) + 1e-9)

    num_actions = len(intents_trajectory[idx][0])

    with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
        for t in range(max_episode_length + 1):
            fig, axes = plt.subplots(
                nrows=len(milestone_idx),
                figsize=(7, 1.27 * len(milestone_idx)),
                sharex=True,
                gridspec_kw={'height_ratios': [0] + [1]*(len(milestone_idx)-1)}
            )

            # print(f"Step: {t +1 }")


            for i, idx in enumerate(milestone_idx):

                if i == 0:
                    # Legend
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
                    continue

                ax = axes[i]

                ax.set_yticks(np.arange(num_actions))
                ax.set_yticklabels([f"Action {j}" for j in range(num_actions)], fontsize=7)
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(num_actions, -1)
                ax.yaxis.grid(True, alpha=0.75)
                ax.set_ylabel(f"{idx}%", rotation=0, labelpad=30,
                            va='center', ha='right', fontsize=10, fontweight='bold')
                ax.text(0.1, 1.02, f"{idx}% Training Steps", transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=8)
                
                if t >= len(intents_trajectory[idx]):
                    # ax.clear()
                    # ax.set_xticks([])
                    # ax.set_xticks([])
                    ax.text(0.03, num_actions/2, f"Episode Terminated. Mean Reward (all sub-envs): {data[0]['reward_curve'][idx][1]:.2f}")
                    continue

                intents = np.array(intents_trajectory[idx][t])
                confidences = np.array(confidences_trajectory[idx][t])
                probs = np.array(probs_trajectory[idx][t])
                selected_action = selected_actions_trajectory[idx][t]

                norm_intents = (np.log(intents + 1e-9) - min_intent) / (max_intent - min_intent + 1e-7)
                action_indices = np.arange(len(intents))
                color_values = [cmap(norm(c)) for c in confidences]

                for j in range(len(intents)):
                    if j == selected_action:
                        continue
                    ax.hlines(y=action_indices[j], xmin=min(norm_intents[j], probs[j]), xmax=max(norm_intents[j], probs[j]),
                            colors=colors[3], alpha=0.5, linewidth=0.3)
                    ax.scatter(norm_intents[j], action_indices[j], s=40, c=[color_values[j]],
                            alpha=1.0, marker='o', edgecolors='black', linewidths=0.5)
                    ax.scatter(probs[j], action_indices[j], s=25, c=colors[3],
                            alpha=1, marker='+', linewidths=0.5)

                if 0 <= selected_action < len(intents):
                    ax.hlines(y=action_indices[selected_action], xmin=min(norm_intents[selected_action], probs[selected_action]),
                            xmax=max(norm_intents[selected_action], probs[selected_action]),
                            colors=colors[3], alpha=0.5, linewidth=0.3)
                    ax.scatter(norm_intents[selected_action], action_indices[selected_action], s=50,
                            c=[color_values[selected_action]], alpha=1.0, marker='D',
                            edgecolors='black', linewidths=0.6)
                    ax.scatter(probs[selected_action], action_indices[selected_action], s=40,
                            c=colors[3], alpha=1.0, marker='x', linewidths=0.5)


            axes[-1].set_xlabel('Intent (Normalized Logarithmic Scale)', fontsize=12)
            fig.suptitle("Intent–Confidence Evolution over Selected Milestones", fontsize=13, fontweight='bold')
            fig.text(0.5, 0.92, f"{get_individual_title(dirname, ablation=ablation_plot)}",
                    fontsize=9, ha='center', fontweight='bold')
            fig.text(0.99, 0.02, f"Step {t+1}", ha='right', va='bottom', fontsize=9, alpha=0.9)

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes[1:], orientation='vertical', pad=0.02, shrink=1.0, aspect=25)
            cbar.set_label("Confidence", fontsize=10, fontweight='bold')
            cbar.ax.invert_yaxis()

            # Save frame to buffer
            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]  # Drop alpha if not needed


            # fig.tight_layout()
            # fig.canvas.draw()
            # image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # all_frames.append(image)
            writer.append_data(image)
            plt.close(fig)

    print(f"Succesfully saved: {gif_path}")
    


def plot_reward_comparison(data):

    env = data['env']
    print(env)
    C = CONSTANTS[env]
    x = data['x']
    total_steps = data['total_steps']

    for ptype in ['GNN', 'GNN_K']:
        fig = plt.figure(figsize=(9,6))

        for mod in mods:
            curve = data[ptype][mod]['reward_curve']
            baseline_curve = data['logits'][mod]['reward_curve']

            std_low, std_high = data[ptype][mod]['std_curve']
            baseline_std_low, baseline_std_high = data['logits'][mod]['std_curve']

            plt.plot(x, smooth(curve, s=1), color=colors_dict[mod], label=f"{clean_policy_type(ptype)} ({clean_mod(mod)})", linewidth=2)
            plt.fill_between(x, smooth(std_low, 1), smooth(std_high, 1), color = colors_dict[mod], alpha = 0.1, edgecolor = 'none')

            plt.plot(x, smooth(baseline_curve, s=1), color = colors_dict[mod], label=f"Baseline ({clean_mod(mod)})", linestyle='--', linewidth=2.5)
            # plt.fill_between(x, baseline_std_low, baseline_std_high, color = colors_dict[mod], alpha = 0.2, edgecolor = 'none')

        plt.axhline(C['threshold'], linewidth=1.25, color='black', alpha=0.6, linestyle='dotted')

        if env == 'Pendulum-v1':
            plt.ylim(top=0 + C['top_offset'], bottom=-1250 + C['bot_offset'])
            plt.yticks(range(-1200, 0+1, 200), fontsize=14)
        else:
            plt.ylim(top=C['max_reward'] + C['top_offset'], bottom=C['min_reward'] + C['bot_offset'])
            plt.yticks(range(C['min_reward'],C['max_reward']+1, C['ytick_interval']), fontsize=14)

        # plt.ylim(top=C['max_reward'] + C['top_offset'], bottom=C['min_reward'] + C['bot_offset'])
        plt.xlim(left=C['left_offset'], right=total_steps+C['right_offset'])
        plt.xticks(range(0, total_steps+1, C['xtick_interval']), fontsize=14)
        # plt.yticks(range(C['min_reward'],C['max_reward']+1, C['ytick_interval']), fontsize=14)

        plt.suptitle(f'{env_title[env]} ({clean_policy_type(ptype)})', fontweight='bold', fontsize=18)
        plt.xlabel('Steps', fontsize=16)
        plt.ylabel('Reward', fontsize=16)
        # fig.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 0.95),
        #     ncol=2, frameon=False, fontsize=14)

        plt.tight_layout()
        # plt.subplots_adjust(top=0.75)


        savedir = f"{DIR}/{env}/plots/reward_curves"
        name = f"all_rewards_{flatten_env(env)}_{ptype}"
        save_and_close(fig, savedir, name)

    # for mod in mods:

    #     fig = plt.figure(figsize=(9,6))
    #     to_plot = data[mod]

    #     for ptype in policy_types:

    #         curve = to_plot[ptype]['reward_curve'] if ptype in to_plot.keys() and 'reward_curve' in to_plot[ptype].keys() else None
    #         if curve is not None:
    #             std_low, std_high = to_plot[ptype]['std_curve'] 
    #             plt.plot(x, curve, color=colors_dict[ptype], label=clean_policy_type(ptype))
    #             plt.fill_between(x, std_low, std_high, color = colors_dict[ptype], alpha = 0.2, edgecolor = 'none')

    #     plt.ylim(top=C['max_reward'] + C['top_offset'], bottom=C['min_reward'] + C['bot_offset'])
    #     plt.xlim(left=C['left_offset'], right=total_steps+C['right_offset'])
    #     plt.xticks(range(0, total_steps+1, C['xtick_interval']))
    #     plt.yticks(range(C['min_reward'],C['max_reward']+1, C['ytick_interval']))

    #     plt.suptitle(f'Rewards: {env_title[env]} ({clean_mod(mod)})', fontweight='bold')
    #     plt.xlabel('Steps')
    #     plt.ylabel('Reward')
    #     plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.08),
    #         ncol=4, frameon=False, fontsize='small')
    #     plt.tight_layout()



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
            plt.plot(x, smooth(curve, 0.75), color=colors[i], label=clean_mod(mod), linewidth=2)
            # plt.fill_between(x, smooth(std_low, 0.75), smooth(std_high, 0.75), color = colors[i], alpha = 0.1, edgecolor = 'none')

    plt.plot(x, smooth(baseline_mean_reward_curve, 0.75), color='black', label='Logits (Baseline)', linestyle='--', linewidth=2)

    plt.axhline(C['threshold'], linewidth=1.25, color='black', alpha=0.6, linestyle='dotted')


    if env == 'pendulum':
        plt.ylim(top=0 + C['top_offset'], bottom=-1250 + C['bot_offset'])
        plt.yticks(range(-1200, 0+1, 200), fontsize=14)
    else:
        plt.ylim(top=C['max_reward'] + C['top_offset'], bottom=C['min_reward'] + C['bot_offset'])
        plt.yticks(range(C['min_reward'],C['max_reward']+1, C['ytick_interval']), fontsize=14)
    


    plt.xlim(left=C['left_offset'], right=total_steps+C['right_offset'])
    plt.xticks(range(0, total_steps+1, C['xtick_interval']), fontsize=14)

    plt.suptitle(f'Rewards: {env_title[get_env(env)]} | CMU-Net', fontweight='bold', fontsize=18)
    plt.xlabel('Steps', fontsize=16)
    # plt.ylabel('Reward', fontsize=16)
    plt.ylabel('')
    plt.tight_layout()
    # fig.subplots_adjust(right=0.75)
    # fig.legend(title=None, loc='upper right', bbox_to_anchor=(1.10, 0.75),
    #     ncol=1, frameon=False, fontsize=14)

    dir_ablation = f"{DIR_ABLATION}_{env}"
    savedir = f"{dir_ablation}/plots/reward_curves"
    name = f"all_rewards_ablation_{env}"
    save_and_close(fig, savedir, name)



def main_plots():

    metrics = {}

    for environment in environments:

        print(environment)
        flat_env = flatten_env(environment)
        
        if flat_env != 'pong' and flat_env != 'acrobot':
            metrics[flat_env] = {}

        # Normal Plots:
        load_from = f"{DIR}/{environment}/pickle/"
        dirnames = os.listdir(load_from)

        data = {
            'env' : environment,
            'x' : None,
            'logits' : {
                'no_mod' : {},
                'entropy' : {},
                'noise' : {},
                'noise_entropy' : {},
            },
            'GNN' : {
                'no_mod' : {},
                'entropy' : {},
                'noise' : {},
                'noise_entropy' : {},
            },
            'GNN_K' : {
                'no_mod' : {},
                'entropy' : {},
                'noise' : {},
                'noise_entropy' : {},
            },
        }

        if flat_env != 'pong' and flat_env != 'acrobot':
            metrics[flat_env]['logits'] = {}
            metrics[flat_env]['GNN'] = {}
            metrics[flat_env]['GNN_K'] = {}

        for dirname in dirnames:

            env, ptype, mod, x, total_steps, mean_reward_curve, std_fill_y1, std_fill_y2, stacked_reward_curves = plot_individual_reward_curves(dirname, load_from)
            
            if flat_env != 'pong' and flat_env != 'acrobot':
                data['x'] = x
                data['total_steps'] = total_steps
                data[ptype][mod]['reward_curve'] = mean_reward_curve
                data[ptype][mod]['std_curve'] = (std_fill_y1, std_fill_y2)


                if flatten_env(env) == 'cartpole':
                    threshold = 475.0
                elif flatten_env(env) == 'pendulum':
                    threshold = -200.0

                metrics[flatten_env(env)][ptype][mod] = compute_metrics(stacked_reward_curves, threshold=threshold)
        if flat_env != 'pong' and flat_env != 'acrobot':    
            plot_reward_comparison(data)

    return metrics


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
        # gif_CMU_IC_evolution(dirname, load_from, ablation_plot=True)  # GIF generation
        # continue
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

def quantitative_analysis_CMU(metrics, env):

    formatted_metrics = {'CMU-Net (' + mod_short(get_ablation_mod(dirname)) + ')': results for dirname,results in metrics.items()}

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


def quantitative_analysis_main(metrics):




    for env in ['cartpole', 'pendulum']:

        dfs = []

        for ptype in ['GNN', 'GNN_K']:

            formatted_metrics = {f"{clean_policy_type(ptype)} (" + mod_short(mod) + ')': results for mod,results in metrics[env][ptype].items()}

            # baseline_dirname = f"{DIR}/{get_env(env)}/pickle/{env}_logits_{'cnn' if env == 'pong' else 'mlp'}"
            # baseline_data = load_runs(baseline_dirname)
            # baseline_reward_curves = [[s[1] for s in d['reward_curve'] ] for d in baseline_data]
            # baseline_stacked_reward_curves = np.stack(baseline_reward_curves)

            # steps = len(baseline_stacked_reward_curves[0])
            if env == 'cartpole':
                steps = 50
            elif env == 'pendulum':
                steps = 100

            # baseline_metrics = compute_metrics(baseline_stacked_reward_curves, threshold=threshold)

            baseline_metrics = metrics[env]['logits']

            for mod in mods:
                formatted_metrics[f"Baseline ({mod_short(mod)})"] = baseline_metrics[mod]
            
            for mod in mods:
                
                short_mod = f"({mod_short(mod)})"

                mod_metrics = {k : v for k,v in formatted_metrics.items() if short_mod in k}

                df = pd.DataFrame.from_dict(data=mod_metrics, orient='index')
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

                dfs.append(df)

        df = pd.concat(dfs)

        # Generate LaTeX
        latex_code = df.to_latex(index=False, escape=False)  # escape=False to allow LaTeX in cells

        lines = latex_code.splitlines()

        start_idx = next(i for i, line in enumerate(lines) if r'\toprule' in line) +2
        end_idx = next(i for i, line in enumerate(lines) if r'\bottomrule' in line)

        # Insert \midrule every 2 rows (starting from data start)
        data_lines = lines[start_idx:end_idx]
        new_data_lines = []
        for i, line in enumerate(data_lines):
            new_data_lines.append(line)
            if (i + 1) % 2 == 1 and (i + 1) != len(data_lines) and (i + 1) != 1:
                new_data_lines.append(r'\midrule')

        # Reconstruct final LaTeX
        final_lines = lines[:start_idx] + new_data_lines + lines[end_idx:]
        final_latex = "\n".join(final_lines)

        print(env)
        print(final_latex)
        print(df)


    # print(df)
    # print(final_latex)

def make_gaussian_comparison_legend():
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=2, label='GNN (No Modification)'),
        Line2D([0], [0], color=colors[0], lw=2, label='Baseline (No Modification)', linestyle='--'),

        Line2D([0], [0], color=colors[1], lw=2, label='GNN (Entropy)'),
        Line2D([0], [0], color=colors[1], lw=2, label='Baseline (Entropy)', linestyle='--'),

        Line2D([0], [0], color=colors[2], lw=2, label='GNN-K (Noise)'),
        Line2D([0], [0], color=colors[2], lw=2, label='Baseline (Noise)', linestyle='--'),

        Line2D([0], [0], color=colors[3], lw=2, label='GNN-K (Noise + Entropy)'),
        Line2D([0], [0], color=colors[3], lw=2, label='Baseline (Noise + Entropy)', linestyle='--'),
    ]

    # Create a blank figure and hide axes
    fig = plt.figure(figsize=(12,2))
    plt.axis('off')

    # Place the legend in the center of the figure
    plt.suptitle("Reward Curves for Modifications of Gaussian Networks", fontsize=40, fontweight='bold')
    fig.legend(handles=legend_elements, loc='lower center', frameon=False, fontsize=28, ncol=4, bbox_to_anchor=(0.5,-0.15))

    # Tight layout and save (or display)
    plt.tight_layout()
    plt.savefig("legends/legend_only_gnn.svg", bbox_inches='tight')  # vector-friendly
    # plt.show()
    plt.close()

def make_aux_comparison_legend():
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=2, label='Alignment'),
        Line2D([0], [0], color=colors[1], lw=2, label='Penalty'),
        Line2D([0], [0], color=colors[2], lw=2, label='Margin'),
        Line2D([0], [0], color=colors[3], lw=2, label='Alignment + Penalty'),
        Line2D([0], [0], color=colors[4], lw=2, label='Alignment + Margin'),
        Line2D([0], [0], color=colors[5], lw=2, label='Penalty + Margin'),
        Line2D([0], [0], color=colors[6], lw=2, label='Alignment + Penalty + Margin'),
        Line2D([0], [0], color=colors[7], lw=2, label='No modification'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Logits (Baseline)'),

    ]

    # Create a blank figure and hide axes
    fig = plt.figure(figsize=(9,6))
    plt.axis('off')

    # Place the legend in the center of the figure
    fig.legend(handles=legend_elements, loc='center', frameon=False, fontsize=24)

    # Tight layout and save (or display)
    plt.tight_layout()
    plt.savefig("legends/legend_only.svg", bbox_inches='tight')  # vector-friendly
    # plt.show()
    plt.close()

def make_individual_reward_legend(env):

    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=2, label='Mean Reward (5 runs)'),
        Line2D([0], [0], color=colors[2], lw=1.25, label='Reward (Individual run)'),
        Line2D([0], [0], color=colors[4], lw=1.5, label='Mean Entropy (5 runs)'),
    ]

    # Create a blank figure and hide axes
    fig = plt.figure(figsize=(9,1))
    plt.axis('off')

    # Place the legend in the center of the figure
    fig.legend(handles=legend_elements, loc='center', frameon=False, fontsize=24, ncol=3, bbox_to_anchor=(0.5,0.1))
    plt.suptitle(f"Individual Reward Curves per Configuration for {env}", fontweight='bold', fontsize=32)

    # Tight layout and save (or display)
    plt.tight_layout()
    plt.savefig(f"legends/legend_only_individual{env}.svg", bbox_inches='tight')  # vector-friendly
    # plt.show()
    plt.close()

def make_policy_loss_legend(env):
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=1, label='PPO Loss'),
        Line2D([0], [0], color=colors[1], lw=1, label='Mixed Loss'),
        Line2D([0], [0], color=colors[2], lw=2, label='Alignment Loss'),
        Line2D([0], [0], color=colors[3], lw=2, label='Penalty Loss'),
        Line2D([0], [0], color=colors[5], lw=2, label='Margin Loss'),
        Line2D([0], [0], color='black', lw=1.5, label='Reward'),
        Line2D([0], [0], color=colors[4], lw=1.5, label='Entropy'),
    ]

    # Create a blank figure and hide axes
    fig = plt.figure(figsize=(9,1))
    plt.axis('off')

    row1 = legend_elements[:3]
    row2 = legend_elements[3:]


    fig.legend(
        handles=row1,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.75),
        ncol=3,
        frameon=False,
        fontsize=16
    )

    fig.legend(
        handles=row2,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.4),
        ncol=4,
        frameon=False,
        fontsize=16
    )

    # Place the legend in the center of the figure
    plt.suptitle(f"Loss Curves per Configuration for {env}", fontweight='bold', fontsize=24)

    # Tight layout and save (or display)
    plt.tight_layout()
    plt.savefig(f"legends/legend_only_policy{env}.svg", bbox_inches='tight')  # vector-friendly
    # plt.show()
    plt.close()

def make_legends():
    make_aux_comparison_legend()
    make_gaussian_comparison_legend()
    
    make_individual_reward_legend('Cartpole-v1')
    make_individual_reward_legend('Pendulum-v1')
    make_individual_reward_legend('Acrobot-v1')
    make_individual_reward_legend('Pong-v5')

    make_policy_loss_legend('Pendulum-v1')
    make_policy_loss_legend('Acrobot-v1')
    make_policy_loss_legend('Pong-v5')

def main():
    
    # main_metrics = main_plots()
    
    # quantitative_analysis_main(main_metrics)

    
    # pendulum_metrics = ablation_plots('pendulum')
    # acrobot_metrics = ablation_plots('acrobot')
    # pong_metrics = ablation_plots('pong')

    # quantitative_analysis_CMU(pendulum_metrics, 'pendulum')
    # quantitative_analysis_CMU(acrobot_metrics, 'acrobot')
    # quantitative_analysis_CMU(pong_metrics, 'pong')

    make_legends()




if __name__ == '__main__':
    main()