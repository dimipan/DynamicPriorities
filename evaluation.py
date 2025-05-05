import os
import pickle
import time
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter  # Consider using this for smoothing
from robot_utils import _get_action_name


def evaluate_trained_policy(agent, learned_policy_files, test_episodes=1):
    """
    Evaluate Q-learning agent (supports both flat and hierarchical agents)
    Args:
        agent: The Q-learning agent (flat or hierarchical)
        learned_policy_files: Dict for hierarchical (with 'manager' and 'workers' keys) or string for flat
        test_episodes: Number of test episodes to run
    """
    try:
        if isinstance(learned_policy_files, dict):
            # Hierarchical agent
            print("\nLoading Q-tables for hierarchical agent:")
            manager_policy = np.load(learned_policy_files['manager'])
            print(f"Manager Q-table loaded from: {learned_policy_files['manager']}")
            
            worker_policies = {}
            for option, filepath in learned_policy_files['workers'].items():
                worker_policies[option] = np.load(filepath)
                print(f"Worker {option} Q-table loaded from: {filepath}")
        else:
            # Flat agent
            print(f"\nLoading Q-table from: {learned_policy_files}")
            learned_policy = np.load(learned_policy_files)
    except FileNotFoundError as e:
        print(f"Error: Q-table not found - {e}")
        return None

    total_rewards = []
    successes = 0  # Track number of successful episodes

    for episode in range(test_episodes):
        obs, _ = agent.env.reset(seed=episode)
        state = agent._get_state(obs)
        terminated = False
        total_return, step, collision_count = 0, 0, 0
        collisions = []

        while not terminated:
            # Flat agent
            action = np.argmax(learned_policy[state])
            next_obs, reward, terminated, _, _ = agent.env.step(action)
            next_state = agent._get_state(next_obs)
            print(f"Step {step+1}: || State={state} || Action={_get_action_name(None, action)} || "
                    f"Reward={reward} || Next State={next_state} || Done={terminated}")

            total_return += reward
            time.sleep(0.5)  # Optional: slow down the evaluation for better visualization
            state = next_state
            step += 1

        total_rewards.append(total_return)
        
        # Check if task was successful based on has_saved state
        is_successful = state[3] == 1  # state[3] is has_saved
        
        if is_successful:
            successes += 1
            print(colored(f"\nTest {episode}: SUCCESS! Robot successfully completed the rescue task", "green"))
        else:
            print(colored(f"\nTest {episode}: FAILURE! Robot did not complete the rescue task", "red"))
            
        print(f"Episode finished after {step} steps with total reward {total_return}")
        print(f"Collisions: {collision_count} at positions {collisions}")

    avg_reward = sum(total_rewards) / test_episodes
    success_rate = (successes / test_episodes) * 100
    
    print(f"\nResults over {test_episodes} testing episodes:")
    print(f"Average reward: {avg_reward:.2f}")
    print(colored(f"Success rate: {success_rate:.1f}%", "green" if success_rate > 0 else "red"))
    
    return avg_reward, success_rate



## 1st option -- original function
def plot_accumulated_rewards(reward_list, labels, colors=None, window_size=100, 
                             figsize=(12, 6), save_path=None, use_savgol=False):
    """
    Plot accumulated rewards for multiple agents with smoothing.
    
    Parameters:
    - reward_list: list of arrays (or lists), each of shape (num_runs, num_episodes)
    - labels: list of strings, labels for each agent
    - colors: list of colors for each agent's plot (optional)
    - window_size: int, window size for smoothing
    - figsize: tuple, figure size in inches
    - save_path: string, path to save the figure (None = don't save)
    - use_savgol: bool, whether to use Savitzky-Golay filter instead of rolling mean
    
    Returns:
    - fig, ax: matplotlib figure and axis objects
    """
    # Set style similar to your existing plots
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'lines.linewidth': 2.5,
    })
    
    # Create default colors if not provided
    if colors is None:
        colors = plt.cm.tab10.colors[:len(reward_list)]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    
    # Iterate over rewards, labels, and colors simultaneously.
    for rewards, label, color in zip(reward_list, labels, colors):
        # Skip entry if rewards is None or empty.
        if rewards is None or (hasattr(rewards, '__len__') and len(rewards) == 0):
            continue
        
        rewards_array = np.array(rewards)
        # Skip if the converted array is empty.
        if rewards_array.size == 0:
            continue
        
        # Calculate mean and std across runs for each episode.
        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)
        episodes = np.arange(len(mean_rewards))
        
        # Apply smoothing.
        if use_savgol and len(mean_rewards) > window_size:
            polyorder = min(3, window_size - 1)
            smooth_mean = savgol_filter(mean_rewards, window_size, polyorder)
            smooth_std = savgol_filter(std_rewards, window_size, polyorder)
        else:
            smooth_mean = pd.Series(mean_rewards).rolling(window=window_size, min_periods=1).mean().values
            smooth_std = pd.Series(std_rewards).rolling(window=window_size, min_periods=1).mean().values
        
        # Plot the smoothed mean line.
        ax.plot(episodes, smooth_mean, color=color, label=label, zorder=3)
        # Plot the shaded area for standard deviation.
        ax.fill_between(episodes, smooth_mean - smooth_std, smooth_mean + smooth_std,
                        color=color, alpha=0.07, zorder=2)
    
    # Set labels, title, and style for the plot.
    ax.set_xlabel('Episodes', fontweight='bold')
    ax.set_ylabel('Accumulated Reward', fontweight='bold')
    ax.set_title('Agent Performance Comparison', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(True, which='major', alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', length=6, width=1.5)
    
    # Add legend if more than one label exists.
    if len(labels) > 1:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)
    
    plt.tight_layout()
    
    # Uncomment these lines to save the figure if desired.
    # if save_path:
    #     plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
    #     plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    return fig, ax



def save_training_results(base_name, rewards, steps, metrics, save_dir='saved_results'):
    """
    Save training results to files.
    
    Args:
        base_name (str): Base name for the saved files (e.g., 'flat', 'att')
        rewards (list): List of rewards from training runs
        steps (list): List of steps from training runs
        metrics (list): List of metrics from training runs
        save_dir (str): Directory to save results
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as pickle files (preserves exact structure)
    with open(f'{save_dir}/{base_name}_rewards.pkl', 'wb') as f:
        pickle.dump(rewards, f)
    
    with open(f'{save_dir}/{base_name}_steps.pkl', 'wb') as f:
        pickle.dump(steps, f)
    
    with open(f'{save_dir}/{base_name}_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    # Save as numpy arrays (for numerical analysis)
    np.save(f'{save_dir}/{base_name}_rewards.npy', np.array(rewards))
    np.save(f'{save_dir}/{base_name}_steps.npy', np.array(steps))
    
    print(f"Results for {base_name} agent saved to '{save_dir}' directory.")


def load_training_results(base_name, data_type='all', file_format='pickle', save_dir='saved_results'):
    """
    Load training results from files.
    
    Args:
        base_name (str): Base name for the files to load (e.g., 'flat', 'att')
        data_type (str): Type of data to load ('rewards', 'steps', 'metrics', or 'all')
        file_format (str): Format to load ('pickle' or 'numpy')
        save_dir (str): Directory containing saved results
    
    Returns:
        The loaded data or a dictionary containing all data types if data_type='all'
    """
    result = {}
    
    if data_type in ['rewards', 'all']:
        try:
            if file_format == 'pickle':
                with open(f'{save_dir}/{base_name}_rewards.pkl', 'rb') as f:
                    result['rewards'] = pickle.load(f)
            else:  # numpy
                result['rewards'] = np.load(f'{save_dir}/{base_name}_rewards.npy')
        except FileNotFoundError:
            print(f"{base_name}_rewards file not found.")
            result['rewards'] = None
    
    if data_type in ['steps', 'all']:
        try:
            if file_format == 'pickle':
                with open(f'{save_dir}/{base_name}_steps.pkl', 'rb') as f:
                    result['steps'] = pickle.load(f)
            else:  # numpy
                result['steps'] = np.load(f'{save_dir}/{base_name}_steps.npy')
        except FileNotFoundError:
            print(f"{base_name}_steps file not found.")
            result['steps'] = None
    
    if data_type in ['metrics', 'all']:
        # Metrics are only handled with pickle here.
        try:
            with open(f'{save_dir}/{base_name}_metrics.pkl', 'rb') as f:
                result['metrics'] = pickle.load(f)
        except FileNotFoundError:
            print(f"{base_name}_metrics file not found.")
            result['metrics'] = None
    
    # Return specific data type if requested, otherwise return the dictionary
    if data_type != 'all':
        return result.get(data_type)
    return result

