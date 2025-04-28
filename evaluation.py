import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.signal import savgol_filter  # Consider using this for smoothing
from termcolor import colored
from enum import Enum
from pathlib import Path
from typing import Dict, Union, Optional
from agents import (
    QLearningAgentFlat, 
    QLearningAgentFlatLLM, 
    QLearningAgentFlatAttention,
    QLearningAgentFlatActionToggle
)
from hierarchical_agents import (
    QLearningAgentHierarchical,
    QLearningAgentHierarchicalLLM,
    QLearningAgentHierarchicalAttention,
    QLearningAgentHierarchicalActionToggle
)
from environment_sar import SARrobotEnv
from robot_utils import RunningParameters, RobotOption, _get_action_name


class AgentType(Enum):
    FLAT = "flat"
    FLAT_LLM = "flat_llm"
    FLAT_ATTENTION = "flat_attention"  ## POLICY SHAPING
    FLAT_ACTION_TOGGLE = "flat_action_toggle" ## ACTION SHAPING
    HIERARCHICAL = "hierarchical"
    HIERARCHICAL_LLM = "hierarchical_llm"
    HIERARCHICAL_ATTENTION = "hierarchical_attention"  ## POLICY SHAPING
    HIERARCHICAL_ACTION_TOGGLE = "hierarchical_action_toggle"  ## ACTION SHAPING


class AgentEvaluationManager:
    def __init__(self):
        # Get the directory where this script is located and use it as base path
        self.base_path = Path(os.path.dirname(os.path.abspath(__file__)))
        
        self.agent_mapping = {
            AgentType.FLAT: QLearningAgentFlat,
            AgentType.FLAT_LLM: QLearningAgentFlatLLM,
            AgentType.FLAT_ATTENTION: QLearningAgentFlatAttention,
            AgentType.FLAT_ACTION_TOGGLE: QLearningAgentFlatActionToggle,
            AgentType.HIERARCHICAL: QLearningAgentHierarchical,
            AgentType.HIERARCHICAL_LLM: QLearningAgentHierarchicalLLM,
            AgentType.HIERARCHICAL_ATTENTION: QLearningAgentHierarchicalAttention,
            AgentType.HIERARCHICAL_ACTION_TOGGLE: QLearningAgentHierarchicalActionToggle
        }
        self.dir_mapping = {
            AgentType.FLAT: "flat",
            AgentType.FLAT_LLM: "FLAT-LLM",
            AgentType.FLAT_ATTENTION: "flat-ATT-PS",
            AgentType.FLAT_ACTION_TOGGLE: "flat-ATT-AS",
            AgentType.HIERARCHICAL: "HIER-manager",
            AgentType.HIERARCHICAL_LLM: "HIER-LLM-manager",
            AgentType.HIERARCHICAL_ATTENTION: "HIER-PS-manager",
            AgentType.HIERARCHICAL_ACTION_TOGGLE: "HIER-manager-AS"
        }
        self.params = RunningParameters()

    def create_environment(self, 
                         grid_rows: int = 4,
                         grid_cols: int = 4,
                         info_number_needed: int = 3,
                         sparse_reward: bool = False,
                         reward_shaping: bool = False,
                         attention: bool = False,
                         hierarchical: bool = False,
                         render_mode: str = 'None') -> SARrobotEnv:
        """Create environment with specified parameters."""
        return SARrobotEnv(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            info_number_needed=info_number_needed,
            sparse_reward=sparse_reward,
            reward_shaping=reward_shaping,
            attention=attention,
            hierarchical=hierarchical,
            render_mode=render_mode
        )

    def get_policy_path(self, agent_type: AgentType, episode: int) -> Union[str, Dict]:
        """Get policy file path(s) based on agent type."""
        base_dir = self.dir_mapping[agent_type]
        
        if agent_type.value.startswith("hierarchical"):
            return {
                'manager': str(self.base_path / "NON-SPARSE" / "results4x4_3info" / "policies" / base_dir / f"manager_q_table_episode_{episode}.npy"),
                'workers': {
                    0: str(self.base_path / "NON-SPARSE" / "results4x4_3info" / "policies" / base_dir / f"worker_0_q_table_episode_{episode}.npy"),
                    1: str(self.base_path / "NON-SPARSE" / "results4x4_3info" / "policies" / base_dir / f"worker_1_q_table_episode_{episode}.npy"),
                    2: str(self.base_path / "NON-SPARSE" / "results4x4_3info" / "policies" / base_dir / f"worker_2_q_table_episode_{episode}.npy")
                }
            }
        return str(self.base_path / "NON-SPARSE" / "results4x4_3info" / "policies" / base_dir / f"q_table_episode_{episode}.npy")

    def create_agent(self, agent_type: AgentType, env: SARrobotEnv) -> Union[QLearningAgentFlat, QLearningAgentHierarchical]:
        """Create agent with proper initialization parameters."""
        agent_class = self.agent_mapping[agent_type]
        
        if agent_type.value.startswith("hierarchical"):
            return agent_class(
                env=env,
                action_space_size=self.params.manager_action_space_size,
                ALPHA=self.params.ALPHA,
                GAMMA=self.params.GAMMA,
                EPSILON_MAX=self.params.EPSILON_MAX,
                DECAY_RATE=self.params.DECAY_RATE,
                EPSILON_MIN=self.params.EPSILON_MIN
            )
        else:
            return agent_class(
                env=env,
                ALPHA=self.params.ALPHA,
                GAMMA=self.params.GAMMA,
                EPSILON_MAX=self.params.EPSILON_MAX,
                DECAY_RATE=self.params.DECAY_RATE,
                EPSILON_MIN=self.params.EPSILON_MIN
            )

    def evaluate_agent(self, agent_type: AgentType, episode: int, 
                      env_params: Optional[Dict] = None) -> float:
        """Evaluate an agent with specified configuration."""
        # Use default environment parameters if none provided
        env_params = env_params or {}
        env_params['hierarchical'] = agent_type.value.startswith("hierarchical")
        
        # Create environment and agent
        env = self.create_environment(**env_params)
        agent = self.create_agent(agent_type, env)
        
        # Get policy files
        policy_files = self.get_policy_path(agent_type, episode)
        
        # Evaluate agent
        return evaluate_trained_policy(agent, policy_files, self.params.evaluation_runs)


def main_evaluation():
    manager = AgentEvaluationManager()
    
    env_params = {
        'grid_rows': 4,
        'grid_cols': 4,
        'info_number_needed': 3,
        'sparse_reward': False,
        'reward_shaping': False,
        'attention': False,
        'render_mode': 'None'
    }
    
    agent_types = [
        AgentType.FLAT,
        AgentType.HIERARCHICAL,
        AgentType.FLAT_ATTENTION,
        AgentType.HIERARCHICAL_ATTENTION,
        AgentType.FLAT_ACTION_TOGGLE,
        AgentType.HIERARCHICAL_ACTION_TOGGLE,
        AgentType.HIERARCHICAL_LLM,
        AgentType.FLAT_LLM,
        
    ]

    for agent_type in agent_types:
        try:
            print(f"\nEvaluating {agent_type.value} agent...")
            result = manager.evaluate_agent(
                agent_type=agent_type,
                episode=1000,  # Example episode number
                env_params=env_params
            )
            print(f"{agent_type.value} agent evaluation result: {result}")
        except Exception as e:
            print(f"Error evaluating {agent_type.value} agent: {str(e)}")


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
            if isinstance(learned_policy_files, dict):
                # Hierarchical agent
                current_option = np.argmax(manager_policy[state])
                # Use worker policy for current option
                action = np.argmax(worker_policies[current_option][state])
                next_obs, reward, terminated, _, _ = agent.env.step(action)
                next_state = agent._get_state(next_obs)
                print(f"Step {step+1}: || State={state} || Option={RobotOption(current_option).name} || "
                      f"Action={_get_action_name(current_option, action)} || Reward={reward} || "
                      f"Next State={next_state} || Done={terminated}")
            else:
                # Flat agent
                action = np.argmax(learned_policy[state])
                next_obs, reward, terminated, _, _ = agent.env.step(action)
                next_state = agent._get_state(next_obs)
                print(f"Step {step+1}: || State={state} || Action={_get_action_name(None, action)} || "
                      f"Reward={reward} || Next State={next_state} || Done={terminated}")

            # Check for collisions
            if tuple([state[0], state[1]]) in agent.env.sar_robot.GENERAL_FIRES_UNKNOWN_TO_THE_AGENT and state[2] == agent.env.sar_robot.info_number_needed:
                print(colored("Robot is in fire!", "red"))
                collision_count += 1
                collisions.append(tuple([state[0], state[1]]))

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


## 2nd option -- alternative to the above function
def plot_accumulated_rewards_v2(total_rewards_list, labels, colors, EPISODES, window_length=200, 
                        polyorder=3, save_path='new_trials', fig_size=(10, 6),
                        use_decorations=False):
    """
    Creates publication-style learning curve plots common in RL papers.
    
    Parameters:
    - total_rewards_list: list of arrays, each array has shape (num_runs, EPISODES)
    - labels: list of strings, labels for each agent
    - colors: list of colors for each agent's plot
    - EPISODES: int, number of episodes
    - window_length: int, window length for smoothing (must be odd)
    - polyorder: int, polynomial order for smoothing
    - save_path: string, base path for saving the plot
    - fig_size: tuple, figure size in inches
    """
    # Set the style to match typical RL paper plots
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
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.facecolor': '#F0F0F0',  # Set the background color to a light grey
    })

    # Define decorative elements if needed
    if use_decorations:
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), 
                      (0, (3, 5, 1, 5)), (0, (1, 1))]
        markers = ['o', 's', '^', 'v', 'D', '*', 'p', 'h']
    else:
        line_styles = ['-'] * len(labels)  # All solid lines
        markers = [None] * len(labels)     # No markers

    episodes = np.arange(0, EPISODES)
    fig, ax = plt.subplots(figsize=fig_size, dpi=150)

    # Plot each learning curve
    for idx, (rewards, label, color) in enumerate(zip(total_rewards_list, labels, colors)):
        rewards = np.array(rewards)
        mean_rewards = np.mean(rewards, axis=0)
        std_rewards = np.std(rewards, axis=0)
        
        # Smooth the curves
        smooth_mean = savgol_filter(mean_rewards, window_length, polyorder)
        smooth_std = savgol_filter(std_rewards, window_length, polyorder)
        
        # Plot mean line with decorations if flag is True
        if use_decorations:
            line = ax.plot(episodes, smooth_mean, 
                         label=label, 
                         color=color,
                         linestyle=line_styles[idx],
                         marker=markers[idx],
                         markersize=6,
                         markevery=1000,
                         zorder=3)
        else:
            # Original plotting style
            line = ax.plot(episodes, smooth_mean, 
                         label=label, 
                         color=color,
                         zorder=3)
        
        # Plot confidence intervals (shaded area)
        ax.fill_between(episodes,
                       smooth_mean - smooth_std,
                       smooth_mean + smooth_std,
                       color=color, alpha=0.07, zorder=2)


    # Customize the plot
    ax.set_xlabel('Training Episodes', fontweight='bold')
    ax.set_ylabel('Average Return', fontweight='bold')
    
    # Format axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Set axis limits with some padding
    ax.set_xlim(0, EPISODES)
    # ymin = min([np.min(r)/3 for r in total_rewards_list])
    # ymax = max([np.max(r) for r in total_rewards_list])
    # y_padding = (ymax - ymin) * 0.1
    # ax.set_ylim(ymin - y_padding, ymax + y_padding)
    
    # Customize grid
    ax.grid(True, which='major', axis='both', alpha=0.3, linestyle='--')
    
    # Format ticks
    ax.tick_params(axis='both', which='major', length=6, width=1.5)
    
    # Add legend
    legend = ax.legend(bbox_to_anchor=(1.02, 1), 
                      loc='upper left',
                      borderaxespad=0,
                      frameon=False,
                      ncol=1)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # # Save high-quality versions
    # plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    # plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
    
    return fig, ax


def compute_avg_metrics(all_metrics, metric_keys):
    """
    Computes the average and standard deviation of specified metrics across multiple runs.
    Supports nested metric keys using dot notation (e.g., 'predictor_stats.overall_success_rate').

    Parameters:
    - all_metrics (list of dict): List of metrics dictionaries from multiple runs.
    - metric_keys (list of str): List of metric keys to compute the statistics for.

    Returns:
    - dict: Dictionary containing the average and std values for each specified metric.
    """
    result_metrics = {}
    
    for key in metric_keys:
        # Handle nested keys (either as tuple or dot notation)
        if isinstance(key, tuple):
            key_parts = key
            display_key = '_'.join(key)
        elif '.' in key:
            key_parts = key.split('.')
            display_key = key.replace('.', '_')
        else:
            key_parts = (key,)
            display_key = key
        
        # Extract values for this key (handling nested access)
        values = []
        for metrics in all_metrics:
            value = metrics
            try:
                for part in key_parts:
                    value = value[part]
                values.append(value)
            except (KeyError, TypeError):
                values.append(np.nan)  # Use NaN if key doesn't exist
        
        # Compute stats (ignoring NaN values)
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            avg_value = np.mean(valid_values)
            std_value = np.std(valid_values)
        else:
            avg_value = np.nan
            std_value = np.nan
        
        # Store both average and standard deviation
        result_metrics[f"{display_key}_avg"] = avg_value
        result_metrics[f"{display_key}_std"] = std_value
        
    return result_metrics


def compute_all_agents_metrics(agent_metrics_dict, metric_keys):
    """
    Computes average metrics and standard deviations for multiple agents.
    Supports nested metric keys using either tuples or dot notation.
    
    Parameters:
    - agent_metrics_dict: Dictionary mapping agent names to their metrics lists
    - metric_keys: List of metric keys to compute statistics for
    
    Returns:
    - Dictionary of dictionaries with average metrics and standard deviations for each agent
    """
    results = {}
    
    for agent_name, metrics_list in agent_metrics_dict.items():
        # Skip if metrics list is empty or None
        if not metrics_list:
            print(f"No metrics available for {agent_name}")
            continue
            
        try:
            # Compute average metrics and standard deviations for this agent
            result_metrics = compute_avg_metrics(metrics_list, metric_keys)
            results[agent_name] = result_metrics
            
            print(f"\nResults for {agent_name}:")
            for key in metric_keys:
                # Determine the display key
                if isinstance(key, tuple):
                    display_key = '_'.join(key)
                elif '.' in key:
                    display_key = key.replace('.', '_')
                else:
                    display_key = key
                
                avg_key = f"{display_key}_avg"
                std_key = f"{display_key}_std"
                
                # Only print if we have valid values
                if not np.isnan(result_metrics[avg_key]):
                    print(f"  {display_key}: {result_metrics[avg_key]:.2f}% ± {result_metrics[std_key]:.2f}%")
                
        except Exception as e:
            print(f"Error computing metrics for {agent_name}: {str(e)}")
    
    return results


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


def plot_average_steps(all_agent_metrics, agent_config, figsize=(12, 6), dpi=150, 
                       save_path=None, show_values=True, rotate_labels=45):
    """
    Creates a bar chart showing average steps per episode for different agents.
    
    Parameters:
    - all_agent_metrics: dict, metrics data for all agents
    - agent_config: dict, configuration containing labels and colors
    - figsize: tuple, figure size in inches
    - dpi: int, figure resolution
    - save_path: str, base path for saving the plot (None = don't save)
    - show_values: bool, whether to show values on top of bars
    - rotate_labels: int, degrees to rotate x-axis labels
    
    Returns:
    - fig, ax: matplotlib figure and axis objects
    """
    # Extract data
    agent_labels = []
    avg_steps = []
    std_steps = []
    
    for label in agent_config['labels']:
        if label in all_agent_metrics:
            # Extract average steps from each run
            steps_per_run = []
            
            # Loop through each run's metrics
            for run_metrics in all_agent_metrics[label]:
                if isinstance(run_metrics, dict) and 'average_steps_per_episode' in run_metrics:
                    steps_per_run.append(run_metrics['average_steps_per_episode'])
            
            # If we found step data for this agent
            if steps_per_run:
                agent_labels.append(label)
                avg_steps.append(np.mean(steps_per_run))
                std_steps.append(np.std(steps_per_run))
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Use the agent_config colors, but ensure we have enough colors
    colors = agent_config['colors'][:len(agent_labels)] if len(agent_config['colors']) >= len(agent_labels) else plt.cm.tab10.colors[:len(agent_labels)]
    
    # Create the bars
    bars = ax.bar(np.arange(len(agent_labels)), avg_steps, yerr=std_steps, 
                  capsize=5, color=colors, alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Agent Type', fontweight='bold', fontsize=14)
    ax.set_ylabel('Average Steps per Episode', fontweight='bold', fontsize=14)
    ax.set_title('Comparison of Agent Efficiency', fontweight='bold', fontsize=16)
    ax.set_xticks(np.arange(len(agent_labels)))
    ax.set_xticklabels(agent_labels, rotation=rotate_labels, ha='right' if rotate_labels > 0 else 'center')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Style the plot similarly to other functions
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', length=6, width=1.5)
    
    # Add value labels on top of each bar if requested
    if show_values:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_steps[i] + 0.5,
                   f'{avg_steps[i]:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.show()


def plot_metric_bars(all_agent_metrics, agent_config, metric_keys, 
                     figsize=(12, 6), dpi=150, save_dir=None, 
                     show_values=True, rotate_labels=45):
    """
    Creates bar charts for each metric in metric_keys using the same structure
    as plot_average_steps.
    
    Parameters:
    - all_agent_metrics: dict, metrics data for all agents
    - agent_config: dict, configuration containing labels and colors
    - metric_keys: list, metrics to plot
    - figsize: tuple, figure size in inches
    - dpi: int, figure resolution
    - save_dir: str, directory for saving plots (None = don't save)
    - show_values: bool, whether to show values on top of bars
    - rotate_labels: int, degrees to rotate x-axis labels
    """
    for metric_name in metric_keys:
        # Extract data for this metric
        agent_labels = []
        metric_values = []
        metric_stds = []
        
        for label in agent_config['labels']:
            if label in all_agent_metrics:
                # Extract metric values from each run
                values_per_run = []

                # Parse nested keys if needed
                if '.' in metric_name:
                    key_parts = metric_name.split('.')
                else:
                    key_parts = [metric_name]
                
                # Loop through each run's metrics
                for run_metrics in all_agent_metrics[label]:
                    if isinstance(run_metrics, dict):
                        # Navigate to the nested value
                        value = run_metrics
                        found = True
                        for part in key_parts:
                            if part in value:
                                value = value[part]
                            else:
                                found = False
                                break
                        
                        if found:
                            values_per_run.append(value)

                # # Loop through each run's metrics
                # for run_metrics in all_agent_metrics[label]:
                #     if isinstance(run_metrics, dict) and metric_name in run_metrics:
                #         values_per_run.append(run_metrics[metric_name])
                
                # If we found data for this agent
                if values_per_run:
                    agent_labels.append(label)
                    metric_values.append(np.mean(values_per_run))
                    metric_stds.append(np.std(values_per_run))
        
        # Skip if no data found
        if not agent_labels:
            print(f"No data found for metric: {metric_name}")
            continue
            
        # Create figure and plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Use the agent_config colors
        colors = agent_config['colors'][:len(agent_labels)] if len(agent_config['colors']) >= len(agent_labels) else plt.cm.tab10.colors[:len(agent_labels)]
        
        # Create the bars
        bars = ax.bar(np.arange(len(agent_labels)), metric_values, yerr=metric_stds,
                     capsize=5, color=colors, alpha=0.7)
        
        # Format metric name for display
        display_metric = metric_name.replace('_', ' ').title()
        
        # Customize the plot
        ax.set_xlabel('Agent Type', fontweight='bold', fontsize=14)
        ax.set_ylabel(display_metric, fontweight='bold', fontsize=14)
        ax.set_title(f'Comparison of Agent {display_metric}', fontweight='bold', fontsize=16)
        ax.set_xticks(np.arange(len(agent_labels)))
        ax.set_xticklabels(agent_labels, rotation=rotate_labels, ha='right' if rotate_labels > 0 else 'center')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Style the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(axis='both', which='major', length=6, width=1.5)
        
        # Add value labels on top of each bar if requested
        if show_values:
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + metric_stds[i] + (max(metric_values) * 0.02),
                       f'{metric_values[i]:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save if requested
        if save_dir:
            save_path = f"{save_dir}/{metric_name}"
            plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
            plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        
        plt.show()