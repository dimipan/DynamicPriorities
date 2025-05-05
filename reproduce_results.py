import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from agents import QLearningAgentFlat, QLearningAgentMaxInfoRL
from robot_utils import RunningParameters
param = RunningParameters()


# Define environment configurations
env_configs = [
    # Config 1 (Original)
    {
        'init_positions': [[2, 1]],
        'target_pos': [0, 3],
        'info_locations': [
            {'position': [1, 1], 'info_type': 'X', 'collection_order': 0},
            {'position': [3, 0], 'info_type': 'Y', 'collection_order': 1},
            {'position': [3, 2], 'info_type': 'Z', 'collection_order': 2},
        ],
        'ditches': [(1, 0), (2, 0), (1, 2)],
        'hazards': [(1, 3), (2, 3), (3, 1)]
    },
    # Config 2
    {
        'init_positions': [[0, 0]],
        'target_pos': [3, 3],
        'info_locations': [
            {'position': [1, 2], 'info_type': 'X', 'collection_order': 0},
            {'position': [2, 1], 'info_type': 'Y', 'collection_order': 1},
            {'position': [3, 1], 'info_type': 'Z', 'collection_order': 2},
        ],
        'ditches': [(0, 1), (1, 3), (2, 2)],
        'hazards': [(0, 3), (2, 0), (3, 2)]
    },
    # Config 3
    {
        'init_positions': [[3, 3]],
        'target_pos': [0, 0],
        'info_locations': [
            {'position': [0, 1], 'info_type': 'X', 'collection_order': 0},
            {'position': [1, 3], 'info_type': 'Y', 'collection_order': 1},
            {'position': [2, 0], 'info_type': 'Z', 'collection_order': 2},
        ],
        'ditches': [(1, 1), (2, 3), (3, 0)],
        'hazards': [(0, 2), (2, 2), (3, 1)]
    },
    # Config 4
    {
        'init_positions': [[1, 0]],
        'target_pos': [2, 3],
        'info_locations': [
            {'position': [0, 2], 'info_type': 'X', 'collection_order': 0},
            {'position': [2, 2], 'info_type': 'Y', 'collection_order': 1},
            {'position': [3, 0], 'info_type': 'Z', 'collection_order': 2},
        ],
        'ditches': [(0, 0), (1, 1), (3, 3)],
        'hazards': [(0, 3), (1, 3), (3, 2)]
    },
    # Config 5
    {
        'init_positions': [[0, 3]],
        'target_pos': [3, 0],
        'info_locations': [
            {'position': [1, 1], 'info_type': 'X', 'collection_order': 0},
            {'position': [2, 3], 'info_type': 'Y', 'collection_order': 1},
            {'position': [3, 2], 'info_type': 'Z', 'collection_order': 2},
        ],
        'ditches': [(0, 1), (2, 1), (3, 3)],
        'hazards': [(1, 0), (1, 3), (2, 0)]
    }
]

# Define agent configurations
agent_types = [
    {
        "name": "Baseline_Static",
        "agent_class": QLearningAgentFlat,
        "change_priorities": None  # No changes
    },

    {
        "name": "Baseline-Boost_Static",
        "agent_class": QLearningAgentFlat,
        "change_priorities": None  # No changes
    },

    {
        "name": "CA-MIQ_Static (Ours)",
        "agent_class": QLearningAgentMaxInfoRL,
        "change_priorities": None  # No changes
    },

    {
        "name": "Baseline_Dynamic",
        "agent_class": QLearningAgentFlat,
        "change_priorities": {
            1700: {'X': 2, 'Y': 0, 'Z': 1},  # Change from X-Y-Z to Y-Z-X
            # 3500: {'X': 1, 'Y': 2, 'Z': 0},  # Change to Z-X-Y
        }
    },

    {
        "name": "Baseline-Boost_Dynamic",
        "agent_class": QLearningAgentFlat,
        "change_priorities": {
            1700: {'X': 2, 'Y': 0, 'Z': 1},  # Change from X-Y-Z to Y-Z-X
            # 3500: {'X': 1, 'Y': 2, 'Z': 0},  # Change to Z-X-Y
        }
    },

    {
        "name": "CA-MIQ_Dynamic (Ours)",
        "agent_class": QLearningAgentMaxInfoRL,
        "change_priorities": {
            1700: {'X': 2, 'Y': 0, 'Z': 1},  # Change from X-Y-Z to Y-Z-X
            # 3500: {'X': 1, 'Y': 2, 'Z': 0},  # Change to Z-X-Y
        }
    }
]

######   S    O    S   ######
# Path to the results file   #### when extracting from the pickle file, use the path to the directory where the results are stored
# Note: The path should be updated to the actual location of the results file
log_dir = "logs_new_single/logs/multi_env_comparison_parallel_20250502_194121"
results_file = os.path.join(log_dir, "all_results.pkl")

# Load the pickle file
with open(results_file, 'rb') as f:
    results = pickle.load(f)
######   S    O    S   ######


def generate_summary_visualizations(all_results, agent_types, env_configs, log_dir):
    """Generate comprehensive visualizations summarizing the experiment results."""
    # Create visualization directory
    vis_dir = os.path.join(log_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Organize data by agent type
    agent_data = {agent["name"]: {"rewards": [], "steps": [], "metrics": []} for agent in agent_types}
    
    # Group results by environment config
    env_results = {i: {"agent_performance": {}} for i in range(len(env_configs))}
    
    # Extract data from all trials
    for trial, data in all_results.items():
        config_idx = data["config_idx"]
        
        for agent_name, agent_result in data["results"].items():
            # Store raw rewards and steps data
            agent_data[agent_name]["rewards"].append(agent_result["rewards"])
            agent_data[agent_name]["steps"].append(agent_result["steps"])
            
            # Store metrics
            if "metrics" in agent_result:
                agent_data[agent_name]["metrics"].append(agent_result["metrics"])
            
            # Store performance by environment
            if agent_name not in env_results[config_idx]["agent_performance"]:
                env_results[config_idx]["agent_performance"][agent_name] = []
            
            # Use last 100 episodes as final performance measure
            last_100_rewards = agent_result["rewards"][-100:]
            env_results[config_idx]["agent_performance"][agent_name].append(np.mean(last_100_rewards))
    
    # 1. Overall Agent Performance Summary (average across all environments)
    print("\nAgent Performance Summary (across all environments):")
    for agent_name, data in agent_data.items():
        # Calculate average of last 100 episodes for each trial
        final_performances = []
        for reward_history in data["rewards"]:
            final_performances.append(np.mean(reward_history[-100:]))
        
        avg_performance = np.mean(final_performances)
        std_performance = np.std(final_performances)
        print(f"Agent {agent_name}: {avg_performance:.2f} ± {std_performance:.2f}")
    
    # 2. Create comparison chart for overall performance
    plt.figure(figsize=(10, 6))
    
    # Plot bars for each agent
    agent_names = list(agent_data.keys())
    x_pos = np.arange(len(agent_names))
    
    for i, agent_name in enumerate(agent_names):
        # Calculate average of last 100 episodes for each trial
        final_performances = []
        for reward_history in agent_data[agent_name]["rewards"]:
            final_performances.append(np.mean(reward_history[-100:]))
        
        plt.bar(x_pos[i], np.mean(final_performances), 
                yerr=np.std(final_performances), 
                capsize=10, 
                label=agent_name)
    
    plt.title("Average Agent Performance Across All Environment Configurations")
    plt.ylabel("Average Reward (Last 100 Episodes)")
    plt.xticks(x_pos, agent_names, rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(vis_dir, "overall_agent_comparison.png"))
    
    # 3. Static vs Dynamic Agent Comparison
    window_size = 15  # For smoothing
    
    # 3.1 Static environment agents
    plt.figure(figsize=(12, 5))
    
    for agent_name, data in agent_data.items():
        if "Static" in agent_name:  # Only include static agents
            # Average rewards across all trials
            avg_rewards = np.mean(data["rewards"], axis=0)
            
            # Smooth rewards using moving average
            smoothed_rewards = np.convolve(avg_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_rewards, label=agent_name)
            
            # Calculate confidence intervals
            all_smoothed = []
            for reward_history in data["rewards"]:
                run_smoothed = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
                all_smoothed.append(run_smoothed)
            
            all_smoothed = np.array(all_smoothed)
            std_dev = np.std(all_smoothed, axis=0)
            
            # Plot confidence interval (±1 std dev)
            x = np.arange(len(smoothed_rewards))
            plt.fill_between(x, smoothed_rewards - std_dev, smoothed_rewards + std_dev, alpha=0.2)
    
    plt.title('Learning Curves (w/o Priority Shift)')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, 'reward_trends_static.png'))
    
    # 3.2 Dynamic environment agents
    plt.figure(figsize=(12, 5))
    dynamic_smoothed_rewards = {}  # Store for min/max calculation
    
    for agent_name, data in agent_data.items():
        if "Dynamic" in agent_name:  # Only include dynamic agents
            # Average rewards across all trials
            avg_rewards = np.mean(data["rewards"], axis=0)
            
            # Smooth rewards using moving average
            smoothed_rewards = np.convolve(avg_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_rewards, label=agent_name)
            dynamic_smoothed_rewards[agent_name] = smoothed_rewards
            
            # Calculate confidence intervals
            all_smoothed = []
            for reward_history in data["rewards"]:
                run_smoothed = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
                all_smoothed.append(run_smoothed)
            
            all_smoothed = np.array(all_smoothed)
            std_dev = np.std(all_smoothed, axis=0)
            
            # Plot confidence interval (±1 std dev)
            x = np.arange(len(smoothed_rewards))
            plt.fill_between(x, smoothed_rewards - std_dev, smoothed_rewards + std_dev, alpha=0.2)
    
    # Add priority change markers (if we have data for dynamic agents)
    if dynamic_smoothed_rewards:
        # Calculate global min/max for consistent text placement
        all_rewards = np.concatenate(list(dynamic_smoothed_rewards.values()))
        min_reward = np.min(all_rewards)
        max_reward = np.max(all_rewards)
        
        # Get the change priority episodes from one of the dynamic agents
        for agent in agent_types:
            if agent["change_priorities"] is not None:
                for episode in agent["change_priorities"].keys():
                    if episode >= window_size//2:
                        adjusted_episode = episode - window_size//2
                        
                        bullet_y = max_reward + (max_reward - min_reward) * 0.05  # Slightly above highest reward
                        plt.plot(adjusted_episode, 
                                bullet_y, 
                                marker='o', 
                                markersize=10, 
                                color='red', 
                                label='Priority Change' if episode == list(agent["change_priorities"].keys())[0] else "")
                break  # Only need one agent's change points as they're the same
    
    plt.title('Learning Curves (with 1 Priority Shift)')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, 'reward_trends_dynamic.png'))
    
    # 4. Performance by Environment Configuration
    plt.figure(figsize=(14, 8))
    
    # Setup
    env_labels = [f"Config {i+1}" for i in range(len(env_configs))]
    agent_names = list(agent_data.keys())
    x = np.arange(len(env_labels))
    width = 0.2  # width of the bars
    
    # Plot bars for each agent grouped by environment
    for i, agent_name in enumerate(agent_names):
        # Collect performance across environments
        env_perf = []
        env_std = []
        
        for env_idx in range(len(env_configs)):
            if agent_name in env_results[env_idx]["agent_performance"]:
                perf_values = env_results[env_idx]["agent_performance"][agent_name]
                if perf_values:
                    env_perf.append(np.mean(perf_values))
                    env_std.append(np.std(perf_values))
                else:
                    env_perf.append(0)
                    env_std.append(0)
            else:
                env_perf.append(0)
                env_std.append(0)
        
        # Calculate the offset for this agent's bars
        offset = width * (i - len(agent_names)/2 + 0.5)
        
        # Plot with error bars
        plt.bar(x + offset, env_perf, width, label=agent_name, yerr=env_std, capsize=5)
    
    plt.title('Agent Performance by Environment Configuration')
    plt.xlabel('Environment')
    plt.ylabel('Average Final Reward')
    plt.xticks(x, env_labels)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(agent_names))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'performance_by_environment.png'))

    
    # Replace section 5 in generate_summary_visualizations with this enhanced adaptation metrics analysis

    # 5. Enhanced Adaptation Metrics Analysis
    # Collect adaptation metrics if available
    adaptation_metrics = {agent_name: [] for agent_name in agent_names if "Dynamic" in agent_name}

    print("\nProcessing adaptation metrics for analysis...")
    for agent_name in adaptation_metrics.keys():
        for metrics_list in agent_data[agent_name]["metrics"]:
            # Check both possible fields where the priority changes might be stored
            if "priority_changes" in metrics_list:
                adaptation_metrics[agent_name].append(metrics_list["priority_changes"])
            elif "all_priority_changes" in metrics_list:
                # Alternative field from our updated agent
                processed_changes = []
                for change in metrics_list["all_priority_changes"]:
                    # Convert raw change data to the expected format
                    processed_change = {
                        'episode': change.get('episode', 0),
                        'completed': change.get('adaptation_completed', False),
                        'success_rate_before': change.get('success_rate_before', 0),
                    }
                    
                    if change.get('adaptation_completed', False):
                        processed_change.update({
                            'steps_to_adapt': change.get('steps_to_adapt', 0),
                            'episodes_to_adapt': change.get('episodes_to_adapt', 0),
                            'success_rate_after': change.get('success_rate_after', 0)
                        })
                    else:
                        # For incomplete adaptations
                        processed_change.update({
                            'steps_without_recovery': change.get('steps_without_recovery', 0),
                            'episodes_without_recovery': change.get('episodes_without_recovery', 0)
                        })
                    
                    processed_changes.append(processed_change)
                
                adaptation_metrics[agent_name].append(processed_changes)

    # If we have adaptation metrics, create enhanced visualizations
    if any(metrics for metrics in adaptation_metrics.values()):
        print(f"Found adaptation metrics, creating visualizations...")
        
        # Find max number of changes
        max_changes = 0
        for metrics_list in adaptation_metrics.values():
            for run_metrics in metrics_list:
                max_changes = max(max_changes, len(run_metrics))
        
        if max_changes > 0:
            print(f"Found {max_changes} priority changes to analyze")
            
            # Setup
            labels = [f"Change {i+1}" for i in range(max_changes)]
            x = np.arange(len(labels))
            width = 0.35 / len(adaptation_metrics)
            
            # 5.1 Episodes to Adapt Analysis
            plt.figure(figsize=(14, 12))
            plt.subplot(3, 1, 1)
            
            # Plot episodes to adapt for each agent
            for i, (agent_name, metrics_list) in enumerate(adaptation_metrics.items()):
                # Calculate average episodes to adapt for each change
                avg_episodes = []
                std_episodes = []
                
                for change_idx in range(max_changes):
                    episode_values = []
                    
                    for run_metrics in metrics_list:
                        if change_idx < len(run_metrics):
                            change = run_metrics[change_idx]
                            if change.get('completed', False) and 'episodes_to_adapt' in change:
                                episode_values.append(change['episodes_to_adapt'])
                    
                    if episode_values:
                        avg_episodes.append(np.mean(episode_values))
                        std_episodes.append(np.std(episode_values))
                    else:
                        avg_episodes.append(0)
                        std_episodes.append(0)
                
                # Plot with offset for each agent
                offset = width * (i - len(adaptation_metrics)/2 + 0.5)
                plt.bar(x + offset, avg_episodes, width, label=agent_name, yerr=std_episodes, capsize=5)
            
            plt.ylabel('Episodes to Adapt')
            plt.title('Number of Episodes Required to Adapt After Priority Changes')
            plt.xticks(x, labels)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            # 5.2 Steps to Adapt Analysis (original plot)
            plt.subplot(3, 1, 2)
            
            # Plot steps to adapt for each agent
            for i, (agent_name, metrics_list) in enumerate(adaptation_metrics.items()):
                # Calculate average steps to adapt for each change
                avg_steps = []
                std_steps = []
                
                for change_idx in range(max_changes):
                    steps_values = []
                    
                    for run_metrics in metrics_list:
                        if change_idx < len(run_metrics):
                            change = run_metrics[change_idx]
                            if change.get('completed', False) and 'steps_to_adapt' in change:
                                steps_values.append(change['steps_to_adapt'])
                    
                    if steps_values:
                        avg_steps.append(np.mean(steps_values))
                        std_steps.append(np.std(steps_values))
                    else:
                        avg_steps.append(0)
                        std_steps.append(0)
                
                # Plot with offset for each agent
                offset = width * (i - len(adaptation_metrics)/2 + 0.5)
                plt.bar(x + offset, avg_steps, width, label=agent_name, yerr=std_steps, capsize=5)
            
            plt.ylabel('Steps to Adapt')
            plt.title('Total Environment Steps Required to Adapt After Priority Changes')
            plt.xticks(x, labels)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            # 5.3 Success Rate Improvement
            plt.subplot(3, 1, 3)
            
            # Plot success rate improvement for each agent
            for i, (agent_name, metrics_list) in enumerate(adaptation_metrics.items()):
                # Calculate success rate improvement for each change
                avg_improvement = []
                std_improvement = []
                
                for change_idx in range(max_changes):
                    improvement_values = []
                    
                    for run_metrics in metrics_list:
                        if change_idx < len(run_metrics):
                            change = run_metrics[change_idx]
                            if change.get('completed', False) and 'success_rate_before' in change and 'success_rate_after' in change:
                                improvement = change['success_rate_after'] - change['success_rate_before']
                                improvement_values.append(improvement)
                    
                    if improvement_values:
                        avg_improvement.append(np.mean(improvement_values))
                        std_improvement.append(np.std(improvement_values))
                    else:
                        avg_improvement.append(0)
                        std_improvement.append(0)
                
                # Plot with offset for each agent
                offset = width * (i - len(adaptation_metrics)/2 + 0.5)
                plt.bar(x + offset, avg_improvement, width, label=agent_name, yerr=std_improvement, capsize=5)
            
            plt.ylabel('Success Rate Improvement (%)')
            plt.title('Performance Improvement After Adaptation')
            plt.xticks(x, labels)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'adaptation_metrics.png'))
            
            # 5.4 Adaptation Success Rate Analysis
            plt.figure(figsize=(12, 6))
            
            # Create a matrix of adaptation success rates for each agent and change
            for i, (agent_name, metrics_list) in enumerate(adaptation_metrics.items()):
                # Calculate success rate for each change
                success_rates = []
                
                for change_idx in range(max_changes):
                    total_runs = 0
                    success_count = 0
                    
                    for run_metrics in metrics_list:
                        if change_idx < len(run_metrics):
                            total_runs += 1
                            change = run_metrics[change_idx]
                            if change.get('completed', False):
                                success_count += 1
                    
                    if total_runs > 0:
                        success_rates.append((success_count / total_runs) * 100)
                    else:
                        success_rates.append(0)
                
                # Plot with offset for each agent
                offset = width * (i - len(adaptation_metrics)/2 + 0.5)
                plt.bar(x + offset, success_rates, width, label=agent_name)
            
            plt.ylabel('Adaptation Success Rate (%)')
            plt.title('Percentage of Runs Where Agent Successfully Adapted')
            plt.xticks(x, labels)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.savefig(os.path.join(vis_dir, 'adaptation_success_rates.png'))
            
            # 5.5 Adaptation Time Comparison
            plt.figure(figsize=(14, 10))
            
            # First subplot: Recovery time in episodes
            plt.subplot(2, 1, 1)
            
            # For each agent, create a box plot of episodes to adapt for each change
            boxplot_data = []
            agent_colors = plt.cm.tab10(np.linspace(0, 1, len(adaptation_metrics)))
            
            for agent_idx, (agent_name, metrics_list) in enumerate(adaptation_metrics.items()):
                for change_idx in range(max_changes):
                    episode_values = []
                    
                    for run_metrics in metrics_list:
                        if change_idx < len(run_metrics):
                            change = run_metrics[change_idx]
                            if change.get('completed', False) and 'episodes_to_adapt' in change:
                                episode_values.append(change['episodes_to_adapt'])
                    
                    if episode_values:
                        boxplot_data.append({
                            'label': f"{agent_name}\nChange {change_idx+1}",
                            'data': episode_values,
                            'color': agent_colors[agent_idx]
                        })
            
            if boxplot_data:
                # Create box plots
                boxes = plt.boxplot([item['data'] for item in boxplot_data], 
                                labels=[item['label'] for item in boxplot_data],
                                patch_artist=True,
                                showfliers=False)  # Hide outliers for clarity
                
                # Color boxes by agent
                for box, item in zip(boxes['boxes'], boxplot_data):
                    box.set(facecolor=item['color'])
            
            plt.ylabel('Episodes to Adapt')
            plt.title('Distribution of Adaptation Time Across Runs (Episodes)')
            plt.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            
            # Second subplot: Recovery time in steps
            plt.subplot(2, 1, 2)
            
            # For each agent, create a box plot of steps to adapt for each change
            boxplot_data = []
            
            for agent_idx, (agent_name, metrics_list) in enumerate(adaptation_metrics.items()):
                for change_idx in range(max_changes):
                    step_values = []
                    
                    for run_metrics in metrics_list:
                        if change_idx < len(run_metrics):
                            change = run_metrics[change_idx]
                            if change.get('completed', False) and 'steps_to_adapt' in change:
                                step_values.append(change['steps_to_adapt'])
                    
                    if step_values:
                        boxplot_data.append({
                            'label': f"{agent_name}\nChange {change_idx+1}",
                            'data': step_values,
                            'color': agent_colors[agent_idx]
                        })
            
            if boxplot_data:
                # Create box plots
                boxes = plt.boxplot([item['data'] for item in boxplot_data], 
                                labels=[item['label'] for item in boxplot_data],
                                patch_artist=True,
                                showfliers=False)  # Hide outliers for clarity
                
                # Color boxes by agent
                for box, item in zip(boxes['boxes'], boxplot_data):
                    box.set(facecolor=item['color'])
            
            plt.ylabel('Steps to Adapt')
            plt.title('Distribution of Adaptation Time Across Runs (Steps)')
            plt.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'adaptation_time_distributions.png'))
            
            # 5.6 Print Detailed Adaptation Summary Table
            print("\n" + "="*100)
            print("ADAPTATION METRICS SUMMARY")
            print("="*100)
            
            header = f"{'Agent':<20} | {'Change':<10} | {'Success Rate':<15} | {'Avg Episodes':<15} | {'Avg Steps':<15} | {'Improvement':<15}"
            print(header)
            print("="*100)
            
            for agent_name, metrics_list in adaptation_metrics.items():
                for change_idx in range(max_changes):
                    # Calculate metrics for this change and agent
                    success_count = 0
                    total_runs = 0
                    episode_values = []
                    step_values = []
                    improvement_values = []
                    
                    for run_metrics in metrics_list:
                        if change_idx < len(run_metrics):
                            total_runs += 1
                            change = run_metrics[change_idx]
                            
                            if change.get('completed', False):
                                success_count += 1
                                
                                if 'episodes_to_adapt' in change:
                                    episode_values.append(change['episodes_to_adapt'])
                                
                                if 'steps_to_adapt' in change:
                                    step_values.append(change['steps_to_adapt'])
                                
                                if 'success_rate_before' in change and 'success_rate_after' in change:
                                    improvement = change['success_rate_after'] - change['success_rate_before']
                                    improvement_values.append(improvement)
                    
                    # Calculate statistics
                    if total_runs > 0:
                        success_rate = f"{(success_count / total_runs) * 100:.1f}%"
                    else:
                        success_rate = "N/A"
                    
                    avg_episodes = f"{np.mean(episode_values):.2f}" if episode_values else "N/A"
                    avg_steps = f"{np.mean(step_values):.2f}" if step_values else "N/A"
                    avg_improvement = f"{np.mean(improvement_values):.2f}%" if improvement_values else "N/A"
                    
                    # Print row
                    row = f"{agent_name:<20} | {f'Change {change_idx+1}':<10} | {success_rate:<15} | {avg_episodes:<15} | {avg_steps:<15} | {avg_improvement:<15}"
                    print(row)
            
            # 5.7 Overall Adaptation Effectiveness Summary
            print("\n" + "="*100)
            print("OVERALL ADAPTATION EFFECTIVENESS")
            print("="*100)
            
            for agent_name, metrics_list in adaptation_metrics.items():
                # Calculate overall metrics across all changes
                total_changes = 0
                successful_adaptations = 0
                total_episodes = []
                total_steps = []
                
                for run_metrics in metrics_list:
                    for change in run_metrics:
                        total_changes += 1
                        
                        if change.get('completed', False):
                            successful_adaptations += 1
                            
                            if 'episodes_to_adapt' in change:
                                total_episodes.append(change['episodes_to_adapt'])
                            
                            if 'steps_to_adapt' in change:
                                total_steps.append(change['steps_to_adapt'])
                
                # Calculate overall statistics
                if total_changes > 0:
                    overall_success_rate = (successful_adaptations / total_changes) * 100
                    print(f"\nAgent: {agent_name}")
                    print(f"  Total priority changes: {total_changes}")
                    print(f"  Successfully adapted: {successful_adaptations} ({overall_success_rate:.1f}%)")
                    
                    if successful_adaptations > 0:
                        avg_episodes = np.mean(total_episodes) if total_episodes else "N/A"
                        avg_steps = np.mean(total_steps) if total_steps else "N/A"
                        print(f"  Average episodes to adapt: {avg_episodes:.2f}")
                        print(f"  Average steps to adapt: {avg_steps:.2f}")
                        
                        # Calculate statistics by change type
                        if max_changes > 1:
                            print("\n  Breakdown by change:")
                            for change_idx in range(max_changes):
                                change_episodes = []
                                change_steps = []
                                change_success = 0
                                change_total = 0
                                
                                for run_metrics in metrics_list:
                                    if change_idx < len(run_metrics):
                                        change_total += 1
                                        change = run_metrics[change_idx]
                                        
                                        if change.get('completed', False):
                                            change_success += 1
                                            
                                            if 'episodes_to_adapt' in change:
                                                change_episodes.append(change['episodes_to_adapt'])
                                            
                                            if 'steps_to_adapt' in change:
                                                change_steps.append(change['steps_to_adapt'])
                                
                                if change_total > 0:
                                    change_success_rate = (change_success / change_total) * 100
                                    avg_change_episodes = np.mean(change_episodes) if change_episodes else "N/A"
                                    print(f"    Change {change_idx+1}: {change_success_rate:.1f}% success rate, {avg_change_episodes:.2f} avg episodes")
    
    # 6. Performance metrics comparison
    # Define key metrics to compare
    metrics_to_compare = [
        'mission_success_rate', 
        'info_collection_success_rate',
        'average_steps_per_episode',
        'mission_success_no_collisions_rate'
    ]
    
    metric_labels = {
        'mission_success_rate': 'Mission Success (%)',
        'info_collection_success_rate': 'Info Collection (%)',
        'average_steps_per_episode': 'Avg Steps',
        'mission_success_no_collisions_rate': 'Success Without Collisions (%)'
    }
    
    # Check if we have these metrics
    have_metrics = True
    for agent_name, data in agent_data.items():
        if not data["metrics"] or not all(metric in data["metrics"][0] for metric in metrics_to_compare):
            have_metrics = False
            break
    
    if have_metrics:
        plt.figure(figsize=(14, 7))
        
        # Setup
        x = np.arange(len(metrics_to_compare))
        width = 0.2  # width of the bars
        
        # Calculate average metrics for each agent
        for i, agent_name in enumerate(agent_names):
            # Collect all values for each metric
            metric_values = []
            metric_stds = []
            
            for metric in metrics_to_compare:
                values = [m.get(metric, 0) for m in agent_data[agent_name]["metrics"]]
                metric_values.append(np.mean(values))
                metric_stds.append(np.std(values))
            
            # Calculate the offset for this agent's bars
            offset = width * (i - len(agent_names)/2 + 0.5)
            
            # Plot with error bars
            plt.bar(x + offset, metric_values, width, label=agent_name, yerr=metric_stds, capsize=5)
        
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, [metric_labels[metric] for metric in metrics_to_compare])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(agent_names))
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'performance_metrics.png'))
        
        # Print summary table
        print("\n" + "="*100)
        print("PERFORMANCE COMPARISON SUMMARY (AVERAGED ACROSS ALL ENVIRONMENTS)")
        print("="*100)
        
        header = f"{'Metric':<40} | " + " | ".join([f"{name:<15}" for name in agent_names])
        print(header)
        print("="*100)
        
        metrics_to_print = metrics_to_compare + ['average_reward_per_episode']
        
        for metric in metrics_to_print:
            values = []
            for agent_name in agent_names:
                # Calculate average value across all trials
                metric_values = [m.get(metric, 0) for m in agent_data[agent_name]["metrics"] if metric in m]
                if metric_values:
                    avg_value = np.mean(metric_values)
                    values.append(f"{avg_value:.2f}")
                else:
                    values.append("N/A")
            
            metric_name = metric_labels.get(metric, metric)
            row = f"{metric_name:<40} | " + " | ".join([f"{val:<15}" for val in values])
            print(row)
    
    print(f"\nResults and visualizations saved to {log_dir}")
    return vis_dir

generate_summary_visualizations(results, agent_types, env_configs, log_dir)