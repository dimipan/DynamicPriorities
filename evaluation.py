import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.signal import savgol_filter  # Consider using this for smoothing
from termcolor import colored
from robot_utils import RobotOption, _get_action_name

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