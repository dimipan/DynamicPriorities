import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from typing import Dict, Union
from tqdm.auto import tqdm
from robot_utils import _get_action_name

##########################################################

class QLearningAgentFlat:
    def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, DECAY_RATE, EPSILON_MIN, log_rewards_dir=None, learned_policy_dir=None,
                 boost=False):
        self.env = env
        self.log_rewards_dir = log_rewards_dir
        self.learned_policy_dir = learned_policy_dir 
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA 
        self.EPSILON_MAX = EPSILON_MAX
        self.EPSILON = EPSILON_MAX
        self.DECAY_RATE = DECAY_RATE
        self.EPSILON_MIN = EPSILON_MIN
        self.num_states = (self.env.observation_space.high[0] + 1, 
                           self.env.observation_space.high[1] + 1, 
                           self.env.observation_space.high[2] + 1,
                           self.env.observation_space.high[3] + 1)  # 7*7*4*2
        
        # Initialize Q-table
        if hasattr(self.env, 'action_space'):
            self.Q = np.zeros((*self.num_states, self.env.action_space.n))  # Standard Q-table
        else:
            self.Q = None
            
        if self.log_rewards_dir:
            self.writer = tf.summary.create_file_writer(log_rewards_dir)
            
        self.save_interval = 500  # Save Q-table every 500 episodes
        self.exploration_count = 0  # Exploration counter
        self.exploitation_count = 0 # Exploitation counter

        # Additional tracking metrics
        self.successful_episodes = 0  # Count of episodes where mission was completed
        self.info_collection_completed = 0  # Count of episodes where all required info was collected
        self.required_info_count = self.env.sar_robot.info_system.get_required_info_count()

        # Initialize exploration statistics per information location
        self.predictor_stats = {
            tuple(loc.position): {
                'calls': 0,
                'successes': 0,
                'collection_order': loc.collection_order,
                'info_type': loc.info_type
            }
            for loc in self.env.sar_robot.info_system.info_locations
        }
        
        # Add tracking for priority changes (for comparison with intrinsic reward agent)
        self.priority_changes = []  # Track when and how priorities have changed
        self.current_collection_order = {
            loc.info_type: loc.collection_order
            for loc in self.env.sar_robot.info_system.info_locations
        }
        self.collection_order_attempts = 0
        self.collection_order_successes = 0

        # Epsilon boost configuration (used when boost=True)
        self.boost = boost
        if self.boost:
            # Epsilon boost parameters for priority changes
            self.epsilon_boost_factor = 2.0  # How much to multiply epsilon by
            self.epsilon_boost_duration = 50 # How many episodes the boost lasts
            self.epsilon_boost_active = False
            self.epsilon_boost_episodes_remaining = 0
            self.original_epsilon_before_boost = None
    
    def save_learned_policy(self, episode: Union[int, str]):
        """Save Q-table for flat agent"""
        if not self.learned_policy_dir:
            return
        if not os.path.exists(self.learned_policy_dir):
            os.makedirs(self.learned_policy_dir)
        
        # Save Q-table
        q_filename = os.path.join(self.learned_policy_dir, f'q_table_episode_{episode}.npy')
        np.save(q_filename, self.Q)
    
    @staticmethod
    def load_learned_policy(config_file: str) -> Dict:
        """Load Q-table based on configuration file"""
        with open(config_file, 'r') as file:
            config = json.load(file)
        loaded_policies = {}
        
        # Load Q-table if specified
        if 'q_table_file' in config:
            q_filepath = os.path.expandvars(config['q_table_file'])
            
            if os.path.exists(q_filepath):
                loaded_policies['flat'] = np.load(q_filepath)
            else:
                raise FileNotFoundError(f"No Q-table found at {q_filepath}")
                
        return loaded_policies
    
    def change_collection_priorities(self, new_order_map: Dict[str, int], episode: int):
        """
        Record when collection priorities change (for comparison with intrinsic agent).
        
        Args:
            new_order_map: Dict mapping info_type to new collection_order
            episode: Current episode number when change occurs
        """
        # Store the old collection order for metrics
        old_order = self.current_collection_order.copy()
        
        # Update the env's info system with new collection priorities
        self.env.sar_robot.info_system.reorder_collection_priorities(new_order_map)
        
        # Update our tracking of the current collection order
        self.current_collection_order = new_order_map
        
        # Record the change for analytics
        self.priority_changes.append({
            'episode': episode,
            'old_order': old_order,
            'new_order': new_order_map.copy(),
            'adaptation_started': True,
            'adaptation_completed': False,
            'steps_to_adapt': 0,
            'episodes_to_adapt': 0
        })
        
        # Reset success rate tracking for measuring adaptation
        if self.priority_changes:
            change_index = len(self.priority_changes) - 1
            success_rate = self.env.sar_robot.info_system.get_collection_success_rate()
            self.priority_changes[change_index]['success_rate_before'] = success_rate
            self.priority_changes[change_index]['success_count_after'] = 0
            
        # Update predictor stats to reflect new collection order
        for loc in self.env.sar_robot.info_system.info_locations:
            pos = tuple(loc.position)
            if pos in self.predictor_stats:
                self.predictor_stats[pos]['collection_order'] = loc.collection_order
                
        print(f"\nCollection priorities changed at episode {episode}")
        print(f"New collection order: {new_order_map}")

        # --- Activate Epsilon Boost (if boosting is enabled) ---
        if self.boost and not self.epsilon_boost_active:
            self.epsilon_boost_active = True
            self.epsilon_boost_episodes_remaining = self.epsilon_boost_duration
            self.original_epsilon_before_boost = self.EPSILON
            boosted_epsilon = min(self.EPSILON_MAX, self.EPSILON * self.epsilon_boost_factor)
            self.EPSILON = boosted_epsilon
            print(f"Priority change detected: Boosting Epsilon to {self.EPSILON:.4f} for {self.epsilon_boost_duration} episodes.")


    def _check_adaptation_progress(self, episode, steps_in_episode, mission_successful):
        """
        Check and update adaptation progress metrics after priority changes.
        Similar to the intrinsic reward agent for comparison purposes.
        """
        if not self.priority_changes:
            return
        
        # Get the most recent priority change
        change_index = len(self.priority_changes) - 1
        change_info = self.priority_changes[change_index]
        
        # Skip if adaptation already completed
        if change_info['adaptation_completed']:
            return
        
        # Check for successful information collection in correct order
        info_collected = self.env.sar_robot.info_system.get_collected_info_count()
        required_info = self.env.sar_robot.info_system.get_required_info_count()
        collected_in_order = (info_collected == required_info)
        
        # Count successful episodes after the change
        if mission_successful and collected_in_order:
            if not hasattr(change_info, 'success_count_after'):
                change_info['success_count_after'] = 0
            
            change_info['success_count_after'] += 1
            
            # Check if adaptation is complete (3 consecutive successes with new priorities)
            if change_info['success_count_after'] >= 2:
                change_info['adaptation_completed'] = True
                change_info['steps_to_adapt'] = (episode - change_info['episode']) * self.env.max_steps + steps_in_episode
                change_info['episodes_to_adapt'] = episode - change_info['episode']  # Calculate episodes to adapt
                
                # Calculate order alignment rate
                if self.collection_order_attempts > 0:
                    order_alignment_rate = (self.collection_order_successes / self.collection_order_attempts) * 100
                else:
                    order_alignment_rate = 0
                
                # Record metrics after adaptation
                success_rate = self.env.sar_robot.info_system.get_collection_success_rate()
                change_info['success_rate_after'] = success_rate
                change_info['order_alignment_rate'] = order_alignment_rate
                
                print(f"\nAdaptation completed after {change_info['steps_to_adapt']} steps ({change_info['episodes_to_adapt']} episodes)")
                print(f"Success rate before change: {change_info['success_rate_before']:.1f}%")
                print(f"Success rate after adaptation: {success_rate:.1f}%")
                print(f"Order alignment rate: {order_alignment_rate:.1f}%")
                
                # Reset tracking for next potential change
                self.collection_order_attempts = 0
                self.collection_order_successes = 0
    
    def _get_state(self, observation):
        """Get state from observation"""
        return tuple(observation)

    def epsilon_greedy_policy(self, state):
        """
        Epsilon-greedy policy:
        - With probability epsilon, choose random action (exploration)
        - With probability 1-epsilon, choose best action (exploitation)
        """
        if np.random.rand() < self.EPSILON:
            self.exploration_count += 1
            selected_action = np.random.randint(0, self.env.action_space.n)
                
            # Check if we're at an info location during exploration
            if self.env.sar_robot.info_system.is_at_info_location(state):
                current_pos = tuple([state[0], state[1]])
                if current_pos in self.predictor_stats:
                    self.predictor_stats[current_pos]['calls'] += 1
                
                action_name = _get_action_name(self, selected_action)
                
                # Check if prediction is correct
                is_correct = False
                info_locations = self.env.sar_robot.info_system.info_locations
                for location in info_locations:
                    if (current_pos == tuple(location.position) and 
                        state[2] == location.collection_order and 
                        action_name == f'COLLECT_{location.info_type}'):
                        is_correct = True
                        break
                
                # Update success statistics
                if is_correct and current_pos in self.predictor_stats:
                    self.predictor_stats[current_pos]['successes'] += 1
                    
            return selected_action
        else:
            # Exploitation: choose action that maximizes Q-value
            self.exploitation_count += 1
            return np.argmax(self.Q[state])

    def _decay_epsilon(self, episodes):
        """Decay epsilon"""
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON -= self.DECAY_RATE/episodes
        else:
            self.EPSILON = self.EPSILON_MIN
        return self.EPSILON
    
    def _do_q_learning(self, state):
        """Execute one step of standard Q-learning"""
        # Select action using epsilon-greedy policy
        action = self.epsilon_greedy_policy(state)
        
        # Track collection order alignment
        action_name = _get_action_name(self, action)
        if action_name and action_name.startswith("COLLECT_"):
            self.collection_order_attempts += 1
            
            # Extract the info type from action name (e.g., "COLLECT_X" -> "X")
            attempted_info_type = action_name.replace("COLLECT_", "")
            
            # Get the next expected info type according to current priorities
            expected_info_type = self.env.sar_robot.info_system.get_current_priority_info_type()
            
            # Check if collection aligns with expected order
            if expected_info_type and attempted_info_type == expected_info_type:
                self.collection_order_successes += 1
        
        # Take action in environment
        obs_, reward, terminated, _, info = self.env.step(action)
        next_state = self._get_state(obs_)
        
        # Standard Q-learning update
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.GAMMA * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.ALPHA * td_error
        
        return next_state, reward, terminated, info, {
            "action": action, 
            "action_name": action_name
        }
    
    def train(self, num_episodes, change_priorities_at=None):
        """
        Train the agent with the option to change information collection priorities mid-training.
        
        Args:
            num_episodes: Number of episodes to train for
            change_priorities_at: Dict mapping episode numbers to new priority orders
                                 e.g. {500: {'X': 2, 'Y': 0, 'Z': 1}}
        """
        total_rewards_per_episode = np.zeros(num_episodes)
        total_steps_per_episode = np.zeros(num_episodes)
        Rewards = 0
        
        # For tracking success rates
        successful_episodes = 0
        info_collection_completed_episodes = 0

        # Add tracking variables for hazard collisions
        successful_episode_collisions = 0
        successful_episodes_with_collisions = 0
        collision_counts_successful = []

        for episode in tqdm(range(num_episodes)):
            # Check if we need to change priorities at this episode
            if change_priorities_at and episode in change_priorities_at:
                self.change_collection_priorities(change_priorities_at[episode], episode)
                
            if episode % 500 == 0:
                print(f"episode: {episode} | reward: {Rewards} | epsilon: {self.EPSILON}")
                
            # Save Q-table periodically
            if self.learned_policy_dir and episode > 0 and episode % self.save_interval == 0:
                self.save_learned_policy(episode)
                
            obs, _ = self.env.reset(seed=episode)
            s = self._get_state(obs)
            terminated = False
            Rewards, steps_cnt, episode_return_Q = 0, 0, 0

            # Episode-specific tracking
            episode_mission_completed = False
            episode_info_collected = False
            
            while not terminated:
                s_, r, terminated, info, step_info = self._do_q_learning(s)
                
                Rewards += r
                episode_return_Q += r
                s = s_
                steps_cnt += 1

                # Check if the episode completed successfully
                if terminated and self.env.sar_robot.has_saved == 1:
                    episode_mission_completed = True
                    successful_episodes += 1
                
                # Check if all required information was collected
                if self.env.sar_robot.info_system.get_collected_info_count() >= self.required_info_count and not episode_info_collected:
                    episode_info_collected = True
                    info_collection_completed_episodes += 1

            # Check adaptation progress if priorities have changed
            self._check_adaptation_progress(episode, steps_cnt, episode_mission_completed)
            
            # Handle epsilon decay with or without boosting
            if self.boost and self.epsilon_boost_active:
                boost_elapsed = self.epsilon_boost_duration - self.epsilon_boost_episodes_remaining
                self.epsilon_boost_episodes_remaining -= 1

                # Exponential decay of boosted epsilon
                if boost_elapsed == 0:
                    self.EPSILON = self.original_epsilon_before_boost * self.epsilon_boost_factor
                else:
                    D_boost = self.epsilon_boost_duration
                    epsilon_boosted = self.original_epsilon_before_boost * self.epsilon_boost_factor
                    self.EPSILON = epsilon_boosted * np.exp(-boost_elapsed / D_boost)

                if self.epsilon_boost_episodes_remaining <= 0:
                    self.epsilon_boost_active = False
                    self.original_epsilon_before_boost = None
            else:
                self.EPSILON = self._decay_epsilon(num_episodes)

            # Log the rewards and steps to Tensorboard
            if self.log_rewards_dir:
                with self.writer.as_default():
                    tf.summary.scalar('Episode Return', Rewards, step=episode)
                    tf.summary.scalar('Steps per Episode', steps_cnt, step=episode)
                    tf.summary.scalar('Epsilon', self.EPSILON, step=episode)

            total_rewards_per_episode[episode] = Rewards
            total_steps_per_episode[episode] = steps_cnt
            
        # Save final Q-table
        if self.learned_policy_dir:
            self.save_learned_policy(num_episodes)

        # Calculate success rates
        mission_success_rate = (successful_episodes / num_episodes) * 100
        info_collection_success_rate = (info_collection_completed_episodes / num_episodes) * 100

        # Get collection statistics from the environment
        collection_stats = self.env.sar_robot.info_system.get_collection_stats()
        collection_success_rate = self.env.sar_robot.info_system.get_collection_success_rate()

        # Calculate exploration-specific collection success rate statistics
        total_exploration_calls = sum(stats['calls'] for stats in self.predictor_stats.values())
        total_exploration_successes = sum(stats['successes'] for stats in self.predictor_stats.values())
        exploration_success_rate = (total_exploration_successes / max(1, total_exploration_calls)) * 100
        
        # Process ALL priority changes for metrics, including incomplete ones
        priority_change_metrics = []
        if self.priority_changes:
            for change in self.priority_changes:
                # Create base data for all changes
                change_data = {
                    'episode': change.get('episode', 0),
                    'completed': change.get('adaptation_completed', False),
                    'success_rate_before': change.get('success_rate_before', 0),
                }
                
                if change.get('adaptation_completed', False):
                    # For completed adaptations
                    change_data.update({
                        'steps_to_adapt': change.get('steps_to_adapt', 0),
                        'episodes_to_adapt': change.get('episodes_to_adapt', 0),
                        'success_rate_after': change.get('success_rate_after', 0)
                    })
                else:
                    # For incomplete adaptations, calculate how many episodes passed without recovery
                    episodes_elapsed = num_episodes - change.get('episode', 0)
                    steps_elapsed = episodes_elapsed * self.env.max_steps
                    change_data.update({
                        'steps_without_recovery': steps_elapsed,
                        'episodes_without_recovery': episodes_elapsed
                    })
                
                priority_change_metrics.append(change_data)
        
        # Store metrics in dictionary for return
        metrics = {
            'total_exploration_actions': self.exploration_count,
            'total_exploitation_actions': self.exploitation_count,
            'exploration_exploitation_ratio': self.exploration_count / (self.exploration_count + self.exploitation_count),
            'average_reward_per_episode': np.mean(total_rewards_per_episode),
            'average_steps_per_episode': np.mean(total_steps_per_episode),
            'best_episode_reward': np.max(total_rewards_per_episode),
            'worst_episode_reward': np.min(total_rewards_per_episode),
            'mission_success_rate': mission_success_rate,
            'info_collection_success_rate': info_collection_success_rate,
            'collection_success_rate': collection_success_rate,
            'collection_stats': collection_stats,
            'total_hazard_collisions_in_successful_episodes': successful_episode_collisions,
            'successful_episodes_with_collisions': successful_episodes_with_collisions,
            'predictor_stats': {
                'total_calls': total_exploration_calls,
                'total_successes': total_exploration_successes,
                'overall_success_rate': exploration_success_rate,
                'by_location': self.predictor_stats
            },
            'priority_changes': priority_change_metrics
        }

        # Print final statistics
        print("\nTraining Complete!")
        print("\nExploration and Exploitation Statistics:")
        print(f"Total exploration actions: {self.exploration_count}")
        print(f"Total exploitation actions: {self.exploitation_count}")
        print(f"Final exploration/exploitation ratio: {self.exploration_count/(self.exploration_count + self.exploitation_count):.2f}")
        
        print("\nInformation Collection Statistics:")
        self.env.sar_robot.info_system.print_collection_stats()

        print("\nTask Success Rates:")
        print(f"Information collection success rate: {info_collection_success_rate:.2f}%")
        print(f"Overall mission success rate: {mission_success_rate:.2f}%")
        
        print("\nHazard Collision Statistics (Successful Episodes):")
        print(f"Total hazard collisions in successful episodes: {successful_episode_collisions}")
        print(f"Number of successful episodes with collisions: {successful_episodes_with_collisions} out of {successful_episodes}")

        
        # Debug info on priority changes
        print(f"\nPriority changes detected: {len(self.priority_changes)}")
        for i, change in enumerate(self.priority_changes):
            print(f"Change {i+1} at episode {change.get('episode', 'unknown')}:")
            print(f"  Adaptation completed: {change.get('adaptation_completed', False)}")
            print(f"  Success count after change: {change.get('success_count_after', 0)}")
            print(f"  Required success count: 3")
            
            # Calculate adaptation success rate if possible
            if self.collection_order_attempts > 0:
                order_alignment_rate = (self.collection_order_successes / self.collection_order_attempts) * 100
                print(f"  Current order alignment rate: {order_alignment_rate:.1f}%")
            
            # Show final status
            if change.get('adaptation_completed', False):
                print(f"  COMPLETED: Adapted after {change.get('episodes_to_adapt', 'unknown')} episodes")
            else:
                episodes_without_recovery = num_episodes - change.get('episode', 0)
                print(f"  NOT COMPLETED: Failed to adapt after {episodes_without_recovery} episodes")

        # Detailed priority change metrics
        print("\nPriority Change Adaptation Metrics:")
        if not priority_change_metrics:
            print("No adaptation metrics available.")
        else:
            # Group by completion status
            completed = [c for c in priority_change_metrics if c.get('completed', False)]
            incomplete = [c for c in priority_change_metrics if not c.get('completed', False)]
            
            print(f"Total number of priority changes: {len(priority_change_metrics)}")
            print(f"Successfully adapted: {len(completed)} / {len(priority_change_metrics)} ({len(completed)/len(priority_change_metrics)*100 if priority_change_metrics else 0:.1f}%)")
            print(f"Failed to adapt: {len(incomplete)} / {len(priority_change_metrics)} ({len(incomplete)/len(priority_change_metrics)*100 if priority_change_metrics else 0:.1f}%)")
            
            # Report on successful adaptations
            if completed:
                print("\n--- Successful Adaptations ---")
                for change in completed:
                    print(f"Change at episode {change.get('episode', 'unknown')}:")
                    print(f"  Episodes to adapt: {change.get('episodes_to_adapt', 'unknown')}")
                    print(f"  Steps to adapt: {change.get('steps_to_adapt', 'unknown')}")
                    print(f"  Success rate before: {change.get('success_rate_before', 0):.1f}%")
                    print(f"  Success rate after: {change.get('success_rate_after', 0):.1f}%")
                    print(f"  Improvement: {change.get('success_rate_after', 0) - change.get('success_rate_before', 0):.1f}%")
            
            # Report on unsuccessful/incomplete adaptations
            if incomplete:
                print("\n--- Incomplete Adaptations ---")
                for change in incomplete:
                    print(f"Change at episode {change.get('episode', 'unknown')}:")
                    print(f"  Episodes without recovery: {change.get('episodes_without_recovery', 'unknown')}")
                    print(f"  Steps without recovery: {change.get('steps_without_recovery', 'unknown')}")
                    print(f"  Success rate before change: {change.get('success_rate_before', 0):.1f}%")
                    print(f"  Status: Did not meet adaptation criteria by end of training")

            # Summary statistics
            if completed:
                avg_episodes_to_adapt = sum(change.get('episodes_to_adapt', 0) for change in completed) / len(completed)
                print(f"\nAverage episodes to adapt (successful only): {avg_episodes_to_adapt:.2f}")
            
            if incomplete:
                avg_episodes_without_recovery = sum(change.get('episodes_without_recovery', 0) for change in incomplete) / len(incomplete)
                print(f"Average episodes without recovery (unsuccessful): {avg_episodes_without_recovery:.2f}")

        print("\nTraining Summary:")
        print(f"Average reward per episode: {np.mean(total_rewards_per_episode):.2f}")
        print(f"Average steps per episode: {np.mean(total_steps_per_episode):.2f}")
        print(f"Best episode reward: {np.max(total_rewards_per_episode):.2f}")
        print(f"Worst episode reward: {np.min(total_rewards_per_episode):.2f}")
        
        return total_rewards_per_episode, total_steps_per_episode, metrics


class QLearningAgentMaxInfoRL:
    def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, DECAY_RATE, EPSILON_MIN, log_rewards_dir=None, learned_policy_dir=None,
                 detect_shifts=False):
        self.env = env
        self.log_rewards_dir = log_rewards_dir
        self.learned_policy_dir = learned_policy_dir 
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA 
        self.EPSILON_MAX = EPSILON_MAX
        self.EPSILON = EPSILON_MAX
        self.DECAY_RATE = DECAY_RATE
        self.EPSILON_MIN = EPSILON_MIN
        self.num_states = (self.env.observation_space.high[0] + 1, 
                           self.env.observation_space.high[1] + 1, 
                           self.env.observation_space.high[2] + 1,
                           self.env.observation_space.high[3] + 1)  # 7*7*4*2
        
        # Initialize two Q-tables: extrinsic and intrinsic
        if hasattr(self.env, 'action_space'):
            self.Q_extrinsic = np.zeros((*self.num_states, self.env.action_space.n))  # For task rewards
            self.Q_intrinsic = np.zeros((*self.num_states, self.env.action_space.n))  # For information gain
        else:
            self.Q_extrinsic = None
            self.Q_intrinsic = None
            
        if self.log_rewards_dir:
            self.writer = tf.summary.create_file_writer(log_rewards_dir)
            
        self.save_interval = 500  # Save Q-table every 500 episodes
        self.exploration_count = 0  # Exploration counter
        self.exploitation_count = 0 # Exploitation counter

        # Initialize visit counts specifically for information locations
        self.info_location_visits = {}  # Will be populated as locations are discovered

        # Visit counts for computing intrinsic rewards (information gain proxy)
        self.visit_counts = {}
        for s0 in range(self.num_states[0]):
            for s1 in range(self.num_states[1]):
                for s2 in range(self.num_states[2]):
                    for s3 in range(self.num_states[3]):
                        for a in range(self.env.action_space.n):
                            self.visit_counts[((s0, s1, s2, s3), a)] = 0

        # Additional tracking metrics
        self.successful_episodes = 0  # Count of episodes where mission was completed
        self.info_collection_completed = 0  # Count of episodes where all required info was collected
        self.required_info_count = self.env.sar_robot.info_system.get_required_info_count()

        # Initialize exploration statistics per information location
        self.predictor_stats = {
            tuple(loc.position): {
                'calls': 0,
                'successes': 0,
                'collection_order': loc.collection_order,
                'info_type': loc.info_type
            }
            for loc in self.env.sar_robot.info_system.info_locations
        }
        
        # New attributes for dynamic information priority tracking
        self.priority_changes = []  # Track when and how priorities have changed
        self.priority_adaptation_metrics = {
            'adaptation_time': [],  # Steps needed to adapt after priority change
            'success_rate_before_change': [],
            'success_rate_after_change': []
        }
        
        # Initial collection order
        self.current_collection_order = {
            loc.info_type: loc.collection_order
            for loc in self.env.sar_robot.info_system.info_locations
        }
        
        # Tracking for order alignment success
        self.collection_order_attempts = 0
        self.collection_order_successes = 0
        
        # Track the number of steps taken to find each info location after a priority change
        self.discovery_steps_after_change = {}

        # Epsilon boost parameters for priority changes
        self.epsilon_boost_factor = 2.0  # How much to multiply epsilon by
        self.epsilon_boost_duration = 50 # How many episodes the boost lasts
        self.epsilon_boost_active = False
        self.epsilon_boost_episodes_remaining = 0
        self.original_epsilon_before_boost = None

        # Q-table reset strength after context change
        self.q_table_reset_factor = 0.5  # How much to reset Q-values (0=complete reset, 1=no reset)


        # --- New attributes for performance-based detection ---
        # --- New attributes for performance-based detection ---
        self.performance_history = []        # Store episode returns
        self.context_detection_window = 100  # CHANGED: Smaller window (was 500)
        self.recent_window_size = 20         # CHANGED: Smaller recent window (was 50)
        self.performance_drop_threshold = 0.8 # CHANGED: More sensitive threshold (was 0.7)
        self.last_detection_episode = -1     # Episode where the last shift was detected
        self.min_episodes_between_detections = 10 # CHANGED: More frequent checks (was 50)

        self.detect_shifts = detect_shifts  # Flag to enable/disable detection
        # Add this new attribute to track detection metrics
        self.detection_metrics = {
            'episode': [],
            'recent_performance': [],
            'previous_performance': [],
            'ratio': [],
            'detection_triggered': []
        }
        # --- End new attributes ---
    
    def save_learned_policy(self, episode: Union[int, str], manager=None, workers=None):
        """Save Q-tables for either flat or hierarchical agent"""
        if not self.learned_policy_dir:
            return
        if not os.path.exists(self.learned_policy_dir):
            os.makedirs(self.learned_policy_dir)
        
        # Save both Q-tables for MaxInfoRL
        extrinsic_filename = os.path.join(self.learned_policy_dir, f'q_extrinsic_table_episode_{episode}.npy')
        intrinsic_filename = os.path.join(self.learned_policy_dir, f'q_intrinsic_table_episode_{episode}.npy')
        
        np.save(extrinsic_filename, self.Q_extrinsic)
        np.save(intrinsic_filename, self.Q_intrinsic)
    
    @staticmethod
    def load_learned_policy(config_file: str) -> Dict:
        """Load Q-tables based on configuration file"""
        with open(config_file, 'r') as file:
            config = json.load(file)
        loaded_policies = {}
        
        # Load MaxInfoRL Q-tables if specified
        if 'extrinsic_q_table_file' in config and 'intrinsic_q_table_file' in config:
            extrinsic_filepath = os.path.expandvars(config['extrinsic_q_table_file'])
            intrinsic_filepath = os.path.expandvars(config['intrinsic_q_table_file'])
            
            if os.path.exists(extrinsic_filepath) and os.path.exists(intrinsic_filepath):
                loaded_policies['maxinforl'] = {
                    'extrinsic': np.load(extrinsic_filepath),
                    'intrinsic': np.load(intrinsic_filepath)
                }
            else:
                if not os.path.exists(extrinsic_filepath):
                    raise FileNotFoundError(f"No extrinsic Q-table found at {extrinsic_filepath}")
                if not os.path.exists(intrinsic_filepath):
                    raise FileNotFoundError(f"No intrinsic Q-table found at {intrinsic_filepath}")
                
        return loaded_policies
    
    def change_collection_priorities(self, new_order_map: Dict[str, int], episode: int):
        """
        Change the collection priorities in the Information Space.
        
        Args:
            new_order_map: Dict mapping info_type to new collection_order
            episode: Current episode number when change occurs
        """
        # Store the old collection order for metrics
        old_order = self.current_collection_order.copy()
        
        # Update the env's info system with new collection priorities
        self.env.sar_robot.info_system.reorder_collection_priorities(new_order_map)
        
        # Update our tracking of the current collection order
        self.current_collection_order = new_order_map
        
        # Record the change for analytics
        self.priority_changes.append({
            'episode': episode,
            'old_order': old_order,
            'new_order': new_order_map.copy(),
            'adaptation_started': True,
            'adaptation_completed': False,
            'steps_to_adapt': 0,
            'episodes_to_adapt': 0
        })
        
        # Reset success rate tracking for measuring adaptation
        if self.priority_changes:
            change_index = len(self.priority_changes) - 1
            success_rate = self.env.sar_robot.info_system.get_collection_success_rate()
            self.priority_changes[change_index]['success_rate_before'] = success_rate
            self.priority_changes[change_index]['success_count_after'] = 0
            
        # Update predictor stats to reflect new collection order
        for loc in self.env.sar_robot.info_system.info_locations:
            pos = tuple(loc.position)
            if pos in self.predictor_stats:
                self.predictor_stats[pos]['collection_order'] = loc.collection_order
                
        print(f"\nCollection priorities changed at episode {episode}")
        print(f"New collection order: {new_order_map}")

        # --- Activate Epsilon Boost ---
        if not self.epsilon_boost_active: # Avoid boosting if already boosting
            self.epsilon_boost_active = True
            self.epsilon_boost_episodes_remaining = self.epsilon_boost_duration
            self.original_epsilon_before_boost = self.EPSILON
            boosted_epsilon = min(self.EPSILON_MAX, self.EPSILON * self.epsilon_boost_factor)
            self.EPSILON = boosted_epsilon
            print(f"Priority change detected: Boosting Epsilon to {self.EPSILON:.4f} for {self.epsilon_boost_duration} episodes.")
        # --- End Epsilon Boost Activation ---
    
    
    def _partial_reset_q_tables(self):
        """
        Partially reset Q-tables to encourage re-exploration after context shift.
        Focuses on resetting values for info collection actions.
        """
        # Identify information collection related actions
        info_actions = []
        for action_idx in range(self.env.action_space.n):
            action_name = _get_action_name(self, action_idx)
            if action_name and action_name.startswith("COLLECT_"):
                info_actions.append(action_idx)
                
        # Partially reset Q-values for information collection actions
        for s0 in range(self.num_states[0]):
            for s1 in range(self.num_states[1]):
                for s2 in range(self.num_states[2]):
                    for s3 in range(self.num_states[3]):
                        for action_idx in info_actions:
                            # Multiply by reset factor to preserve some knowledge
                            self.Q_extrinsic[s0, s1, s2, s3, action_idx] *= self.q_table_reset_factor
                            self.Q_intrinsic[s0, s1, s2, s3, action_idx] *= self.q_table_reset_factor
                            
        print(f"Partially reset Q-tables for information collection actions")


    def compute_intrinsic_reward(self, state, action):
        current_pos = [state[0], state[1]]
        
        # 1. General exploration reward (for all state-action pairs)
        self.visit_counts[(state, action)] += 1
        general_count = self.visit_counts[(state, action)]
        general_novelty_reward = 1.0 / np.sqrt(general_count)
        
        # 2. Information location-specific exploration reward
        info_location_reward = 0.0
        is_at_info_location = self.env.sar_robot.info_system.is_at_info_location(state)
        
        if is_at_info_location:
            # Track visits to information locations specifically
            pos_tuple = tuple(current_pos)
            if pos_tuple not in self.info_location_visits:
                self.info_location_visits[pos_tuple] = 0
            self.info_location_visits[pos_tuple] += 1
            
            # Higher reward for less-visited information locations
            info_count = self.info_location_visits[pos_tuple]
            info_location_reward = 1.0 / np.sqrt(info_count)  # Higher multiplier for info locations
        
        # 3. Collection order alignment reward (unchanged)
        order_alignment_reward = 0.0
        action_name = _get_action_name(self, action)
        if action_name and action_name.startswith("COLLECT_"):
            attempted_info_type = action_name.replace("COLLECT_", "")
            expected_info_type = self.env.sar_robot.info_system.get_current_priority_info_type()
            
            if expected_info_type and attempted_info_type == expected_info_type:
                order_alignment_reward = 2.0
            elif expected_info_type and attempted_info_type != expected_info_type:
                order_alignment_reward = -0.5
        
        # Combine all rewards with adjusted weights
        combined_reward = (
            0.3 * general_novelty_reward +     # General exploration (10%)
            0.4 * info_location_reward +       # Info location exploration (40%)
            0.3 * order_alignment_reward       # Collection order alignment (50%)
        )

        return combined_reward
    
    def epsilon_maxinforl_policy(self, state):
        """
        Epsilon-MAXINFORL policy:
        - With probability epsilon, choose action that maximizes intrinsic Q-value
        - With probability 1-epsilon, choose action that maximizes extrinsic Q-value
        """
        if np.random.rand() < self.EPSILON:
            self.exploration_count += 1
            
            # Get action that maximizes intrinsic Q-value (directed exploration)
            selected_action = np.argmax(self.Q_intrinsic[state])
            
            # Check if we're at an info location during exploration
            if self.env.sar_robot.info_system.is_at_info_location(state):
                current_pos = tuple([state[0], state[1]])
                if current_pos in self.predictor_stats:
                    self.predictor_stats[current_pos]['calls'] += 1
                
                action_name = _get_action_name(self, selected_action)
                
                # Check if prediction is correct
                is_correct = False
                info_locations = self.env.sar_robot.info_system.info_locations
                for location in info_locations:
                    if (current_pos == tuple(location.position) and 
                        state[2] == location.collection_order and 
                        action_name == f'COLLECT_{location.info_type}'):
                        is_correct = True
                        break
                
                # Update success statistics
                if is_correct and current_pos in self.predictor_stats:
                    self.predictor_stats[current_pos]['successes'] += 1
                    
            return selected_action
        else:
            # Exploitation: choose action that maximizes extrinsic Q-value
            self.exploitation_count += 1
            return np.argmax(self.Q_extrinsic[state])
    
    def _get_state(self, observation):
        """Get state from observation"""
        return tuple(observation)

    def _decay_epsilon(self, episodes):
        """Decay epsilon"""
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON -= self.DECAY_RATE/episodes
        else:
            self.EPSILON = self.EPSILON_MIN
        return self.EPSILON
 
    def _update_q_tables(self, state, action, reward, next_state, intrinsic_reward):
        """
        Update both Q-tables:
        - Extrinsic Q-table uses environment reward
        - Intrinsic Q-table uses information gain reward
        """
        # Update extrinsic Q-table (standard Q-learning update)
        best_next_action_extrinsic = np.argmax(self.Q_extrinsic[next_state])
        td_target_extrinsic = reward + self.GAMMA * self.Q_extrinsic[next_state][best_next_action_extrinsic]
        td_error_extrinsic = td_target_extrinsic - self.Q_extrinsic[state][action]
        self.Q_extrinsic[state][action] += self.ALPHA * td_error_extrinsic
        
        # Update intrinsic Q-table using information gain as reward
        best_next_action_intrinsic = np.argmax(self.Q_intrinsic[next_state])
        td_target_intrinsic = intrinsic_reward + self.GAMMA * self.Q_intrinsic[next_state][best_next_action_intrinsic]
        td_error_intrinsic = td_target_intrinsic - self.Q_intrinsic[state][action]
        self.Q_intrinsic[state][action] += self.ALPHA * td_error_intrinsic
    
    def _do_q_learning(self, state):
        """Execute one step of epsilon-MAXINFORL Q-learning"""
        # Select action using epsilon-MAXINFORL policy
        action = self.epsilon_maxinforl_policy(state)
        
        # Track collection order alignment
        action_name = _get_action_name(self, action)
        if action_name and action_name.startswith("COLLECT_"):
            self.collection_order_attempts += 1
            
            # Extract the info type from action name (e.g., "COLLECT_X" -> "X")
            attempted_info_type = action_name.replace("COLLECT_", "")
            
            # Get the next expected info type according to current priorities
            expected_info_type = self.env.sar_robot.info_system.get_current_priority_info_type()
            
            # Check if collection aligns with expected order
            if expected_info_type and attempted_info_type == expected_info_type:
                self.collection_order_successes += 1
        
        # Take action in environment
        obs_, reward, terminated, _, info = self.env.step(action)
        next_state = self._get_state(obs_)
        
        # Compute intrinsic reward (information gain)
        intrinsic_reward = self.compute_intrinsic_reward(state, action)
        
        # Update both Q-tables
        self._update_q_tables(state, action, reward, next_state, intrinsic_reward)
        
        return next_state, reward, terminated, info, {
            "action": action, 
            "action_name": action_name,
            "intrinsic_reward": intrinsic_reward
        }
    
    def _check_adaptation_progress(self, episode, steps_in_episode, mission_successful):
        """
        Check and update adaptation progress metrics after priority changes.
        Focus on how well the agent aligns with the new collection order.
        """
        if not self.priority_changes:
            return
        
        # Get the most recent priority change
        change_index = len(self.priority_changes) - 1
        change_info = self.priority_changes[change_index]
        
        # Skip if adaptation already completed
        if change_info.get('adaptation_completed', False):
            return
        
        # Check for successful information collection in correct order
        info_collected = self.env.sar_robot.info_system.get_collected_info_count()
        required_info = self.env.sar_robot.info_system.get_required_info_count()
        collected_in_order = (info_collected == required_info)
        
        # Count successful episodes after the change
        if mission_successful and collected_in_order:
            # Use dictionary access rather than attribute access
            if 'success_count_after' not in change_info:
                change_info['success_count_after'] = 0
            
            change_info['success_count_after'] += 1
            
            # Check if adaptation is complete (3 consecutive successes with new priorities)
            if change_info['success_count_after'] >= 2:
                change_info['adaptation_completed'] = True
                change_info['steps_to_adapt'] = (episode - change_info['episode']) * self.env.max_steps + steps_in_episode
                change_info['episodes_to_adapt'] = episode - change_info['episode']  # Calculate episodes to adapt
                
                # Calculate order alignment rate (successful collections in order / total collection attempts)
                if self.collection_order_attempts > 0:
                    order_alignment_rate = (self.collection_order_successes / self.collection_order_attempts) * 100
                else:
                    order_alignment_rate = 0
                
                # Record metrics after adaptation
                success_rate = self.env.sar_robot.info_system.get_collection_success_rate()
                change_info['success_rate_after'] = success_rate
                change_info['order_alignment_rate'] = order_alignment_rate
                
                print(f"\nAdaptation completed after {change_info['steps_to_adapt']} steps ({change_info['episodes_to_adapt']} episodes)")
                print(f"Success rate before change: {change_info['success_rate_before']:.1f}%")
                print(f"Success rate after adaptation: {success_rate:.1f}%")
                print(f"Order alignment rate: {order_alignment_rate:.1f}%")
                
                # Reset tracking for next potential change
                self.collection_order_attempts = 0
                self.collection_order_successes = 0
    
    def _detect_priority_shift(self, current_episode):
        """
        Detect a potential priority shift based on performance degradation.
        Also logs detection metrics for analysis.

        Args:
            current_episode (int): The current episode number.

        Returns:
            bool: True if a significant performance drop is detected, False otherwise.
        """
        # Initialize metrics for this check
        metrics = {
            'recent_performance': None,
            'previous_performance': None,
            'ratio': None,
            'triggered': False
        }
        
        # Check if enough history is available and enough time passed since last detection
        if (len(self.performance_history) < self.context_detection_window or
            current_episode < self.last_detection_episode + self.min_episodes_between_detections):
            # Log skipped check
            self.detection_metrics['episode'].append(current_episode)
            self.detection_metrics['recent_performance'].append(None)
            self.detection_metrics['previous_performance'].append(None)
            self.detection_metrics['ratio'].append(None)
            self.detection_metrics['detection_triggered'].append(False)
            return False

        # Calculate performance averages
        recent_performance = np.mean(self.performance_history[-self.recent_window_size:])
        metrics['recent_performance'] = recent_performance
        
        previous_window_end = len(self.performance_history) - self.recent_window_size
        previous_window_start = max(0, previous_window_end - (self.context_detection_window - self.recent_window_size))
        
        # Ensure the previous window has data
        if previous_window_start >= previous_window_end:
            # Log incomplete data
            self.detection_metrics['episode'].append(current_episode)
            self.detection_metrics['recent_performance'].append(recent_performance)
            self.detection_metrics['previous_performance'].append(None)
            self.detection_metrics['ratio'].append(None)
            self.detection_metrics['detection_triggered'].append(False)
            return False
            
        previous_performance = np.mean(self.performance_history[previous_window_start:previous_window_end])
        metrics['previous_performance'] = previous_performance

        # Avoid division by zero or near-zero
        if previous_performance < 1e-6:
            # Log data without ratio
            self.detection_metrics['episode'].append(current_episode)
            self.detection_metrics['recent_performance'].append(recent_performance)
            self.detection_metrics['previous_performance'].append(previous_performance)
            self.detection_metrics['ratio'].append(None)
            self.detection_metrics['detection_triggered'].append(False)
            return False

        # Calculate ratio
        ratio = recent_performance / previous_performance
        metrics['ratio'] = ratio

        # Store complete metrics regardless of detection
        self.detection_metrics['episode'].append(current_episode)
        self.detection_metrics['recent_performance'].append(recent_performance)
        self.detection_metrics['previous_performance'].append(previous_performance)
        self.detection_metrics['ratio'].append(ratio)
        
        # Add debug output when ratio is potentially concerning
        if ratio < 1.0 and current_episode > 1500:  # Only log when relevant
            print(f"DEBUG - Episode {current_episode}: Recent avg: {recent_performance:.2f}, "
                f"Previous avg: {previous_performance:.2f}, Ratio: {ratio:.2f}, "
                f"Threshold: {self.performance_drop_threshold:.2f}")

        # Check for significant drop
        if recent_performance < previous_performance * self.performance_drop_threshold:
            self.detection_metrics['detection_triggered'].append(True)
            
            print(f"\n--- Performance Drop Detected at Episode {current_episode} ---")
            print(f"  Recent Avg Reward ({self.recent_window_size} episodes): {recent_performance:.2f}")
            print(f"  Previous Avg Reward ({self.context_detection_window - self.recent_window_size} episodes): {previous_performance:.2f}")
            print(f"  Ratio: {ratio:.2f} (Threshold: {self.performance_drop_threshold:.2f})")
            
            # Record this detection
            self.last_detection_episode = current_episode
            return True
        
        self.detection_metrics['detection_triggered'].append(False)
        return False
    
    

    def plot_detection_metrics(self, save_path=None):
        """
        Plot the detection metrics over time to visualize the performance detection mechanism.
        
        Args:
            save_path (str, optional): Path to save the plot. If None, plot is displayed.
        """
        if not hasattr(self, 'detection_metrics') or len(self.detection_metrics['episode']) == 0:
            print("No detection metrics available.")
            return
                
        plt.figure(figsize=(12, 8))
        
        # Filter out None values for plotting
        episodes = self.detection_metrics['episode']
        valid_indices = []
        
        for i in range(len(episodes)):
            if (self.detection_metrics['recent_performance'][i] is not None and
                self.detection_metrics['previous_performance'][i] is not None):
                valid_indices.append(i)
        
        if not valid_indices:
            print("No valid metrics to plot.")
            return
        
        valid_episodes = [episodes[i] for i in valid_indices]
        valid_recent = [self.detection_metrics['recent_performance'][i] for i in valid_indices]
        valid_previous = [self.detection_metrics['previous_performance'][i] for i in valid_indices]
        
        # Filter ratio indices separately (might have more None values)
        ratio_indices = [i for i in valid_indices if self.detection_metrics['ratio'][i] is not None]
        ratio_episodes = [episodes[i] for i in ratio_indices]
        ratio_values = [self.detection_metrics['ratio'][i] for i in ratio_indices]
        
        # Get detection episodes
        detection_episodes = [episodes[i] for i in range(len(episodes)) 
                            if self.detection_metrics['detection_triggered'][i]]
        
        # Plot 1: Recent and Previous Performance
        plt.subplot(2, 1, 1)
        plt.plot(valid_episodes, valid_recent, 
                label=f'Recent ({self.recent_window_size} episodes)', color='blue')
        plt.plot(valid_episodes, valid_previous, 
                label=f'Previous ({self.context_detection_window - self.recent_window_size} episodes)', color='green')
        
        # Mark detection points
        if detection_episodes:
            for e in detection_episodes:
                plt.axvline(x=e, color='red', linestyle='--', alpha=0.7)
        
        plt.title('Performance Window Comparison')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Performance Ratio
        plt.subplot(2, 1, 2)
        plt.plot(ratio_episodes, ratio_values, color='purple')
        plt.axhline(y=self.performance_drop_threshold, color='red', linestyle='--', 
                    label=f'Threshold ({self.performance_drop_threshold})')
        
        # Mark detection points
        if detection_episodes:
            for e in detection_episodes:
                plt.axvline(x=e, color='red', linestyle='--', alpha=0.7)
        
        plt.title('Performance Ratio (Recent/Previous)')
        plt.xlabel('Episodes')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def train(self, num_episodes, change_priorities_at=None):
    # def train(self, num_episodes, change_priorities_at=None):
        """
        Train the agent with the option to change information collection priorities mid-training.
        
        Args:
            num_episodes: Number of episodes to train for
            change_priorities_at: Dict mapping episode numbers to new priority orders
                                 e.g. {500: {'X': 2, 'Y': 0, 'Z': 1}}
        """
        total_rewards_per_episode = np.zeros(num_episodes)
        total_steps_per_episode = np.zeros(num_episodes)
        total_intrinsic_rewards_per_episode = np.zeros(num_episodes)
        Rewards = 0
        
        # For tracking success rates
        successful_episodes = 0
        info_collection_completed_episodes = 0

        # Add tracking variables for hazard collisions
        successful_episode_collisions = 0
        successful_episodes_with_collisions = 0
        collision_counts_successful = []

        for episode in tqdm(range(num_episodes)):
            
            priority_changed_this_episode = False # Flag for this episode   #### neww

            # Check if we need to change priorities at this episode
            if change_priorities_at and episode in change_priorities_at:
                print(f"\n--- Explicit Priority Change Triggered at Episode {episode} ---")
                self.change_collection_priorities(change_priorities_at[episode], episode)
                # Partially reset Q-tables to encourage re-exploration
                self._partial_reset_q_tables()
        
            if episode % 500 == 0:
                print(f"episode: {episode} | reward: {Rewards} | epsilon: {self.EPSILON}")
                
            # Save Q-tables periodically
            if self.learned_policy_dir and episode > 0 and episode % self.save_interval == 0:
                self.save_learned_policy(episode)
                
            obs, _ = self.env.reset(seed=episode)
            s = self._get_state(obs)
            terminated = False
            Rewards, steps_cnt, episode_return_Q = 0, 0, 0
            intrinsic_rewards_sum = 0

            # Episode-specific tracking
            episode_mission_completed = False
            episode_info_collected = False
            
            while not terminated:
                s_, r, terminated, info, step_info = self._do_q_learning(s)
                
                Rewards += r
                intrinsic_rewards_sum += step_info["intrinsic_reward"]
                episode_return_Q += r
                s = s_
                steps_cnt += 1

                # Check if the episode completed successfully
                if terminated and self.env.sar_robot.has_saved == 1:
                    episode_mission_completed = True
                    successful_episodes += 1
                
                # Check if all required information was collected
                if self.env.sar_robot.info_system.get_collected_info_count() >= self.required_info_count and not episode_info_collected:
                    episode_info_collected = True
                    info_collection_completed_episodes += 1


            # --- Store performance and Check for Autonomous Detection ---
            self.performance_history.append(Rewards) # Store episode reward
            if self.detect_shifts and not priority_changed_this_episode:
                if self._detect_priority_shift(episode):
                    print(f"Autonomous shift detection triggered adaptation at episode {episode}.")
                    # React to the detected shift
                    self._partial_reset_q_tables()

                    # Activate Epsilon Boost (if not already active)
                    if not self.epsilon_boost_active:
                        self.epsilon_boost_active = True
                        self.epsilon_boost_episodes_remaining = self.epsilon_boost_duration
                        self.original_epsilon_before_boost = self.EPSILON
                        boosted_epsilon = min(self.EPSILON_MAX, self.EPSILON * self.epsilon_boost_factor)
                        self.EPSILON = boosted_epsilon
                        print(f"Boosting Epsilon to {self.EPSILON:.4f} for {self.epsilon_boost_duration} episodes.")

                    self.last_detection_episode = episode # Record detection episod


            # Check adaptation progress if priorities have changed
            self._check_adaptation_progress(episode, steps_cnt, episode_mission_completed)
            
            
            if not self.detect_shifts:
                # Epsilon handling (Boost check and Decay)
                if self.epsilon_boost_active:
                    boost_elapsed = self.epsilon_boost_duration - self.epsilon_boost_episodes_remaining
                    self.epsilon_boost_episodes_remaining -= 1

                    if boost_elapsed == 0:
                        self.EPSILON = self.original_epsilon_before_boost * self.epsilon_boost_factor
                    else:
                        D_boost = self.epsilon_boost_duration
                        epsilon_boosted = self.original_epsilon_before_boost * self.epsilon_boost_factor
                        self.EPSILON = epsilon_boosted * np.exp(-boost_elapsed / D_boost)

                    if self.epsilon_boost_episodes_remaining <= 0:
                        self.epsilon_boost_active = False
                        self.original_epsilon_before_boost = None
                else:
                    self.EPSILON = self._decay_epsilon(num_episodes)

            # Log the rewards and steps to Tensorboard
            if self.log_rewards_dir:
                with self.writer.as_default():
                    tf.summary.scalar('Episode Return', Rewards, step=episode)
                    tf.summary.scalar('Episode Return (Intrinsic)', intrinsic_rewards_sum, step=episode)
                    tf.summary.scalar('Steps per Episode', steps_cnt, step=episode)
                    tf.summary.scalar('Epsilon', self.EPSILON, step=episode)

            total_rewards_per_episode[episode] = Rewards
            total_intrinsic_rewards_per_episode[episode] = intrinsic_rewards_sum
            total_steps_per_episode[episode] = steps_cnt
            
        # Save final Q-tables
        if self.learned_policy_dir:
            self.save_learned_policy(num_episodes)

        # Calculate success rates
        mission_success_rate = (successful_episodes / num_episodes) * 100
        info_collection_success_rate = (info_collection_completed_episodes / num_episodes) * 100

        # Get collection statistics from the environment
        collection_stats = self.env.sar_robot.info_system.get_collection_stats()
        collection_success_rate = self.env.sar_robot.info_system.get_collection_success_rate()

        # Calculate exploration-specific collection success rate statistics
        total_exploration_calls = sum(stats['calls'] for stats in self.predictor_stats.values())
        total_exploration_successes = sum(stats['successes'] for stats in self.predictor_stats.values())
        exploration_success_rate = (total_exploration_successes / max(1, total_exploration_calls)) * 100
        
        # Process ALL priority changes for metrics, including incomplete ones
        priority_change_metrics = []
        if self.priority_changes:
            for change in self.priority_changes:
                # Create base data for all changes
                change_data = {
                    'episode': change.get('episode', 0),
                    'completed': change.get('adaptation_completed', False),
                    'success_rate_before': change.get('success_rate_before', 0),
                }
                
                if change.get('adaptation_completed', False):
                    # For completed adaptations
                    change_data.update({
                        'steps_to_adapt': change.get('steps_to_adapt', 0),
                        'episodes_to_adapt': change.get('episodes_to_adapt', 0),
                        'success_rate_after': change.get('success_rate_after', 0)
                    })
                else:
                    # For incomplete adaptations, calculate how many episodes passed without recovery
                    episodes_elapsed = num_episodes - change.get('episode', 0)
                    steps_elapsed = episodes_elapsed * self.env.max_steps
                    change_data.update({
                        'steps_without_recovery': steps_elapsed,
                        'episodes_without_recovery': episodes_elapsed
                    })
                
                priority_change_metrics.append(change_data)
        
        # Store metrics in dictionary for return
        metrics = {
            'total_exploration_actions': self.exploration_count,
            'total_exploitation_actions': self.exploitation_count,
            'exploration_exploitation_ratio': self.exploration_count / (self.exploration_count + self.exploitation_count),
            'average_reward_per_episode': np.mean(total_rewards_per_episode),
            'average_intrinsic_reward_per_episode': np.mean(total_intrinsic_rewards_per_episode),
            'average_steps_per_episode': np.mean(total_steps_per_episode),
            'best_episode_reward': np.max(total_rewards_per_episode),
            'worst_episode_reward': np.min(total_rewards_per_episode),
            'mission_success_rate': mission_success_rate,
            'info_collection_success_rate': info_collection_success_rate,
            'collection_success_rate': collection_success_rate,
            'collection_stats': collection_stats,
            'total_hazard_collisions_in_successful_episodes': successful_episode_collisions,
            'successful_episodes_with_collisions': successful_episodes_with_collisions,
            'predictor_stats': {
                'total_calls': total_exploration_calls,
                'total_successes': total_exploration_successes,
                'overall_success_rate': exploration_success_rate,
                'by_location': self.predictor_stats
            },
            'all_priority_changes': self.priority_changes,  # Include ALL changes 
            'priority_changes': priority_change_metrics     # Processed metrics
        }

        # Print final statistics
        print("\nTraining Complete!")
        print("\nExploration and Exploitation Statistics:")
        print(f"Total exploration actions: {self.exploration_count}")
        print(f"Total exploitation actions: {self.exploitation_count}")
        print(f"Final exploration/exploitation ratio: {self.exploration_count/(self.exploration_count + self.exploitation_count):.2f}")
        
        print("\nInformation Collection Statistics:")
        self.env.sar_robot.info_system.print_collection_stats()

        print("\nExploration Collection Statistics by Location:")
        for pos, stats in self.predictor_stats.items():
            success_rate = (stats['successes'] / max(1, stats['calls'])) * 100
            print(f"\nLocation {pos} (Info {stats['info_type']}, Collection Order {stats['collection_order']}):")
            print(f"  Predictor calls: {stats['calls']}")
            print(f"  Successful predictions: {stats['successes']}")
            print(f"  Success rate: {success_rate:.1f}%")
        print(f"\nTotal predictor calls across all locations: {total_exploration_calls}")
        print(f"Total successful predictions: {total_exploration_successes}")
        print(f"Overall predictor success rate: {exploration_success_rate:.1f}%")

        print("\nTask Success Rates:")
        print(f"Information collection success rate: {info_collection_success_rate:.2f}%")
        print(f"Overall mission success rate: {mission_success_rate:.2f}%")
        
        print("\nHazard Collision Statistics (Successful Episodes):")
        print(f"Total hazard collisions in successful episodes: {successful_episode_collisions}")
        print(f"Number of successful episodes with collisions: {successful_episodes_with_collisions} out of {successful_episodes}")

        # Debug info on priority changes
        print(f"\nPriority changes detected: {len(self.priority_changes)}")
        for i, change in enumerate(self.priority_changes):
            print(f"Change {i+1} at episode {change.get('episode', 'unknown')}:")
            print(f"  Adaptation completed: {change.get('adaptation_completed', False)}")
            print(f"  Success count after change: {change.get('success_count_after', 0)}")
            print(f"  Required success count: 3")
            
            # Calculate adaptation success rate if possible
            if self.collection_order_attempts > 0:
                order_alignment_rate = (self.collection_order_successes / self.collection_order_attempts) * 100
                print(f"  Current order alignment rate: {order_alignment_rate:.1f}%")
            
            # Show final status
            if change.get('adaptation_completed', False):
                print(f"  COMPLETED: Adapted after {change.get('episodes_to_adapt', 'unknown')} episodes")
            else:
                episodes_without_recovery = num_episodes - change.get('episode', 0)
                print(f"  NOT COMPLETED: Failed to adapt after {episodes_without_recovery} episodes")

        # Detailed priority change metrics
        print("\nPriority Change Adaptation Metrics:")
        if not priority_change_metrics:
            print("No adaptation metrics available.")
        else:
            # Group by completion status
            completed = [c for c in priority_change_metrics if c.get('completed', False)]
            incomplete = [c for c in priority_change_metrics if not c.get('completed', False)]
            
            print(f"Total number of priority changes: {len(priority_change_metrics)}")
            print(f"Successfully adapted: {len(completed)} / {len(priority_change_metrics)} ({len(completed)/len(priority_change_metrics)*100 if priority_change_metrics else 0:.1f}%)")
            print(f"Failed to adapt: {len(incomplete)} / {len(priority_change_metrics)} ({len(incomplete)/len(priority_change_metrics)*100 if priority_change_metrics else 0:.1f}%)")
            
            # Report on successful adaptations
            if completed:
                print("\n--- Successful Adaptations ---")
                for change in completed:
                    print(f"Change at episode {change.get('episode', 'unknown')}:")
                    print(f"  Episodes to adapt: {change.get('episodes_to_adapt', 'unknown')}")
                    print(f"  Steps to adapt: {change.get('steps_to_adapt', 'unknown')}")
                    print(f"  Success rate before: {change.get('success_rate_before', 0):.1f}%")
                    print(f"  Success rate after: {change.get('success_rate_after', 0):.1f}%")
                    print(f"  Improvement: {change.get('success_rate_after', 0) - change.get('success_rate_before', 0):.1f}%")
            
            # Report on unsuccessful/incomplete adaptations
            if incomplete:
                print("\n--- Incomplete Adaptations ---")
                for change in incomplete:
                    print(f"Change at episode {change.get('episode', 'unknown')}:")
                    print(f"  Episodes without recovery: {change.get('episodes_without_recovery', 'unknown')}")
                    print(f"  Steps without recovery: {change.get('steps_without_recovery', 'unknown')}")
                    print(f"  Success rate before change: {change.get('success_rate_before', 0):.1f}%")
                    print(f"  Status: Did not meet adaptation criteria by end of training")

            # Summary statistics
            if completed:
                avg_episodes_to_adapt = sum(change.get('episodes_to_adapt', 0) for change in completed) / len(completed)
                print(f"\nAverage episodes to adapt (successful only): {avg_episodes_to_adapt:.2f}")
            
            if incomplete:
                avg_episodes_without_recovery = sum(change.get('episodes_without_recovery', 0) for change in incomplete) / len(incomplete)
                print(f"Average episodes without recovery (unsuccessful): {avg_episodes_without_recovery:.2f}")

        print("\nTraining Summary:")
        print(f"Average extrinsic reward per episode: {np.mean(total_rewards_per_episode):.2f}")
        print(f"Average intrinsic reward per episode: {np.mean(total_intrinsic_rewards_per_episode):.2f}")
        print(f"Average steps per episode: {np.mean(total_steps_per_episode):.2f}")
        print(f"Best episode reward: {np.max(total_rewards_per_episode):.2f}")
        print(f"Worst episode reward: {np.min(total_rewards_per_episode):.2f}")
        
        # Create visualization if priority changes occurred
        if priority_change_metrics and len(total_rewards_per_episode) > 0:
            self._plot_training_with_priority_changes(
                total_rewards_per_episode, 
                total_intrinsic_rewards_per_episode,
                [c.get('episode', 0) for c in self.priority_changes]
            )
        
        return total_rewards_per_episode, total_steps_per_episode, metrics
    
    def _plot_training_with_priority_changes(self, rewards, intrinsic_rewards, change_episodes):
        """Create a visualization showing how rewards changed with priority changes"""
        plt.figure(figsize=(12, 8))
        
        # Plot rewards
        plt.subplot(2, 1, 1)
        plt.plot(rewards, label='Extrinsic Rewards')
        
        # Mark where priority changes occurred
        for episode in change_episodes:
            plt.axvline(x=episode, color='r', linestyle='--', alpha=0.7)
            plt.text(episode, min(rewards) + (max(rewards)-min(rewards))*0.1, 
                     f"Priority\nChange", rotation=90, color='r')
            
        plt.title('Rewards Over Training with Priority Changes')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        
        # Plot intrinsic rewards
        plt.subplot(2, 1, 2)
        plt.plot(intrinsic_rewards, label='Intrinsic Rewards', color='green')
        
        # Mark where priority changes occurred
        for episode in change_episodes:
            plt.axvline(x=episode, color='r', linestyle='--', alpha=0.7)
            plt.text(episode, min(intrinsic_rewards) + (max(intrinsic_rewards)-min(intrinsic_rewards))*0.1, 
                     f"Priority\nChange", rotation=90, color='r')
            
        plt.xlabel('Episodes')
        plt.ylabel('Intrinsic Reward')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the figure if a directory is available
        if self.log_rewards_dir:
            plt.savefig(os.path.join(self.log_rewards_dir, 'rewards_with_priority_changes.png'))
        
        plt.show()

"""
Implementation of epsilon-MAXINFORL agent based on Section 3.1 of the paper "MAXINFORL: BOOSTING EXPLORATION IN
REINFORCEMENT LEARNING THROUGH INFORMATION GAIN MAXIMIZATION" -- https://arxiv.org/abs/2412.12098
Here we use it for discrete state-action spaces.

1) 
The benefit of epsilon-MAXINFORL should manifest as higher extrinsic rewards achieved faster than the baseline agent. 
The agent isn't directly optimising for extrinsic rewards during exploration, but its more efficient exploration 
strategy should lead to better overall performance on the task.

2) 
When comparing against a standard learning agent, you should only use the extrinsic rewards (the environment rewards) 
for a fair comparison. The intrinsic rewards are an internal mechanism to guide exploration and don't reflect actual 
task performance.

3)
When we're evaluating the agent based on its learned understanding of how to maximise task performance, not on its 
exploration strategy. Said that we should ONLY use the extrinsic Q-table (Q_extrinsic) for evaluation since 
this table contains the agent's knowledge about which actions maximise task rewards. In contrast, the intrinsic Q-table
(Q_intrinsic) is used to guide exploration and doesn't directly correlate with task performance. 
"""