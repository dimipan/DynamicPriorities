import os
import json
import time
import numpy as np
import tensorflow as tf
import numpy as np
from typing import Dict, Union
from tqdm.auto import tqdm
from LLM_InformationPredictor import InformationPredictor
from shaping_mechanisms import AttentionSpace, ActionToggleMechanism
from robot_utils import _get_action_name

### THIS IS THE MAIN FLAT AGENT ------------------------------
class LearningAgentFlat:
    def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, DECAY_RATE, EPSILON_MIN, log_rewards_dir=None, learned_policy_dir=None,
                 use_llm=False,
                 use_attention=False,
                 use_action_toggle=False,
                 llm_model=None):
        """
        Initialize the Q-learning agent with optional enhancements.
        
        Args:
            env: The environment
            alpha: Learning rate
            gamma: Discount factor
            epsilon_max: Initial exploration rate
            decay_rate: Epsilon decay rate
            epsilon_min: Minimum exploration rate
            log_rewards_dir: Directory for logging rewards
            learned_policy_dir: Directory for saving learned policies
            use_llm: Whether to use LLM-based prediction
            use_attention: Whether to use attention mechanism
            use_action_toggle: Whether to use action toggle mechanism
            llm_model: LLM model to use for action predicton (if use_llm is True)
        """
        # base parameters
        self.env = env
        self.log_rewards_dir = log_rewards_dir
        self.learned_policy_dir = learned_policy_dir 
        self.ALPHA = ALPHA
        self.GAMMA = GAMMA 
        self.EPSILON_MAX = EPSILON_MAX
        self.EPSILON = EPSILON_MAX
        self.DECAY_RATE = DECAY_RATE
        self.EPSILON_MIN = EPSILON_MIN

        # Optional enhancements (mechanisms)
        self.use_llm = use_llm
        self.use_attention = use_attention
        self.use_action_toggle = use_action_toggle

        # initialise state space
        self.num_states = (self.env.observation_space.high[0] + 1, 
                           self.env.observation_space.high[1] + 1, 
                           self.env.observation_space.high[2] + 1,
                           self.env.observation_space.high[3] + 1)  # 7*7*4*2
        
        # Initialize Q-table only if action_space exists
        if hasattr(self.env, 'action_space'):
            self.Q_table = np.zeros((*self.num_states, self.env.action_space.n))
        else:
            self.Q_table = None
        
        # Initialize TensorBoard writer if log_rewards_dir is provided
        if self.log_rewards_dir:
            self.writer = tf.summary.create_file_writer(log_rewards_dir)

        # Initialize tracking variables
        self.save_interval = 500  # Save Q-table every 500 episodes
        self.exploration_count = 0  # Exploration counter
        self.exploitation_count = 0 # exploitation counter
        self.successful_episodes = 0  # Count of episodes where mission was completed
        self.info_collection_completed = 0  # Count of episodes where all required info was collected
        self.required_info_count = self.env.sar_robot.info_system.get_required_info_count()

        # Initialize exploration statistics per information location (useful for tracking exploration actions on these locations)
        self.predictor_stats = {
            tuple(loc.position): {
                'calls': 0,
                'successes': 0,
                'collection_order': loc.collection_order,
                'info_type': loc.info_type
            }
            for loc in self.env.sar_robot.info_system.info_locations
        }

        # Initialise LLM predictor if needed
        if self.use_llm:
            self.info_predictor = InformationPredictor(model=llm_model)
            self.llm_call_times = []
            self.total_llm_calls = 0
            self.total_llm_time = 0
        
        # Initialise attention space mechanism if needed
        if self.use_attention:
            self.attention_space = AttentionSpace(self.env) # instantiate attention space
            self.input_received = False
            self.episode_exploit_mode = False  # Flag to indicate if the agent is in exploit mode for the current episode
            self.global_epsilon_exploit = False  # Flag to indicate if the agent is in exploit mode for the entire training
        
        if self.use_action_toggle:
            self.action_toggle = ActionToggleMechanism(self.env)
            self.input_received = False

    
    def save_learned_policy(self, episode: Union[int, str], manager=None, workers=None):
        """Save Q-tables for either flat or hierarchical agent"""
        if not self.learned_policy_dir:
            return
        if not os.path.exists(self.learned_policy_dir):
            os.makedirs(self.learned_policy_dir)
        if manager is None or workers is None:  # Flat agent
            filename = os.path.join(self.learned_policy_dir, f'q_table_episode_{episode}.npy')
            np.save(filename, self.Q_table)
            # print(f"Flat Q-table saved at episode {episode}")
        else:  # Hierarchical agent
            # Save manager Q-table
            manager_filename = os.path.join(self.learned_policy_dir, 
                                          f'manager_q_table_episode_{episode}.npy')
            np.save(manager_filename, manager.Q_table)
            # Save worker Q-tables
            for option, worker in workers.items():
                worker_filename = os.path.join(self.learned_policy_dir, 
                                             f'worker_{option}_q_table_episode_{episode}.npy')
                np.save(worker_filename, worker.Q_table)
            # print(f"Hierarchical Q-tables saved at episode {episode}")
    
    @staticmethod
    def load_learned_policy(config_file: str) -> Dict:
        """Load Q-tables based on configuration file"""
        with open(config_file, 'r') as file:
            config = json.load(file)
        loaded_policies = {}
        # Load flat Q-table if specified
        if 'flat_q_table_file' in config:
            filepath = os.path.expandvars(config['flat_q_table_file'])
            if os.path.exists(filepath):
                loaded_policies['flat'] = np.load(filepath)
            else:
                raise FileNotFoundError(f"No flat Q-table found at {filepath}")
        # Load hierarchical Q-tables if specified
        if 'hierarchical_q_tables' in config:
            hierarchical_tables = {}
            for key, filepath in config['hierarchical_q_tables'].items():
                expanded_path = os.path.expandvars(filepath)
                if os.path.exists(expanded_path):
                    hierarchical_tables[key] = np.load(expanded_path)
                else:
                    raise FileNotFoundError(f"No Q-table found at {expanded_path}")
            loaded_policies['hierarchical'] = hierarchical_tables
        return loaded_policies


    def _standard_epsilon_greedy_policy(self, state, valid_actions=None): # Epsilon-greedy policy
        """Standard epsilon-greedy policy with optional LLM predictor call"""
        # if valid actions are provided (meaning we are toggling the action space), use them instead of all actions
        if valid_actions is None:
            valid_actions = list(range(self.env.action_space.n))

        if np.random.rand() < self.EPSILON:
            self.exploration_count += 1

            # Check if we're at an info location during exploration
            if self.env.sar_robot.info_system.is_at_info_location(state):
                current_pos = tuple([state[0], state[1]])
                # print(f"\n** At info location {current_pos} **")
                if current_pos in self.predictor_stats:
                    self.predictor_stats[current_pos]['calls'] += 1
                
                ### Case where LLM predictor is ON -----------
                # Use LLM to predict action if enabled
                if self.use_llm:
                    start_time = time.time()
                    predicted_action, is_correct = self.info_predictor.predict(state, self.env) # Get prediction from InformationPredictor - is_correct tells us if this matches ground truth
                    end_time = time.time()

                    elapsed_time = end_time - start_time
                    self.llm_call_times.append(elapsed_time)
                    self.total_llm_calls += 1
                    self.total_llm_time += elapsed_time

                    if predicted_action != 'NONE':
                        action_index = self.env.sar_robot.info_system.get_collection_action_index(predicted_action, hierarchical=False)
                        if action_index is not None and action_index in valid_actions: # update success statistics
                            if is_correct and current_pos in self.predictor_stats:
                                self.predictor_stats[current_pos]['successes'] += 1

                            if predicted_action.startswith("COLLECT_"):  # Extract info type from the action name
                                info_type = predicted_action.replace("COLLECT_", "")
                                self.info_predictor.record_attempt(current_pos, info_type, is_correct) # Record attempt for cross-episodic memory InformationPredictor

                            return action_index
                ### Case where LLM predictor is ON -----------

                # Default random exploration for info location
                selected_action = np.random.choice(valid_actions) # Explore: choose a random action
                action_name = _get_action_name(None, selected_action)
                is_correct = False
                info_locations = self.env.sar_robot.info_system.info_locations
                for location in info_locations:
                    if (current_pos == tuple(location.position) and 
                        state[2] == location.collection_order and 
                        action_name == f'COLLECT_{location.info_type}'):
                        is_correct = True
                        break
                # update success statistics
                if is_correct and current_pos in self.predictor_stats:
                    self.predictor_stats[current_pos]['successes'] += 1
                return selected_action
            return np.random.choice(valid_actions) # Explore: choose a random action # Explore: choose a random action
        else:
            self.exploitation_count += 1
            if len(valid_actions) == self.env.action_space.n:
                return np.argmax(self.Q_table[state]) # Exploit: choose the action with max Q-value from all actions 
            else:
                # Exploit: choose best action from valid actions only
                q_values = [self.Q_table[state][a] for a in valid_actions]
                return valid_actions[np.argmax(q_values)]
    
    
    def _epsilon_greedy_policy(self, state): # Epsilon-greedy policy used by the agents
        """
        Epsilon-greedy policy with optional enhancements:
        - LLM prediction when at info locations
        - Attention-based exploitation after info collection
        - Action toggle to avoid dangerous states
        """
        # Check if we should use action toggle to filter available actions
        if self.use_action_toggle and self.env.sar_robot.visited_information_state:
            valid_actions = self.action_toggle.get_valid_actions(state) # Get the valid actions for this state (excluding those leading to fires)
            # print(f"Valid actions: {valid_actions} at state {state}")
            # if len(valid_actions) == self.env.action_space.n:
                # print(f"This {state} is no adjacent to any danger and has the original action space")
            return self._standard_epsilon_greedy_policy(state, valid_actions)
        
        # Handle attention mechanism logic
        elif self.use_attention:
            if self.global_epsilon_exploit:
                if not self.env.sar_robot.visited_information_state:
                    return self._standard_epsilon_greedy_policy(state)

                else:
                    self.exploitation_count += 1
                    return np.argmax(self.Q_table[state]) # Exploit: choose the action with max Q-value
            else:
                if not self.episode_exploit_mode:
                    return self._standard_epsilon_greedy_policy(state)
                else:
                    # Exploit: choose action with max Q-value
                    self.exploitation_count += 1
                    return np.argmax(self.Q_table[state])
                             
        # Standard epsilon-greedy with LLM if enabled
        else:
            return self._standard_epsilon_greedy_policy(state)
        
    
    def _get_state(self, observation): # Get state from observation
        return tuple(observation)

    def _decay_epsilon(self, episodes): # Decay epsilon
        if self.EPSILON > 0.1:
            self.EPSILON -= self.DECAY_RATE/episodes
        else:
            self.EPSILON = self.EPSILON_MIN
        return self.EPSILON
    
    def _decay_epsilon_exploit(self):
        """Set very low epsilon for exploit mode"""
        self.EPSILON = 0.005
        return self.EPSILON
    
    def _update_attention(self, sensor_readings):
        """Update attention space based on sensor readings (when attention mechanism is used)"""
        if not self.use_attention:
            return
        changed_states = self.attention_space._identify_changed_states(sensor_readings)
        # print(f"Changed states: {changed_states}\n")
        if changed_states:
            for state in changed_states:
                self.attention_space._update_attention_space(state, sensor_readings)
            self.attention_space._apply_attention_to_q_table(self.Q_table)
 
    def _update(self, state, action, reward, next_state): ## normal Q-learning update
        """
        Update Q-values with various enhancements:
        - Standard Q-learning update
        - Action toggle-aware update
        """
        if self.use_action_toggle and self.env.sar_robot.visited_information_state:
            valid_next_actions = self.action_toggle.get_valid_actions(next_state)

            # calculate the target Q-value using the valid actions only
            next_q_values = [self.Q_table[next_state][a] for a in valid_next_actions]
            best_next_action = valid_next_actions[np.argmax(next_q_values)]

            # update the Q-table
            td_target = reward + self.GAMMA * self.Q_table[next_state][best_next_action]
            td_error = td_target - self.Q_table[state][action]
            self.Q_table[state][action] += self.ALPHA * td_error

        else:
            # Q-learning update
            best_next_action = np.argmax(self.Q_table[next_state])
            td_target = reward + self.GAMMA * self.Q_table[next_state][best_next_action]
            td_error = td_target - self.Q_table[state][action]
            self.Q_table[state][action] += self.ALPHA * td_error
    
    # Execute one step of the Q-learning algorithm
    def _do_q_learning(self, state):
        action = self._epsilon_greedy_policy(state)
        obs_, reward, terminated, _, info = self.env.step(action)
        next_state = self._get_state(obs_)

        if self.use_llm:
            self.info_predictor.record_step(state, action, next_state, reward) # Just record the step in the trajectory for InformationPredictor

        self._update(state, action, reward, next_state)
        return next_state, reward, terminated, info, {
            "action": action, 
            "action_name": _get_action_name(None, action)}
    
    def train(self, num_episodes):
        total_rewards_per_episode = np.zeros(num_episodes)
        total_steps_per_episode = np.zeros(num_episodes)
        Rewards, steps_cnt, episode_return_Q = 0, 0, 0

        # For tracking success rates
        successful_episodes = 0
        info_collection_completed_episodes = 0

        # Add tracking variables for hazard collisions -- new
        successful_episode_collisions = 0
        successful_episodes_with_collisions = 0
        collision_counts_successful = []

        for episode in tqdm(range(num_episodes)):

            if self.use_attention:
                self.episode_exploit_mode = False  # Reset exploit mode for each episode
            
            if episode % 100 == 0:
                print(f"episode: {episode} | reward: {Rewards} | epsilon: {self.EPSILON}")
            
            if self.use_llm:
                # Reset episode-specific trajectory at the start of each episode
                self.info_predictor.reset_episode()

            # Save Q-tables periodically
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
                
                if self.env.sar_robot.visited_information_state: # Check if we have new sensor readings and update attention/action toggle

                    # Update action toggle mechanism if enabled
                    if self.use_action_toggle and not self.input_received: # Update action toggle mechanism when fires are discovered
                        self.action_toggle.update_from_sensor_readings(self.env.sar_robot.sensor_readings)
                        self.input_received = True
                        print(f"Updated action toggle with fire information at episode {episode}")
                    
                     # Update attention mechanism if enabled
                    if self.use_attention and not self.input_received: # Check if we have new sensor readings and update attention
                        self._update_attention(self.env.sar_robot.sensor_readings)
                        print("=============================================")
                        self.input_received = True
                        print(f"Updated attention space with new information at episode {episode}")

                    # Set exploit mode for attention
                    if self.use_attention and not self.global_epsilon_exploit:
                        self.episode_exploit_mode = True

                s_, r, terminated, info, step_info = self._do_q_learning(s)
                # print(f'state: {s} | action: {step_info["action"]} - {step_info["action_name"]} | reward: {r} | next state: {s_} | done: {terminated}')
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

            # After episode completes, check if successful and track collisions -- new
            if episode_mission_completed:
                episode_collision_count = self.env.sar_robot.episode_collisions
                collision_counts_successful.append(episode_collision_count)
                successful_episode_collisions += episode_collision_count
                if episode_collision_count > 0:
                    successful_episodes_with_collisions += 1

            # Log the rewards and steps to Tensorboard
            if self.log_rewards_dir:
                with self.writer.as_default():
                    tf.summary.scalar('Episode Return', Rewards, step=episode)
                    tf.summary.scalar('Steps per Episode', steps_cnt, step=episode)
                    if episode_mission_completed:
                        tf.summary.scalar('Collisions in Successful Episode', self.env.sar_robot.episode_collisions, step=episode)
            
            # Update epsilon
            if self.use_attention and self.global_epsilon_exploit:
                # For attention with global exploit
                if not self.env.sar_robot.visited_information_state:
                    self.EPSILON = self._decay_epsilon(num_episodes)
                else:
                    self.EPSILON = self._decay_epsilon_exploit()
            else:
                # Standard epsilon decay
                self.EPSILON = self._decay_epsilon(num_episodes)
            
            total_rewards_per_episode[episode] = Rewards
            total_steps_per_episode[episode] = steps_cnt
            
        # Save final Q-tables
        if self.learned_policy_dir:
            self.save_learned_policy(num_episodes)

        # Calculate success rates
        mission_success_rate = (successful_episodes / num_episodes) * 100
        info_collection_success_rate = (info_collection_completed_episodes / num_episodes) * 100

        # Calculate collision metrics -- new
        collision_rate_successful = 0
        avg_collisions_per_success = 0
        mission_success_no_collisions_rate = 0
        if successful_episodes > 0:
            collision_rate_successful = (successful_episodes_with_collisions / successful_episodes) * 100
            avg_collisions_per_success = successful_episode_collisions / successful_episodes
            # Calculate mission success rate without collisions -- new
            mission_success_no_collisions_rate = ((successful_episodes - successful_episodes_with_collisions) / num_episodes) * 100
        ## -- new

        # Get collection statistics from the environment
        collection_stats = self.env.sar_robot.info_system.get_collection_stats()
        collection_success_rate = self.env.sar_robot.info_system.get_collection_success_rate()

        # Calculate exploration-specific collection success rate statistics -- new
        total_exploration_calls = sum(stats['calls'] for stats in self.predictor_stats.values())
        total_exploration_successes = sum(stats['successes'] for stats in self.predictor_stats.values())
        exploration_success_rate = (total_exploration_successes / max(1, total_exploration_calls)) * 100
        
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
            'total_hazard_collisions_in_successful_episodes': successful_episode_collisions, ## -- new
            'successful_episodes_with_collisions': successful_episodes_with_collisions, ## -- new
            'collision_rate_in_successful_episodes': collision_rate_successful, ## -- new
            'average_collisions_per_successful_episode': avg_collisions_per_success, ## -- new
            'collision_counts_per_successful_episode': collision_counts_successful, ## -- new
            'mission_success_no_collisions_rate': mission_success_no_collisions_rate,
            'llm_active': False,  # Flag indicating this was not an LLM-based agent
            'attention_active': self.use_attention,
            'action_toggle_active': self.use_action_toggle,
            'predictor_stats': {
                'total_calls': total_exploration_calls,
                'total_successes': total_exploration_successes,
                'overall_success_rate': exploration_success_rate,
                'by_location': self.predictor_stats
            }
        }

        # Add LLm-specific metrics if LLM is used
        if self.use_llm:
            # Calculate LLM timing statistics
            avg_llm_time = 0
            min_llm_time = 0
            max_llm_time = 0
            if self.llm_call_times:
                avg_llm_time = self.total_llm_time / self.total_llm_calls if self.total_llm_calls > 0 else 0
                min_llm_time = min(self.llm_call_times) if self.llm_call_times else 0
                max_llm_time = max(self.llm_call_times) if self.llm_call_times else 0
            
            metrics['collection_memory'] = self.info_predictor.collection_memory
            metrics['llm_timing'] = {
                'total_calls': self.total_llm_calls,
                'total_time': self.total_llm_time,
                'average_time_per_call': avg_llm_time,
                'min_time': min_llm_time,
                'max_time': max_llm_time
            }
        
        # print finla statistics
        self._print_training_statistics(metrics, successful_episodes)

        return total_rewards_per_episode, total_steps_per_episode, metrics


    def _print_training_statistics(self, metrics, successful_episodes):
        # Print final statistics
        print("\nTraining Complete!")

        print("\nExploration and Exploitation Statistics:")
        print(f"Total exploration actions: {self.exploration_count}")
        print(f"Total exploitation actions: {self.exploitation_count}")
        print(f"Final exploration/exploitation ratio: {self.exploration_count/(self.exploration_count + self.exploitation_count):.2f}")

        print("\nInformation Collection Statistics:") # Print training-wide collection statistics
        self.env.sar_robot.info_system.print_collection_stats()

        print("\nExploration Collection Statistics by Location:")
        for pos, stats in self.predictor_stats.items():
            success_rate = (stats['successes'] / max(1, stats['calls'])) * 100
            print(f"\nLocation {pos} (Info {stats['info_type']}, Collection Order {stats['collection_order']}):")
            print(f"  Predictor calls: {stats['calls']}")
            print(f"  Successful predictions: {stats['successes']}")
            print(f"  Success rate: {success_rate:.1f}%")

        total_exploration_calls = sum(stats['calls'] for stats in self.predictor_stats.values())
        total_exploration_successes = sum(stats['successes'] for stats in self.predictor_stats.values())
        exploration_success_rate = (total_exploration_successes / max(1, total_exploration_calls)) * 100
        print(f"\nTotal predictor calls across all locations: {total_exploration_calls}")
        print(f"Total successful predictions: {total_exploration_successes}")
        print(f"Overall predictor success rate: {exploration_success_rate:.1f}%")

        print("\nTask Success Rates:")
        print(f"Information collection success rate: {metrics['info_collection_success_rate']:.2f}%")
        print(f"Overall mission success rate: {metrics['mission_success_rate']:.2f}%")

        print("\nHazard Collision Statistics (Successful Episodes):")
        print(f"Total hazard collisions in successful episodes: {metrics['total_hazard_collisions_in_successful_episodes']}")
        print(f"Number of successful episodes with collisions: {metrics['successful_episodes_with_collisions']} out of {successful_episodes}")
        print(f"Collision rate in successful episodes: {metrics['collision_rate_in_successful_episodes']:.2f}%")
        print(f"Average collisions per successful episode: {metrics['average_collisions_per_successful_episode']:.2f}")
        print(f"Mission success rate *without collisions*: {metrics['mission_success_no_collisions_rate']:.2f}%")

        print("\nTraining Summary:")
        print(f"Average reward per episode: {metrics['average_reward_per_episode']:.2f}")
        print(f"Average steps per episode: {metrics['average_steps_per_episode']:.2f}")
        print(f"Best episode reward: {metrics['best_episode_reward']:.2f}")
        print(f"Worst episode reward: {metrics['worst_episode_reward']:.2f}")

        # Print LLM-specific statistics if enabled
        if self.use_llm:
            print("\nLLM Timing Statistics:")
            print(f"Total LLM calls: {metrics['llm_timing']['total_calls']}")
            print(f"Total LLM time: {metrics['llm_timing']['total_time']:.2f} seconds")
            print(f"Average time per LLM call: {metrics['llm_timing']['average_time_per_call']:.4f} seconds")
            print(f"Min time per call: {metrics['llm_timing']['min_time']:.4f} seconds")
            print(f"Max time per call: {metrics['llm_timing']['max_time']:.4f} seconds")

            print("\nFinal Collection Memory:")
            for loc, data in self.info_predictor.collection_memory.items():
                print(f"Location {loc}: {data}")
            print(f"Final collection order: {self.info_predictor.collection_order}")
        
### THIS IS THE MAIN FLAT AGENT ------------------------------


##########################################################

class QLearningAgentFlat:
    def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, DECAY_RATE, EPSILON_MIN, log_rewards_dir=None, learned_policy_dir=None):
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


        ### NEW
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
            'steps_to_adapt': 0
        })
        
        # Reset success rate tracking for measuring adaptation
        if self.priority_changes:
            change_index = len(self.priority_changes) - 1
            success_rate = self.env.sar_robot.info_system.get_collection_success_rate()
            self.priority_changes[change_index]['success_rate_before'] = success_rate
            self.priority_changes[change_index]['success_count_before'] = 0
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
            if change_info['success_count_after'] >= 3:
                change_info['adaptation_completed'] = True
                change_info['steps_to_adapt'] = (episode - change_info['episode']) * self.env.max_steps + steps_in_episode
                
                # Calculate order alignment rate
                if self.collection_order_attempts > 0:
                    order_alignment_rate = (self.collection_order_successes / self.collection_order_attempts) * 100
                else:
                    order_alignment_rate = 0
                
                # Record metrics after adaptation
                success_rate = self.env.sar_robot.info_system.get_collection_success_rate()
                change_info['success_rate_after'] = success_rate
                change_info['order_alignment_rate'] = order_alignment_rate
                
                print(f"\nAdaptation completed after {change_info['steps_to_adapt']} steps")
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
            
            # After episode completes, check if successful and track collisions
            if episode_mission_completed:
                episode_collision_count = self.env.sar_robot.episode_collisions
                collision_counts_successful.append(episode_collision_count)
                successful_episode_collisions += episode_collision_count
                if episode_collision_count > 0:
                    successful_episodes_with_collisions += 1
            ###########
            #### NEW #####
            # --- Epsilon handling (Boost check and Decay) ---
            if self.epsilon_boost_active:
                self.epsilon_boost_episodes_remaining -= 1
                if self.epsilon_boost_episodes_remaining <= 0:
                    self.epsilon_boost_active = False
                    restored_epsilon = self.original_epsilon_before_boost
                    print(f"\nEpisode {episode}: Epsilon boost ended. Restoring Epsilon from {self.EPSILON:.4f} towards {restored_epsilon:.4f}.")
                    self.EPSILON = restored_epsilon # Restore pre-boost value
                    self.original_epsilon_before_boost = None
                    # Now apply standard decay to the restored value for this episode
                    self.EPSILON = self._decay_epsilon(num_episodes)
                # else: Epsilon remains boosted, no decay applied this episode
                
            else: # Boost not active, apply standard decay
                self.EPSILON = self._decay_epsilon(num_episodes)
            # --- End Epsilon handling ---
            ### END NEW ###
            ###########

            # Log the rewards and steps to Tensorboard
            if self.log_rewards_dir:
                with self.writer.as_default():
                    tf.summary.scalar('Episode Return', Rewards, step=episode)
                    tf.summary.scalar('Steps per Episode', steps_cnt, step=episode)
                    tf.summary.scalar('Epsilon', self.EPSILON, step=episode)
                    if episode_mission_completed:
                        tf.summary.scalar('Collisions in Successful Episode', self.env.sar_robot.episode_collisions, step=episode)

            # self.EPSILON = self._decay_epsilon(num_episodes)
            total_rewards_per_episode[episode] = Rewards
            total_steps_per_episode[episode] = steps_cnt
            
        # Save final Q-table
        if self.learned_policy_dir:
            self.save_learned_policy(num_episodes)

        # Calculate success rates
        mission_success_rate = (successful_episodes / num_episodes) * 100
        info_collection_success_rate = (info_collection_completed_episodes / num_episodes) * 100

        # Calculate collision metrics
        collision_rate_successful = 0
        avg_collisions_per_success = 0
        mission_success_no_collisions_rate = 0
        if successful_episodes > 0:
            collision_rate_successful = (successful_episodes_with_collisions / successful_episodes) * 100
            avg_collisions_per_success = successful_episode_collisions / successful_episodes
            mission_success_no_collisions_rate = ((successful_episodes - successful_episodes_with_collisions) / num_episodes) * 100

        # Get collection statistics from the environment
        collection_stats = self.env.sar_robot.info_system.get_collection_stats()
        collection_success_rate = self.env.sar_robot.info_system.get_collection_success_rate()

        # Calculate exploration-specific collection success rate statistics
        total_exploration_calls = sum(stats['calls'] for stats in self.predictor_stats.values())
        total_exploration_successes = sum(stats['successes'] for stats in self.predictor_stats.values())
        exploration_success_rate = (total_exploration_successes / max(1, total_exploration_calls)) * 100
        
        # Analyze adaptation to priority changes
        priority_change_metrics = []
        if self.priority_changes:
            for change in self.priority_changes:
                if change['adaptation_completed']:
                    priority_change_metrics.append({
                        'episode': change['episode'],
                        'steps_to_adapt': change['steps_to_adapt'],
                        'success_rate_before': change.get('success_rate_before', 0),
                        'success_rate_after': change.get('success_rate_after', 0),
                        'order_alignment_rate': change.get('order_alignment_rate', 0)
                    })
        
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
            'collision_rate_in_successful_episodes': collision_rate_successful,
            'average_collisions_per_successful_episode': avg_collisions_per_success,
            'collision_counts_per_successful_episode': collision_counts_successful,
            'mission_success_no_collisions_rate': mission_success_no_collisions_rate,
            'llm_active': self.env.attention if hasattr(self.env, 'attention') else False,
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
        print(f"Collision rate in successful episodes: {collision_rate_successful:.2f}%")
        print(f"Average collisions per successful episode: {avg_collisions_per_success:.2f}")
        print(f"Mission success rate *without collisions*: {mission_success_no_collisions_rate:.2f}%")
        
        # Print priority change adaptation metrics
        if priority_change_metrics:
            print("\nPriority Change Adaptation Metrics:")
            for change in priority_change_metrics:
                print(f"Change at episode {change['episode']}:")
                print(f"  Steps to adapt: {change['steps_to_adapt']}")
                print(f"  Success rate before: {change['success_rate_before']:.1f}%")
                print(f"  Success rate after: {change['success_rate_after']:.1f}%")
                print(f"  Order alignment rate: {change['order_alignment_rate']:.1f}%")
                print(f"  Improvement: {change['success_rate_after'] - change['success_rate_before']:.1f}%")

        print("\nTraining Summary:")
        print(f"Average reward per episode: {np.mean(total_rewards_per_episode):.2f}")
        print(f"Average steps per episode: {np.mean(total_steps_per_episode):.2f}")
        print(f"Best episode reward: {np.max(total_rewards_per_episode):.2f}")
        print(f"Worst episode reward: {np.min(total_rewards_per_episode):.2f}")
        
        return total_rewards_per_episode, total_steps_per_episode, metrics


import matplotlib.pyplot as plt
class QLearningAgentMaxInfoRL_ADVANCED:
    def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, DECAY_RATE, EPSILON_MIN, log_rewards_dir=None, learned_policy_dir=None):
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

        ### NEW
        # Epsilon boost parameters for priority changes
        self.epsilon_boost_factor = 2.0  # How much to multiply epsilon by
        self.epsilon_boost_duration = 50 # How many episodes the boost lasts
        self.epsilon_boost_active = False
        self.epsilon_boost_episodes_remaining = 0
        self.original_epsilon_before_boost = None
    
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
            'steps_to_adapt': 0
        })
        
        # Reset success rate tracking for measuring adaptation
        if self.priority_changes:
            change_index = len(self.priority_changes) - 1
            success_rate = self.env.sar_robot.info_system.get_collection_success_rate()
            self.priority_changes[change_index]['success_rate_before'] = success_rate
            self.priority_changes[change_index]['success_count_before'] = 0
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
            info_location_reward = 2.0 / np.sqrt(info_count)  # Higher multiplier for info locations
        
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
            0.1 * general_novelty_reward +     # General exploration (30%)
            0.4 * info_location_reward +       # Info location exploration (30%)
            0.5 * order_alignment_reward       # Collection order alignment (40%)
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
            if change_info['success_count_after'] >= 3:
                change_info['adaptation_completed'] = True
                change_info['steps_to_adapt'] = (episode - change_info['episode']) * self.env.max_steps + steps_in_episode
                
                # Calculate order alignment rate (successful collections in order / total collection attempts)
                if self.collection_order_attempts > 0:
                    order_alignment_rate = (self.collection_order_successes / self.collection_order_attempts) * 100
                else:
                    order_alignment_rate = 0
                
                # Record metrics after adaptation
                success_rate = self.env.sar_robot.info_system.get_collection_success_rate()
                change_info['success_rate_after'] = success_rate
                change_info['order_alignment_rate'] = order_alignment_rate
                
                print(f"\nAdaptation completed after {change_info['steps_to_adapt']} steps")
                print(f"Success rate before change: {change_info['success_rate_before']:.1f}%")
                print(f"Success rate after adaptation: {success_rate:.1f}%")
                print(f"Order alignment rate: {order_alignment_rate:.1f}%")
                
                # Reset tracking for next potential change
                self.collection_order_attempts = 0
                self.collection_order_successes = 0
    
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
            # Check if we need to change priorities at this episode
            if change_priorities_at and episode in change_priorities_at:
                self.change_collection_priorities(change_priorities_at[episode], episode)
        
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

            # Check adaptation progress if priorities have changed
            self._check_adaptation_progress(episode, steps_cnt, episode_mission_completed)
            
            # After episode completes, check if successful and track collisions
            if episode_mission_completed:
                episode_collision_count = self.env.sar_robot.episode_collisions
                collision_counts_successful.append(episode_collision_count)
                successful_episode_collisions += episode_collision_count
                if episode_collision_count > 0:
                    successful_episodes_with_collisions += 1


            #### NEW #####
            # --- Epsilon handling (Boost check and Decay) ---
            if self.epsilon_boost_active:
                self.epsilon_boost_episodes_remaining -= 1
                if self.epsilon_boost_episodes_remaining <= 0:
                    self.epsilon_boost_active = False
                    restored_epsilon = self.original_epsilon_before_boost
                    print(f"\nEpisode {episode}: Epsilon boost ended. Restoring Epsilon from {self.EPSILON:.4f} towards {restored_epsilon:.4f}.")
                    self.EPSILON = restored_epsilon # Restore pre-boost value
                    self.original_epsilon_before_boost = None
                    # Now apply standard decay to the restored value for this episode
                    self.EPSILON = self._decay_epsilon(num_episodes)
                # else: Epsilon remains boosted, no decay applied this episode
                
            else: # Boost not active, apply standard decay
                self.EPSILON = self._decay_epsilon(num_episodes)
            # --- End Epsilon handling ---
            ### END NEW ###


            # Log the rewards and steps to Tensorboard
            if self.log_rewards_dir:
                with self.writer.as_default():
                    tf.summary.scalar('Episode Return', Rewards, step=episode)
                    tf.summary.scalar('Episode Return (Intrinsic)', intrinsic_rewards_sum, step=episode)
                    tf.summary.scalar('Steps per Episode', steps_cnt, step=episode)
                    tf.summary.scalar('Epsilon', self.EPSILON, step=episode)
                    if episode_mission_completed:
                        tf.summary.scalar('Collisions in Successful Episode', self.env.sar_robot.episode_collisions, step=episode)


            # self.EPSILON = self._decay_epsilon(num_episodes)
            total_rewards_per_episode[episode] = Rewards
            total_intrinsic_rewards_per_episode[episode] = intrinsic_rewards_sum
            total_steps_per_episode[episode] = steps_cnt
            
        # Save final Q-tables
        if self.learned_policy_dir:
            self.save_learned_policy(num_episodes)

        # Calculate success rates
        mission_success_rate = (successful_episodes / num_episodes) * 100
        info_collection_success_rate = (info_collection_completed_episodes / num_episodes) * 100

        # Calculate collision metrics
        collision_rate_successful = 0
        avg_collisions_per_success = 0
        mission_success_no_collisions_rate = 0
        if successful_episodes > 0:
            collision_rate_successful = (successful_episodes_with_collisions / successful_episodes) * 100
            avg_collisions_per_success = successful_episode_collisions / successful_episodes
            mission_success_no_collisions_rate = ((successful_episodes - successful_episodes_with_collisions) / num_episodes) * 100

        # Get collection statistics from the environment
        collection_stats = self.env.sar_robot.info_system.get_collection_stats()
        collection_success_rate = self.env.sar_robot.info_system.get_collection_success_rate()

        # Calculate exploration-specific collection success rate statistics
        total_exploration_calls = sum(stats['calls'] for stats in self.predictor_stats.values())
        total_exploration_successes = sum(stats['successes'] for stats in self.predictor_stats.values())
        exploration_success_rate = (total_exploration_successes / max(1, total_exploration_calls)) * 100
        
        # Analyze adaptation to priority changes
        priority_change_metrics = []
        if self.priority_changes:
            for change in self.priority_changes:
                if change['adaptation_completed']:
                    priority_change_metrics.append({
                        'episode': change['episode'],
                        'steps_to_adapt': change['steps_to_adapt'],
                        'success_rate_before': change.get('success_rate_before', 0),
                        'success_rate_after': change.get('success_rate_after', 0)
                    })
        
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
            'collision_rate_in_successful_episodes': collision_rate_successful,
            'average_collisions_per_successful_episode': avg_collisions_per_success,
            'collision_counts_per_successful_episode': collision_counts_successful,
            'mission_success_no_collisions_rate': mission_success_no_collisions_rate,
            'llm_active': self.env.attention if hasattr(self.env, 'attention') else False,
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
        print(f"Collision rate in successful episodes: {collision_rate_successful:.2f}%")
        print(f"Average collisions per successful episode: {avg_collisions_per_success:.2f}")
        print(f"Mission success rate *without collisions*: {mission_success_no_collisions_rate:.2f}%")
        
        # Print priority change adaptation metrics
        if priority_change_metrics:
            print("\nPriority Change Adaptation Metrics:")
            for change in priority_change_metrics:
                print(f"Change at episode {change['episode']}:")
                print(f"  Steps to adapt: {change['steps_to_adapt']}")
                print(f"  Success rate before: {change['success_rate_before']:.1f}%")
                print(f"  Success rate after: {change['success_rate_after']:.1f}%")
                print(f"  Improvement: {change['success_rate_after'] - change['success_rate_before']:.1f}%")

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
                [c['episode'] for c in self.priority_changes]
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
##########################################################

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

class QLearningAgentMaxInfoRL:
    def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, DECAY_RATE, EPSILON_MIN, log_rewards_dir=None, learned_policy_dir=None):
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
    


    # ## alternative 1 intrinsic reward function ------
    # """
    # Goal-Directed Exploration with Discovered Information
    # This only uses information the agent has actually discovered, combining pure exploration with informed exploration 
    # once information is discovered.
    # """
    # def compute_intrinsic_reward(self, state, action):
    #     # Only initialize once we've discovered some information
    #     if not hasattr(self, 'discovered_info_locations'):
    #         self.discovered_info_locations = []
        
    #     # Add info point to discovered list when we find one
    #     current_pos = (state[0], state[1])
    #     for loc in self.env.sar_robot.info_system.info_locations:
    #         if tuple(loc.position) == current_pos and tuple(loc.position) not in self.discovered_info_locations:
    #             # We've discovered a new info location!
    #             self.discovered_info_locations.append(tuple(loc.position))
    #             return 2.0  # High reward for discovery
        
    #     # Once we know where some info points are, start using them to guide exploration
    #     if self.discovered_info_locations and len(self.discovered_info_locations) < self.env.sar_robot.info_number_needed:
    #         # Calculate distance to nearest discovered but uncollected info
    #         uncollected_locations = [loc for loc in self.discovered_info_locations 
    #                             if not any(tuple(info.position) == loc and info.is_collected 
    #                                         for info in self.env.sar_robot.info_system.info_locations)]
            
    #         if uncollected_locations:
    #             distances = [abs(current_pos[0] - loc[0]) + abs(current_pos[1] - loc[1]) 
    #                         for loc in uncollected_locations]
    #             min_distance = min(distances)
    #             return 0.5 / (min_distance + 1)  # Higher reward when closer
        
    #     # Default: novelty-based exploration
    #     state_action = (state, action)
    #     if state_action not in self.visit_counts:
    #         self.visit_counts[state_action] = 0
    #     self.visit_counts[state_action] += 1
    #     return 1.0 / np.sqrt(self.visit_counts[state_action])
    # ## alternative 1 intrinsic reward function ------ 



    # ## alternative 2 intrinsic reward function ------
    # """
    # Learning Progress Motivation
    # This rewards the agent not just for visiting new states, but for finding states where its understanding 
    # is actively improving.
    # """
    # def compute_intrinsic_reward(self, state, action):
    #     if not hasattr(self, 'q_value_changes'):
    #         self.q_value_changes = {}
            
    #     state_action = (state, action)
    #     old_q_value = self.Q_extrinsic[state][action]
        
    #     # Take action and update Q-value
    #     next_obs, reward, _, _, _ = self.env.step(action)
    #     next_state = self._get_state(next_obs)
        
    #     # Perform Q-learning update
    #     best_next_action = np.argmax(self.Q_extrinsic[next_state])
    #     td_target = reward + self.GAMMA * self.Q_extrinsic[next_state][best_next_action]
    #     self.Q_extrinsic[state][action] += self.ALPHA * (td_target - old_q_value)
        
    #     # Calculate absolute change in Q-value
    #     new_q_value = self.Q_extrinsic[state][action]
    #     q_change = abs(new_q_value - old_q_value)
        
    #     # Store history of changes for this state-action
    #     if state_action not in self.q_value_changes:
    #         self.q_value_changes[state_action] = []
    #     self.q_value_changes[state_action].append(q_change)
        
    #     # Calculate change relative to recent changes (learning progress)
    #     if len(self.q_value_changes[state_action]) > 1:
    #         recent_changes = self.q_value_changes[state_action][-10:]
    #         learning_progress = q_change / (np.mean(recent_changes) + 1e-8)
    #         return min(learning_progress, 2.0)  # Cap reward for stability
        
    #     return 1.0  # Default for first visit
    ## alternative 2 intrinsic reward function ------
    
    def compute_intrinsic_reward(self, state, action):
        """
        Compute intrinsic reward based on visit counts as a proxy for information gain.
        Uses uncertainty estimate 1/sqrt(N(s,a)) as intrinsic reward.
        """
        # Increment visit count for this state-action pair
        self.visit_counts[(state, action)] += 1
        
        # Add small constant to avoid division by zero
        count = self.visit_counts[(state, action)]
        
        # Return information gain (inverse of visit count)
        return 1.0 / np.sqrt(count)
    
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
                
                action_name = _get_action_name(None, selected_action)
                
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
        
        # Take action in environment
        obs_, reward, terminated, _, info = self.env.step(action)
        next_state = self._get_state(obs_)
        
        # Compute intrinsic reward (information gain)
        intrinsic_reward = self.compute_intrinsic_reward(state, action)
        
        # Update both Q-tables
        self._update_q_tables(state, action, reward, next_state, intrinsic_reward)
        
        return next_state, reward, terminated, info, {
            "action": action, 
            "action_name": _get_action_name(None, action),
            "intrinsic_reward": intrinsic_reward
        }
    
    def train(self, num_episodes):
        total_rewards_per_episode = np.zeros(num_episodes)
        total_steps_per_episode = np.zeros(num_episodes)
        Rewards, steps_cnt, episode_return_Q = 0, 0, 0
        
        # For tracking intrinsic rewards
        total_intrinsic_rewards_per_episode = np.zeros(num_episodes)

        # For tracking success rates
        successful_episodes = 0
        info_collection_completed_episodes = 0

        # Add tracking variables for hazard collisions
        successful_episode_collisions = 0
        successful_episodes_with_collisions = 0
        collision_counts_successful = []

        for episode in tqdm(range(num_episodes)):
            if episode % 100 == 0:
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

            # After episode completes, check if successful and track collisions
            if episode_mission_completed:
                episode_collision_count = self.env.sar_robot.episode_collisions
                collision_counts_successful.append(episode_collision_count)
                successful_episode_collisions += episode_collision_count
                if episode_collision_count > 0:
                    successful_episodes_with_collisions += 1

            # Log the rewards and steps to Tensorboard
            if self.log_rewards_dir:
                with self.writer.as_default():
                    tf.summary.scalar('Episode Return', Rewards, step=episode)
                    tf.summary.scalar('Episode Return (Intrinsic)', intrinsic_rewards_sum, step=episode)
                    tf.summary.scalar('Steps per Episode', steps_cnt, step=episode)
                    if episode_mission_completed:
                        tf.summary.scalar('Collisions in Successful Episode', self.env.sar_robot.episode_collisions, step=episode)

            self.EPSILON = self._decay_epsilon(num_episodes)
            total_rewards_per_episode[episode] = Rewards
            total_intrinsic_rewards_per_episode[episode] = intrinsic_rewards_sum
            total_steps_per_episode[episode] = steps_cnt
            
        # Save final Q-tables
        if self.learned_policy_dir:
            self.save_learned_policy(num_episodes)

        # Calculate success rates
        mission_success_rate = (successful_episodes / num_episodes) * 100
        info_collection_success_rate = (info_collection_completed_episodes / num_episodes) * 100

        # Calculate collision metrics
        collision_rate_successful = 0
        avg_collisions_per_success = 0
        mission_success_no_collisions_rate = 0
        if successful_episodes > 0:
            collision_rate_successful = (successful_episodes_with_collisions / successful_episodes) * 100
            avg_collisions_per_success = successful_episode_collisions / successful_episodes
            mission_success_no_collisions_rate = ((successful_episodes - successful_episodes_with_collisions) / num_episodes) * 100

        # Get collection statistics from the environment
        collection_stats = self.env.sar_robot.info_system.get_collection_stats()
        collection_success_rate = self.env.sar_robot.info_system.get_collection_success_rate()

        # Calculate exploration-specific collection success rate statistics
        total_exploration_calls = sum(stats['calls'] for stats in self.predictor_stats.values())
        total_exploration_successes = sum(stats['successes'] for stats in self.predictor_stats.values())
        exploration_success_rate = (total_exploration_successes / max(1, total_exploration_calls)) * 100
        
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
            'collision_rate_in_successful_episodes': collision_rate_successful,
            'average_collisions_per_successful_episode': avg_collisions_per_success,
            'collision_counts_per_successful_episode': collision_counts_successful,
            'mission_success_no_collisions_rate': mission_success_no_collisions_rate,
            'llm_active': False,
            'predictor_stats': {
                'total_calls': total_exploration_calls,
                'total_successes': total_exploration_successes,
                'overall_success_rate': exploration_success_rate,
                'by_location': self.predictor_stats
            }
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
        print(f"Collision rate in successful episodes: {collision_rate_successful:.2f}%")
        print(f"Average collisions per successful episode: {avg_collisions_per_success:.2f}")
        print(f"Mission success rate *without collisions*: {mission_success_no_collisions_rate:.2f}%")

        print("\nTraining Summary:")
        print(f"Average extrinsic reward per episode: {np.mean(total_rewards_per_episode):.2f}")
        print(f"Average intrinsic reward per episode: {np.mean(total_intrinsic_rewards_per_episode):.2f}")
        print(f"Average steps per episode: {np.mean(total_steps_per_episode):.2f}")
        print(f"Best episode reward: {np.max(total_rewards_per_episode):.2f}")
        print(f"Worst episode reward: {np.min(total_rewards_per_episode):.2f}")
        
        return total_rewards_per_episode, total_steps_per_episode, metrics


# class QLearningAgentFlat:
#     def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, DECAY_RATE, EPSILON_MIN, log_rewards_dir=None, learned_policy_dir=None):
#         self.env = env
#         self.log_rewards_dir = log_rewards_dir
#         self.learned_policy_dir = learned_policy_dir 
#         self.ALPHA = ALPHA
#         self.GAMMA = GAMMA 
#         self.EPSILON_MAX = EPSILON_MAX
#         self.EPSILON = EPSILON_MAX
#         self.DECAY_RATE = DECAY_RATE
#         self.EPSILON_MIN = EPSILON_MIN
#         self.num_states = (self.env.observation_space.high[0] + 1, 
#                            self.env.observation_space.high[1] + 1, 
#                            self.env.observation_space.high[2] + 1,
#                            self.env.observation_space.high[3] + 1)  # 7*7*4*2
#         # Initialize Q-table only if action_space exists
#         if hasattr(self.env, 'action_space'):
#             self.Q_table = np.zeros((*self.num_states, self.env.action_space.n))
#         else:
#             self.Q_table = None
#         if self.log_rewards_dir:
#             self.writer = tf.summary.create_file_writer(log_rewards_dir)
#         self.save_interval = 500  # Save Q-table every 500 episodes
#         self.exploration_count = 0  # Exploration counter
#         self.exploitation_count = 0 # exploitation counter

#         # Additional tracking metrics
#         self.successful_episodes = 0  # Count of episodes where mission was completed
#         self.info_collection_completed = 0  # Count of episodes where all required info was collected
#         self.required_info_count = self.env.sar_robot.info_system.get_required_info_count()

#         # Initialize exploration statistics per information location (useful for tracking exploration actions on these locations)
#         self.predictor_stats = {
#             tuple(loc.position): {
#                 'calls': 0,
#                 'successes': 0,
#                 'collection_order': loc.collection_order,
#                 'info_type': loc.info_type
#             }
#             for loc in self.env.sar_robot.info_system.info_locations
#         }
        
    
#     def save_learned_policy(self, episode: Union[int, str], manager=None, workers=None):
#         """Save Q-tables for either flat or hierarchical agent"""
#         if not self.learned_policy_dir:
#             return
#         if not os.path.exists(self.learned_policy_dir):
#             os.makedirs(self.learned_policy_dir)
#         if manager is None or workers is None:  # Flat agent
#             filename = os.path.join(self.learned_policy_dir, f'q_table_episode_{episode}.npy')
#             np.save(filename, self.Q_table)
#             # print(f"Flat Q-table saved at episode {episode}")
#         else:  # Hierarchical agent
#             # Save manager Q-table
#             manager_filename = os.path.join(self.learned_policy_dir, 
#                                           f'manager_q_table_episode_{episode}.npy')
#             np.save(manager_filename, manager.Q_table)
#             # Save worker Q-tables
#             for option, worker in workers.items():
#                 worker_filename = os.path.join(self.learned_policy_dir, 
#                                              f'worker_{option}_q_table_episode_{episode}.npy')
#                 np.save(worker_filename, worker.Q_table)
#             # print(f"Hierarchical Q-tables saved at episode {episode}")
    
#     @staticmethod
#     def load_learned_policy(config_file: str) -> Dict:
#         """Load Q-tables based on configuration file"""
#         with open(config_file, 'r') as file:
#             config = json.load(file)
#         loaded_policies = {}
#         # Load flat Q-table if specified
#         if 'flat_q_table_file' in config:
#             filepath = os.path.expandvars(config['flat_q_table_file'])
#             if os.path.exists(filepath):
#                 loaded_policies['flat'] = np.load(filepath)
#             else:
#                 raise FileNotFoundError(f"No flat Q-table found at {filepath}")
#         # Load hierarchical Q-tables if specified
#         if 'hierarchical_q_tables' in config:
#             hierarchical_tables = {}
#             for key, filepath in config['hierarchical_q_tables'].items():
#                 expanded_path = os.path.expandvars(filepath)
#                 if os.path.exists(expanded_path):
#                     hierarchical_tables[key] = np.load(expanded_path)
#                 else:
#                     raise FileNotFoundError(f"No Q-table found at {expanded_path}")
#             loaded_policies['hierarchical'] = hierarchical_tables
#         return loaded_policies
    
#     def _epsilon_greedy_policy(self, state): # Epsilon-greedy policy
#         if np.random.rand() < self.EPSILON:
#             self.exploration_count += 1
#             # Check if we're at an info location during exploration
#             if self.env.sar_robot.info_system.is_at_info_location(state):
#                 current_pos = tuple([state[0], state[1]])
#                 # print(f"\n** At info location {current_pos} **")
#                 if current_pos in self.predictor_stats:
#                     self.predictor_stats[current_pos]['calls'] += 1
#                 selected_action = self.env.action_space.sample() # Explore: choose a random action
#                 action_name = _get_action_name(None, selected_action)
#                 # Check if prediction is correct
#                 is_correct = False
#                 info_locations = self.env.sar_robot.info_system.info_locations
#                 for location in info_locations:
#                     if (current_pos == tuple(location.position) and 
#                         state[2] == location.collection_order and 
#                         action_name == f'COLLECT_{location.info_type}'):
#                         is_correct = True
#                         break
#                 # update success statistics
#                 if is_correct and current_pos in self.predictor_stats:
#                     self.predictor_stats[current_pos]['successes'] += 1
#                 return selected_action
#             return self.env.action_space.sample() # Explore: choose a random action
#         else:
#             self.exploitation_count += 1
#             return np.argmax(self.Q_table[state]) # Exploit: choose the action with max Q-value
    
#     def _get_state(self, observation): # Get state from observation
#         return tuple(observation)

#     def _decay_epsilon(self, episodes): # Decay epsilon
#         if self.EPSILON > 0.1:
#             self.EPSILON -= self.DECAY_RATE/episodes
#         else:
#             self.EPSILON = self.EPSILON_MIN
#         return self.EPSILON
 
#     def _update(self, state, action, reward, next_state): 
#         # Q-learning update
#         best_next_action = np.argmax(self.Q_table[next_state])
#         td_target = reward + self.GAMMA * self.Q_table[next_state][best_next_action]
#         td_error = td_target - self.Q_table[state][action]
#         self.Q_table[state][action] += self.ALPHA * td_error
    
#     # Execute one step of the Q-learning algorithm
#     def _do_q_learning(self, state):
#         action = self._epsilon_greedy_policy(state)
#         obs_, reward, terminated, _, info = self.env.step(action)
#         next_state = self._get_state(obs_)
#         self._update(state, action, reward, next_state)
#         return next_state, reward, terminated, info, {
#             "action": action, 
#             "action_name": _get_action_name(None, action)}
    
#     def train(self, num_episodes):
#         total_rewards_per_episode = np.zeros(num_episodes)
#         total_steps_per_episode = np.zeros(num_episodes)
#         Rewards, steps_cnt, episode_return_Q = 0, 0, 0

#         # For tracking success rates
#         successful_episodes = 0
#         info_collection_completed_episodes = 0

#         # Add tracking variables for hazard collisions -- new
#         successful_episode_collisions = 0
#         successful_episodes_with_collisions = 0
#         collision_counts_successful = []

#         for episode in tqdm(range(num_episodes)):
#             # print(f"Episode {episode}")
#             if episode % 100 == 0:
#                 print(f"episode: {episode} | reward: {Rewards} | epsilon: {self.EPSILON}")
#             # Save Q-tables periodically
#             if self.learned_policy_dir and episode > 0 and episode % self.save_interval == 0:
#                 self.save_learned_policy(episode)
#             obs, _ = self.env.reset(seed=episode)
#             s = self._get_state(obs)
#             terminated = False
#             Rewards, steps_cnt, episode_return_Q = 0, 0, 0

#             # Episode-specific tracking
#             episode_mission_completed = False
#             episode_info_collected = False
#             while not terminated:
#                 s_, r, terminated, info, step_info = self._do_q_learning(s)
#                 # print(f'state: {s} | action: {step_info["action"]} - {step_info["action_name"]} | reward: {r} | next state: {s_} | done: {terminated}')
#                 Rewards += r
#                 episode_return_Q += r
#                 s = s_
#                 steps_cnt += 1

#                 # Check if the episode completed successfully
#                 if terminated and self.env.sar_robot.has_saved == 1:
#                     episode_mission_completed = True
#                     successful_episodes += 1
                
#                 # Check if all required information was collected
#                 if self.env.sar_robot.info_system.get_collected_info_count() >= self.required_info_count and not episode_info_collected:
#                     episode_info_collected = True
#                     info_collection_completed_episodes += 1

#             # After episode completes, check if successful and track collisions -- new
#             if episode_mission_completed:
#                 episode_collision_count = self.env.sar_robot.episode_collisions
#                 collision_counts_successful.append(episode_collision_count)
#                 successful_episode_collisions += episode_collision_count
#                 if episode_collision_count > 0:
#                     successful_episodes_with_collisions += 1

#             # Log the rewards and steps to Tensorboard
#             if self.log_rewards_dir:
#                 with self.writer.as_default():
#                     tf.summary.scalar('Episode Return', Rewards, step=episode)
#                     tf.summary.scalar('Steps per Episode', steps_cnt, step=episode)
#                     if episode_mission_completed:
#                         tf.summary.scalar('Collisions in Successful Episode', self.env.sar_robot.episode_collisions, step=episode)

#             self.EPSILON = self._decay_epsilon(num_episodes)
#             total_rewards_per_episode[episode] = Rewards
#             total_steps_per_episode[episode] = steps_cnt
            
#         # Save final Q-tables
#         if self.learned_policy_dir:
#             self.save_learned_policy(num_episodes)

#         # Calculate success rates
#         mission_success_rate = (successful_episodes / num_episodes) * 100
#         info_collection_success_rate = (info_collection_completed_episodes / num_episodes) * 100

#         # Calculate collision metrics -- new
#         collision_rate_successful = 0
#         avg_collisions_per_success = 0
#         mission_success_no_collisions_rate = 0
#         if successful_episodes > 0:
#             collision_rate_successful = (successful_episodes_with_collisions / successful_episodes) * 100
#             avg_collisions_per_success = successful_episode_collisions / successful_episodes
#             # Calculate mission success rate without collisions -- new
#             mission_success_no_collisions_rate = ((successful_episodes - successful_episodes_with_collisions) / num_episodes) * 100
#         ## -- new

#         # Get collection statistics from the environment
#         collection_stats = self.env.sar_robot.info_system.get_collection_stats()
#         collection_success_rate = self.env.sar_robot.info_system.get_collection_success_rate()

#         # Calculate exploration-specific collection success rate statistics -- new
#         total_exploration_calls = sum(stats['calls'] for stats in self.predictor_stats.values())
#         total_exploration_successes = sum(stats['successes'] for stats in self.predictor_stats.values())
#         exploration_success_rate = (total_exploration_successes / max(1, total_exploration_calls)) * 100
        
#         # Store metrics in dictionary for return
#         metrics = {
#             'total_exploration_actions': self.exploration_count,
#             'total_exploitation_actions': self.exploitation_count,
#             'exploration_exploitation_ratio': self.exploration_count / (self.exploration_count + self.exploitation_count),
#             'average_reward_per_episode': np.mean(total_rewards_per_episode),
#             'average_steps_per_episode': np.mean(total_steps_per_episode),
#             'best_episode_reward': np.max(total_rewards_per_episode),
#             'worst_episode_reward': np.min(total_rewards_per_episode),
#             'mission_success_rate': mission_success_rate,
#             'info_collection_success_rate': info_collection_success_rate,
#             'collection_success_rate': collection_success_rate,
#             'collection_stats': collection_stats,
#             'total_hazard_collisions_in_successful_episodes': successful_episode_collisions, ## -- new
#             'successful_episodes_with_collisions': successful_episodes_with_collisions, ## -- new
#             'collision_rate_in_successful_episodes': collision_rate_successful, ## -- new
#             'average_collisions_per_successful_episode': avg_collisions_per_success, ## -- new
#             'collision_counts_per_successful_episode': collision_counts_successful, ## -- new
#             'mission_success_no_collisions_rate': mission_success_no_collisions_rate,
#             'llm_active': False,  # Flag indicating this was not an LLM-based agent
#             'predictor_stats': {
#                 'total_calls': total_exploration_calls,
#                 'total_successes': total_exploration_successes,
#                 'overall_success_rate': exploration_success_rate,
#                 'by_location': self.predictor_stats
#             }
#         }

#         # Print final statistics
#         print("\nTraining Complete!")
#         print("\nExploration and Exploitation Statistics:")
#         print(f"Total exploration actions: {self.exploration_count}")
#         print(f"Total exploitation actions: {self.exploitation_count}")
#         print(f"Final exploration/exploitation ratio: {self.exploration_count/(self.exploration_count + self.exploitation_count):.2f}")
        
#         print("\nInformation Collection Statistics:") # Print training-wide collection statistics
#         self.env.sar_robot.info_system.print_collection_stats()

#         ## -- new
#         print("\nExploration Collection Statistics by Location:")
#         for pos, stats in self.predictor_stats.items():
#             success_rate = (stats['successes'] / max(1, stats['calls'])) * 100
#             print(f"\nLocation {pos} (Info {stats['info_type']}, Collection Order {stats['collection_order']}):")
#             print(f"  Predictor calls: {stats['calls']}")
#             print(f"  Successful predictions: {stats['successes']}")
#             print(f"  Success rate: {success_rate:.1f}%")
#         print(f"\nTotal predictor calls across all locations: {total_exploration_calls}")
#         print(f"Total successful predictions: {total_exploration_successes}")
#         print(f"Overall predictor success rate: {exploration_success_rate:.1f}%")
#         ## -- new

#         print("\nTask Success Rates:")
#         print(f"Information collection success rate: {info_collection_success_rate:.2f}%")
#         print(f"Overall mission success rate: {mission_success_rate:.2f}%")
        
#         print("\nHazard Collision Statistics (Successful Episodes):")
#         print(f"Total hazard collisions in successful episodes: {successful_episode_collisions}")
#         print(f"Number of successful episodes with collisions: {successful_episodes_with_collisions} out of {successful_episodes}")
#         print(f"Collision rate in successful episodes: {collision_rate_successful:.2f}%")
#         print(f"Average collisions per successful episode: {avg_collisions_per_success:.2f}")
#         print(f"Mission success rate *without collisions*: {mission_success_no_collisions_rate:.2f}%")

#         print("\nTraining Summary:") # Print summary metrics
#         print(f"Average reward per episode: {np.mean(total_rewards_per_episode):.2f}")
#         print(f"Average steps per episode: {np.mean(total_steps_per_episode):.2f}")
#         print(f"Best episode reward: {np.max(total_rewards_per_episode):.2f}")
#         print(f"Worst episode reward: {np.min(total_rewards_per_episode):.2f}")   
#         return total_rewards_per_episode, total_steps_per_episode, metrics


## new experimental with experience accumulated so that the LLM is indeed an exploration facilitator
class QLearningAgentFlatLLM(QLearningAgentFlat):
    def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, DECAY_RATE, EPSILON_MIN, log_rewards_dir=None, learned_policy_dir=None):
        super().__init__(env, ALPHA, GAMMA, EPSILON_MAX, DECAY_RATE, EPSILON_MIN, log_rewards_dir, learned_policy_dir)
        self.info_predictor = InformationPredictor()

        # Add timing metrics
        self.llm_call_times = []
        self.total_llm_calls = 0
        self.total_llm_time = 0

    def _epsilon_greedy_policy(self, state):
        if np.random.rand() < self.EPSILON:
            self.exploration_count += 1

            # Check if we're at an info location during exploration
            if self.env.sar_robot.info_system.is_at_info_location(state):
                current_pos = tuple([state[0], state[1]])
                # print(f"\n** At info location {current_pos} **")
                if current_pos in self.predictor_stats:
                    self.predictor_stats[current_pos]['calls'] += 1
                    # print(f"Predictor calls for this location: {self.predictor_stats[current_pos]['calls']}")

                start_time = time.time()  # Start timing the LLM call
                predicted_action, is_correct = self.info_predictor.predict(state, self.env) # Get prediction from InformationPredictor
                end_time = time.time()  # End timing the LLM call

                # Record timing
                elapsed_time = end_time - start_time
                self.llm_call_times.append(elapsed_time)
                self.total_llm_calls += 1
                self.total_llm_time += elapsed_time

                if predicted_action != 'NONE':
                    action_index = self.env.sar_robot.info_system.get_collection_action_index(predicted_action, hierarchical=False)
                    if action_index is not None:
                        # update success statistics
                        if is_correct and current_pos in self.predictor_stats:
                            self.predictor_stats[current_pos]['successes'] += 1

                        # Extract info type from the action name
                        if predicted_action.startswith("COLLECT_"):
                            info_type = predicted_action.replace("COLLECT_", "")

                            # Record the prediction attempt in memory
                            # is_correct tells us if this matches ground truth
                            self.info_predictor.record_attempt(current_pos, info_type, is_correct) # Record attempt for InformationPredictor

                        return action_index
            return self.env.action_space.sample() # If not at info location or prediction failed, use random action
        else:
            self.exploitation_count += 1
            return np.argmax(self.Q_table[state])
    
    def _do_q_learning(self, state):
        action = self._epsilon_greedy_policy(state)
        obs_, reward, terminated, _, info = self.env.step(action)
        next_state = self._get_state(obs_)

        self.info_predictor.record_step(state, action, next_state, reward) # Just record the step in the trajectory for InformationPredictor

        self._update(state, action, reward, next_state)
        return next_state, reward, terminated, info, {
            "action": action, 
            "action_name": _get_action_name(None, action)}

    
    def train(self, num_episodes):
        """Train the agent"""
        total_rewards_per_episode = np.zeros(num_episodes)
        total_steps_per_episode = np.zeros(num_episodes)
        Rewards, steps_cnt, episode_return_Q = 0, 0, 0

        # For tracking success rates
        successful_episodes = 0
        info_collection_completed_episodes = 0

        # Add tracking variables for hazard collisions -- new
        successful_episode_collisions = 0
        successful_episodes_with_collisions = 0
        collision_counts_successful = []

        for episode in tqdm(range(num_episodes)):
            self.current_episode = episode
            # print(f"\n==== Starting Episode {episode} ====")
            if episode % 100 == 0:
                print(f"episode: {episode} | reward: {Rewards} | epsilon: {self.EPSILON}")
            
            # Reset episode-specific trajectory at the start of each episode
            self.info_predictor.reset_episode()
            
            # Save Q-tables periodically
            if self.learned_policy_dir and episode > 0 and episode % self.save_interval == 0:
                self.save_learned_policy(episode)
            obs, _ = self.env.reset(seed=episode)
            s = self._get_state(obs)
            terminated = False
            Rewards, steps_cnt, episode_return_Q = 0, 0, 0
            self.linguistic_descriptions = []

            # Episode-specific tracking
            episode_mission_completed = False
            episode_info_collected = False
            while not terminated:
                s_, r, terminated, info, step_info = self._do_q_learning(s)
                Rewards += r
                # print(f'state: {s} | action: {step_info["action"]} - {step_info["action_name"]} | reward: {r} | next state: {s_} | done: {terminated}')
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
            
            # After episode completes, check if successful and track collisions -- new
            if episode_mission_completed:
                episode_collision_count = self.env.sar_robot.episode_collisions
                collision_counts_successful.append(episode_collision_count)
                successful_episode_collisions += episode_collision_count
                if episode_collision_count > 0:
                    successful_episodes_with_collisions += 1

            # Log episode results
            if self.log_rewards_dir:
                with self.writer.as_default():
                    tf.summary.scalar('Episode Return', Rewards, step=episode)
                    tf.summary.scalar('Steps per Episode', steps_cnt, step=episode)
                    if episode_mission_completed:
                        tf.summary.scalar('Collisions in Successful Episode', self.env.sar_robot.episode_collisions, step=episode)
            # Update tracking variables
            self.EPSILON = self._decay_epsilon(num_episodes)
            total_rewards_per_episode[episode] = Rewards
            total_steps_per_episode[episode] = steps_cnt
            
        # Save final Q-tables
        if self.learned_policy_dir:
            self.save_learned_policy(num_episodes)
        

        # Calculate success rates
        mission_success_rate = (successful_episodes / num_episodes) * 100
        info_collection_success_rate = (info_collection_completed_episodes / num_episodes) * 100

        # Calculate collision metrics -- new
        collision_rate_successful = 0
        avg_collisions_per_success = 0
        mission_success_no_collisions_rate = 0
        if successful_episodes > 0:
            collision_rate_successful = (successful_episodes_with_collisions / successful_episodes) * 100
            avg_collisions_per_success = successful_episode_collisions / successful_episodes
            # Calculate mission success rate without collisions -- new
            mission_success_no_collisions_rate = ((successful_episodes - successful_episodes_with_collisions) / num_episodes) * 100
        ## -- new
        
        # Get collection statistics from the environment
        collection_stats = self.env.sar_robot.info_system.get_collection_stats()
        collection_success_rate = self.env.sar_robot.info_system.get_collection_success_rate()
        
        # Calculate LLM predictor statistics
        total_predictor_calls = sum(stats['calls'] for stats in self.predictor_stats.values())
        total_predictor_successes = sum(stats['successes'] for stats in self.predictor_stats.values())
        predictor_success_rate = (total_predictor_successes / max(1, total_predictor_calls)) * 100

        # Calculate LLM timing statistics
        avg_llm_time = 0
        min_llm_time = 0
        max_llm_time = 0
        if self.llm_call_times:
            avg_llm_time = self.total_llm_time / self.total_llm_calls if self.total_llm_calls > 0 else 0
            min_llm_time = min(self.llm_call_times) if self.llm_call_times else 0
            max_llm_time = max(self.llm_call_times) if self.llm_call_times else 0
        

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
            'total_hazard_collisions_in_successful_episodes': successful_episode_collisions, ## -- new
            'successful_episodes_with_collisions': successful_episodes_with_collisions, ## -- new
            'collision_rate_in_successful_episodes': collision_rate_successful, ## -- new
            'average_collisions_per_successful_episode': avg_collisions_per_success, ## -- new
            'collision_counts_per_successful_episode': collision_counts_successful, ## -- new
            'mission_success_no_collisions_rate': mission_success_no_collisions_rate,
            'llm_active': True,  # Flag indicating this was an LLM-based agent
            'predictor_stats': {
                'total_calls': total_predictor_calls,
                'total_successes': total_predictor_successes,
                'overall_success_rate': predictor_success_rate,
                'by_location': self.predictor_stats
            },
            'collection_memory': self.info_predictor.collection_memory,
            'llm_timing': {
                'total_calls': self.total_llm_calls,
                'total_time': self.total_llm_time,
                'average_time_per_call': avg_llm_time,
                'min_time': min_llm_time,
                'max_time': max_llm_time
            }
        }

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

        ## -- new
        print("\nHazard Collision Statistics (Successful Episodes):")
        print(f"Total hazard collisions in successful episodes: {successful_episode_collisions}")
        print(f"Number of successful episodes with collisions: {successful_episodes_with_collisions} out of {successful_episodes}")
        print(f"Collision rate in successful episodes: {collision_rate_successful:.2f}%")
        print(f"Average collisions per successful episode: {avg_collisions_per_success:.2f}")
        print(f"Mission success rate *without collisions*: {mission_success_no_collisions_rate:.2f}%")
        ## -- new
        
        print("\nTraining Summary:")
        print(f"Average reward per episode: {np.mean(total_rewards_per_episode):.2f}")
        print(f"Average steps per episode: {np.mean(total_steps_per_episode):.2f}")
        print(f"Best episode reward: {np.max(total_rewards_per_episode):.2f}")
        print(f"Worst episode reward: {np.min(total_rewards_per_episode):.2f}")

        print("\nLLM Predictor in ON - Statistics by Location:")
        for pos, stats in self.predictor_stats.items():
            success_rate = (stats['successes'] / max(1, stats['calls'])) * 100
            print(f"\nLocation {pos} (Info {stats['info_type']}, Collection Order {stats['collection_order']}):")
            print(f"  Predictor calls: {stats['calls']}")
            print(f"  Successful predictions: {stats['successes']}")
            print(f"  Success rate: {success_rate:.1f}%")
        print(f"\nTotal predictor calls across all locations: {total_predictor_calls}")
        print(f"Total successful predictions: {total_predictor_successes}")
        print(f"Overall predictor success rate: {predictor_success_rate:.1f}%")

        # Print LLM timing statistics
        print("\nLLM Timing Statistics:")
        print(f"Total LLM calls: {self.total_llm_calls}")
        print(f"Total LLM time: {self.total_llm_time:.2f} seconds")
        print(f"Average time per LLM call: {avg_llm_time:.4f} seconds")
        print(f"Min time per call: {min_llm_time:.4f} seconds")
        print(f"Max time per call: {max_llm_time:.4f} seconds")

        # Print final collection memory
        print("\nFinal Collection Memory:")
        for loc, data in self.info_predictor.collection_memory.items():
            print(f"Location {loc}: {data}")
        print(f"Final collection order: {self.info_predictor.collection_order}")
        
        return total_rewards_per_episode, total_steps_per_episode, metrics


## done + new
class QLearningAgentFlatAttention(QLearningAgentFlat):
    def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, DECAY_RATE, EPSILON_MIN, log_rewards_dir=None, learned_policy_dir=None):
        super().__init__(env, ALPHA, GAMMA, EPSILON_MAX, DECAY_RATE, EPSILON_MIN, log_rewards_dir, learned_policy_dir)
        self.attention_space = AttentionSpace(self.env)  # Instantiate the new AttentionSpace class
        self.input_received = False
        self.episode_exploit_mode = False  # Flag to indicate if the agent is in exploit mode for the current episode
        self.global_epsilon_exploit = False  # Flag to indicate if the agent is in exploit mode for the entire training

    # Modified epsilon-greedy policy that considers information state
    def _epsilon_greedy_policy(self, state):
        if self.global_epsilon_exploit:
            if not self.env.sar_robot.visited_information_state:
                return super()._epsilon_greedy_policy(state)
            else:
                self.exploitation_count += 1
                return np.argmax(self.Q_table[state])
        else:
            if not self.episode_exploit_mode:
                return super()._epsilon_greedy_policy(state)
            else:
                self.exploitation_count += 1
                return np.argmax(self.Q_table[state])

    def _decay_epsilon_exploit(self):
        self.EPSILON = 0.005 # Set epsilon to a very low value for checking if it indeed works
        return self.EPSILON

    def _update_attention(self, sensor_readings):
        changed_states = self.attention_space._identify_changed_states(sensor_readings)
        # print(f"Changed states: {changed_states}\n")
        if changed_states:
            for state in changed_states:
                self.attention_space._update_attention_space(state, sensor_readings)
            self.attention_space._apply_attention_to_q_table(self.Q_table)

    def train(self, num_episodes):
        total_rewards_per_episode = np.zeros(num_episodes)
        total_steps_per_episode = np.zeros(num_episodes)
        Rewards = 0

        # For tracking success rates
        successful_episodes = 0
        info_collection_completed_episodes = 0

        # Add tracking variables for hazard collisions -- new
        successful_episode_collisions = 0
        successful_episodes_with_collisions = 0
        collision_counts_successful = []

        for episode in tqdm(range(num_episodes)):

            self.episode_exploit_mode = False  # Reset exploit mode for each episode
            
            if episode % 100 == 0:
                print(f"episode: {episode} | reward: {Rewards} | epsilon: {self.EPSILON}")
            # Save Q-tables periodically
            if self.learned_policy_dir and episode > 0 and episode % self.save_interval == 0:
                self.save_learned_policy(episode)
            obs, _ = self.env.reset(seed=episode)
            s = self._get_state(obs)
            terminated = False
            Rewards, steps_cnt = 0, 0

            # Episode-specific tracking
            episode_mission_completed = False
            episode_info_collected = False
            while not terminated:

                # Check if we have new sensor readings and update attention
                if self.env.sar_robot.visited_information_state and not self.input_received:
                    self._update_attention(self.env.sar_robot.sensor_readings)
                    print("=============================================")
                    self.input_received = True
                    print(f"Updated attention space with new information at episode {episode}")
                
                if self.env.sar_robot.visited_information_state and not self.global_epsilon_exploit:
                    self.episode_exploit_mode = True  # Set exploit mode for the episode
                
                s_, r, terminated, info, step_info = self._do_q_learning(s)
                # print(f'state: {s} | action: {step_info["action"]} - {step_info["action_name"]} | reward: {r} | next state: {s_} | done: {terminated}')
                s = s_
                Rewards += r
                steps_cnt += 1

                # Check if the episode completed successfully
                if terminated and self.env.sar_robot.has_saved == 1:
                    episode_mission_completed = True
                    successful_episodes += 1
                
                # Check if all required information was collected
                if self.env.sar_robot.info_system.get_collected_info_count() >= self.required_info_count and not episode_info_collected:
                    episode_info_collected = True
                    info_collection_completed_episodes += 1

            # After episode completes, check if successful and track collisions -- new
            if episode_mission_completed:
                episode_collision_count = self.env.sar_robot.episode_collisions
                collision_counts_successful.append(episode_collision_count)
                successful_episode_collisions += episode_collision_count
                if episode_collision_count > 0:
                    successful_episodes_with_collisions += 1

            # Log rewards and steps to Tensorboard
            if self.log_rewards_dir:
                with self.writer.as_default():
                    tf.summary.scalar('Episode Return', Rewards, step=episode)
                    tf.summary.scalar('Steps per Episode', steps_cnt, step=episode)
                    if episode_mission_completed:
                        tf.summary.scalar('Collisions in Successful Episode', self.env.sar_robot.episode_collisions, step=episode)

            if not self.global_epsilon_exploit:
                self.EPSILON = self._decay_epsilon(num_episodes)
            else:
                # Adjust epsilon for exploration
                if not self.env.sar_robot.visited_information_state:
                    self.EPSILON = self._decay_epsilon(num_episodes)
                else:
                    self.EPSILON = self._decay_epsilon_exploit()

            total_rewards_per_episode[episode] = Rewards
            total_steps_per_episode[episode] = steps_cnt
        # Save final Q-table
        if self.learned_policy_dir:
            self.save_learned_policy(num_episodes)

        # Calculate success rates
        mission_success_rate = (successful_episodes / num_episodes) * 100
        info_collection_success_rate = (info_collection_completed_episodes / num_episodes) * 100

        # Calculate collision metrics -- new
        collision_rate_successful = 0
        avg_collisions_per_success = 0
        mission_success_no_collisions_rate = 0
        if successful_episodes > 0:
            collision_rate_successful = (successful_episodes_with_collisions / successful_episodes) * 100
            avg_collisions_per_success = successful_episode_collisions / successful_episodes
            # Calculate mission success rate without collisions -- new
            mission_success_no_collisions_rate = ((successful_episodes - successful_episodes_with_collisions) / num_episodes) * 100
        ## -- new
        
        # Get collection statistics from the environment
        collection_stats = self.env.sar_robot.info_system.get_collection_stats()
        collection_success_rate = self.env.sar_robot.info_system.get_collection_success_rate()

        # Calculate exploration-specific collection success rate statistics -- new
        total_exploration_calls = sum(stats['calls'] for stats in self.predictor_stats.values())
        total_exploration_successes = sum(stats['successes'] for stats in self.predictor_stats.values())
        exploration_success_rate = (total_exploration_successes / max(1, total_exploration_calls)) * 100

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
            'total_hazard_collisions_in_successful_episodes': successful_episode_collisions, ## -- new
            'successful_episodes_with_collisions': successful_episodes_with_collisions, ## -- new
            'collision_rate_in_successful_episodes': collision_rate_successful, ## -- new
            'average_collisions_per_successful_episode': avg_collisions_per_success, ## -- new
            'collision_counts_per_successful_episode': collision_counts_successful, ## -- new
            'mission_success_no_collisions_rate': mission_success_no_collisions_rate,
            'attention_active': True,  # Flag indicating this was an attention-based agent
            'llm_active': False,  # Flag indicating this was not an LLM-based agent
            'predictor_stats': {
                'total_calls': total_exploration_calls,
                'total_successes': total_exploration_successes,
                'overall_success_rate': exploration_success_rate,
                'by_location': self.predictor_stats
            }
        }

        # Print final statistics
        print("\nTraining Complete!")
        print("\nExploration and Exploitation Statistics:")
        print(f"Total exploration actions: {self.exploration_count}")
        print(f"Total exploitation actions: {self.exploitation_count}")
        print(f"Final exploration/exploitation ratio: {self.exploration_count/(self.exploration_count + self.exploitation_count):.2f}")
        
        print("\nInformation Collection Statistics:") # Print training-wide collection statistics
        self.env.sar_robot.info_system.print_collection_stats()

        ## -- new
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
        ## -- new

        print("\nTask Success Rates:")
        print(f"Information collection success rate: {info_collection_success_rate:.2f}%")
        print(f"Overall mission success rate: {mission_success_rate:.2f}%")

        ## -- new
        print("\nHazard Collision Statistics (Successful Episodes):")
        print(f"Total hazard collisions in successful episodes: {successful_episode_collisions}")
        print(f"Number of successful episodes with collisions: {successful_episodes_with_collisions} out of {successful_episodes}")
        print(f"Collision rate in successful episodes: {collision_rate_successful:.2f}%")
        print(f"Average collisions per successful episode: {avg_collisions_per_success:.2f}")
        print(f"Mission success rate *without collisions*: {mission_success_no_collisions_rate:.2f}%")
        ## -- new
        
        print("\nTraining Summary:") # Print summary metrics
        print(f"Average reward per episode: {np.mean(total_rewards_per_episode):.2f}")
        print(f"Average steps per episode: {np.mean(total_steps_per_episode):.2f}")
        print(f"Best episode reward: {np.max(total_rewards_per_episode):.2f}")
        print(f"Worst episode reward: {np.min(total_rewards_per_episode):.2f}") 
        return total_rewards_per_episode, total_steps_per_episode, metrics


class QLearningAgentFlatActionToggle(QLearningAgentFlat):
    """
    Flat Q-learning agent with action toggling mechanism.
    This agent completely removes actions that would lead to undesirable states.
    """
    def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, DECAY_RATE, EPSILON_MIN, log_rewards_dir=None, learned_policy_dir=None):
        super().__init__(env, ALPHA, GAMMA, EPSILON_MAX, DECAY_RATE, EPSILON_MIN, log_rewards_dir, learned_policy_dir)
        # self.attention_space = AttentionSpace(self.env)  # Instantiate the new AttentionSpace class
        self.action_toggle = ActionToggleMechanism(self.env)
        self.input_received = False
    
    def _epsilon_greedy_policy(self, state):
        if not self.env.sar_robot.visited_information_state:
            return super()._epsilon_greedy_policy(state)
        else:
            # Get the valid actions for this state (excluding those leading to fires)
            valid_actions = self.action_toggle.get_valid_actions(state)
            # print(f"Valid actions: {valid_actions} at state {state}")
            # if len(valid_actions) == self.env.action_space.n:
                # print(f"This {state} is no adjacent to any danger and has the original action space")
        
            if np.random.rand() < self.EPSILON:
                self.exploration_count += 1
                # Check if we're at an info location during exploration
                if self.env.sar_robot.info_system.is_at_info_location(state):
                    current_pos = tuple([state[0], state[1]])
                    # print(f"\n** At info location {current_pos} **")
                    if current_pos in self.predictor_stats:
                        self.predictor_stats[current_pos]['calls'] += 1
                    selected_action = np.random.choice(valid_actions) # Explore: choose a random action
                    action_name = _get_action_name(None, selected_action)
                    # Check if prediction is correct
                    is_correct = False
                    info_locations = self.env.sar_robot.info_system.info_locations
                    for location in info_locations:
                        if (current_pos == tuple(location.position) and 
                            state[2] == location.collection_order and 
                            action_name == f'COLLECT_{location.info_type}'):
                            is_correct = True
                            break
                    # update success statistics
                    if is_correct and current_pos in self.predictor_stats:
                        self.predictor_stats[current_pos]['successes'] += 1
                    return selected_action

                # Explore: randomly sample from valid actions only
                return np.random.choice(valid_actions)
            else:
                self.exploitation_count += 1
                # Exploit: choose best action from valid actions only
                q_values = [self.Q_table[state][a] for a in valid_actions]
                return valid_actions[np.argmax(q_values)]
    
    def _update(self, state, action, reward, next_state):
        """
        Override the parent method to update the Q-table only for valid actions when computing the target Q-value.
        """
        if not self.env.sar_robot.visited_information_state:
            super()._update(state, action, reward, next_state)
        else:
            # get valid actions for the next state
            valid_next_actions = self.action_toggle.get_valid_actions(next_state)

            # calculate the target Q-value using the valid actions only
            next_q_values = [self.Q_table[next_state][a] for a in valid_next_actions]
            best_next_action = valid_next_actions[np.argmax(next_q_values)]

            # update the Q-table
            td_target = reward + self.GAMMA * self.Q_table[next_state][best_next_action]
            td_error = td_target - self.Q_table[state][action]
            self.Q_table[state][action] += self.ALPHA * td_error
    
    def train(self, num_episodes):
        total_rewards_per_episode = np.zeros(num_episodes)
        total_steps_per_episode = np.zeros(num_episodes)
        Rewards = 0

        # For tracking success rates
        successful_episodes = 0
        info_collection_completed_episodes = 0

        # Add tracking variables for hazard collisions -- new
        successful_episode_collisions = 0
        successful_episodes_with_collisions = 0
        collision_counts_successful = []
        
        for episode in tqdm(range(num_episodes)):
            if episode % 100 == 0:
                print(f"Episode: {episode} | Reward: {Rewards} | Epsilon: {self.EPSILON}")
            
            # Save Q-tables periodically
            if self.learned_policy_dir and episode > 0 and episode % self.save_interval == 0:
                self.save_learned_policy(episode)
            
            obs, _ = self.env.reset(seed=episode)
            s = self._get_state(obs)
            terminated = False
            Rewards, steps_cnt = 0, 0
            
            # Episode-specific tracking
            episode_mission_completed = False
            episode_info_collected = False
            while not terminated:
                # Update action toggle mechanism when fires are discovered
                if self.env.sar_robot.visited_information_state and not self.input_received:
                    self.action_toggle.update_from_sensor_readings(self.env.sar_robot.sensor_readings)
                    self.input_received = True
                    print(f"Updated action toggle with fire information at episode {episode}")
                
                s_, r, terminated, info, step_info = self._do_q_learning(s)
                # print(f'state: {s} | action: {step_info["action"]} - {step_info["action_name"]} | reward: {r} | next state: {s_} | done: {terminated}')
                s = s_
                Rewards += r
                steps_cnt += 1

                # Check if the episode completed successfully
                if terminated and self.env.sar_robot.has_saved == 1:
                    episode_mission_completed = True
                    successful_episodes += 1
                
                # Check if all required information was collected
                if self.env.sar_robot.info_system.get_collected_info_count() >= self.required_info_count and not episode_info_collected:
                    episode_info_collected = True
                    info_collection_completed_episodes += 1
            
            # After episode completes, check if successful and track collisions -- new
            if episode_mission_completed:
                episode_collision_count = self.env.sar_robot.episode_collisions
                collision_counts_successful.append(episode_collision_count)
                successful_episode_collisions += episode_collision_count
                if episode_collision_count > 0:
                    successful_episodes_with_collisions += 1

            # Logging and metrics
            if self.log_rewards_dir:
                with self.writer.as_default():
                    tf.summary.scalar('Episode Return', Rewards, step=episode)
                    tf.summary.scalar('Steps per Episode', steps_cnt, step=episode)
                    if episode_mission_completed:
                        tf.summary.scalar('Collisions in Successful Episode', self.env.sar_robot.episode_collisions, step=episode)
            
            self.EPSILON = self._decay_epsilon(num_episodes)
            total_rewards_per_episode[episode] = Rewards
            total_steps_per_episode[episode] = steps_cnt
        
        # Save final Q-table
        if self.learned_policy_dir:
            self.save_learned_policy(num_episodes)
        
        # Calculate success rates
        mission_success_rate = (successful_episodes / num_episodes) * 100
        info_collection_success_rate = (info_collection_completed_episodes / num_episodes) * 100

        # Calculate collision metrics -- new
        collision_rate_successful = 0
        avg_collisions_per_success = 0
        mission_success_no_collisions_rate = 0
        if successful_episodes > 0:
            collision_rate_successful = (successful_episodes_with_collisions / successful_episodes) * 100
            avg_collisions_per_success = successful_episode_collisions / successful_episodes
            # Calculate mission success rate without collisions -- new
            mission_success_no_collisions_rate = ((successful_episodes - successful_episodes_with_collisions) / num_episodes) * 100
        ## -- new
        
        # Get collection statistics from the environment
        collection_stats = self.env.sar_robot.info_system.get_collection_stats()
        collection_success_rate = self.env.sar_robot.info_system.get_collection_success_rate()

        # Calculate exploration-specific collection success rate statistics -- new
        total_exploration_calls = sum(stats['calls'] for stats in self.predictor_stats.values())
        total_exploration_successes = sum(stats['successes'] for stats in self.predictor_stats.values())
        exploration_success_rate = (total_exploration_successes / max(1, total_exploration_calls)) * 100
        
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
            'total_hazard_collisions_in_successful_episodes': successful_episode_collisions, ## -- new
            'successful_episodes_with_collisions': successful_episodes_with_collisions, ## -- new
            'collision_rate_in_successful_episodes': collision_rate_successful, ## -- new
            'average_collisions_per_successful_episode': avg_collisions_per_success, ## -- new
            'collision_counts_per_successful_episode': collision_counts_successful, ## -- new
            'mission_success_no_collisions_rate': mission_success_no_collisions_rate,
            'attention_active': True,  # Flag indicating this was an attention-based agent
            'llm_active': False,  # Flag indicating this was not an LLM-based agent
            'predictor_stats': {
                'total_calls': total_exploration_calls,
                'total_successes': total_exploration_successes,
                'overall_success_rate': exploration_success_rate,
                'by_location': self.predictor_stats
            }
        }

        # Print final statistics
        print("\nTraining Complete!")
        print("\nExploration and Exploitation Statistics:")
        print(f"Total exploration actions: {self.exploration_count}")
        print(f"Total exploitation actions: {self.exploitation_count}")
        print(f"Final exploration/exploitation ratio: {self.exploration_count/(self.exploration_count + self.exploitation_count):.2f}")
        
        print("\nInformation Collection Statistics:")
        self.env.sar_robot.info_system.print_collection_stats()

        ## -- new
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
        ## -- new

        print("\nTask Success Rates:")
        print(f"Information collection success rate: {info_collection_success_rate:.2f}%")
        print(f"Overall mission success rate: {mission_success_rate:.2f}%")

        ## -- new
        print("\nHazard Collision Statistics (Successful Episodes):")
        print(f"Total hazard collisions in successful episodes: {successful_episode_collisions}")
        print(f"Number of successful episodes with collisions: {successful_episodes_with_collisions} out of {successful_episodes}")
        print(f"Collision rate in successful episodes: {collision_rate_successful:.2f}%")
        print(f"Average collisions per successful episode: {avg_collisions_per_success:.2f}")
        print(f"Mission success rate *without collisions*: {mission_success_no_collisions_rate:.2f}%")
        ## -- new

        print("\nTraining Summary:")
        print(f"Average reward per episode: {np.mean(total_rewards_per_episode):.2f}")
        print(f"Average steps per episode: {np.mean(total_steps_per_episode):.2f}")
        print(f"Best episode reward: {np.max(total_rewards_per_episode):.2f}")
        print(f"Worst episode reward: {np.min(total_rewards_per_episode):.2f}")
        
        return total_rewards_per_episode, total_steps_per_episode, metrics