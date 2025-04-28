import numpy as np
from robot_utils import RobotOption, RobotAction, NavigationActions, InformationCollectionActions, OperationTriageActions, RunningParameters

'''
Mechanism shapes the Q-table values based on sensor readings in the environment - 
identifies states where sensor readings have changed and creates connections between states
helps guide the agent away from dangerous areas and towards beneficial states

- works for both flat and hierarchical agents
POLICY SHAPING MECHANISM
'''

class AttentionSpace:
    # Initialize attention space with environment dimensions
    def __init__(self, env, action_space_size=None):
        self.env = env
        self.num_states = (self.env.observation_space.high[0] + 1, 
                           self.env.observation_space.high[1] + 1, 
                           self.env.observation_space.high[2] + 1,
                           self.env.observation_space.high[3] + 1)  # 7*7*4*2
        self.param = RunningParameters()
        if self.env.hierarchical:
            if action_space_size is None:
                raise ValueError("action_space_size must be provided when using hierarchical mode")
            self.action_space_size = action_space_size
            self.attention_space_low = np.zeros((*self.num_states, self.action_space_size))
        else:
            self.num_actions = self.env.action_space.n
            self.attention_space_low = np.zeros((*self.num_states, self.num_actions))
    
    def _identify_changed_states(self, readings):
        """Identify states where sensor readings have changed."""
        return [i for i, value in readings.items() if value != 1]
    
    def _get_connected_states(self, target_state):
        """Find states that are connected to the target state."""
        if self.env.hierarchical:
            inverse_actions = {
                NavigationActions.UP.value: NavigationActions.DOWN.value,
                NavigationActions.DOWN.value: NavigationActions.UP.value,
                NavigationActions.LEFT.value: NavigationActions.RIGHT.value,
                NavigationActions.RIGHT.value: NavigationActions.LEFT.value
            }
            # action_range = range(len(NavigationActions))
            action_type = NavigationActions
        else:
            inverse_actions = {
                RobotAction.UP.value: RobotAction.DOWN.value, 
                RobotAction.DOWN.value: RobotAction.UP.value, 
                RobotAction.LEFT.value: RobotAction.RIGHT.value, 
                RobotAction.RIGHT.value: RobotAction.LEFT.value
            }
            # action_range = range(4)  # Movement actions only
            action_type = RobotAction
        connected_states_pairs = []
        # for action in action_range:
        for action in inverse_actions.keys():
            possible_prev_state = self.env.sar_robot._next_state_vision(
                list(target_state[:2]), 
                action_type(inverse_actions[action])
            )
            if tuple(possible_prev_state) != tuple(target_state[:2]) and tuple(possible_prev_state) not in self.env.sar_robot.ditches:
                connected_states_pairs.append((tuple(possible_prev_state), action))        
        return connected_states_pairs
    
    def _update_attention_space(self, connection, readings):
        """Update attention values for connected states based on sensor readings."""
        connected_states = self._get_connected_states(connection)
        print(f"Connected states for {connection} are: {connected_states}\n")
        value_to_add = self.param.POSITIVE_ATTENTION_VALUE if readings[connection] > 0 else self.param.NEGATIVE_ATTENTION_VALUE
        for connected_state, action in connected_states:
            full_state = tuple([*connected_state, connection[2], connection[3]])
            # In hierarchical mode, avoid overwriting existing values
            if self.env.hierarchical:
                if self.attention_space_low[full_state][action] == 0:
                    self.attention_space_low[full_state][action] = value_to_add
            else:
                self.attention_space_low[full_state][action] = value_to_add
        # Handle target state
        if list((connection[0], connection[1])) == self.env.sar_robot.target_pos:
            if self.env.hierarchical:
                self.attention_space_low[connection][OperationTriageActions.SAVE.value] = self.param.SAVE_ACTION_VALUE  # 'save' option
            else:
                self.attention_space_low[connection][RobotAction.SAVE.value] = self.param.SAVE_ACTION_VALUE  # SAVE action

    # Apply attention values to modify the Q-table
    def _apply_attention_to_q_table(self, Q_table):
        for index, value in np.ndenumerate(self.attention_space_low):
            *state_indices, action = index
            if value != 0:
                Q_table[tuple(state_indices)][action] = value
                if not self.env.hierarchical:
                    print(f"Updated Q-table at {tuple(state_indices)}, action {action} with value {value}")
                else:
                    print(f"Updated Q-table at {tuple(state_indices)}, action {action} with value {value} - option {self.env.sar_robot.current_option}")

########
#### new class for action biasing
class ActionToggleMechanism:
    """
    ActionToggleMechanism identifies undesirable states from sensor readings and
    removes actions from connected states that would lead to those.
    """
    def __init__(self, env, action_space_size=None):
        self.env = env
        if self.env.hierarchical:
            if action_space_size is None:
                raise ValueError("action_space_size must be provided when using hierarchical mode")
            self.action_space_size = action_space_size
        else:
            self.action_space_size = self.env.action_space.n
        self.attention_space = AttentionSpace(self.env, self.action_space_size)
        self.undesirable_states = set() # Set of undesirable states with negative sensor readings
        self.connected_states_actions = {}  # Dict mapping connected states to actions to be removed
    
    def identify_hazard_states(self, readings):
        """Identify undesirable states (those with negative sensor readings)"""
        return [state for state, value in readings.items() if value < 0]
    
    def get_connected_states(self, target_state):
        """
        Find states that are connected to the target undesirable state and
        identify which action from each connected state leads there.
        Reuses some logic from AttentionSpace.
        """
        return self.attention_space._get_connected_states(target_state)
    
    def update_from_sensor_readings(self, readings):
        """
        Update the action toggle mechanism based on sensor readings.
        Identifies undesirable states and their connected states.
        """
        # First, identify undesirable states
        self.undesirable_states = self.identify_hazard_states(readings)

        # print(f"1a - Hazardous states: {self.undesirable_states}")

        self.connected_states_actions.clear()
        
        # For each undesirable state, find connected states and the actions to avoid
        for undesirable_state in self.undesirable_states:
            connected_states = self.get_connected_states(undesirable_state)
            # print(f"1b - the connected states for {undesirable_state} are: {connected_states}")

            # For each connected state, store the action to be avoided,
            # but only for states with info_count=info_number_needed (all must have been collected) and has_saved=0
            for connected_state, action in connected_states:
                full_state = (*connected_state, self.env.observation_space.high[2], 0)

                if full_state not in self.connected_states_actions:
                    self.connected_states_actions[full_state] = []
                
                # Add the action to be avoided (actions leading to undesirable state)
                if action not in self.connected_states_actions[full_state]:
                    self.connected_states_actions[full_state].append(action)
                # print(f"1c - To avoid {undesirable_state} from connected state {connected_state} you must avoid actions: {self.connected_states_actions[full_state]}")

        # Print summary
        # print(f"1d - Action Toggle Mechanism: Identified {len(self.undesirable_states)} undesirable states states and {len(self.connected_states_actions)} connected states")
        
        # print("=====================================")

    # function that gets the valid actions for the current state by removing actions that lead to undesirable states
    def get_valid_actions(self, state, option=None):
        """
        Get valid actions for the current state by removing actions that lead to undesirable states.
        """
        # for hierarchical agents, only apply action biasing for navigation actions
        if self.env.hierarchical and option is not None:
            if option != 0:
                # for non-navigation options, all actions are valid
                if option == RobotOption.INFORMATION_COLLECTION.value:
                    return list(range(len(InformationCollectionActions)))
                elif option == RobotOption.OPERATION_TRIAGE.value:
                    return list(range(len(OperationTriageActions)))
                return list(range(len(NavigationActions)))
        
        # Get the actions to avoid this state
        actions_to_avoid = self.connected_states_actions.get(state, [])

        # Determine full action space based on agent type
        if self.env.hierarchical:
            all_actions = list(range(len(NavigationActions)))
        else:
            all_actions = list(range(self.env.action_space.n))
        
        # return only valid actions (removing actions that lead to undesirable states)
        return [action for action in all_actions if action not in actions_to_avoid]
            

########
'''
Uses a potential function based on distances to Points of Interest (POIs) - 
The potential is negative distance to closest POI (closer = higher potential)
Calculates shaping reward using the formula F(s,s') = γΦ(s') - Φ(s) where:
 - Φ(s) is the potential of current state
 - Φ(s') is the potential of next state
 - γ is the discount factor
 encourages the agent to move closer to POIs by providing additional rewards for movements that reduce distance to them
provides additional rewards during learning based on distance metrics, without directly modifying the Q-table

REWARD SHAPING MECHANISM (POTENTIAL-BASED)    
'''
class RewardShapingMechanism:
    def __init__(self, gamma=0.99):
        self.gamma = gamma

    def calculate_potential(self, state, fires):
        """
        Calculate the potential function based on distance to fires.
        Higher potential when farther from fires.
        
        Args:
            state (numpy.array): selected state [x, y, has_info, has_saved]
            fires (list): List of fire/hazard coordinates
            
        Returns:
            float: Potential value based on distance to fires
        """
        if not fires:
            return 0  # No potential if no fires are known
        
        x, y = state[0], state[1]
        min_fire_distance = min([abs(x - fire[0]) + abs(y - fire[1]) for fire in fires])
        
        # Potential is proportional to distance from nearest fire
        # Scale by safety factor to control the influence
        return 1 * min_fire_distance

    # def calculate_potential(self, state, pois):
    #     """
    #     Calculate the potential function for a given state.
        
    #     Args:
    #         state (numpy.array): selected state [x, y, has_info, has_saved]
    #         pois (list): List of points of interest coordinates
            
    #     Returns:
    #         float: Potential value based on minimum distance to POIs
    #     """
    #     if not pois:
    #         return 0  # No potential before POIs are known
        
    #     x, y = state[0], state[1]
    #     min_distance = min([abs(x - poi[0]) + abs(y - poi[1]) for poi in pois])
    #     return -min_distance  # Negative distance as potential
    
    def get_shaping_reward(self, old_state, new_state, fires):
        """
        Calculate the shaping reward between two states using F(s,s') = γΦ(s') - Φ(s)
        
        Args:
            old_state (numpy.array): Previous state
            new_state (numpy.array): Current state
            pois (list): List of points of interest coordinates
            
        Returns:
            float: Shaping reward value
        """
        phi_s = self.calculate_potential(old_state, fires)
        phi_s_prime = self.calculate_potential(new_state, fires)
        return self.gamma * phi_s_prime - phi_s

    # def get_shaping_reward(self, old_state, new_state, pois):
    #     """
    #     Calculate the shaping reward between two states using F(s,s') = γΦ(s') - Φ(s)
        
    #     Args:
    #         old_state (numpy.array): Previous state
    #         new_state (numpy.array): Current state
    #         pois (list): List of points of interest coordinates
            
    #     Returns:
    #         float: Shaping reward value
    #     """
    #     phi_s = self.calculate_potential(old_state, pois)
    #     phi_s_prime = self.calculate_potential(new_state, pois)
    #     return self.gamma * phi_s_prime - phi_s
