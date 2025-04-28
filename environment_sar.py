import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from gymnasium import spaces
from typing import Optional, Literal
from IPython.display import clear_output
from InformationCollection import InfoLocation, InfoCollectionSystem
from LLM_ContextExtractor import DisasterResponseAssistant
from robot_utils import get_file_type, RobotAction, GridTile, RobotOption, NavigationActions, InformationCollectionActions, OperationTriageActions
# from shaping_mechanisms import RewardShapingMechanism

class searchANDrescueRobot:
    def __init__(self, grid_rows: int, grid_cols: int, info_number_needed: int, attention: bool, hierarchical: bool):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.hierarchical = hierarchical
        self.attention = attention
        self.info_number_needed = info_number_needed
        self._reset()

        # Define information locations dynamically based on info_number_needed
        info_locations = self._generate_info_locations(info_number_needed)
        print(info_locations)
        
        self.info_system = InfoCollectionSystem(info_locations)
        
        # Initialize other attributes
        self.ask_action_counter = 0
        self.visited_information_state = False
        self.input_received = False
        self.GENERAL_FIRES_UNKNOWN_TO_THE_AGENT = [(1, 3), (2, 3), (3, 1)] ## for evaluation
        self.POIs, self.fires, self.hazards, self.poi = [], [], [], []
        self.sensor_readings = {}
        self.visited_pois = set()
        
        # Action tracking at info locations
        self.info_location_actions = {tuple(loc.position): 0 for loc in info_locations}
        self.max_info_location_actions = 2
        self.total_info_location_actions = 0

        # Load the disaster response assistant
        document_path = "sar_data.json"
        document_type = get_file_type(document_path)
        self.assistant = DisasterResponseAssistant(document_path, document_type)

        # Add collision counter
        self.hazard_collisions = 0
        self.episode_collisions = 0  # Track collisions per episode
    

    def _generate_info_locations(self, count):
        """
        Generate information locations dynamically based on the requested count.
        
        Args:
            count (int): Number of information locations to generate
            
        Returns:
            list: List of InfoLocation objects
        """
        # Default locations for up to 6 info points
        all_possible_locations = [
            InfoLocation([1, 1], "X", 0),
            InfoLocation([3, 0], "Y", 1),
            InfoLocation([3, 2], "Z", 2),
            InfoLocation([1, 3], "A", 3),
            InfoLocation([3, 3], "B", 4),
            InfoLocation([2, 2], "C", 5)
        ]
        
        # Return only the requested number of locations
        return all_possible_locations[:count]
        
    
    def _reset(self, seed=None):
        self.ask_action_counter = 0
        self.init_positions = [[2,1]]#, [2, 1], [3, 3]]
        self.robot_pos = random.choice(self.init_positions)
        self.has_saved = 0
        if self.hierarchical:
            self.current_option = RobotOption.NAVIGATION.value
        self.target_pos = [0, 3] # 0, 3

        # Generate info locations dynamically
        info_locations = self._generate_info_locations(self.info_number_needed)
        
        if not hasattr(self, 'info_system'):
            # Only create new InfoCollectionSystem if it doesn't exist
            self.info_system = InfoCollectionSystem(info_locations)
        else:
            # Just reset episode-specific stats
            self.info_system.reset_episode()
        
        # Reset other states
        self.ditches = [(1, 0), (2, 0), (1, 2)]
        self.GENERAL_FIRES_UNKNOWN_TO_THE_AGENT = [(1, 3), (2, 3), (3, 1)] ## for evaluation
        self.POIs, self.fires = [], []
        self.visited_information_state = False
        self.visited_pois = set()
        
        # Reset action tracking
        self.info_location_actions = {tuple(loc.position): 0 for loc in self.info_system.info_locations}
        self.max_info_location_actions = 2
        self.total_info_location_actions = 0

        self.episode_collisions = 0 # Reset collision counter for new episode

    def _is_valid_robot_action(self, robot_action: RobotAction) -> bool:
        valid_actions = {RobotAction.SAVE, RobotAction.USE, RobotAction.REMOVE, RobotAction.CARRY} | \
                        set(action for action in RobotAction if action.name.startswith("COLLECT"))
        return robot_action in valid_actions

    def _is_information_action(self, robot_action):
        """Check if the action is a valid information collection action"""
        return robot_action in [action.value for action in InformationCollectionActions]
    
    def _is_valid_robot_action_hier(self, robot_action):
        # Check if the robot action belongs to any of the defined action groups
        return (
            isinstance(robot_action, InformationCollectionActions) or 
            isinstance(robot_action, OperationTriageActions)
        )

    # Update the robot's position -- used by the attention space
    def _next_state_vision(self, target, robot_action:RobotAction) -> bool:
        robot_pos = target
        self.last_action = robot_action
        if not self.hierarchical:
            if robot_action == RobotAction.UP:
                if robot_pos[0] > 0:
                    robot_pos[0] -= 1  
            elif robot_action == RobotAction.DOWN:
                if robot_pos[0] < self.grid_rows-1:
                    robot_pos[0] += 1
            elif robot_action == RobotAction.LEFT:
                if robot_pos[1] > 0:
                    robot_pos[1] -= 1
            elif robot_action == RobotAction.RIGHT:
                if robot_pos[1] < self.grid_cols-1:
                    robot_pos[1] += 1 
            # Use the new helper function
            if self._is_valid_robot_action(robot_action):
                robot_pos = robot_pos
        else:
            if robot_action in [NavigationActions.UP, NavigationActions.DOWN, NavigationActions.LEFT, NavigationActions.RIGHT]:
                if robot_action == NavigationActions.UP:
                    if robot_pos[0] > 0:
                        robot_pos[0] -= 1  
                elif robot_action == NavigationActions.DOWN:
                    if robot_pos[0] < self.grid_rows-1:
                        robot_pos[0] += 1
                elif robot_action == NavigationActions.LEFT:
                    if robot_pos[1] > 0:
                        robot_pos[1] -= 1
                elif robot_action == NavigationActions.RIGHT:
                    if robot_pos[1] < self.grid_cols-1:
                        robot_pos[1] += 1
            if self._is_valid_robot_action_hier(robot_action):
                robot_pos = robot_pos
        return robot_pos

    def perform_action(self, action):
        # Now call the appropriate function
        if self.hierarchical:
            return self.perform_hierarchical_action(action)
        else:
            return self.perform_flat_action(RobotAction(action))

    def _handle_movement(self, action):
        if not self.hierarchical:
            if action == RobotAction.UP and self.robot_pos[0] > 0:
                self.robot_pos[0] -= 1
            elif action == RobotAction.DOWN and self.robot_pos[0] < self.grid_rows-1:
                self.robot_pos[0] += 1
            elif action == RobotAction.LEFT and self.robot_pos[1] > 0:
                self.robot_pos[1] -= 1
            elif action == RobotAction.RIGHT and self.robot_pos[1] < self.grid_cols-1:
                self.robot_pos[1] += 1
        else:
            if action == NavigationActions.UP.value and self.robot_pos[0] > 0:
                self.robot_pos[0] -= 1
            elif action == NavigationActions.DOWN.value and self.robot_pos[0] < self.grid_rows-1:
                self.robot_pos[0] += 1
            elif action == NavigationActions.LEFT.value and self.robot_pos[1] > 0:
                self.robot_pos[1] -= 1
            elif action == NavigationActions.RIGHT.value and self.robot_pos[1] < self.grid_cols-1:
                self.robot_pos[1] += 1
     
    def perform_flat_action(self, robot_action: RobotAction):
        self.last_action = robot_action
        info_collected = {loc.info_type: False for loc in self.info_system.info_locations}
        total_info_collected = illegal_action = action_limit_exceeded = False
        # Handle movement actions
        if robot_action in [RobotAction.UP, RobotAction.DOWN, RobotAction.LEFT, RobotAction.RIGHT]:
            self._handle_movement(robot_action)
        # Handle collection actions
        elif self._is_valid_robot_action(robot_action):
            current_pos_tuple = tuple(self.robot_pos)
            # Check action limits at info locations
            if current_pos_tuple in self.info_location_actions:
                self.info_location_actions[current_pos_tuple] += 1
                if self.info_location_actions[current_pos_tuple] > self.max_info_location_actions:
                    action_limit_exceeded = True
                    return False, *info_collected.values(), False, False, action_limit_exceeded
            # Determine if the action is a collect action and extract the attempted info type
            attempted_info_type = None
            if robot_action.name.startswith("COLLECT_"):
                attempted_info_type = robot_action.name.replace("COLLECT_", "")
            collectible_info = self.info_system.can_collect_at_position(self.robot_pos)
            if collectible_info:
                # If we are at a collectible info location, attempt collection regardless of correctness
                success = self.info_system.collect_info(attempted_info_type, self.robot_pos)
                if success:
                    # Correct info collected
                    info_collected[attempted_info_type] = True
                    if attempted_info_type == self.info_system.info_locations[-1].info_type:
                        if self.attention:
                            self.perform_collect_action()
                        total_info_collected = True
                else:
                    # Wrong info attempted, still counts as an attempt but is incorrect
                    illegal_action = True
            # Handle save action
            elif (self.robot_pos == self.target_pos and self.info_system.get_collected_info_count() >= self.info_system.get_required_info_count()):
                if robot_action == RobotAction.SAVE:
                    self.has_saved = 1
                else:
                    illegal_action = True
            else:
                illegal_action = True
        mission_complete = (self.robot_pos == self.target_pos and 
                          self.info_system.get_collected_info_count() >= self.info_system.get_required_info_count() and 
                          self.has_saved)
        return (mission_complete, *info_collected.values(), total_info_collected, illegal_action, action_limit_exceeded)

    def perform_hierarchical_action(self, robot_action):
        robot_option = self.current_option
        self.last_action = robot_action
        info_collected = {loc.info_type: False for loc in self.info_system.info_locations}
        total_info_collected = illegal_action = action_limit_exceeded = False
        # Check action limits at info locations
        current_pos_tuple = tuple(self.robot_pos)
        is_info_location = current_pos_tuple in self.info_location_actions
        # Increment action counter for non-movement actions at info locations
        if (is_info_location and robot_action not in [NavigationActions.UP.value, NavigationActions.DOWN.value, NavigationActions.LEFT.value, NavigationActions.RIGHT.value]):
            self.info_location_actions[current_pos_tuple] += 1
            if self.info_location_actions[current_pos_tuple] > self.max_info_location_actions:
                action_limit_exceeded = True
                return False, *info_collected.values(), False, False, action_limit_exceeded
        # Handle Navigation Option
        if robot_option == RobotOption.NAVIGATION.value:
            if robot_action in [NavigationActions.UP.value, NavigationActions.DOWN.value, 
                            NavigationActions.LEFT.value, NavigationActions.RIGHT.value]:
                # Check if we should allow movement
                should_move = (
                    (self.info_system.get_collected_info_count() < self.info_system.get_required_info_count() and 
                    not self.info_system.can_collect_at_position(self.robot_pos)) or
                    (self.info_system.get_collected_info_count() >= self.info_system.get_required_info_count() and 
                    self.robot_pos != self.target_pos)
                )
                if should_move:
                    self._handle_movement(robot_action)
                # Check if we've reached a collection point or target
                collectible_info = self.info_system.can_collect_at_position(self.robot_pos)
                if collectible_info:
                    robot_option = RobotOption.INFORMATION_COLLECTION.value
                elif (self.info_system.get_collected_info_count() >= self.info_system.get_required_info_count() and 
                    self.robot_pos == self.target_pos):
                    robot_option = RobotOption.OPERATION_TRIAGE.value
            else:
                illegal_action = True
        # Handle Information Collection Option
        elif robot_option == RobotOption.INFORMATION_COLLECTION.value:
            # Map the action value to the corresponding InformationCollectionActions enum
            try:
                action_enum = InformationCollectionActions(robot_action)
                attempted_info_type = action_enum.name.replace("COLLECT_", "")
                collectible_info = self.info_system.can_collect_at_position(self.robot_pos)
                if collectible_info:
                    success = self.info_system.collect_info(attempted_info_type, self.robot_pos)
                    if success:
                        info_collected[attempted_info_type] = True
                        if attempted_info_type == self.info_system.info_locations[-1].info_type:
                            if self.attention:
                                self.perform_collect_action()
                            total_info_collected = True
                        robot_option = RobotOption.NAVIGATION.value
                    else:
                        illegal_action = True
                else:
                    illegal_action = True
            except ValueError:
                illegal_action = True
        # Handle Operation Triage Option
        elif robot_option == RobotOption.OPERATION_TRIAGE.value:
            valid_triage_actions = [OperationTriageActions.SAVE.value, OperationTriageActions.USE.value,
                                OperationTriageActions.REMOVE.value, OperationTriageActions.CARRY.value]
            if robot_action in valid_triage_actions:
                if (self.robot_pos == self.target_pos and 
                    self.info_system.get_collected_info_count() >= self.info_system.get_required_info_count()):
                    if robot_action == OperationTriageActions.SAVE.value:
                        self.has_saved = 1
                    else:
                        illegal_action = True
                else:
                    illegal_action = True
            else:
                illegal_action = True
        self.current_option = robot_option
        mission_complete = self.has_saved
        return (mission_complete, *info_collected.values(), total_info_collected, illegal_action, action_limit_exceeded)
    
    def perform_collect_action(self):
        self.ask_action_counter += 1
        verbal_inputs = []
        if self.info_system.get_collected_info_count() == self.info_number_needed:  ## should be 2 if total number of infos are 3 
            verbal_input = ("Hey, there's a victim at the hospital. A fire was reported at the train station. There is a fire at the bank. A safe area is the mall. You must go to the access route in the school. Another access route at the restaurant. And there is a shelter in the shop. There are also reports of significant instances of heat at the bakery. Police told us that no access allowed around the petrol station.")
            # print(f"real LLM is about to start handling the input {verbal_input}")
            verbal_inputs.append(verbal_input)
            if self.ask_action_counter <= 1:
                # print('attention ON')
                # print(f"real LLM is about to start handling the input {verbal_input}")
                # self.hazards, self.poi = [(5, 6), (6, 5), (3, 6), (2, 5)], [(0, 3), (4, 1), (3, 0), (2, 0), (1, 2)]
                self.hazards, self.poi = [(1, 3), (2, 3), (3, 1)], [(2, 1), (1, 1), (0, 2)]
                # print(f"real LLM is about to end handling the input {verbal_input}")
                self._update_environment_real(self.hazards, self.poi)
                self.visited_information_state = True
                # for input_text in verbal_inputs:
                #     response = self.assistant.generate_response(input_text)
                #     if response:
                #         self.visited_information_state = True
                #     self.hazards, self.poi = self.assistant.refine_response(response)
                #     print(f"real LLM is about to end handling the input {verbal_input}")
                #     self._update_environment_real(self.hazards, self.poi)
            # else:
            #     # #print(f"input will be handled hereby by pseudoLLM")
            #     # print(self.hazards, self.poi)
            #     self.visited_information_state = True
            #     self._update_environment_real(self.hazards, self.poi)
            
    def _update_environment_real(self, haz, poi):
        for hazardous_location in haz:
            self.sensor_readings[(hazardous_location[0], hazardous_location[1], self.info_number_needed, 0)] = -10.0
            self.fires.append(hazardous_location)
        for safe_location in poi:
            self.sensor_readings[(safe_location[0], safe_location[1], self.info_number_needed, 0)] = 10.0
            self.POIs.append(safe_location)
            
    def _is_in_ditch(self):
        return tuple(self.robot_pos) in self.ditches

    def render(self):
        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(4, 4))
            plt.ion()  # Interactive mode ON

        self.ax.clear()
        self.ax.set_xlim(0, self.grid_rows)
        self.ax.set_ylim(0, self.grid_cols)
        self.ax.set_xticks(range(self.grid_rows))
        self.ax.set_yticks(range(self.grid_cols))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(which='both')

        elements = [
            {'positions': [tuple(self.robot_pos)], 'marker': 'o', 'color': 'blue', 'size': 10, 'label': 'Robot'},
            {'positions': [tuple(self.target_pos)], 'marker': 'P', 'color': 'green', 'size': 10, 'label': 'Target'},
            {'positions': self.ditches, 'marker': 'x', 'color': 'red', 'size': 10, 'label': 'Ditch'},
            {'positions': self.POIs, 'marker': 'P', 'color': 'pink', 'size': 10, 'label': 'POI'},
            {'positions': self.fires, 'marker': 'x', 'color': 'orange', 'size': 10, 'label': 'Fire/Hazard'},
            {'positions': [tuple(info.position) for info in self.info_system.info_locations], 
            'marker': '*', 'color': 'yellow', 'size': 10, 'label': 'Info Location'}
        ]

        for element in elements:
            for pos in element.get('positions', []):
                self.ax.plot(pos[1] + 0.5, pos[0] + 0.5, marker=element['marker'], 
                            color=element['color'], markersize=element['size'])

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Robot', linestyle='None'),
            Line2D([0], [0], marker='P', color='w', markerfacecolor='green', markersize=10, label='Target', linestyle='None'),
            Line2D([0], [0], marker='x', color='red', markersize=10, label='Ditch', linestyle='None'),
            Line2D([0], [0], marker='P', color='w', markerfacecolor='pink', markersize=10, label='POI', linestyle='None'),
            Line2D([0], [0], marker='x', color='orange', markersize=10, label='Fire/Hazard', linestyle='None'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow', markersize=10, label='Info Location', linestyle='None')
        ]

        self.ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        self.ax.invert_yaxis()
        self.fig.tight_layout()
        plt.pause(0.5)


class SARrobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}
    def __init__(self, grid_rows: int, grid_cols: int, info_number_needed: int, sparse_reward: bool, reward_shaping: bool, attention: bool, hierarchical: bool, render_mode: Optional[Literal["human"]] = None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode
        self.info_number_needed = info_number_needed
        self.sparse_reward = sparse_reward  ## mode
        self.reward_shaping = reward_shaping ## mode
        self.hierarchical = hierarchical
        self.attention = attention
        # Print environment configuration
        print("\nEnvironment Configuration:")
        print(f"Sparse Reward Mode: {self.sparse_reward}")
        print(f"Hierarchical Mode: {self.hierarchical}")
        print(f"Reward Shaping Mode: {self.reward_shaping}")
        print(f"Attention Mechanism: {self.attention}")
        print(f"Grid Size: {self.grid_rows}x{self.grid_cols}")
        print(f"Required Information Points: {self.info_number_needed}\n")
        self.sar_robot = searchANDrescueRobot(self.grid_rows, self.grid_cols, self.info_number_needed, self.attention, self.hierarchical)
        # Set up the appropriate action space or option space 
        if self.hierarchical:
            self.option_space = spaces.Discrete(len(RobotOption))
        else:
            self.action_space = spaces.Discrete(len(RobotAction))
        required_info_count = self.sar_robot.info_system.get_required_info_count() # Update observation space to use flexible info count
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.grid_rows-1, self.grid_cols-1, required_info_count, 1]),
            shape=(4,),
            dtype=np.int32
        )
        self.max_steps = 50
        self.current_step = 0
        self.turnPenalty = -1
        self.stepsPenalty = -5
        self.ditchPenalty = -30
        self.illegalActionPenalty = -5  # Penalty for illegal actions
        self.infoLocationActionsPenalty = -5  # New penalty for exceeding action limit at info locations
        self.winReward = 100
        self.gamma = 0.99  # Discount factor for reward shaping
        # self.reward_shaping_mechanism = RewardShapingMechanism(gamma=self.gamma)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sar_robot._reset(seed=seed)
        self.current_step = 0
        obs = np.concatenate((
            self.sar_robot.robot_pos,
            [self.sar_robot.info_system.get_collected_info_count()],
            [self.sar_robot.has_saved]
        )).astype(np.int32)
        info = {'option': self.sar_robot.current_option} if self.hierarchical else {}
        return obs, info

    def step(self, action):
        reward = 0
        self.current_step += 1
        # Save old state for reward shaping if enabled
        if self.reward_shaping:
            old_state = np.concatenate((
                self.sar_robot.robot_pos, 
                [self.sar_robot.info_system.get_collected_info_count()],
                [self.sar_robot.has_saved]
            )).astype(np.int32)
        # Get previous info count for reward calculation
        prev_info_count = self.sar_robot.info_system.get_collected_info_count()
        # Perform action and get results
        result = self.sar_robot.perform_action(action)
        target_reached = result[0]
        info_types = [loc.info_type for loc in self.sar_robot.info_system.info_locations]
        info_collected_dict = {
            info_type: collected 
            for info_type, collected in zip(info_types, result[1:len(info_types)+1])
        }
        total_info_collected = result[len(info_types)+1]
        illegal_action = result[len(info_types)+2]
        action_limit_exceeded = result[len(info_types)+3]
        terminated = False

        # Check for hazard collision after action -- new for tracking collisions
        had_collision = False
        if self.sar_robot.info_system.get_required_info_count() == self.sar_robot.info_system.get_collected_info_count() and tuple(self.sar_robot.robot_pos) in self.sar_robot.GENERAL_FIRES_UNKNOWN_TO_THE_AGENT:
            # print("Hazard Collision!")
            self.sar_robot.hazard_collisions += 1  # Total across all episodes
            self.sar_robot.episode_collisions += 1  # For current episode
            had_collision = True

        # Check for visited information state if reward shaping is enabled
        if self.reward_shaping and self.sar_robot.visited_information_state:
            current_pos = tuple(self.sar_robot.robot_pos)
            if current_pos in self.sar_robot.POIs and current_pos not in self.sar_robot.visited_pois:
                if not self.sparse_reward:
                    reward = 0.1 # Small positive reward for first-time visiting a POI
                    # print(f"Visited POI! {current_pos}")
                self.sar_robot.visited_pois.add(current_pos)
        if action_limit_exceeded:
            reward = self.infoLocationActionsPenalty
            terminated = True
        if self.sar_robot._is_in_ditch():
            reward = self.ditchPenalty
            terminated = True
        if self._is_max_steps_exceeded():
            reward = self.stepsPenalty
            terminated = True
        # Add information collection rewards in non-sparse mode
        if not self.sparse_reward:
            current_info_count = self.sar_robot.info_system.get_collected_info_count()
            if current_info_count > prev_info_count:
                reward = 10
        if target_reached:
            reward = self.winReward
            terminated = True
        # if illegal_action:
        #     reward = self.illegalActionPenalty
        reward += self.turnPenalty
        # Create observation
        obs = np.concatenate((
            self.sar_robot.robot_pos,
            [self.sar_robot.info_system.get_collected_info_count()],
            [self.sar_robot.has_saved]
        )).astype(np.int32)
        info = {'option': self.sar_robot.current_option} if self.hierarchical else {}
        # compute shaping reward if enabled
        if self.reward_shaping:
            # shaping_reward = self.reward_shaping_mechanism.get_shaping_reward(old_state, obs, self.sar_robot.POIs)
            shaping_reward = self.reward_shaping_mechanism.get_shaping_reward(old_state, obs, self.sar_robot.fires)
            reward += shaping_reward
        # Add collection status to info dict
        info.update({
            'info_collected': info_collected_dict,
            'total_info_collected': total_info_collected,
            'collected_count': self.sar_robot.info_system.get_collected_info_count(),
            'required_count': self.sar_robot.info_system.get_required_info_count(),
            'had_collision': had_collision,                             # Add collection status to info dict
            'episode_collisions': self.sar_robot.episode_collisions     # Add collection status to info dict
        })
        if self.render_mode == 'human':
            action_str = f"Option: {self.sar_robot.current_option}, Action: {action}" if self.hierarchical \
                        else f"Action: {action}"
            print(f"{action_str}, Reward: {reward}, Terminated: {terminated}")
            self.render()
        return obs, reward, terminated, False, info

    def _is_max_steps_exceeded(self):
        return self.current_step >= self.max_steps
    
    def render(self):
        if self.render_mode == 'human':
            self.sar_robot.render()