import gymnasium as gym
import random
import numpy as np
from gymnasium import spaces
from InformationCollection import InfoLocation, InfoCollectionSystem
from robot_utils import RobotAction

class searchANDrescueRobot:
    def __init__(self, grid_rows: int, grid_cols: int, info_number_needed: int):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.info_number_needed = info_number_needed
    
        self.init_positions = [[2,1]]
        self.target_pos = [0, 3]
        self.ditches = [(1, 0), (2, 0), (1, 2)]

        # Define information locations dynamically based on info_number_needed
        info_locations = self._generate_info_locations(info_number_needed)
        print(info_locations)
        self.info_system = InfoCollectionSystem(info_locations)
        
        # Action tracking at info locations
        self.info_location_actions = {tuple(loc.position): 0 for loc in info_locations}
        self.max_info_location_actions = 2
        self.total_info_location_actions = 0
        self._reset()
    

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
        self.robot_pos = random.choice(self.init_positions)
        self.has_saved = 0
       
        # Generate info locations dynamically
        info_locations = self._generate_info_locations(self.info_number_needed)
        
        if not hasattr(self, 'info_system'):
            # Only create new InfoCollectionSystem if it doesn't exist
            self.info_system = InfoCollectionSystem(info_locations)
        else:
            # Just reset episode-specific stats
            self.info_system.reset_episode()
        
        # Reset action tracking
        self.info_location_actions = {tuple(loc.position): 0 for loc in self.info_system.info_locations}
        self.max_info_location_actions = 2
        self.total_info_location_actions = 0

    def _is_valid_robot_action(self, robot_action: RobotAction) -> bool:
        valid_actions = {RobotAction.SAVE, RobotAction.USE, RobotAction.REMOVE, RobotAction.CARRY} | \
                        set(action for action in RobotAction if action.name.startswith("COLLECT"))
        return robot_action in valid_actions
    

    def perform_action(self, action):
        # Now call the appropriate function
        return self.perform_flat_action(RobotAction(action))

    def _handle_movement(self, action):
        """Handle movement actions for the robot."""
        if action == RobotAction.UP and self.robot_pos[0] > 0:
            self.robot_pos[0] -= 1
        elif action == RobotAction.DOWN and self.robot_pos[0] < self.grid_rows-1:
            self.robot_pos[0] += 1
        elif action == RobotAction.LEFT and self.robot_pos[1] > 0:
            self.robot_pos[1] -= 1
        elif action == RobotAction.RIGHT and self.robot_pos[1] < self.grid_cols-1:
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

            
    def _is_in_ditch(self):
        return tuple(self.robot_pos) in self.ditches


class SARrobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}
    def __init__(self, grid_rows: int, grid_cols: int, info_number_needed: int):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.info_number_needed = info_number_needed
        
        # Print environment configuration
        print("\nEnvironment Configuration:")
        print(f"Grid Size: {self.grid_rows}x{self.grid_cols}")
        print(f"Required Information Points: {self.info_number_needed}\n")
        self.sar_robot = searchANDrescueRobot(self.grid_rows, self.grid_cols, self.info_number_needed)
       
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
        info = {}
        return obs, info

    def step(self, action):
        reward = 0
        self.current_step += 1
        
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

        if action_limit_exceeded:
            reward = self.infoLocationActionsPenalty
            terminated = True
        if self.sar_robot._is_in_ditch():
            reward = self.ditchPenalty
            terminated = True
        if self._is_max_steps_exceeded():
            reward = self.stepsPenalty
            terminated = True
        
        current_info_count = self.sar_robot.info_system.get_collected_info_count()
        if current_info_count > prev_info_count:
            reward = 10
        if target_reached:
            reward = self.winReward
            terminated = True
        reward += self.turnPenalty
        # Create observation
        obs = np.concatenate((
            self.sar_robot.robot_pos,
            [self.sar_robot.info_system.get_collected_info_count()],
            [self.sar_robot.has_saved]
        )).astype(np.int32)
        info = {}
        
        # Add collection status to info dict
        info.update({
            'info_collected': info_collected_dict,
            'total_info_collected': total_info_collected,
            'collected_count': self.sar_robot.info_system.get_collected_info_count(),
            'required_count': self.sar_robot.info_system.get_required_info_count(),
        })
        return obs, reward, terminated, False, info
        
    def _is_max_steps_exceeded(self):
        return self.current_step >= self.max_steps
    
   