import os
from enum import Enum

# Configuration dictionary for plotting agent rewards
agent_config = {
    # List of agent variable names as strings
    'agent_vars': [
        'all_total_rewards_AGENT_flat',    ### flat
        'all_total_rewards_AGENT_flatLLM', ### flatLLM
        'all_total_rewards_AGENT_tog',     ### flat action shaping
        'all_total_rewards_AGENT_att',     ### flat policy shaping
        'all_total_rewards_AGENT_attRS',   ### flat reward shaping
        'all_total_rewards_AGENT_hier',    ### hierarchical
        'all_total_rewards_AGENT_hierLLM', ### hierarchical LLM
        'all_total_rewards_AGENT_hier_tog', ### hierarchical action shaping
        'all_total_rewards_AGENT_hier_att',  ### hierarchical policy shaping 
        'all_total_rewards_AGENT_hier_attRS'  ### hierarchical reward shaping
    ],
    # Corresponding labels and colors for each agent
    'labels': [
        'Q-learning-Flat', 
        'Q-learning-LLM',
        'Q-learning-ActionToggle',
        'Q-learning-PolicyShaping',
        'Q-learning-RewardShaping',
        'Q-learning-Hierarchical',
        'Q-learning-Hierarchical-LLM',
        'Q-learning-Hierarchical-ActionToggle',
        'Q-learning-Hierarchical-PolicyShaping',
        'Q-learning-Hierarchical-RewardShaping'
    ],
    'colors': [
        'blue',
        'orange',  
        'green',  
        'black',
        'magenta',
        'red',
        'purple',
        'brown',
        'gray',
        'pink'
    ]
}

def get_file_type(document_path):
    # Split the path and get the extension
    _, file_extension = os.path.splitext(document_path)
    # Return the file extension without the period
    return file_extension[1:] if file_extension else None

def _get_action_name(option: int | None, action: int) -> str:
    """
    Helper function to get the name of an action based on the current option.
    For hierarchical agents, returns action name based on the option type.
    For flat agents, returns the base RobotAction name.
    """
    action_map = {
        RobotOption.NAVIGATION.value: NavigationActions,
        RobotOption.INFORMATION_COLLECTION.value: InformationCollectionActions,
        RobotOption.OPERATION_TRIAGE.value: OperationTriageActions,
    }
    if option is None:
        return RobotAction(action).name
    action_class = action_map.get(option)
    return action_class(action).name if action_class else f"Unknown Action ({action})"

class RobotAction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    COLLECT_X = 4
    COLLECT_Y = 5
    COLLECT_Z = 6
    COLLECT_A = 7
    COLLECT_B = 8
    COLLECT_C = 9
    SAVE = 10
    USE = 11
    REMOVE = 12
    CARRY = 13

class GridTile(Enum):
    _FLOOR = 0
    ROBOT = 1
    TARGET = 2
    X_INFO = 3
    Y_INFO = 4
    Z_INFO = 5
    A_INFO = 6
    B_INFO = 7
    C_INFO = 8
    DITCH = 9
    
    def __str__(self):
        return self.name[:1]

# Define Actions for Each Option
class NavigationActions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class InformationCollectionActions(Enum):
    COLLECT_X = 0
    COLLECT_Y = 1
    COLLECT_Z = 2
    COLLECT_A = 3
    COLLECT_B = 4
    COLLECT_C = 5
    COLLECT_D = 6
    COLLECT_E = 7
    COLLECT_F = 8
    COLLECT_G = 9
    COLLECT_H = 10
    COLLECT_I = 11
    COLLECT_J = 12
    COLLECT_K = 13
    COLLECT_L = 14
    COLLECT_M = 15
    COLLECT_N = 16
    COLLECT_O = 17
    COLLECT_P = 18
    COLLECT_Q = 19
    COLLECT_R = 20
    COLLECT_S = 21
    COLLECT_T = 22
    COLLECT_U = 23
    COLLECT_V = 24
    COLLECT_W = 25

class OperationTriageActions(Enum):
    SAVE = 0
    USE = 1
    REMOVE = 2
    CARRY = 3

# Define Robot Options (Subtasks)
class RobotOption(Enum):
    NAVIGATION = 0
    INFORMATION_COLLECTION = 1
    OPERATION_TRIAGE = 2

class RunningParameters:
    def __init__(self):
        self.manager_action_space_size = len(RobotOption)                    # for HRL
        self.explore_action_space_size = len(NavigationActions)              # for HRL
        self.collect_action_space_size = len(InformationCollectionActions)   # for HRL
        self.operate_action_space_size = len(OperationTriageActions)         # for HRL
        self.EPISODES = 5000     # Number of episodes to train the agent
        self.ALPHA = 0.1         # Learning rate
        self.GAMMA = 0.99        # Discount factor
        self.EPSILON_MAX = 1.0   # Exploration rate
        self.EPSILON_MIN = 0.1  # Minimum exploration rate
        self.DECAY_RATE = 2      # Decay rate for exploration rate
        self.POSITIVE_ATTENTION_VALUE = 2.0    # Attention space values/rewards
        self.NEGATIVE_ATTENTION_VALUE = -100.0 # Attention space values/rewards
        self.SAVE_ACTION_VALUE = 100.0         # Attention space values/rewards
        self.testing_runs = 1    # Number of testing runs   
        self.evaluation_runs = 1
        self.sleeping_time = 60.0