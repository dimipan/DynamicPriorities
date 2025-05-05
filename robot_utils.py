from enum import Enum

def _get_action_name(option: int | None, action: int) -> str:
    return RobotAction(action).name

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
    
    def __str__(self):
        return self.name[:1]

class RunningParameters:
    def __init__(self):
        self.EPISODES = 5000     # Number of episodes to train the agent
        self.ALPHA = 0.1         # Learning rate
        self.GAMMA = 0.99        # Discount factor
        self.EPSILON_MAX = 1.0   # Exploration rate
        self.EPSILON_MIN = 0.1  # Minimum exploration rate
        self.DECAY_RATE = 2      # Decay rate for exploration rate
        self.testing_runs = 1    # Number of testing runs   
        self.evaluation_runs = 1
        self.sleeping_time = 60.0