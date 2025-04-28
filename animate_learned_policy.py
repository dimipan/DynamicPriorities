from evaluation import AgentEvaluationManager, AgentType

# Initialize the evaluation manager
manager = AgentEvaluationManager()

# Set up environment parameters to match your training environment
env_params = {
    'grid_rows': 4,
    'grid_cols': 4,
    'info_number_needed': 3,
    'sparse_reward': False,
    'reward_shaping': False,
    'attention': True,  # Set to match the training setting
    'render_mode': 'human'  # Add this line to enable rendering
}

# Choose the agent type you want to evaluate
agent_type = AgentType.HIERARCHICAL_ATTENTION  # Change to the appropriate agent type

# Evaluate the agent
# The episode number corresponds to the saved policy file number
result = manager.evaluate_agent(
    agent_type=agent_type,
    episode=1000,  # Change to match your saved policy episode number
    env_params=env_params
)

print(f"Evaluation result: {result}")