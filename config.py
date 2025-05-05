from agents import QLearningAgentFlat, QLearningAgentMaxInfoRL


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
        'ditches': [(1, 0), (2, 0), (1, 2)]
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
        'ditches': [(0, 1), (1, 3), (2, 2)]
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
        'ditches': [(1, 1), (2, 3), (3, 0)]
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
        'ditches': [(0, 0), (1, 1), (3, 3)]
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
        'ditches': [(0, 1), (2, 1), (3, 3)]
    }
]

# Define agent configurations
agent_types = [
    {
        "name": "Baseline_Static",
        "agent_class": QLearningAgentFlat,
        "change_priorities": None,  # No changes
        "agent_params": {"boost": False}  # Explicitly set boost=False
    },

    {
        "name": "Baseline-Boost_Static",
        "agent_class": QLearningAgentFlat,
        "change_priorities": None,  # No changes
        "agent_params": {"boost": True}  # Explicitly set boost=False
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
        },
        "agent_params": {"boost": False}  # Explicitly set boost=True
    },

    {
        "name": "Baseline-Boost_Dynamic",
        "agent_class": QLearningAgentFlat,
        "change_priorities": {
            1700: {'X': 2, 'Y': 0, 'Z': 1},  # Change from X-Y-Z to Y-Z-X
            # 3500: {'X': 1, 'Y': 2, 'Z': 0},  # Change to Z-X-Y
        },
        "agent_params": {"boost": True}  # Explicitly set boost=True
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