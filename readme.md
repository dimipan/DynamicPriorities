# CA-MIQ

CA‑MIQ (Context‑Aware Max‑Information Q‑learning) is a reinforcement learning framework for tabular environments with a focus on information collection and priority adaptation. The project implements and compares several reinforcement learning approaches including traditional Q-learning and a novel information-maximizing approach in search and rescue robot tasks.

## Project Overview

This project implements a tabular reinforcement learning framework for a search-and-rescue robot environment where the agent must:
- Navigate a grid world
- Collect information in specific priority orders
- Avoid hazards (ditches)
- Reach a target location

The key innovation is in how agents handle changes in information collection priorities during training, with comparison between standard approaches and a context-aware information-maximising approach.

## Key Components

- **Environment**: Grid-based Search and Rescue (SAR) domain
- **Agent Types**: 
  - Baseline Q-learning
  - Baseline with epsilon boost
  - Context-Aware Maximum Information Q-learning (CA-MIQ)
- **Experiment Features**:
  - Static priority settings
  - Dynamic priority changes
  - Parallel training across multiple environments
  - Comprehensive metrics and visualization

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: numpy, matplotlib, gymnasium, tqdm

### Installation

```bash
git clone https://github.com/yourusername/DynamicPriorities.git
cd DynamicPriorities
pip install -r requirements.txt (if exists)
```

## Running Experiments

To run the main experiments comparing different agent types across multiple environments:

```bash
python parallel_training.py
```

To customize experiment parameters, modify the settings in `config.py`.

## Reproducing Results

To visualize and analyze previously generated results:

```bash
python reproduce_results.py
```

The notebook version can also be used:

```bash
jupyter notebook reproduce_resutls_notebook.ipynb
```

## Project Structure

- `parallel_training.py`: Main training framework with parallel execution
- `config.py`: Environment and agent configurations
- `environment_sar.py`: Search and rescue environment implementation
- `InformationCollection.py`: Information collection system logic
- `robot_utils.py`: Robot action definitions and utilities
- `agents.py`: Agent implementations (Q-learning, MaxInfoRL)
- `reproduce_results.py`: Analysis and visualization of results

## Results

The experiments demonstrate how different agent architectures handle:
1. Static environments with fixed information priorities
2. Dynamic environments with changing priorities
3. Adaptation efficiency after priority shifts

Key metrics include:
- Average reward
- Success rate
- Adaptation time (for dynamic scenarios)
- Information collection efficiency

## License

[Add your license information here]

## Acknowledgments

[Add any acknowledgments here]