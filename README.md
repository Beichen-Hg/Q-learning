# Snake Game Q-Learning AI

A reinforcement learning project that trains an AI agent to play Snake using Q-Learning algorithm. Built with Python, featuring PyGame visualization and comprehensive experiment management and analysis systems.

## Features

- Q-Learning implementation for reinforcement learning
- Visual game interface with real-time training display
- Support for parallel testing of multiple experiment configurations
- Complete experiment data collection and analysis system
- Automated generation of training curves and experiment reports

## Project Structure

- `qlearning.py`: Core Q-Learning algorithm implementation
- `game.py`: Snake game environment implementation
- `train.py`: Training process control
- `experiment_manager.py`: Experiment management system
- `run_experiments.py`: Multi-experiment configuration runner
- `analysis.py`: Experiment data analysis tools

## Installation

```bash
pip install numpy pygame matplotlib seaborn pandas
# json and glob are included in Python standard library
```

## Requirements

- Python 3.6+
- Pygame
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Usage

### 1. Run Single Training Experiment

```bash
python train.py
```

### 2. Run Multiple Comparative Experiments

```bash
python run_experiments.py
```

### 3. Analyze Experiment Results

```bash
python analysis.py
```

## Experiment Configurations

Five preset configurations:
- `baseline`: Standard learning parameters
- `fast_learning`: Higher learning rate (0.2)
- `slow_exploration`: Slower exploration decay
- `fast_stable`: Balanced parameters
- `fast_aggressive`: Aggressive learning (0.25 learning rate)

Each runs for 500 episodes.

## Training Parameters

Default configuration:
- episodes: 500 (not 2000)
- alpha: 0.1 (learning rate)
- gamma: 0.99 (discount factor)
- epsilon: 1.0 (initial exploration rate)
- min_epsilon: 0.01 (minimum exploration rate)
- decay_rate: 0.995 (exploration decay rate)

## Output Files

- `experiment_results/`: Stores experiment results and configurations
  - `config.json`: Experiment configuration
  - `stats.csv`: Training statistics
  - `final_model.json`: Trained Q-table
- `training_curves/`: Stores learning curves from training process
- `pretrained/`: Stores trained Q-tables
- `analysis_report.txt`: Experiment analysis report

## Experiment Analysis

The system automatically generates the following analyses:
- Learning performance comparison
- Exploration efficiency analysis
- Best experiment configuration recommendations
- Training curve visualization

## Important Notes

- Ensure sufficient disk space for experiment data storage
- Training process can be controlled by adjusting parameters in `train.py`
- Experiment data is automatically saved and can be interrupted/resumed

## State and Action Space

### State Space
The state consists of binary values representing:
- Danger detection in current direction
- Food relative position (left/right/up/down)
- Current direction
- Collision detection

### Action Space
The snake has three possible actions:
- Keep direction (0)
- Turn left (1)
- Turn right (2)

## Example Results

Different configurations achieve different performance characteristics:
- `baseline`: Best balance of learning speed and final performance
- `fast_learning`: Quick initial learning but lower final scores
- `slow_exploration`: Most stable learning but requires longer training
- `fast_stable`: Good compromise between speed and stability
- `fast_aggressive`: Fastest learning but potentially suboptimal final policy



