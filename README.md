# Q-Learning Snake Game

This project is an implementation of the Q-learning algorithm, a model-free reinforcement learning technique, applied to a Snake game environment.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Q-learning is a reinforcement learning algorithm used to find the optimal action-selection policy for any given finite Markov decision process. This project demonstrates the application of Q-learning in a Snake game environment.

## Installation
To install the necessary dependencies, run:
```bash
pip install numpy pygame matplotlib
```

## Usage
To run the Q-learning algorithm, execute:
```bash
python train.py
```

## Training
The training script initializes the game environment and Q-learning agent, and starts the training process. The Q-table is saved periodically during training.

### Training Parameters
- `EPISODES`: Number of training episodes.
- `SHOW_EVERY`: Frequency of rendering the game during training.
- `SAVE_EVERY`: Frequency of saving the Q-table.
- `STATS_EVERY`: Frequency of recording training statistics.

### Training Script
The training script is located in `train.py`. To start training, run:
```bash
python train.py
```

### Training Output
During training, the script will output the following information for each episode:
- Episode number
- Score
- Steps taken
- Epsilon value

The training script will also save the Q-table periodically and generate training curves showing the score and steps per episode.

## Evaluation
You can evaluate the trained model by loading the saved Q-table and running the evaluation script.

### Evaluation Script
```python
from qlearning import QLearningAgent
from game import SnakeGame

def evaluate(agent, game, episodes=100):
    total_score = 0
    total_steps = 0
    
    for episode in range(episodes):
        state = game.reset()
        done = False
        score = 0
        steps = 0
        
        while not done:
            action = agent.choose_action(state)
            reward, done, score = game.step(action)
            state = game.get_state()
            steps += 1
        
        total_score += score
        total_steps += steps
    
    avg_score = total_score / episodes
    avg_steps = total_steps / episodes
    print(f'Average Score: {avg_score:.2f} | Average Steps: {avg_steps:.2f}')

# Load the model and evaluate
agent = QLearningAgent(state_space=16)
agent.load('pretrained/q_table_final.json')
game = SnakeGame(grid_size=20, block_size=20)
evaluate(agent, game)
```

### Running the Evaluation
To evaluate the trained model, run the following script:
```bash
python evaluate.py
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

