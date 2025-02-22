import numpy as np
import json
from collections import defaultdict

class QLearningAgent:
    def __init__(self, state_space, action_space=4):
        """
        Initialize the Q-learning agent.
        
        Parameters:
        state_space (int): The size of the state space.
        action_space (int): The number of possible actions. Default is 4.
        """
        self.q_table = defaultdict(lambda: np.zeros(action_space))  # Initialize Q-table with zeros
        self.alpha = 0.1    # Learning rate
        self.gamma = 0.99   # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.min_epsilon = 0.01  # Minimum exploration rate
        self.decay_rate = 0.995  # Decay rate for exploration rate
        
    def choose_action(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy policy.
        
        Parameters:
        state (tuple): The current state.
        
        Returns:
        int: The action to be taken.
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(4)  # Random exploration
        else:
            return np.argmax(self.q_table[state])  # Exploitation of learned values
            
    def learn(self, state, action, reward, next_state):
        """
        Update the Q-table based on the agent's experience.
        
        Parameters:
        state (tuple): The current state.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (tuple): The next state after taking the action.
        """
        current_q = self.q_table[state][action]  # Current Q-value
        max_next_q = np.max(self.q_table[next_state])  # Maximum Q-value for the next state
        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q)  # Update Q-value using the Q-learning formula
        self.q_table[state][action] = new_q  # Update the Q-table
        
    def decay_epsilon(self):
        """
        Decay the exploration rate (epsilon) after each episode.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        
    def save(self, filename):
        """
        Save the Q-table to a file.
        
        Parameters:
        filename (str): The name of the file to save the Q-table.
        """
        serializable = {str(k): v.tolist() for k, v in self.q_table.items()}  # Convert Q-table to a serializable format
        with open(filename, 'w') as f:
            json.dump(serializable, f)  # Save Q-table to file
            
    def load(self, filename):
        """
        Load the Q-table from a file.
        
        Parameters:
        filename (str): The name of the file to load the Q-table from.
        """
        with open(filename, 'r') as f:
            data = json.load(f)  # Load Q-table from file
        self.q_table = defaultdict(lambda: np.zeros(4))  # Initialize Q-table
        for k, v in data.items():
            self.q_table[eval(k)] = np.array(v)  # Populate Q-table with loaded data