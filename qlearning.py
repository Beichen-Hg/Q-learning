import numpy as np
import json
from collections import defaultdict

class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=1.0, min_epsilon=0.01, decay_rate=0.995):
        """
        Initialize Q-learning agent
        
        Parameters:
        state_space (int): Size of state space
        action_space (int): Number of possible actions
        alpha (float): Learning rate
        gamma (float): Discount factor
        epsilon (float): Initial exploration rate
        min_epsilon (float): Minimum exploration rate
        decay_rate (float): Exploration rate decay
        """
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.min_epsilon = min_epsilon  # Minimum exploration
        self.decay_rate = decay_rate  # Decay rate
        self.q_table = defaultdict(lambda: np.zeros(action_space))  # Initialize Q-table with zeros
        
    def choose_action(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy policy.
        
        Parameters:
        state (tuple): The current state.
        
        Returns:
        int: The action to be taken.
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)  # Random exploration
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
        self.q_table = defaultdict(lambda: np.zeros(self.action_space))  # Initialize Q-table
        for k, v in data.items():
            self.q_table[eval(k)] = np.array(v)  # Populate Q-table with loaded data