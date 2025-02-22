import numpy as np
import json
from collections import defaultdict

class QLearningAgent:
    def __init__(self, state_space, action_space=4):
        self.q_table = defaultdict(lambda: np.zeros(action_space))
        self.alpha = 0.1    # 学习率
        self.gamma = 0.95   # 折扣因子
        self.epsilon = 0.7  # 初始探索率
        self.min_epsilon = 0.01
        self.decay_rate = 0.995
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(4)  # 随机探索
        else:
            return np.argmax(self.q_table[state])
            
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon*self.decay_rate)
        
    def save(self, filename):
        serializable = {str(k): v.tolist() for k, v in self.q_table.items()}
        with open(filename, 'w') as f:
            json.dump(serializable, f)
            
    def load(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.q_table = defaultdict(lambda: np.zeros(4))
        for k, v in data.items():
            self.q_table[eval(k)] = np.array(v)