import pygame
import numpy as np
from pygame.locals import *

class SnakeGame:
    def __init__(self, grid_size=20, block_size=20):
        """
        Initialize the Snake game environment.
        
        Parameters:
        grid_size (int): The size of the grid.
        block_size (int): The size of each block in the grid.
        """
        self.grid_size = grid_size
        self.block_size = block_size
        self.reset()
        
        pygame.init()
        self.screen = pygame.display.set_mode(
            (grid_size * block_size, grid_size * block_size + 50))
        self.font = pygame.font.SysFont('simhei', 18)
        
    def reset(self):
        """
        Reset the game environment to the initial state.
        
        Returns:
        tuple: The initial state of the game.
        """
        self.snake = [(10, 10), (10, 11), (10, 12)]  # Initial snake position
        self.direction = 0  # Initial direction (0-Up, 1-Right, 2-Down, 3-Left)
        self.food = self._generate_food()  # Generate initial food position
        self.score = 0
        self.steps = 0
        return self.get_state()
        
    def _generate_food(self):
        """
        Generate a new food position that is not on the snake.
        
        Returns:
        tuple: The position of the food.
        """
        while True:
            pos = (np.random.randint(2, self.grid_size - 2), 
                   np.random.randint(2, self.grid_size - 2))
            if pos not in self.snake:
                return pos
                
    def get_state(self):
        """
        Get the current state of the game.
        
        Returns:
        tuple: The current state represented as a tuple of integers.
        """
        head = self.snake[0]
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Directions (Up, Right, Down, Left)
        
        # Danger detection (8 directions)
        danger = []
        for dx, dy in [*dirs, (1, -1), (1, 1), (-1, 1), (-1, -1)]:
            x = head[0] + dx
            y = head[1] + dy
            danger.append(int(
                x < 0 or x >= self.grid_size or 
                y < 0 or y >= self.grid_size or 
                (x, y) in self.snake[:-1]
            ))
            
        # Relative position of the food
        food_rel = [
            self.food[0] < head[0],  # Left
            self.food[0] > head[0],  # Right
            self.food[1] < head[1],  # Up
            self.food[1] > head[1]   # Down
        ]
        
        # Direction encoding (one-hot)
        dir_encoding = [0] * 4
        dir_encoding[self.direction] = 1
        
        state = danger + food_rel + dir_encoding
        return tuple(map(int, state))
        
    def step(self, action):
        """
        Take a step in the game based on the given action.
        
        Parameters:
        action (int): The action to take (0-Keep, 1-Turn Left, 2-Turn Right).
        
        Returns:
        tuple: The reward, whether the game is done, and the current score.
        """
        # Change direction: 0-Keep, 1-Turn Left, 2-Turn Right
        if action == 1:
            self.direction = (self.direction - 1) % 4
        elif action == 2:
            self.direction = (self.direction + 1) % 4
            
        # Move the snake's head
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.direction]
        new_head = (self.snake[0][0] + dx, self.snake[0][1] + dy)
        
        # Collision detection
        if (new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            return -10, True, self.score
            
        # Update the snake's body
        self.snake.insert(0, new_head)
        
        # Food detection
        if new_head == self.food:
            self.score += 1
            self.food = self._generate_food()
            reward = 10
        else:
            self.snake.pop()
            reward = -0.1
            
        # Penalty for looping
        if self.steps > 100 and len(set(self.snake)) < 3:
            reward -= 2
            
        self.steps += 1
        return reward, False, self.score
        
    def render(self, q_values=None, epsilon=None):
        """
        Render the game environment.
        
        Parameters:
        q_values (list): The Q-values for the current state.
        epsilon (float): The current exploration rate.
        """
        self.screen.fill((30, 30, 30))
        
        # Draw the grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = (x * self.block_size, y * self.block_size,
                        self.block_size - 1, self.block_size - 1)
                pygame.draw.rect(self.screen, (60, 60, 60), rect, 1)
                
        # Draw the snake
        for i, (x, y) in enumerate(self.snake):
            color = (100, 200, 100) if i == 0 else (50, 150, 50)
            pygame.draw.rect(self.screen, color,
                             (x * self.block_size, y * self.block_size,
                              self.block_size - 1, self.block_size - 1))
                             
        # Draw the food
        pygame.draw.circle(self.screen, (200, 50, 50),
                           (self.food[0] * self.block_size + self.block_size // 2,
                            self.food[1] * self.block_size + self.block_size // 2),
                            self.block_size // 3)
                           
        # Display score and steps
        text = self.font.render(
            f'Score: {self.score}  Steps: {self.steps}', 
            True, (200, 200, 200))
        self.screen.blit(text, (10, self.grid_size * self.block_size + 5))
        
        if q_values is not None:
            text = self.font.render(
                f'Q-values: [{", ".join(f"{v:.1f}" for v in q_values)}]', 
                True, (150, 150, 250))
            self.screen.blit(text, (10, self.grid_size * self.block_size + 25))
            
        pygame.display.flip()
        
    def close(self):
        """
        Close the game environment.
        """
        pygame.quit()