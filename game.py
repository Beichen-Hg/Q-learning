import pygame
import numpy as np
from pygame.locals import *

class SnakeGame:
    def __init__(self, grid_size=20, block_size=20):
        self.grid_size = grid_size
        self.block_size = block_size
        self.reset()
        
        pygame.init()
        self.screen = pygame.display.set_mode(
            (grid_size*block_size, grid_size*block_size+50))
        self.font = pygame.font.SysFont('simhei', 18)
        
    def reset(self):
        self.snake = [(10,10), (10,11), (10,12)]
        self.direction = 0  # 0-上 1-右 2-下 3-左
        self.food = self._generate_food()
        self.score = 0
        self.steps = 0
        return self.get_state()
        
    def _generate_food(self):
        while True:
            pos = (np.random.randint(2, self.grid_size-2), 
                  np.random.randint(2, self.grid_size-2))
            if pos not in self.snake:
                return pos
                
    def get_state(self):
        head = self.snake[0]
        dirs = [(0,-1), (1,0), (0,1), (-1,0)]  # 上右下左
        
        # 危险检测（8方向）
        danger = []
        for dx, dy in [*dirs, (1,-1), (1,1), (-1,1), (-1,-1)]:
            x = head[0] + dx
            y = head[1] + dy
            danger.append(int(
                x < 0 or x >= self.grid_size or 
                y < 0 or y >= self.grid_size or 
                (x,y) in self.snake[:-1]
            ))
            
        # 食物相对位置
        food_rel = [
            self.food[0] < head[0],  # 左
            self.food[0] > head[0],  # 右
            self.food[1] < head[1],  # 上
            self.food[1] > head[1]   # 下
        ]
        
        # 方向编码 (one-hot)
        dir_encoding = [0]*4
        dir_encoding[self.direction] = 1
        
        state = danger + food_rel + dir_encoding
        return tuple(map(int, state))
        
    def step(self, action):
        # 转换方向：0-保持 1-左转 2-右转
        if action == 1:
            self.direction = (self.direction - 1) % 4
        elif action == 2:
            self.direction = (self.direction + 1) % 4
            
        # 移动蛇头
        dx, dy = [(0,-1), (1,0), (0,1), (-1,0)][self.direction]
        new_head = (self.snake[0][0] + dx, self.snake[0][1] + dy)
        
        # 碰撞检测
        if (new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            return -10, True, self.score
            
        # 更新蛇身
        self.snake.insert(0, new_head)
        
        # 食物检测
        if new_head == self.food:
            self.score += 1
            self.food = self._generate_food()
            reward = 10
        else:
            self.snake.pop()
            reward = -0.1
            
        # 防止循环惩罚
        if self.steps > 100 and len(set(self.snake)) < 3:
            reward -= 2
            
        self.steps += 1
        return reward, False, self.score
        
    def render(self, q_values=None, epsilon=None):
        self.screen.fill((30, 30, 30))
        
        # 绘制网格
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = (x*self.block_size, y*self.block_size,
                       self.block_size-1, self.block_size-1)
                pygame.draw.rect(self.screen, (60,60,60), rect, 1)
                
        # 绘制蛇身
        for i, (x,y) in enumerate(self.snake):
            color = (100, 200, 100) if i ==0 else (50, 150, 50)
            pygame.draw.rect(self.screen, color,
                            (x*self.block_size, y*self.block_size,
                             self.block_size-1, self.block_size-1))
                             
        # 绘制食物
        pygame.draw.circle(self.screen, (200, 50, 50),
                          (self.food[0]*self.block_size + self.block_size//2,
                           self.food[1]*self.block_size + self.block_size//2),
                           self.block_size//3)
                           
        # 状态信息
        text = self.font.render(
            f'得分: {self.score}  步数: {self.steps}', 
            True, (200, 200, 200))
        self.screen.blit(text, (10, self.grid_size*self.block_size + 5))
        
        if q_values is not None:
            text = self.font.render(
                f'Q值: [{", ".join(f"{v:.1f}" for v in q_values)}]', 
                True, (150, 150, 250))
            self.screen.blit(text, (10, self.grid_size*self.block_size + 25))
            
        pygame.display.flip()
        
    def close(self):
        pygame.quit()