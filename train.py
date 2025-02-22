import time
import matplotlib.pyplot as plt
from game import SnakeGame
from qlearning import QLearningAgent
import os

# 训练参数
EPISODES = 1000
SHOW_EVERY = 50    # 每N轮显示一次
SAVE_EVERY = 100   # 每N轮保存一次
STATS_EVERY = 20   # 统计间隔

def train():
    print("Starting training...")
    game = SnakeGame(grid_size=20, block_size=20)
    agent = QLearningAgent(state_space=len(game.get_state()))
    
    # 如果存在之前保存的模型，则加载模型
    pretrained_path = 'pretrained/q_table_final.json'
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained Q-table from {pretrained_path}")
        agent.load(pretrained_path)
    
    stats = {
        'episode': [],
        'scores': [],
        'steps': [],
        'epsilons': []
    }
    
    pretrained_dir = 'pretrained'
    if not os.path.exists(pretrained_dir):
        print(f"Creating directory: {pretrained_dir}")
        os.makedirs(pretrained_dir)
    
    for episode in range(1, EPISODES+1):
        print(f"Starting episode {episode}")
        state = game.reset()
        total_reward = 0
        done = False
        show = episode % SHOW_EVERY == 0
        
        while not done:
            action = agent.choose_action(state)
            reward, done, score = game.step(action)
            next_state = game.get_state()
            
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
            if show:
                game.render(agent.q_table[state], agent.epsilon)
                time.sleep(0.05)
                
        agent.decay_epsilon()
        
        # 记录统计信息
        if episode % STATS_EVERY == 0:
            stats['episode'].append(episode)
            stats['scores'].append(score)
            stats['steps'].append(game.steps)
            stats['epsilons'].append(agent.epsilon)
            
        # 保存模型
        if episode % SAVE_EVERY == 0:
            save_path = os.path.join(pretrained_dir, f'q_table_{episode}.json')
            print(f"Saving Q-table to {save_path}")
            if not os.path.exists(pretrained_dir):
                os.makedirs(pretrained_dir)
            agent.save(save_path)
            
        print(f'Episode: {episode:4d} | Score: {score:2d} | '
              f'Steps: {game.steps:4d} | Epsilon: {agent.epsilon:.2f}')
              
    # 绘制训练曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.plot(stats['episode'], stats['scores'], 'b-')
    plt.title('Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.subplot(1,2,2)
    plt.plot(stats['episode'], stats['steps'], 'r-')
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.savefig('training_curve.png')
    plt.show()
    
    # 保存最终的 Q 表
    final_save_path = os.path.join(pretrained_dir, 'q_table_final.json')
    print(f"Saving final Q-table to {final_save_path}")
    agent.save(final_save_path)
    
    game.close()
    print("Training completed.")

if __name__ == "__main__":
    train()