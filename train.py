import time
import matplotlib.pyplot as plt
from game import SnakeGame
from qlearning import QLearningAgent
import os

# Training parameters
EPISODES = 2000       # Number of training episodes
SHOW_EVERY = 100      # Frequency of rendering the game during training
SAVE_EVERY = 200      # Frequency of saving the Q-table
STATS_EVERY = 20      # Frequency of recording training statistics

def train(config=None):
    """
    Training function for a single training run
    """
    # Default configuration
    default_config = {
        'episodes': 500,
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 1.0,
        'min_epsilon': 0.01,
        'decay_rate': 0.995
    }
    
    # Use provided config or default
    config = config or default_config
    
    start_time = time.time()
    
    # Initialize environment and agent
    game = SnakeGame(grid_size=20, block_size=20)
    agent = QLearningAgent(
        state_space=len(game.get_state()),
        action_space=3,  # Snake has 3 actions: 0-Keep direction, 1-Turn left, 2-Turn right
        alpha=config['alpha'],
        gamma=config['gamma'],
        epsilon=config['epsilon'],
        min_epsilon=config['min_epsilon'],
        decay_rate=config['decay_rate']
    )
    
    # Dictionary to store training statistics
    stats = {
        'episode': [],
        'scores': [],
        'steps': [],
        'epsilons': [],
        'win_rate': []  # Track win rate statistics
    }
    
    # Create directory for saving Q-tables
    pretrained_dir = 'pretrained'
    os.makedirs(pretrained_dir, exist_ok=True)
    
    # Training loop
    for episode in range(1, config['episodes']+1):
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
        
        # Record statistics
        if episode % STATS_EVERY == 0:
            stats['episode'].append(episode)
            stats['scores'].append(score)
            stats['steps'].append(game.steps)
            stats['epsilons'].append(agent.epsilon)
            stats['win_rate'].append(1 if score > 0 else 0)  # Simple win condition based on score
            
        # Save Q-table periodically
        if episode % SAVE_EVERY == 0:
            save_path = os.path.join(pretrained_dir, f'q_table_{episode}.json')
            agent.save(save_path)
            
        print(f'Episode: {episode:4d} | Score: {score:2d} | '
              f'Steps: {game.steps:4d} | Epsilon: {agent.epsilon:.2f}')
    
    # Save final Q-table
    final_save_path = os.path.join(pretrained_dir, 'q_table_final.json')
    agent.save(final_save_path)
    
    game.close()
    print("Training completed.")
    
    # Return training results with required format
    return {
        'config': config,
        'stats': stats,
        'final_model': {str(k): v.tolist() for k, v in agent.q_table.items()},
        'training_time': time.time() - start_time,
        'experiment_id': time.strftime("%Y%m%d-%H%M%S"),
        'training_curve': None
    }

if __name__ == "__main__":
    # Run single training session with default configuration
    training_stats = train()
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_stats['stats']['episode'], training_stats['stats']['scores'])
    plt.title('Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(training_stats['stats']['episode'], training_stats['stats']['steps'])
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()