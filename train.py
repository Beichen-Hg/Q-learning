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

def train():
    print("Starting training...")
    game = SnakeGame(grid_size=20, block_size=20)  # Initialize the game environment
    agent = QLearningAgent(state_space=len(game.get_state()))  # Initialize the Q-learning agent
    
    # Load pretrained Q-table if available
    pretrained_path = 'pretrained/q_table_final.json'
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained Q-table from {pretrained_path}")
        agent.load(pretrained_path)
    
    # Dictionary to store training statistics
    stats = {
        'episode': [],
        'scores': [],
        'steps': [],
        'epsilons': []
    }
    
    # Create directory to save pretrained Q-tables if it doesn't exist
    pretrained_dir = 'pretrained'
    if not os.path.exists(pretrained_dir):
        print(f"Creating directory: {pretrained_dir}")
        os.makedirs(pretrained_dir)
    
    # Training loop
    for episode in range(1, EPISODES+1):
        print(f"Starting episode {episode}")
        state = game.reset()  # Reset the game environment
        total_reward = 0
        done = False
        show = episode % SHOW_EVERY == 0  # Determine if the game should be rendered
        
        while not done:
            action = agent.choose_action(state)  # Choose an action based on the current state
            reward, done, score = game.step(action)  # Execute the action and get the reward and next state
            next_state = game.get_state()  # Get the next state
            
            agent.learn(state, action, reward, next_state)  # Update the Q-table
            state = next_state  # Update the current state
            total_reward += reward  # Accumulate the reward
            
            if show:
                game.render(agent.q_table[state], agent.epsilon)  # Render the game
                time.sleep(0.05)
                
        agent.decay_epsilon()  # Decay the exploration rate
        
        # Record statistics
        if episode % STATS_EVERY == 0:
            stats['episode'].append(episode)
            stats['scores'].append(score)
            stats['steps'].append(game.steps)
            stats['epsilons'].append(agent.epsilon)
            
        # Save the Q-table periodically
        if episode % SAVE_EVERY == 0:
            save_path = os.path.join(pretrained_dir, f'q_table_{episode}.json')
            print(f"Saving Q-table to {save_path}")
            if not os.path.exists(pretrained_dir):
                os.makedirs(pretrained_dir)
            agent.save(save_path)
            
        print(f'Episode: {episode:4d} | Score: {score:2d} | '
              f'Steps: {game.steps:4d} | Epsilon: {agent.epsilon:.2f}')
              
    # Plot training curves
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
    
    # Save the final Q-table
    final_save_path = os.path.join(pretrained_dir, 'q_table_final.json')
    print(f"Saving final Q-table to {final_save_path}")
    agent.save(final_save_path)
    
    game.close()  # Close the game environment
    print("Training completed.")

if __name__ == "__main__":
    train()