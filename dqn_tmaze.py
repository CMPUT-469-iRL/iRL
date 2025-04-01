import torch
import numpy as np
from tmaze_pen import TMazeEnv
from eLSTM_model.model import DQNModel
from dqn_agent import DQNAgent
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import integrate
import time

class TMazeEnvWrapper(TMazeEnv):
    """Wrapper for TMazeEnv that modifies the reward structure:
    - 0 reward for all transition states
    - +1 reward for correct terminal state
    - -1 reward for incorrect terminal state
    """
    def __init__(self, corridor_length=10, delay=0, randomize_goal=True):
        super().__init__(corridor_length, delay, randomize_goal)
        
    def step(self, action):
        self.t += 1
        
        # Update signal after delay
        if self.t == self.delay + 1:
            self.signal = self.goal_location
        
        # Default reward is 0 for transitions
        reward = 0
        
        # Handle movement based on action
        if action == 0:  # Left
            if self.pos[1] == self.corridor_length:  # At junction
                self.pos[0] -= 1
                self.done = True
                # Only give reward at terminal state
                reward = 1.0 if self.goal_location == "left" else -1.0
        elif action == 1:  # Right
            if self.pos[1] == self.corridor_length:  # At junction
                self.pos[0] += 1
                self.done = True
                # Only give reward at terminal state
                reward = 1.0 if self.goal_location == "right" else -1.0
        elif action == 2:  # Forward
            if self.pos[1] < self.corridor_length:
                self.pos[1] += 1
                # No reward for moving forward (0)
            else:
                # No penalty for trying to move forward at junction
                pass
        
        obs = self._get_observation()
        return obs, reward, self.done, {}
    
    def set_corridor_length(self, length):
        """Set corridor length dynamically for curriculum learning"""
        self.corridor_length = length
        self.reset()

def train_dqn(env, agent, num_episodes, eval_interval=100, save_dir='dqn_checkpoints', curriculum_learning=True):
    # Create save directory with timestamp
    print(f"TRAINING DQN for {num_episodes} episodes with eval interval {eval_interval}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(save_dir, f'tmaze_dqn_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Start time for tracking total training time
    start_time = time.time()
    
    # ----- Pre-fill Replay Buffer ----
    MIN_REPLAY_SIZE = 5000  # Increased from 1000 for better initial training
    print(f"Filling replay buffer with {MIN_REPLAY_SIZE} experiences...")
    
    # Start with a shorter corridor for easier pre-training
    if curriculum_learning:
        original_corridor_length = env.corridor_length
        env.set_corridor_length(2)  # Start with a very short corridor
    
    state = env.reset()
    while len(agent.memory) < MIN_REPLAY_SIZE:
        action = np.random.randint(0, agent.action_size)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        if done:
            state = env.reset()
        else:
            state = next_state
    print("Initial replay buffer filled with random experiences.")
    
    # Reset corridor length back after filling buffer if we're using curriculum learning
    if curriculum_learning:
        # Start with corridor length of 2 and gradually increase
        corridor_lengths = [2, 3, 4, original_corridor_length]
        curriculum_stages = len(corridor_lengths)
        episodes_per_stage = num_episodes // curriculum_stages
        
        print(f"Using curriculum learning with stages: {corridor_lengths}")
    else:
        # Keep original corridor length throughout training
        episodes_per_stage = num_episodes
        curriculum_stages = 1

    # Training metrics
    episode_rewards = []
    success_rates = []
    eval_rewards = []
    discounted_returns = []  # Track discounted returns
    
    # For AUC calculations
    success_rate_episodes = []
    eval_reward_episodes = []
    
    # Set a maximum step count per episode to avoid infinite loops
    max_steps_per_episode = 100  # Reduced from 1000 to prevent long episodes
    
    current_stage = 0
    success_count = 0
    success_window = 20  # Number of episodes to consider for advancement
    success_threshold = 0.8  # 80% success rate to advance to next stage
    
    # For large runs, reduce verbosity
    log_interval = 100 if num_episodes <= 10000 else 1000
    checkpoint_interval = eval_interval
    
    for episode in range(num_episodes):
        # Check if we need to update corridor length based on curriculum
        if curriculum_learning and episode > 0 and episode % episodes_per_stage == 0 and current_stage < len(corridor_lengths) - 1:
            current_stage += 1
            env.set_corridor_length(corridor_lengths[current_stage])
            print(f"Curriculum Learning: Advancing to corridor length {corridor_lengths[current_stage]}")
        
        # Alternative advancement based on performance
        if curriculum_learning and episode > success_window and episode % 10 == 0:
            recent_success_rate = np.mean([1 if r > 0 else 0 for r in episode_rewards[-success_window:]])
            if recent_success_rate >= success_threshold and current_stage < len(corridor_lengths) - 1:
                current_stage += 1
                env.set_corridor_length(corridor_lengths[current_stage])
                print(f"Curriculum Learning: Performance threshold reached! Advancing to corridor length {corridor_lengths[current_stage]}")
        
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        # For tracking discounted return
        discount_factor = 1.0
        episode_discounted_return = 0
        
        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition with original reward values
            agent.store_transition(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Track discounted return
            episode_discounted_return += discount_factor * reward
            discount_factor *= agent.gamma
            
            steps += 1
            
            if len(agent.memory) >= agent.batch_size:
                agent.train()
                
        # Update metrics
        episode_rewards.append(episode_reward)
        discounted_returns.append(episode_discounted_return)
        
        # Track success for curriculum advancement
        if episode_reward > 0:
            success_count += 1
        
        # Log progress less frequently for large runs
        if episode % log_interval == 0 or episode == num_episodes - 1:
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / (episode + 1)) * (num_episodes - episode - 1) if episode > 0 else 0
            print(f"Episode {episode}/{num_episodes} ({(episode+1)/num_episodes*100:.1f}%) - Time: {elapsed_time/60:.1f}m - ETA: {eta/60:.1f}m")
            print(f"Reward: {episode_reward}, Discounted Return: {episode_discounted_return:.4f}, Corridor Length: {env.corridor_length}")
            
            # Show recent performance
            window = min(100, len(episode_rewards))
            if window > 0:
                recent_rewards = episode_rewards[-window:]
                recent_discounted_returns = discounted_returns[-window:]
                print(f"Recent {window} episodes - Avg Reward: {np.mean(recent_rewards):.2f}, Success Rate: {np.mean([1 if r > 0 else 0 for r in recent_rewards]):.2f}")
        
        # Evaluate periodically
        if (episode + 1) % eval_interval == 0 or episode == num_episodes - 1:
            # Use original corridor length for evaluation
            if curriculum_learning:
                original_length = env.corridor_length
                env.set_corridor_length(original_corridor_length)
            
            eval_reward = evaluate_dqn(env, agent, num_episodes=30)  # More evaluation episodes for reliability
            eval_rewards.append(eval_reward)
            eval_reward_episodes.append(episode + 1)  # Record episode number for AUC calculation
            
            # Restore corridor length after evaluation
            if curriculum_learning:
                env.set_corridor_length(original_length)
            
            # Here success rate is computed as the fraction of evaluation episodes with positive reward.
            success_rate = sum(1 for r in eval_rewards[-10:] if r > 0) / 10 if len(eval_rewards) >= 10 else 0
            success_rates.append(success_rate)
            success_rate_episodes.append(episode + 1)  # Record episode number for AUC calculation
            
            print(f'Evaluation at Episode {episode + 1}/{num_episodes}')
            print(f'Average Reward (last {eval_interval} episodes): {np.mean(episode_rewards[-eval_interval:]):.2f}')
            print(f'Average Discounted Return (last {eval_interval} episodes): {np.mean(discounted_returns[-eval_interval:]):.2f}')
            print(f'Success Rate: {success_rate:.2f}')
            print(f'Eval Reward: {eval_reward:.2f}')
            
            # Calculate AUC for current data
            if len(success_rate_episodes) > 1:
                success_rate_auc = calculate_auc(success_rate_episodes, success_rates)
                eval_reward_auc = calculate_auc(eval_reward_episodes, eval_rewards)
                print(f'Success Rate AUC: {success_rate_auc:.4f}')
                print(f'Eval Reward AUC: {eval_reward_auc:.4f}')
            
            # Save checkpoint (less frequently for large runs)
            if (episode + 1) % checkpoint_interval == 0 or episode == num_episodes - 1:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_{episode+1}.pt')
                agent.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            
            # Plot training metrics
            plot_metrics(episode_rewards, success_rates, eval_rewards, discounted_returns, 
                         success_rate_episodes, eval_reward_episodes, save_dir)
    
    # Calculate final AUC metrics
    final_success_rate_auc = calculate_auc(success_rate_episodes, success_rates)
    final_eval_reward_auc = calculate_auc(eval_reward_episodes, eval_rewards)
    
    print("Training completed. Final metrics:")
    print(f"Total training time: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"Success Rate AUC: {final_success_rate_auc:.4f}")
    print(f"Eval Reward AUC: {final_eval_reward_auc:.4f}")
    
    # Save final metrics
    np.savez(os.path.join(save_dir, 'training_metrics.npz'),
             episode_rewards=np.array(episode_rewards),
             success_rates=np.array(success_rates),
             success_rate_episodes=np.array(success_rate_episodes),
             eval_rewards=np.array(eval_rewards),
             eval_reward_episodes=np.array(eval_reward_episodes),
             discounted_returns=np.array(discounted_returns),
             success_rate_auc=final_success_rate_auc,
             eval_reward_auc=final_eval_reward_auc)
    
    return episode_rewards, success_rates, eval_rewards, discounted_returns, final_success_rate_auc, final_eval_reward_auc

def calculate_auc(x_values, y_values):
    """Calculate area under the curve using trapezoidal rule."""
    if len(x_values) <= 1:
        return 0.0
    
    # Normalize x-values to [0,1] for fair comparison between different run lengths
    x_normalized = np.array(x_values) / x_values[-1]
    
    # Calculate AUC using scipy's integration
    auc = integrate.trapz(y_values, x_normalized)
    
    return auc

def evaluate_dqn(env, agent, num_episodes=10):
    eval_rewards = []
    max_steps_per_episode = 100  # Reduced from 1000
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            steps += 1
        eval_rewards.append(episode_reward)
    return np.mean(eval_rewards)

def plot_metrics(episode_rewards, success_rates, eval_rewards, discounted_returns, 
                success_rate_episodes, eval_reward_episodes, save_dir):
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    window_size = min(100, len(episode_rewards))
    if window_size > 0:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), smoothed_rewards, 'r-', alpha=0.7)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot success rate with AUC
    plt.subplot(2, 2, 2)
    plt.plot(success_rate_episodes, success_rates, 'b-')
    plt.fill_between(success_rate_episodes, 0, success_rates, alpha=0.2, color='blue')
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)
    auc = calculate_auc(success_rate_episodes, success_rates)
    plt.title(f'Success Rate (AUC: {auc:.4f})')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Rate')
    plt.ylim([-0.1, 1.1])
    
    # Plot eval rewards with AUC
    plt.subplot(2, 2, 3)
    plt.plot(eval_reward_episodes, eval_rewards, 'g-')
    plt.fill_between(eval_reward_episodes, np.minimum(0, np.min(eval_rewards)), eval_rewards, alpha=0.2, color='green')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    eval_auc = calculate_auc(eval_reward_episodes, eval_rewards)
    plt.title(f'Evaluation Rewards (AUC: {eval_auc:.4f})')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Reward')
    
    # Plot discounted returns
    plt.subplot(2, 2, 4)
    plt.plot(discounted_returns)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    window_size = min(100, len(discounted_returns))
    if window_size > 0:
        smoothed_returns = np.convolve(discounted_returns, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(discounted_returns)), smoothed_returns, 'r-', alpha=0.7)
    plt.title('Discounted Returns')
    plt.xlabel('Episode')
    plt.ylabel('Discounted Return')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--corridor_length', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--save_dir', type=str, default='dqn_checkpoints')
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning')
    args = parser.parse_args()
    
    # Initialize environment with wrapper for fixed reward structure
    env = TMazeEnvWrapper(corridor_length=args.corridor_length)
    
    # Initialize DQN model and agent
    model = DQNModel(
        emb_dim=0,
        hidden_size=args.hidden_size,
        in_vocab_size=4,
        out_vocab_size=3,
        no_embedding=True,
        dropout=0.1
    )
    
    agent = DQNAgent(
        state_size=4,
        action_size=3,
        model=model,
        learning_rate=args.lr
    )
    
    # Train the agent
    episode_rewards, success_rates, eval_rewards, discounted_returns, success_rate_auc, eval_reward_auc = train_dqn(
        env, agent, args.episodes, args.eval_interval, args.save_dir, curriculum_learning=args.curriculum
    )
    
    # Final evaluation
    final_eval_reward = evaluate_dqn(env, agent, num_episodes=100)
    print(f'\nFinal Evaluation Reward: {final_eval_reward:.2f}')
    print(f'Final Success Rate: {sum(1 for r in eval_rewards[-10:] if r > 0) / 10:.2f}')
    print(f'Success Rate AUC: {success_rate_auc:.4f}')
    print(f'Eval Reward AUC: {eval_reward_auc:.4f}')

if __name__ == '__main__':
    main()
