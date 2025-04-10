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
import itertools
import json

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

def train_dqn(env, model, agent, num_episodes, eval_interval, corridor_length, hidden_size, lr, curriculum=False):
    episode_rewards = []
    success_rates = []
    eval_rewards = []
    discounted_returns = []
    start_time = time.time()
    
    # Curriculum learning stages with longer training periods
    curriculum_stages = [2, 2, 3, 3, 4, 4, 5] if curriculum else [corridor_length]  # Repeat stages for better learning
    current_stage = 0
    stage_start_episode = 0
    min_episodes_per_stage = 15000  # Increased for better stability
    best_success_rate = 0.0
    
    # Learning rate decay
    initial_lr = lr
    min_lr = lr * 0.1
    
    # Pre-fill replay buffer with successful experiences
    print("Filling replay buffer with initial experiences...")
    successful_experiences = 0
    while successful_experiences < 1000:  # Ensure some successful experiences
        state = env.reset()
        done = False
        trajectory = []
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward, next_state, done))
            state = next_state
        
        # If trajectory was successful, add it to memory
        if reward > 0:
            successful_experiences += 1
            for transition in trajectory:
                agent.memory.push(*transition)
    
    # Fill remaining buffer with random experiences
    while len(agent.memory) < 5000:
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
    
    print("Initial replay buffer filled.")
    
    if curriculum:
        print(f"Using curriculum learning with stages: {curriculum_stages}")
        env.set_corridor_length(curriculum_stages[0])
    
    for episode in range(num_episodes):
        # Decay learning rate
        progress = episode / num_episodes
        current_lr = max(min_lr, initial_lr * (1 - progress))
        agent.optimizer.param_groups[0]['lr'] = current_lr
        
        state = env.reset()
        episode_reward = 0
        episode_return = 0
        done = False
        steps = 0
        max_steps = 100
        
        # Increase exploration for first steps after curriculum change
        if curriculum and episode - stage_start_episode < 1000:
            agent.epsilon = min(0.9, agent.epsilon * 1.2)
        
        while not done and steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Scale rewards for better learning signal
            scaled_reward = reward * 0.1  # Reduce reward magnitude
            if done and reward > 0:
                scaled_reward += 0.5  # Additional bonus for successful episodes
            
            agent.memory.push(state, action, scaled_reward, next_state, done)
            agent.train()
            
            episode_reward += reward  # Track original reward for metrics
            episode_return += reward * (0.99 ** steps)
            state = next_state
            steps += 1
        
        episode_rewards.append(episode_reward)
        discounted_returns.append(episode_return)
        
        # Track success rates for curriculum learning
        if curriculum:
            success = episode_reward > 0
            success_rates.append(1.0 if success else 0.0)
            recent_success_rates = success_rates[-200:] if len(success_rates) >= 200 else success_rates  # Increased window
            current_success_rate = np.mean(recent_success_rates)
            
            # Update best success rate for current stage
            best_success_rate = max(best_success_rate, current_success_rate)
            
            # More conservative curriculum advancement with regression check
            if (episode - stage_start_episode >= min_episodes_per_stage and 
                len(recent_success_rates) >= 200 and 
                current_success_rate > 0.6 and  # Reduced threshold for smoother progression
                np.std(recent_success_rates) < 0.2 and 
                current_success_rate >= best_success_rate * 0.8):  # More lenient regression check
                
                current_stage += 1
                if current_stage < len(curriculum_stages):
                    print(f"\nCurriculum Learning: Advancing to corridor length {curriculum_stages[current_stage]}")
                    print(f"Current Success Rate: {current_success_rate:.2f}, Best Success Rate: {best_success_rate:.2f}")
                    env.set_corridor_length(curriculum_stages[current_stage])
                    stage_start_episode = episode
                    best_success_rate = 0.0
                    # Reset epsilon for exploration in new stage
                    agent.epsilon = min(0.9, agent.epsilon * 2.0)
                else:
                    print("\nCurriculum Learning: Completed all stages!")
                    curriculum = False
        
        # Log progress
        if episode % 1000 == 0:
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / (episode + 1)) * (num_episodes - episode) if episode > 0 else 0
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            recent_success = np.mean(recent_success_rates) if curriculum else 0
            print(f"Episode {episode}/{num_episodes} ({episode/num_episodes*100:.1f}%) - Time: {elapsed_time/60:.1f}m - ETA: {eta/60:.1f}m")
            print(f"Reward: {episode_reward:.1f}, Discounted Return: {episode_return:.4f}, Corridor Length: {env.corridor_length}")
            print(f"Recent 100 episodes - Avg Reward: {np.mean(recent_rewards):.2f}, Success Rate: {recent_success:.2f}")
        
        # Evaluate periodically
        if (episode + 1) % eval_interval == 0 or episode == num_episodes - 1:
            eval_reward = evaluate_dqn(env, agent, num_episodes=30)  # More evaluation episodes for reliability
            eval_rewards.append(eval_reward)
            
            print(f'Evaluation at Episode {episode + 1}/{num_episodes}')
            print(f'Average Reward (last {eval_interval} episodes): {np.mean(episode_rewards[-eval_interval:]):.2f}')
            print(f'Average Discounted Return (last {eval_interval} episodes): {np.mean(discounted_returns[-eval_interval:]):.2f}')
            print(f'Eval Reward: {eval_reward:.2f}')
            
            # Calculate AUC for current data
            if len(success_rates) > 1:
                success_rate_auc = calculate_auc(range(len(success_rates)), success_rates)
                eval_reward_auc = calculate_auc(range(len(eval_rewards)), eval_rewards)
                print(f'Success Rate AUC: {success_rate_auc:.4f}')
                print(f'Eval Reward AUC: {eval_reward_auc:.4f}')
            
            # Save checkpoint (less frequently for large runs)
            if (episode + 1) % eval_interval == 0 or episode == num_episodes - 1:
                checkpoint_path = os.path.join('dqn_checkpoints', f'checkpoint_{episode+1}.pt')
                agent.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            
            # Plot training metrics
            plot_metrics(episode_rewards, success_rates, eval_rewards, discounted_returns, 
                         range(len(success_rates)), range(len(eval_rewards)), 'dqn_checkpoints')
    
    # Calculate final AUC metrics
    final_success_rate_auc = calculate_auc(range(len(success_rates)), success_rates)
    final_eval_reward_auc = calculate_auc(range(len(eval_rewards)), eval_rewards)
    
    print("Training completed. Final metrics:")
    print(f"Total training time: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"Success Rate AUC: {final_success_rate_auc:.4f}")
    print(f"Eval Reward AUC: {final_eval_reward_auc:.4f}")
    
    # Save final metrics
    np.savez(os.path.join('dqn_checkpoints', 'training_metrics.npz'),
             episode_rewards=np.array(episode_rewards),
             success_rates=np.array(success_rates),
             eval_rewards=np.array(eval_rewards),
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

def run_hyperparameter_search(base_args):
    """Run grid search over hyperparameters"""
    # Define hyperparameter grid with focused ranges based on previous experiments
    param_grid = {
        'lr': [0.001, 0.0005],  # Focus on higher learning rates
        'hidden_size': [128, 256],  # These sizes work well for this task
        'batch_size': [64],  # Fix batch size as it's less critical
        'memory_capacity': [50000],  # Fixed value that works well
        'epsilon_decay': [0.998, 0.999],  # Slower decay rates for better exploration
        'gamma': [0.99]  # Fixed to standard value
    }
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    # Store results
    results = []
    best_reward = float('-inf')
    best_params = None
    best_agent = None
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'hyperparam_search_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting hyperparameter search with {len(combinations)} combinations")
    print("Using focused parameter ranges:")
    for param, values in param_grid.items():
        print(f"{param}: {values}")
    
    for i, values in enumerate(combinations):
        params = dict(zip(param_names, values))
        print(f"\nTrying combination {i+1}/{len(combinations)}:")
        for k, v in params.items():
            print(f"{k}: {v}")
        
        # Initialize environment with shorter corridor for faster search
        env = TMazeEnvWrapper(corridor_length=3)  # Start with shorter corridor
        
        # Initialize model and agent with current parameters
        model = DQNModel(
            emb_dim=0,
            hidden_size=params['hidden_size'],
            in_vocab_size=4,
            out_vocab_size=3,
            no_embedding=True,
            dropout=0.1
        )
        
        agent = DQNAgent(
            state_size=4,
            action_size=3,
            model=model,
            learning_rate=params['lr'],
            batch_size=params['batch_size'],
            memory_capacity=params['memory_capacity'],
            epsilon_decay=params['epsilon_decay'],
            gamma=params['gamma']
        )
        
        # Train with current parameters (shorter episodes for search)
        search_episodes = 2000  # Shorter training for quicker search
        episode_rewards, success_rates, eval_rewards, discounted_returns, success_rate_auc, eval_reward_auc = train_dqn(
            env, model, agent, 
            num_episodes=search_episodes,
            eval_interval=200,  # More frequent evaluation
            corridor_length=3,  # Shorter corridor for search
            hidden_size=params['hidden_size'],
            lr=params['lr'],
            curriculum=False  # No curriculum during search
        )
        
        # Evaluate final performance
        final_eval_reward = evaluate_dqn(env, agent, num_episodes=50)
        final_success_rate = sum(1 for r in eval_rewards[-10:] if r > 0) / 10
        
        # Store results
        result = {
            'params': params,
            'final_eval_reward': final_eval_reward,
            'final_success_rate': final_success_rate,
            'success_rate_auc': success_rate_auc,
            'eval_reward_auc': eval_reward_auc,
            'avg_last_100_rewards': np.mean(episode_rewards[-100:])
        }
        results.append(result)
        
        # Update best parameters if current combination is better
        if final_eval_reward > best_reward:
            best_reward = final_eval_reward
            best_params = params
            best_agent = agent
            
            # Save best model
            best_model_path = os.path.join(results_dir, 'best_model.pt')
            agent.save_checkpoint(best_model_path)
            print(f"\nNew best parameters found!")
            print(f"Eval Reward: {final_eval_reward:.2f}")
            print(f"Success Rate: {final_success_rate:.2f}")
            print("Parameters:", params)
        
        # Save current results
        with open(os.path.join(results_dir, 'search_results.json'), 'w') as f:
            json.dump({
                'all_results': results,
                'best_params': best_params,
                'best_reward': float(best_reward)
            }, f, indent=4)
        
        # Plot comparison of results so far
        plot_hyperparameter_results(results, results_dir)
    
    print("\nHyperparameter search completed!")
    print("\nBest parameters found:")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    print(f"Best evaluation reward: {best_reward:.2f}")
    
    return best_params, best_agent

def plot_hyperparameter_results(results, save_dir):
    """Plot comparison of different hyperparameter combinations"""
    plt.figure(figsize=(15, 10))
    
    # Plot final evaluation rewards
    plt.subplot(2, 2, 1)
    rewards = [r['final_eval_reward'] for r in results]
    plt.plot(rewards, 'b-')
    plt.title('Final Evaluation Rewards')
    plt.xlabel('Combination')
    plt.ylabel('Reward')
    
    # Plot final success rates
    plt.subplot(2, 2, 2)
    success_rates = [r['final_success_rate'] for r in results]
    plt.plot(success_rates, 'g-')
    plt.title('Final Success Rates')
    plt.xlabel('Combination')
    plt.ylabel('Success Rate')
    
    # Plot success rate AUC
    plt.subplot(2, 2, 3)
    auc_values = [r['success_rate_auc'] for r in results]
    plt.plot(auc_values, 'r-')
    plt.title('Success Rate AUC')
    plt.xlabel('Combination')
    plt.ylabel('AUC')
    
    # Plot eval reward AUC
    plt.subplot(2, 2, 4)
    eval_auc_values = [r['eval_reward_auc'] for r in results]
    plt.plot(eval_auc_values, 'y-')
    plt.title('Eval Reward AUC')
    plt.xlabel('Combination')
    plt.ylabel('AUC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hyperparameter_comparison.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--corridor_length', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--curriculum', action='store_true', help='Use curriculum learning')
    parser.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization')
    args = parser.parse_args()
    
    if args.optimize:
        print("Running hyperparameter optimization...")
        best_params, best_agent = run_hyperparameter_search(args)
        
        # Save best parameters
        with open('best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=4)
        
        print("\nBest parameters saved to best_hyperparameters.json")
        return
    
    # Regular training with specified or default parameters
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
        env, model, agent, args.episodes, args.eval_interval, args.corridor_length, args.hidden_size, args.lr, args.curriculum
    )
    
    # Final evaluation
    final_eval_reward = evaluate_dqn(env, agent, num_episodes=100)
    print(f'\nFinal Evaluation Reward: {final_eval_reward:.2f}')
    print(f'Final Success Rate: {sum(1 for r in eval_rewards[-10:] if r > 0) / 10:.2f}')
    print(f'Success Rate AUC: {success_rate_auc:.4f}')
    print(f'Eval Reward AUC: {eval_reward_auc:.4f}')

if __name__ == '__main__':
    main()
