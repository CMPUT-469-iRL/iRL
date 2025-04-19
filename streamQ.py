import os
import argparse
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn

# Enable CuDNN benchmarking for faster training on consistent input sizes
torch.backends.cudnn.benchmark = True

from tmaze_pen import TMazeEnv
from eLSTM_model.model import QuasiLSTMModel


class StreamQAgent:
    """
    Optimized Stream Q agent using QuasiLSTM for efficient learning.
    """
    def __init__(self, 
                input_size, 
                embedding_size,
                hidden_size, 
                output_size, 
                lr=0.001, 
                gamma=0.99, 
                epsilon_start=1.0, 
                epsilon_end=0.01, 
                epsilon_decay=0.995,
                memory_size=10000,
                batch_size=64,  # Increased batch size for better GPU utilization
                no_embedding=False,
                target_update_freq=10):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Main Q-network
        self.q_network = QuasiLSTMModel(
            emb_dim=embedding_size,
            hidden_size=hidden_size,
            in_vocab_size=input_size,
            out_vocab_size=output_size,
            no_embedding=no_embedding
        ).to(self.device)
        
        # Target network for stable learning
        self.target_network = QuasiLSTMModel(
            emb_dim=embedding_size,
            hidden_size=hidden_size,
            in_vocab_size=input_size,
            out_vocab_size=output_size,
            no_embedding=no_embedding
        ).to(self.device)
        
        # Copy parameters from Q-network to target network
        self.update_target_network()
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Setup for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.input_size = input_size
        self.output_size = output_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Pre-allocate tensors for common operations
        self.state_tensor = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        
    def reset_hidden_state(self):
        """Reset the hidden state at the beginning of episodes."""
        pass  # QuasiLSTM handles state internally
    
    @torch.no_grad()  # Performance optimization
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.output_size)
        
        # Reuse pre-allocated tensor
        self.state_tensor[0, 0] = state
                
        # Get Q-values from network
        q_values = self.q_network(self.state_tensor)
        return q_values.squeeze(0).argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        
    def update_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        """Update target network with current Q-network parameters."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(self):
        """Perform a single training step with optimizations."""
        if len(self.memory) < self.batch_size:
            return 0
            
        # Sample random batch from memory - use numpy for faster sampling
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        transitions = [self.memory[idx] for idx in indices]
        
        # Prepare batch data more efficiently
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # Convert to tensors efficiently
        states = torch.tensor(states, dtype=torch.long, device=self.device).unsqueeze(0)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).view(1, -1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(1, -1, 1)
        next_states = torch.tensor(next_states, dtype=torch.long, device=self.device).unsqueeze(0)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).view(1, -1, 1)
        
        # Reset gradients
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than setting to zero
        
        # Use mixed precision if available
        if self.scaler:
            with torch.cuda.amp.autocast():
                # Compute Q-values for current states
                q_values = self.q_network(states)
                q_values = q_values.gather(-1, actions)
                
                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = self.target_network(next_states)
                    next_q_values = next_q_values.max(-1)[0].unsqueeze(-1)
                    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                
                # Compute loss
                loss = self.loss_fn(q_values, target_q_values)
            
            # Scale gradients and optimize
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Compute Q-values for current states
            q_values = self.q_network(states)
            q_values = q_values.gather(-1, actions)
            
            # Compute target Q-values
            with torch.no_grad():
                next_q_values = self.target_network(next_states)
                next_q_values = next_q_values.max(-1)[0].unsqueeze(-1)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute loss and backpropagate
            loss = self.loss_fn(q_values, target_q_values)
            loss.backward()
            self.optimizer.step()
        
        return loss.item()


def train_stream_q(env_name="tmaze", 
                  episodes=500, 
                  epsilon_start=1.0,
                  epsilon_end=0.01,
                  epsilon_decay=0.995,
                  lr=0.001,
                  gamma=0.99,
                  embedding_size=32,
                  hidden_size=64,
                  batch_size=64,  # Increased batch size
                  target_update_freq=10,
                  save_path=None,
                  eval_interval=50,  # Reduced evaluation frequency
                  corridor_length=5,
                  max_steps_per_episode=100,
                  max_time_steps=None,
                  disable_plot=True,  # Disable plotting for faster training
                  progress_interval=20,  # Report progress less frequently
                  ):
    """Optimized training function for Stream Q agent."""
    
    start_time = time.time()
    
    # Set up environment
    if env_name == "tmaze":
        env = TMazeEnv(corridor_length=corridor_length)
        input_size = 4  # TMaze has 4 observation states
        output_size = 3  # TMaze has 3 actions
    elif env_name == "pendulum":
        env = gym.make("Pendulum-v1")
        input_size = 1000  # Discretized observations (10^3)
        output_size = 11   # Discretized actions for pendulum
    else:
        raise ValueError(f"Unsupported environment type: {env_name}")
    
    # Construct save path if not provided
    if save_path is None:
        save_path = f"save_models/stream_q_model_{corridor_length}.pt"
        
    # Initialize agent
    agent = StreamQAgent(
        input_size=input_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        output_size=output_size,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        memory_size=100000,  # Increased memory size
        batch_size=batch_size,
        target_update_freq=target_update_freq
    )
    
    # Track metrics
    rewards_history = []
    losses_history = []
    evaluation_rewards = []
    
    # Add a counter for total time steps across all episodes
    total_time_steps = 0
    
    # Pre-evaluation to establish baseline performance
    eval_reward = evaluate_agent(agent, env, episodes=5, max_steps_per_episode=max_steps_per_episode)
    evaluation_rewards.append(eval_reward)
    print(f"Initial evaluation: Average Reward = {eval_reward:.2f}")
    
    # Training loop
    for episode in range(episodes):
        # Reset environment and agent state
        state = env.reset()
        agent.reset_hidden_state()
        
        done = False
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        
        # Episode loop
        while not done and step_count < max_steps_per_episode:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Increment total time steps counter
            total_time_steps += 1
            
            # Check if we've reached the maximum time steps
            if max_time_steps is not None and total_time_steps >= max_time_steps:
                print(f"Reached maximum time steps ({max_time_steps}). Stopping training.")
                
                # Save model before exiting
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': agent.q_network.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'rewards_history': rewards_history,
                        'losses_history': losses_history,
                        'eval_rewards': evaluation_rewards,
                        'total_time_steps': total_time_steps,
                        'training_time': time.time() - start_time,
                        'hyperparams': {
                            'epsilon_start': epsilon_start,
                            'epsilon_end': epsilon_end,
                            'epsilon_decay': epsilon_decay,
                            'lr': lr,
                            'gamma': gamma,
                            'embedding_size': embedding_size,
                            'hidden_size': hidden_size,
                            'batch_size': batch_size,
                            'corridor_length': corridor_length
                        }
                    }, save_path)
                    print(f"Model saved to {save_path} after {total_time_steps} time steps")
                return agent, rewards_history
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent - only backprop every 4 steps for efficiency
            if total_time_steps % 4 == 0:
                loss = agent.train_step()
                if loss > 0:
                    episode_loss += loss
                
            # Update state and reward
            state = next_state
            episode_reward += reward
            step_count += 1
            
        # Update exploration rate
        agent.update_epsilon()
        
        # Update target network periodically
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
            
        # Track metrics
        rewards_history.append(episode_reward)
        losses_history.append(episode_loss / max(1, step_count // 4))  # Normalize by training steps
            
        # Print progress less frequently to reduce overhead
        if (episode + 1) % progress_interval == 0:
            avg_reward = np.mean(rewards_history[-progress_interval:])
            elapsed = time.time() - start_time
            steps_per_sec = total_time_steps / max(1, elapsed)
            print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}, "
                  f"Avg R: {avg_reward:.2f}, Eps: {agent.epsilon:.3f}, "
                  f"Steps: {total_time_steps} ({steps_per_sec:.1f}/s)")
                  
        # Evaluate agent less frequently to save time
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(agent, env, episodes=5, max_steps_per_episode=max_steps_per_episode)
            evaluation_rewards.append(eval_reward)
            print(f"Eval at ep {episode+1}: Avg R = {eval_reward:.2f}")
            
    # Save trained model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': agent.q_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'rewards_history': rewards_history,
            'losses_history': losses_history,
            'eval_rewards': evaluation_rewards,
            'total_time_steps': total_time_steps,
            'training_time': time.time() - start_time,
            'hyperparams': {
                'epsilon_start': epsilon_start,
                'epsilon_end': epsilon_end,
                'epsilon_decay': epsilon_decay,
                'lr': lr,
                'gamma': gamma,
                'embedding_size': embedding_size,
                'hidden_size': hidden_size,
                'batch_size': batch_size,
                'corridor_length': corridor_length
            }
        }, save_path)
        
        print(f"Model saved to {save_path}")
        print(f"Training completed in {(time.time() - start_time):.1f} seconds")
    
    # Plot training metrics only if not disabled
    if not disable_plot:
        plot_training_results(rewards_history, losses_history, evaluation_rewards, eval_interval)
    
    return agent, rewards_history


@torch.no_grad()  # Performance optimization
def evaluate_agent(agent, env, episodes=10, max_steps_per_episode=100):
    """Evaluate a trained agent without exploration."""
    total_rewards = []
    
    for _ in range(episodes):
        state = env.reset()
        agent.reset_hidden_state()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1
            
        total_rewards.append(episode_reward)
        
    return np.mean(total_rewards)


def plot_training_results(rewards, losses, eval_rewards, eval_interval):
    """Plot training metrics."""
    plt.figure(figsize=(15, 5))
    
    # Plot episode rewards
    plt.subplot(131)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot losses
    plt.subplot(132)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot evaluation rewards
    plt.subplot(133)
    eval_episodes = np.arange(0, len(rewards), eval_interval)[:len(eval_rewards)]
    plt.plot(eval_episodes, eval_rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig('stream_q_training_results.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Stream Q agent with RTRL QuasiLSTM")
    parser.add_argument("--env_type", type=str, default="tmaze", choices=["tmaze", "pendulum"], 
                        help="Environment type to train on")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of LSTM")
    parser.add_argument("--embedding_size", type=int, default=32, help="Embedding size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--save_path", type=str, default=None, 
                        help="Path to save the trained model. If None, automatically generated based on corridor length.")
    parser.add_argument("--corridor_length", type=int, default=5, 
                        help="Length of the corridor in TMaze environment")
    parser.add_argument("--max_steps", type=int, default=100, 
                        help="Maximum steps per episode")
    parser.add_argument("--max_time_steps", type=int, default=None,
                        help="Maximum total time steps across all episodes")
    parser.add_argument("--disable_plot", action="store_true", 
                        help="Disable plotting for faster execution")
    parser.add_argument("--eval_interval", type=int, default=50,
                        help="How often to evaluate agent (higher = faster training)")
    args = parser.parse_args()
    
    # Train agent
    agent, rewards = train_stream_q(
        env_name=args.env_type,
        episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        batch_size=args.batch_size,
        save_path=args.save_path,
        corridor_length=args.corridor_length,
        max_steps_per_episode=args.max_steps,
        max_time_steps=args.max_time_steps,
        disable_plot=args.disable_plot,
        eval_interval=args.eval_interval
    )