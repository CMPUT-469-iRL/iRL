import os
import time
import argparse
import logging
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
# import gym
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmaze_pen import TMazeEnv
# from eLSTM_model.model import RTRLQuasiLSTMModel
from rtu_complex import RTRLRTU


class StreamQAgent:
    """
    Stream Q agent using RTRL QuasiLSTM for efficient learning with recurrent connections.
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
                batch_size=32,
                no_embedding=False,
                target_update_freq=10):
        
        # Main Q-network
        self.q_network = RTRLRTU(
            emb_dim=embedding_size,
            hidden_size=hidden_size,
            in_vocab_size=input_size,
            out_vocab_size=output_size,
            no_embedding=no_embedding
        )
        
        # Target network for stable learning
        self.target_network = RTRLRTU(
            emb_dim=embedding_size,
            hidden_size=hidden_size,
            in_vocab_size=input_size,
            out_vocab_size=output_size,
            no_embedding=no_embedding
        )
        
        # Copy parameters from Q-network to target network
        self.update_target_network()
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move networks to device
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
        # Current hidden state
        self.hidden_state = None
        
    def reset_hidden_state(self, batch_size=1):
        """Reset the hidden state at the beginning of episodes."""
        self.hidden_state = self.q_network.get_init_states(batch_size, self.device)
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.output_size)
        
        # Convert state to tensor
        if isinstance(state, int):
            state_tensor = torch.tensor([state], dtype=torch.long).to(self.device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.long).to(self.device)
            
        # Get Q-values from network
        with torch.no_grad():
            q_values, _, self.hidden_state = self.q_network(state_tensor, self.hidden_state)
            
        # Return action with highest Q-value
        return q_values.argmax().item()
    
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
        """Perform a single training step using RTRL."""
        if len(self.memory) < self.batch_size:
            return 0
            
        # Sample random batch from memory
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        
        # Convert to tensors
        states = torch.tensor(batch[0], dtype=torch.long).to(self.device)
        actions = torch.tensor(batch[1], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(batch[2], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(batch[3], dtype=torch.long).to(self.device)
        dones = torch.tensor(batch[4], dtype=torch.float32).view(-1, 1).to(self.device)
        
        # Reset gradients
        self.q_network.reset_grad()
        self.q_network.rtrl_reset_grad()
        
        # Initialize hidden states for batch processing
        hidden_state = self.q_network.get_init_states(self.batch_size, self.device)
        target_hidden_state = self.target_network.get_init_states(self.batch_size, self.device)
        
        # Compute Q-values for current states
        q_values, cell_output, rtrl_state = self.q_network(states, hidden_state)
        q_values = q_values.gather(1, actions)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values, _, _ = self.target_network(next_states, target_hidden_state)
            next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and backpropagate
        loss = self.loss_fn(q_values, target_q_values)
        loss.backward()
        
        # Apply RTRL gradient updates
        if cell_output.grad is not None:
            self.q_network.compute_gradient_rtrl(cell_output.grad, rtrl_state)
        
        # Update parameters
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
                  batch_size=32,
                  target_update_freq=10,
                  save_path="save_models/stream_q_model.pt",
                  eval_interval=20):
    """Train a Stream Q agent using RTRL QuasiLSTM."""
    
    # Set up environment
    if env_name == "tmaze":
        env = TMazeEnv(corridor_length=10)
        input_size = 4  # TMaze has 4 observation states
        output_size = 3  # TMaze has 3 actions
    # elif env_name == "pendulum":
    #     env = gym.make("Pendulum-v1")
    #     input_size = 1000  # Discretized observations (10^3)
    #     output_size = 11   # Discretized actions for pendulum
        # We'll use a discretized action space for pendulum
    else:
        raise ValueError(f"Unsupported environment type: {env_name}")
        
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
        memory_size=10000,
        batch_size=batch_size,
        target_update_freq=target_update_freq
    )
    
    # Track metrics
    rewards_history = []
    losses_history = []
    evaluation_rewards = []
    
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
        while not done:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            if loss > 0:
                episode_loss += loss
                step_count += 1
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
        # Update exploration rate
        agent.update_epsilon()
        
        # Update target network periodically
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
            
        # Track metrics
        rewards_history.append(episode_reward)
        if step_count > 0:
            losses_history.append(episode_loss / step_count)
        else:
            losses_history.append(0)
            
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}, "
                  f"Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
                  
        # Evaluate agent periodically
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(agent, env, episodes=10)
            evaluation_rewards.append(eval_reward)
            print(f"Evaluation at episode {episode+1}: Average Reward = {eval_reward:.2f}")
            
    # Save trained model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'rewards_history': rewards_history,
        'losses_history': losses_history,
        'eval_rewards': evaluation_rewards,
        'hyperparams': {
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'lr': lr,
            'gamma': gamma,
            'embedding_size': embedding_size,
            'hidden_size': hidden_size,
            'batch_size': batch_size
        }
    }, save_path)
    
    print(f"Model saved to {save_path}")
    
    # Plot training metrics
    plot_training_results(rewards_history, losses_history, evaluation_rewards, eval_interval)
    
    return agent, rewards_history


def evaluate_agent(agent, env, episodes=10):
    """Evaluate a trained agent without exploration."""
    total_rewards = []
    
    for _ in range(episodes):
        state = env.reset()
        agent.reset_hidden_state()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
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
    eval_episodes = np.arange(eval_interval, len(rewards) + 1, eval_interval)
    plt.plot(eval_episodes, eval_rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig('stream_q_training_results.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Stream Q agent with RTRL QuasiLSTM")
    parser.add_argument("--env_type", type=str, default="tmaze", choices=["tmaze", "pendulum"], 
                        help="Environment type to train on")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of LSTM")
    parser.add_argument("--embedding_size", type=int, default=32, help="Embedding size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--save_path", type=str, default="save_models/stream_q_model.pt", 
                        help="Path to save the trained model")
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
        save_path=args.save_path
    )