import os
import argparse
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from tmaze_pen import TMazeEnv
from eLSTM_model.model import QuasiLSTMModel


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
        self.q_network = QuasiLSTMModel(
            emb_dim=embedding_size,
            hidden_size=hidden_size,
            in_vocab_size=input_size,
            out_vocab_size=output_size,
            no_embedding=no_embedding
        )
        
        # Target network for stable learning
        self.target_network = QuasiLSTMModel(
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
        # For QuasiLSTM, we don't need to maintain explicit hidden state
        # The state is maintained internally in the forward pass
        self.hidden_state = None
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.output_size)
        
        try:
            # Convert state to tensor and add sequence dimension
            if isinstance(state, int):
                state_tensor = torch.tensor([[state]], dtype=torch.long).to(self.device)
            else:
                state_tensor = torch.tensor([state], dtype=torch.long).unsqueeze(0).to(self.device)
                
            # Get Q-values from network
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                q_values = q_values.squeeze(0)  # Remove sequence dimension
                
            # Return action with highest Q-value
            return q_values.argmax().item()
        except Exception as e:
            print(f"Error in select_action: {e}")
            # Return a random action as fallback
            return random.randrange(self.output_size)
    
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
        """Perform a single training step using standard backpropagation."""
        if len(self.memory) < self.batch_size:
            return 0
            
        # Sample random batch from memory
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        
        # Convert to tensors and add sequence dimension
        states = torch.tensor(batch[0], dtype=torch.long).unsqueeze(0).to(self.device)  # [1, B, 1]
        actions = torch.tensor(batch[1], dtype=torch.long).view(1, -1, 1).to(self.device)  # [1, B, 1]
        rewards = torch.tensor(batch[2], dtype=torch.float32).view(1, -1, 1).to(self.device)  # [1, B, 1]
        next_states = torch.tensor(batch[3], dtype=torch.long).unsqueeze(0).to(self.device)  # [1, B, 1]
        dones = torch.tensor(batch[4], dtype=torch.float32).view(1, -1, 1).to(self.device)  # [1, B, 1]
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Compute Q-values for current states
        q_values = self.q_network(states)  # [1, B, num_actions]
        q_values = q_values.gather(-1, actions)  # [1, B, 1]
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)  # [1, B, num_actions]
            next_q_values = next_q_values.max(-1)[0].unsqueeze(-1)  # [1, B, 1]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and backpropagate
        loss = self.loss_fn(q_values, target_q_values)
        loss.backward()
        
        # Just use standard backpropagation - no RTRL
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
                  save_path=None,
                  eval_interval=20,
                  corridor_length=5,
                  max_steps_per_episode=100,
                  max_time_steps=None):  # Add max_time_steps parameter
    """Train a Stream Q agent using RTRL QuasiLSTM."""
    
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
        memory_size=10000,
        batch_size=batch_size,
        target_update_freq=target_update_freq
    )
    
    # Track metrics
    rewards_history = []
    losses_history = []
    evaluation_rewards = []
    
    # Add a counter for total time steps across all episodes
    total_time_steps = 0
    
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
            try:
                # Select and perform action
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
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
                
                # Train agent
                loss = agent.train_step()
                if loss > 0:
                    episode_loss += loss
                    step_count += 1
                
                # Update state and reward
                state = next_state
                episode_reward += reward
            except Exception as e:
                print(f"Error during training: {e}")
                break
            
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
                  f"Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}, "
                  f"Total Steps: {total_time_steps}")
                  
        # Evaluate agent periodically
        if (episode + 1) % eval_interval == 0:
            try:
                eval_reward = evaluate_agent(agent, env, episodes=10, max_steps_per_episode=max_steps_per_episode)
                evaluation_rewards.append(eval_reward)
                print(f"Evaluation at episode {episode+1}: Average Reward = {eval_reward:.2f}")
            except Exception as e:
                print(f"Error during evaluation: {e}")
                evaluation_rewards.append(0)
            
    # Save trained model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'rewards_history': rewards_history,
        'losses_history': losses_history,
        'eval_rewards': evaluation_rewards,
        'total_time_steps': total_time_steps,
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
    
    # Plot training metrics
    plot_training_results(rewards_history, losses_history, evaluation_rewards, eval_interval)
    
    return agent, rewards_history


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
            try:
                action = agent.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
                steps += 1
            except Exception as e:
                print(f"Error during evaluation: {e}")
                break
            
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
    parser.add_argument("--save_path", type=str, default=None, 
                        help="Path to save the trained model. If None, automatically generated based on corridor length.")
    parser.add_argument("--corridor_length", type=int, default=5, 
                        help="Length of the corridor in TMaze environment")
    parser.add_argument("--max_steps", type=int, default=100, 
                        help="Maximum steps per episode")
    parser.add_argument("--max_time_steps", type=int, default=None,
                        help="Maximum total time steps across all episodes")
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
        max_time_steps=args.max_time_steps
    )