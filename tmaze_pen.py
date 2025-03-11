# run using tmaze: python elstm_rl_eval.py --model_path save_models/model.pt --env_type tmaze --episodes 100
# run using pendulum: python elstm_rl_eval.py --model_path save_models/model.pt --env_type pendulum --episodes 100
# change directory 

import os
import time
import argparse
import logging
import random
import numpy as np
from datetime import datetime
import gym
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from eLSTM_model.model import QuasiLSTMModel


# T-maze environment
class TMazeEnv:
    def __init__(self, corridor_length=10, delay=0, randomize_goal=True):
        """
        T-maze environment:
        - Agent starts at the bottom of T
        - Signal at the start indicates which way to turn at junction (left/right)
        - Reward is given only if agent turns the correct way
        
        Args:
            corridor_length: Length of the corridor before the T-junction
            delay: Number of steps with no signal before showing the signal
            randomize_goal: Whether to randomize the goal location each episode
        """
        self.corridor_length = corridor_length
        self.delay = delay
        self.randomize_goal = randomize_goal
        
        # State space: 6 features
        self.observation_space = 6
        
        # Action space: 4 discrete actions (up, down, left, right)
        self.action_space = 4
        
        self.reset()
        
    def reset(self):
        """Reset the environment for a new episode."""
        # Initialize the maze layout
        self.maze = np.zeros((self.corridor_length + 1, 3), dtype=np.int32)
        
        # Add T-junction
        self.maze[self.corridor_length, 0] = 3
        self.maze[self.corridor_length, 1] = 3
        self.maze[self.corridor_length, 2] = 3
        
        # Randomize goal location (left or right)
        if self.randomize_goal:
            self.goal_pos = random.choice([0, 2])
        else:
            self.goal_pos = 0  # Always left for debugging
        
        self.maze[self.corridor_length, self.goal_pos] = 4
        
        # Set agent at the start
        self.agent_pos = [0, 1]
        
        # Determine signal based on goal
        self.signal = 1 if self.goal_pos == 0 else 2  # 1=left, 2=right
        
        # Current step
        self.step_count = 0
        self.done = False
        
        # Initial observation
        if self.delay == 0:
            # Show signal immediately
            obs = self._get_observation(show_signal=True)
        else:
            obs = self._get_observation(show_signal=False)
        
        return obs
    
    def _get_observation(self, show_signal=True):
        """Create observation vector from current state."""
        # Create flat encoding of the environment state
        obs = np.zeros(self.observation_space, dtype=np.float32)
        
        # Agent's position in corridor (normalized)
        obs[0] = self.agent_pos[0] / self.corridor_length
        
        # Signal (only at beginning)
        if self.step_count < 1 and show_signal:
            obs[self.signal] = 1.0
        
        # Is agent at junction?
        if self.agent_pos[0] == self.corridor_length:
            obs[3] = 1.0
        
        # Is agent at goal?
        if (self.agent_pos[0] == self.corridor_length and 
            self.agent_pos[1] == self.goal_pos):
            obs[4] = 1.0
        
        # Agent position marker
        obs[5] = 1.0
        
        return obs
    
    def step(self, action):
        """Take a step in the environment."""
        self.step_count += 1
        reward = 0
        
        # Move agent according to action
        if action == 0:  # up
            if self.agent_pos[0] < self.corridor_length:
                self.agent_pos[0] += 1
        elif action == 1:  # down
            if self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
        elif action == 2:  # left
            if self.agent_pos[0] == self.corridor_length and self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
        elif action == 3:  # right
            if self.agent_pos[0] == self.corridor_length and self.agent_pos[1] < 2:
                self.agent_pos[1] += 1
        
        # Check if agent reached the goal
        if (self.agent_pos[0] == self.corridor_length and 
            self.agent_pos[1] == self.goal_pos):
            reward = 1.0
            self.done = True
        
        # Check if agent made wrong turn at junction
        elif (self.agent_pos[0] == self.corridor_length and 
              self.agent_pos[1] != 1 and 
              self.agent_pos[1] != self.goal_pos):
            reward = -1.0
            self.done = True
        
        # Timeout
        if self.step_count > 2 * self.corridor_length + 5:
            self.done = True
        
        # Get observation
        show_signal = self.step_count <= 1 and self.delay <= self.step_count
        obs = self._get_observation(show_signal=show_signal)
        
        info = {
            'goal_pos': self.goal_pos,
            'agent_pos': self.agent_pos,
            'step_count': self.step_count
        }
        
        return obs, reward, self.done, info


# Pendulum Wrapper for discrete actions
class DiscretePendulumWrapper(gym.Wrapper):
    def __init__(self, env, n_actions=5):
        super().__init__(env)
        self.n_actions = n_actions
        self.action_space = n_actions  # Number of discrete actions
        
        # Map discrete actions to continuous values
        self.actions = np.linspace(-2.0, 2.0, n_actions)
    
    def step(self, action):
        # Convert discrete action to continuous action
        continuous_action = np.array([self.actions[action]])
        return self.env.step(continuous_action)


# eLSTM-based Policy Network for discrete action spaces
class eLSTMDiscretePolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_size=32):
        super(eLSTMDiscretePolicy, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Input embedding
        self.embedding = nn.Linear(input_size, embedding_size)
        
        # Use eLSTM model as the backbone
        self.elstm_model = QuasiLSTMModel(
            emb_dim=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            in_vocab_size=100,  # placeholder, not directly used
            out_vocab_size=100, # placeholder, not directly used
            dropout=0.0,
            no_embedding=True   # We'll use our own embedding
        )
        
        # Policy head
        self.policy_head = nn.Linear(hidden_size, output_size)
        
        # Value head for actor-critic
        self.value_head = nn.Linear(hidden_size, 1)
    
    def get_embedding(self, x):
        # Preprocess continuous input
        return self.embedding(x)
    
    def forward(self, x, state=None):
        # Initialize state if not provided
        batch_size = x.size(0)
        if state is None:
            state = self.elstm_model.get_init_states(batch_size, x.device)
        
        # Create embedding for input
        embed = self.get_embedding(x)
        
        # Process through eLSTM
        # For evaluation, we need to bypass the embedding in elstm_model
        # and feed our embedding directly to the lstm
        # We need to access internal components of the model
        h, c = state
        
        # Forward pass through LSTM
        h_out, c_out = self.elstm_model.model.lstm_cell(embed, (h, c))
        new_state = (h_out, c_out)
        
        # Get policy logits and value
        policy_logits = self.policy_head(h_out)
        value = self.value_head(h_out)
        
        return policy_logits, value, new_state
    
    def act(self, obs, state=None, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value, new_state = self.forward(obs_tensor, state)
            
            if deterministic:
                action = torch.argmax(policy_logits, dim=-1)
            else:
                probs = F.softmax(policy_logits, dim=-1)
                action_dist = Categorical(probs)
                action = action_dist.sample()
            
        return action.item(), new_state, value.item()


# eLSTM-based Policy Network for continuous action spaces
class eLSTMContinuousPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_size=32):
        super(eLSTMContinuousPolicy, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Input embedding
        self.embedding = nn.Linear(input_size, embedding_size)
        
        # Use eLSTM model as the backbone
        self.elstm_model = QuasiLSTMModel(
            emb_dim=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            in_vocab_size=100,  # placeholder, not directly used
            out_vocab_size=100, # placeholder, not directly used
            dropout=0.0,
            no_embedding=True   # We'll use our own embedding
        )
        
        # Mean and std for continuous action distribution
        self.action_mean = nn.Linear(hidden_size, output_size)
        self.action_log_std = nn.Parameter(torch.zeros(1, output_size))
        
        # Value head for actor-critic
        self.value_head = nn.Linear(hidden_size, 1)
    
    def get_embedding(self, x):
        # Preprocess continuous input
        return self.embedding(x)
    
    def forward(self, x, state=None):
        # Initialize state if not provided
        batch_size = x.size(0)
        if state is None:
            state = self.elstm_model.get_init_states(batch_size, x.device)
        
        # Create embedding for input
        embed = self.get_embedding(x)
        
        # Process through eLSTM internals
        h, c = state
        
        # Forward pass through LSTM
        h_out, c_out = self.elstm_model.model.lstm_cell(embed, (h, c))
        new_state = (h_out, c_out)
        
        # Get action distribution parameters and value
        action_mean = self.action_mean(h_out)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        value = self.value_head(h_out)
        
        return action_mean, action_std, value, new_state
    
    def act(self, obs, state=None, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action_mean, action_std, value, new_state = self.forward(obs_tensor, state)
            
            if deterministic:
                action = action_mean
            else:
                normal = Normal(action_mean, action_std)
                action = normal.sample()
            
            # Clip action to valid range (-2, 2) for pendulum
            action = torch.clamp(action, -2.0, 2.0)
            
        return action.squeeze().numpy(), new_state, value.item()


# Adapter class to apply the pretrained eLSTM to RL tasks
class eLSTMAdapter:
    def __init__(self, model_path, env_type, hidden_size=2048, device='cuda'):
        self.model_path = model_path
        self.env_type = env_type
        self.hidden_size = hidden_size
        self.device = device
        
        # Load the pretrained model
        self.pretrained_model = self.load_pretrained_model()
        
        # Create appropriate policy network based on environment type
        if env_type == 'tmaze':
            self.env = TMazeEnv()
            self.policy = eLSTMDiscretePolicy(
                input_size=self.env.observation_space,
                hidden_size=hidden_size,
                output_size=self.env.action_space
            )
        elif env_type == 'pendulum':
            self.env = gym.make('Pendulum-v1')
            
            # Use discretized pendulum if specified
            use_discrete = True  # Can be made configurable
            if use_discrete:
                self.env = DiscretePendulumWrapper(self.env, n_actions=5)
                self.policy = eLSTMDiscretePolicy(
                    input_size=self.env.observation_space.shape[0],
                    hidden_size=hidden_size,
                    output_size=self.env.action_space
                )
            else:
                self.policy = eLSTMContinuousPolicy(
                    input_size=self.env.observation_space.shape[0],
                    hidden_size=hidden_size,
                    output_size=self.env.action_space.shape[0]
                )
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
        
        # Initialize the policy
        self.transfer_weights()
        self.policy = self.policy.to(device)
    
    def load_pretrained_model(self):
        """Load the pretrained eLSTM model."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model with the same architecture
        model = QuasiLSTMModel(
            emb_dim=128,  # Default from original script
            hidden_size=self.hidden_size,
            num_layers=1, 
            in_vocab_size=100,  # Placeholder, will be overridden
            out_vocab_size=100, # Placeholder, will be overridden
            dropout=0.0,
            no_embedding=True
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def transfer_weights(self):
        """Transfer weights from pretrained eLSTM to policy network."""
        # Transfer LSTM weights
        self.policy.elstm_model.model.lstm_cell.weight_ih = \
            self.pretrained_model.model.lstm_cell.weight_ih
        self.policy.elstm_model.model.lstm_cell.weight_hh = \
            self.pretrained_model.model.lstm_cell.weight_hh
        self.policy.elstm_model.model.lstm_cell.bias_ih = \
            self.pretrained_model.model.lstm_cell.bias_ih
        self.policy.elstm_model.model.lstm_cell.bias_hh = \
            self.pretrained_model.model.lstm_cell.bias_hh
        
        # Freeze LSTM weights
        for param in self.policy.elstm_model.parameters():
            param.requires_grad = False
    
    def evaluate(self, num_episodes=100):
        """Evaluate the policy on the environment."""
        self.policy.eval()
        total_rewards = []
        
        for ep in range(num_episodes):
            obs = self.env.reset()
            done = False
            ep_reward = 0
            state = self.policy.elstm_model.get_init_states(1, self.device)
            
            while not done:
                action, state, _ = self.policy.act(
                    obs, state, deterministic=True
                )
                obs, reward, done, _ = self.env.step(action)
                ep_reward += reward
            
            total_rewards.append(ep_reward)
            
            if (ep + 1) % 10 == 0:
                print(f"Episode {ep+1}/{num_episodes}, Reward: {ep_reward:.2f}")
        
        avg_reward = sum(total_rewards) / num_episodes
        print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
        
        return avg_reward, total_rewards


def evaluate_tmaze_memory(adapter, corridor_lengths=[5, 10, 20, 50], delays=[0, 5, 10, 20], 
                          episodes_per_config=50):
    """Evaluate how well the model can handle increasing memory demands in TMaze."""
    results = {}
    
    for length in corridor_lengths:
        for delay in delays:
            print(f"\nEvaluating T-maze with corridor length {length} and delay {delay}...")
            # Create new environment with specified parameters
            env = TMazeEnv(corridor_length=length, delay=delay)
            adapter.env = env
            
            # Evaluate
            successes = 0
            for ep in range(episodes_per_config):
                obs = env.reset()
                done = False
                ep_reward = 0
                state = adapter.policy.elstm_model.get_init_states(1, adapter.device)
                
                while not done:
                    action, state, _ = adapter.policy.act(
                        obs, state, deterministic=True
                    )
                    obs, reward, done, _ = env.step(action)
                    ep_reward += reward
                
                if ep_reward > 0:  # Successful navigation
                    successes += 1
            
            success_rate = 100 * successes / episodes_per_config
            print(f"Success rate: {success_rate:.2f}%")
            
            # Store results
            key = f"length_{length}_delay_{delay}"
            results[key] = success_rate
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate pretrained eLSTM model on RL tasks')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained eLSTM model')
    parser.add_argument('--env_type', type=str, choices=['tmaze', 'pendulum'], required=True,
                        help='Environment type to evaluate on')
    parser.add_argument('--hidden_size', type=int, default=2048,
                        help='Hidden size of the eLSTM model')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(args.output_dir, f"{args.env_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        filename=f"{output_dir}/log.txt",
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    logger = logging.getLogger()
    logger.info(f"Arguments: {vars(args)}")
    
    # Create adapter
    adapter = eLSTMAdapter(
        model_path=args.model_path,
        env_type=args.env_type,
        hidden_size=args.hidden_size,
        device=args.device
    )
    
    # Run evaluation
    logger.info(f"Starting evaluation on {args.env_type} for {args.episodes} episodes...")
    
    if args.env_type == 'tmaze':
        # For T-maze, also evaluate memory capacity
        avg_reward, rewards = adapter.evaluate(num_episodes=args.episodes)
        logger.info(f"Average reward: {avg_reward:.2f}")
        
        # Save rewards
        np.savetxt(f"{output_dir}/rewards.txt", rewards)
        
        # Test memory capacity
        memory_results = evaluate_tmaze_memory(adapter)
        
        # Save memory test results
        with open(f"{output_dir}/memory_results.json", 'w') as f:
            json.dump(memory_results, f)
        
        # Log memory results
        for key, value in memory_results.items():
            logger.info(f"{key}: {value:.2f}%")
    else:
        # For other environments, just run standard evaluation
        avg_reward, rewards = adapter.evaluate(num_episodes=args.episodes)
        logger.info(f"Average reward: {avg_reward:.2f}")
        
        # Save rewards
        np.savetxt(f"{output_dir}/rewards.txt", rewards)
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")
    print(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
