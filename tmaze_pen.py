# run using tmaze: python tmaze_pen.py --model_path save_models/model.pt --env_type tmaze --episodes 100
# run using pendulum: python tmaze_pen.py --model_path save_models/model.pt --env_type pendulum --episodes 100

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

# Change the import to use RTRLQuasiLSTMModel
from eLSTM_model.model import RTRLQuasiLSTMModel


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
        self.goal_location = None
        self.pos = None
        self.signal = None
        self.t = None
        self.done = None
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # Left, Right, Forward
        self.observation_space = gym.spaces.Discrete(4)  # No signal, Left signal, Right signal, Junction
        
    def reset(self):
        self.goal_location = random.choice(["left", "right"]) if self.randomize_goal else "right"
        self.pos = [0, 0]  # Start at bottom of T
        self.signal = self.goal_location if self.delay == 0 else "none"
        self.t = 0
        self.done = False
        obs = self._get_observation()
        return obs
    
    def step(self, action):
        self.t += 1
        
        # Update signal after delay
        if self.t == self.delay + 1:
            self.signal = self.goal_location
        
        reward = 0
        
        # Handle movement based on action
        if action == 0:  # Left
            if self.pos[1] == self.corridor_length:  # At junction
                self.pos[0] -= 1
                self.done = True
                reward = 1.0 if self.goal_location == "left" else -1.0
            else:
                reward = -0.1  # Penalty for invalid move
        elif action == 1:  # Right
            if self.pos[1] == self.corridor_length:  # At junction
                self.pos[0] += 1
                self.done = True
                reward = 1.0 if self.goal_location == "right" else -1.0
            else:
                reward = -0.1  # Penalty for invalid move
        elif action == 2:  # Forward
            if self.pos[1] < self.corridor_length:
                self.pos[1] += 1
            else:
                reward = -0.1  # Penalty for invalid move
        
        obs = self._get_observation()
        return obs, reward, self.done, {}
    
    def _get_observation(self):
        if self.pos[1] == self.corridor_length:
            return 3  # Junction
        elif self.signal == "left":
            return 1
        elif self.signal == "right":
            return 2
        else:
            return 0  # No signal


# eLSTM-based policy for discrete action spaces (used for T-maze)
class eLSTMDiscretePolicy:
    def __init__(self, input_size, embedding_size, hidden_size, output_size, no_embedding=False):
        # Initialize eLSTM model with RTRLQuasiLSTMModel instead of QuasiLSTMModel
        self.elstm_model = RTRLQuasiLSTMModel(
            emb_dim=embedding_size,
            hidden_size=hidden_size,
            in_vocab_size=input_size,
            out_vocab_size=output_size,
            no_embedding=no_embedding
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state = None
        
    def reset_state(self, batch_size=1, device="cpu"):
        # Use the new get_init_states method
        self.state = self.elstm_model.get_init_states(batch_size, device)
        
    def act(self, observation):
        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            obs_tensor = torch.from_numpy(observation).long()
        elif isinstance(observation, int):
            obs_tensor = torch.tensor([observation]).long()
        else:
            obs_tensor = observation.long()
            
        # Initialize state if needed
        if self.state is None:
            self.reset_state(batch_size=obs_tensor.size(0), device=obs_tensor.device)
            
        # Forward pass through eLSTM
        logits, _, self.state = self.elstm_model(obs_tensor, self.state)
        
        # Sample action from categorical distribution
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action).item()


# eLSTM-based policy for continuous action spaces (used for Pendulum)
class eLSTMContinuousPolicy:
    def __init__(self, input_size, embedding_size, hidden_size, output_size, no_embedding=False):
        # Initialize eLSTM model with RTRLQuasiLSTMModel
        self.elstm_model = RTRLQuasiLSTMModel(
            emb_dim=embedding_size,
            hidden_size=hidden_size,
            in_vocab_size=input_size,
            out_vocab_size=output_size * 2,  # Output mean and log_std
            no_embedding=no_embedding
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state = None
        
    def reset_state(self, batch_size=1, device="cpu"):
        # Use the new get_init_states method
        self.state = self.elstm_model.get_init_states(batch_size, device)
        
    def act(self, observation):
        # Preprocess observation
        if isinstance(observation, np.ndarray):
            # Discretize continuous observations for eLSTM input
            obs_discrete = self._discretize_observation(observation)
            obs_tensor = torch.from_numpy(obs_discrete).long()
        else:
            obs_tensor = observation.long()
            
        # Initialize state if needed
        if self.state is None:
            self.reset_state(batch_size=obs_tensor.size(0), device=obs_tensor.device)
            
        # Forward pass through eLSTM
        outputs, _, self.state = self.elstm_model(obs_tensor, self.state)
        
        # Split outputs into means and log_stds
        means = outputs[:, :self.output_size]
        log_stds = outputs[:, self.output_size:]
        log_stds = torch.clamp(log_stds, -20, 2)  # Clamp for stability
        
        # Create normal distribution
        dist = Normal(means, log_stds.exp())
        
        # Sample action and apply tanh to bound it
        raw_action = dist.sample()
        action = torch.tanh(raw_action)
        
        # Compute log probability with adjustment for tanh
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1)
        
        return action.squeeze().cpu().numpy(), log_prob.item()
    
    def _discretize_observation(self, obs, bins=10):
        # Simple discretization for continuous observations
        # Maps each continuous value to discrete indices
        obs_min = np.array([-1.0, -8.0, -0.5])  # Example ranges for pendulum
        obs_max = np.array([1.0, 8.0, 0.5])
        
        # Scale to [0, bins-1] and convert to integers
        scaled = (obs - obs_min) / (obs_max - obs_min) * (bins - 1)
        discrete = np.clip(scaled, 0, bins - 1).astype(int)
        
        # Encode as a single integer (for embedding lookup)
        encoded = discrete[0] * bins**2 + discrete[1] * bins + discrete[2]
        return np.array([encoded])


# Adapter for loading and running eLSTM models
class eLSTMAdapter:
    def __init__(self, model_path, env_type="tmaze"):
        self.model_path = model_path
        self.env_type = env_type
        self.model = None
        self.env = None
        
        if env_type == "tmaze":
            self.env = TMazeEnv(corridor_length=10)
            self.model = self._create_discrete_policy(
                input_size=4,  # TMaze has 4 observation states
                output_size=3   # TMaze has 3 actions
            )
        elif env_type == "pendulum":
            self.env = gym.make("Pendulum-v1")
            # For pendulum, discretize the continuous space
            self.model = self._create_continuous_policy(
                input_size=1000,  # Discretized observations (10^3)
                output_size=1     # Pendulum has 1 action dimension
            )
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
            
        self.load_pretrained_model()
    
    def _create_discrete_policy(self, input_size, output_size):
        return eLSTMDiscretePolicy(
            input_size=input_size,
            embedding_size=32,
            hidden_size=64,
            output_size=output_size
        )
    
    def _create_continuous_policy(self, input_size, output_size):
        return eLSTMContinuousPolicy(
            input_size=input_size,
            embedding_size=32,
            hidden_size=64,
            output_size=output_size,
            no_embedding=False
        )
    
    def load_pretrained_model(self):
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        # Update this to handle the RTRLQuasiLSTMModel structure
        self.model.elstm_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model from {self.model_path}")
    
    def run_episode(self, render=False):
        obs = self.env.reset()
        self.model.reset_state()
        done = False
        total_reward = 0
        
        while not done:
            if render and hasattr(self.env, 'render'):
                self.env.render()
                
            action, _ = self.model.act(obs)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            
        return total_reward


# Evaluation function
def evaluate_model(model_path, env_type="tmaze", episodes=100, render=False):
    adapter = eLSTMAdapter(model_path, env_type)
    rewards = []
    
    for i in range(episodes):
        reward = adapter.run_episode(render=render)
        rewards.append(reward)
        if (i + 1) % 10 == 0:
            print(f"Episode {i+1}/{episodes}, Reward: {reward:.2f}")
    
    print(f"Average reward over {episodes} episodes: {np.mean(rewards):.2f}")
    return rewards


# Main function for command-line usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate eLSTM models on RL tasks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--env_type", type=str, default="tmaze", choices=["tmaze", "pendulum"], 
                        help="Environment type")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.env_type, args.episodes, args.render)