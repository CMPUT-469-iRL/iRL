import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size, model, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.1  # Increased minimum exploration
        self.epsilon_decay = 0.995  # Slower decay
        self.learning_rate = learning_rate
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create target model with same parameters as main model
        self.target_model = type(model)(
            emb_dim=0,  # No embedding needed for this task
            hidden_size=model.hidden_size,
            in_vocab_size=model.in_vocab_size,
            out_vocab_size=model.out_vocab_size,
            no_embedding=True
        )
        self.target_model.load_state_dict(model.state_dict())
        self.update_target_counter = 0
        self.target_update_frequency = 5  # Update target network more frequently

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            # During exploration, prefer forward action
            if random.random() < 0.7:  # 70% chance to choose forward
                return 2  # Forward action
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values, _ = self.model(state)
            return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([data[0] for data in minibatch])
        actions = torch.LongTensor([data[1] for data in minibatch])
        rewards = torch.FloatTensor([data[2] for data in minibatch])
        next_states = torch.FloatTensor([data[3] for data in minibatch])
        dones = torch.FloatTensor([data[4] for data in minibatch])

        # Convert states to LongTensor for one-hot encoding
        states = states.long()
        next_states = next_states.long()

        # Ensure actions are within valid range
        actions = torch.clamp(actions, 0, self.action_size - 1)

        # Get current Q-values
        current_q_values, _ = self.model(states)
        next_q_values, _ = self.target_model(next_states)

        # Create target Q-values
        target_q_values = current_q_values.clone()
        
        # Update Q-values for each experience in the batch
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                # Double DQN: use current network to select action, target network to evaluate
                next_state_q_values, _ = self.model(next_states[i:i+1])
                next_action = next_state_q_values.argmax().item()
                # Ensure next_action is within valid range
                next_action = min(next_action, self.action_size - 1)
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * next_q_values[i][next_action]

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_target_counter += 1
        if self.update_target_counter % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon'] 
    