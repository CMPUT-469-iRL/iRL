import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.nn.functional as F

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = uniform sampling)
        self.beta = beta    # Correction for importance sampling bias
        self.beta_increment = beta_increment
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities[self.size] = max_priority
            self.size += 1
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
            
        self.position = (self.position + 1) % self.capacity
            
    def sample(self, batch_size):
        if self.size < batch_size:
            idx = np.random.randint(0, self.size, size=self.size)
        else:
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities /= np.sum(probabilities)
            
            idx = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
            
        # Increase beta over time to reduce bias
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate importance sampling weights
        weights = np.zeros(batch_size, dtype=np.float32)
        for i, index in enumerate(idx):
            weights[i] = (self.size * probabilities[index]) ** (-self.beta)
        weights /= np.max(weights)
        
        samples = [self.buffer[i] for i in idx]
        return samples, idx, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return self.size

class DQNAgent:
    def __init__(self, state_size, action_size, model, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=50000)  # Larger memory with prioritization
        self.batch_size = 128  # Larger batch size for more stable learning
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.05  # Lower minimum exploration
        self.epsilon_decay = 0.998  # Even slower decay for more exploration
        self.learning_rate = learning_rate
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay
        
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
        self.error_clip = 1.0  # For prioritized replay

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            # Different exploration strategies based on state
            if state == 3:  # If at junction
                # At junction, bias exploration 50/50 between left and right
                return random.randint(0, 1)  # Choose only left or right
            elif state == 1:  # Left signal
                # Bias toward left at junction if left signal seen
                if random.random() < 0.8:
                    return 0 if random.random() < 0.7 else 2  # Left or forward
                return random.randrange(self.action_size)
            elif state == 2:  # Right signal
                # Bias toward right at junction if right signal seen
                if random.random() < 0.8:
                    return 1 if random.random() < 0.7 else 2  # Right or forward
                return random.randrange(self.action_size)
            else:
                # In corridor with no signal, heavily favor moving forward
                if random.random() < 0.9:  # 90% chance to move forward
                    return 2  # Forward action
                return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor([state]).unsqueeze(0)
            q_values, _ = self.model(state)
            return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        samples, indices, weights = self.memory.sample(self.batch_size)
        weights = torch.FloatTensor(weights)
        
        states = torch.LongTensor([data[0] for data in samples])
        actions = torch.LongTensor([data[1] for data in samples])
        rewards = torch.FloatTensor([data[2] for data in samples])
        next_states = torch.LongTensor([data[3] for data in samples])
        dones = torch.FloatTensor([data[4] for data in samples])

        # Ensure actions are within valid range
        actions = torch.clamp(actions, 0, self.action_size - 1)

        # Get current Q-values
        current_q_values, _ = self.model(states)
        current_q_actions = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Get next state Q-values using Double DQN
        with torch.no_grad():
            next_q_values, _ = self.target_model(next_states)
            best_actions = self.model(next_states)[0].max(1)[1]
            next_q_actions = next_q_values.gather(1, best_actions.unsqueeze(1)).squeeze()
            
            # Calculate target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_actions
        
        # Calculate TD errors for prioritized replay
        td_errors = torch.abs(current_q_actions - target_q_values).detach().numpy()
        
        # Clip errors for stability in prioritized replay
        clipped_errors = np.minimum(td_errors, self.error_clip)
        
        # Update priorities
        new_priorities = clipped_errors + 1e-6  # Add small constant to avoid zero priority
        self.memory.update_priorities(indices, new_priorities)
        
        # Calculate loss with importance sampling weights
        losses = F.smooth_l1_loss(current_q_actions, target_q_values, reduction='none')
        weighted_loss = (losses * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
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
    