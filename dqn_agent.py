import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.nn.functional as F

class EpisodeBuffer:
    """Buffer to store sequences of transitions within an episode"""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer = []
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def get_sequence(self):
        return self.buffer
    
    def clear(self):
        self.buffer = []

class PrioritizedReplayBuffer:
    def __init__(self, capacity, sequence_length=8, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def push(self, sequence):
        """Add a sequence of transitions"""
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(sequence)
            self.priorities[self.size] = max_priority
            self.size += 1
        else:
            self.buffer[self.position] = sequence
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
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        weights = np.zeros(batch_size, dtype=np.float32)
        for i, index in enumerate(idx):
            weights[i] = (self.size * probabilities[index]) ** (-self.beta)
        weights /= np.max(weights)
        
        sequences = [self.buffer[i] for i in idx]
        return sequences, idx, weights

class DQNAgent:
    def __init__(self, state_size, action_size, model, learning_rate=0.001, batch_size=32, 
                 memory_capacity=50000, gamma=0.99, epsilon_decay=0.998, sequence_length=8):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.memory = PrioritizedReplayBuffer(capacity=memory_capacity, sequence_length=sequence_length)
        self.episode_buffer = EpisodeBuffer()
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Create target model
        self.target_model = type(model)(
            emb_dim=0,
            hidden_size=model.hidden_size,
            in_vocab_size=model.in_vocab_size,
            out_vocab_size=model.out_vocab_size,
            no_embedding=True
        )
        self.target_model.load_state_dict(model.state_dict())
        
        # Initialize hidden states
        self.hidden_state = None
        self.target_hidden_state = None
        self.reset_hidden_states()
    
    def reset_hidden_states(self):
        """Reset hidden states for both main and target networks"""
        self.hidden_state = None
        self.target_hidden_state = None
    
    def store_transition(self, state, action, reward, next_state, done):
        # Add to episode buffer
        self.episode_buffer.add(state, action, reward, next_state, done)
        
        if done:
            # Get sequence from episode buffer
            sequence = self.episode_buffer.get_sequence()
            if len(sequence) >= self.sequence_length:
                # Add full sequence to replay buffer
                self.memory.push(sequence)
            self.episode_buffer.clear()
            self.reset_hidden_states()

    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            # Exploration strategies remain the same
            if state == 3:
                return random.randint(0, 1)
            elif state == 1:
                if random.random() < 0.8:
                    return 0 if random.random() < 0.7 else 2
                return random.randrange(self.action_size)
            elif state == 2:
                if random.random() < 0.8:
                    return 1 if random.random() < 0.7 else 2
                return random.randrange(self.action_size)
            else:
                if random.random() < 0.9:
                    return 2
                return random.randrange(self.action_size)
        
        with torch.no_grad():
            # Process state as part of sequence
            state_seq = torch.FloatTensor([state]).unsqueeze(0).unsqueeze(0)  # [1, 1, state_size]
            q_values, self.hidden_state = self.model(state_seq, self.hidden_state)
            return q_values.squeeze(0).argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        sequences, indices, weights = self.memory.sample(self.batch_size)
        weights = torch.FloatTensor(weights)
        
        # Process sequences
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        for seq in sequences:
            # Take last sequence_length transitions
            seq = seq[-self.sequence_length:]
            states, actions, rewards, next_states, dones = zip(*seq)
            
            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_next_states.append(next_states)
            batch_dones.append(dones)
        
        # Convert to tensors with sequence dimension
        states = torch.LongTensor(batch_states)  # [batch_size, seq_len]
        actions = torch.LongTensor(batch_actions)
        rewards = torch.FloatTensor(batch_rewards)
        next_states = torch.LongTensor(batch_next_states)
        dones = torch.FloatTensor(batch_dones)
        
        # Process sequences through networks
        current_q_values, _ = self.model(states)  # [batch_size, seq_len, action_size]
        current_q_actions = current_q_values.gather(2, actions.unsqueeze(2)).squeeze(2)
        
        with torch.no_grad():
            next_q_values, _ = self.target_model(next_states)
            best_actions = self.model(next_states)[0].max(2)[1]
            next_q_actions = next_q_values.gather(2, best_actions.unsqueeze(2)).squeeze(2)
            
            # Calculate target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_actions
        
        # Calculate TD errors and update priorities
        td_errors = torch.abs(current_q_actions - target_q_values).mean(dim=1).detach().numpy()
        clipped_errors = np.minimum(td_errors, self.error_clip)
        new_priorities = clipped_errors + 1e-6
        self.memory.update_priorities(indices, new_priorities)
        
        # Calculate loss with importance sampling
        losses = F.smooth_l1_loss(current_q_actions, target_q_values, reduction='none').mean(dim=1)
        weighted_loss = (losses * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_target_counter += 1
        if self.update_target_counter % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.reset_hidden_states()

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'hidden_state': self.hidden_state,
            'target_hidden_state': self.target_hidden_state
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.hidden_state = checkpoint['hidden_state']
        self.target_hidden_state = checkpoint['target_hidden_state'] 
    