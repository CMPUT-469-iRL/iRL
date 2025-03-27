import torch
import numpy as np
from tmaze_pen import TMazeEnv
from eLSTM_model.model import DQNModel
from dqn_agent import DQNAgent
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt

def train_dqn(env, agent, num_episodes, eval_interval=100, save_dir='dqn_checkpoints'):
    # Create save directory with timestamp
    print("WE ARE TRAINING THE DQN", agent, num_episodes, eval_interval)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(save_dir, f'tmaze_dqn_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    MIN_REPLAY_SIZE = 1000
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

    # Training metrics
    episode_rewards = []
    success_rates = []
    eval_rewards = []
    
    # Set a maximum step count per episode to avoid infinite loops
    max_steps_per_episode = 1000
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1
            
            if len(agent.memory) >= agent.batch_size:
                agent.train()
                
        episode_rewards.append(episode_reward)
        print(f"Episode {episode} reward: {episode_reward}")
        
        # Evaluate periodically
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_dqn(env, agent, num_episodes=10)
            eval_rewards.append(eval_reward)
            # Here success rate is computed as the fraction of evaluation episodes with positive reward.
            success_rate = sum(1 for r in eval_rewards[-10:] if r > 0) / 10
            success_rates.append(success_rate)
            
            print(f'Episode {episode + 1}/{num_episodes}')
            print(f'Average Reward (last {eval_interval} episodes): {np.mean(episode_rewards[-eval_interval:]):.2f}')
            print(f'Success Rate: {success_rate:.2f}')
            print(f'Eval Reward: {eval_reward:.2f}')
            
            # Save checkpoint
            agent.save_checkpoint(os.path.join(save_dir, f'checkpoint_{episode+1}.pt'))
    
    print("Training completed. Returning metrics.")
    return episode_rewards, success_rates, eval_rewards

def evaluate_dqn(env, agent, num_episodes=10):
    eval_rewards = []
    max_steps_per_episode = 1000
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--corridor_length', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--save_dir', type=str, default='dqn_checkpoints')
    args = parser.parse_args()
    
    # Initialize environment
    env = TMazeEnv(corridor_length=args.corridor_length)
    
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
    episode_rewards, success_rates, eval_rewards = train_dqn(
        env, agent, args.episodes, args.eval_interval, args.save_dir
    )
    
    # Final evaluation
    final_eval_reward = evaluate_dqn(env, agent, num_episodes=100)
    print(f'\nFinal Evaluation Reward: {final_eval_reward:.2f}')
    print(f'Final Success Rate: {sum(1 for r in eval_rewards[-10:] if r > 0) / 10:.2f}')

if __name__ == '__main__':
    main()