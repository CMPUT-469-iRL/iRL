import torch
import numpy as np
import argparse
from tmaze_pen import TMazeEnv
from eLSTM_model.model import DQNModel
from dqn.dqn_agent import DQNAgent

def evaluate(env, agent, episodes=100):
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.tensor([state]).long()
            action = agent.select_action(state_tensor, evaluate=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    print(f"Average Reward over {episodes} episodes: {np.mean(rewards):.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--corridor_length', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()

    # ENV
    env = TMazeEnv(corridor_length=args.corridor_length)

    # MODEL
    model = DQNModel(
        emb_dim=0,
        hidden_size=64,
        in_vocab_size=4,
        out_vocab_size=3,
        no_embedding=True
    )

    # AGENT (no training, just evaluation)
    agent = DQNAgent(model=model)
    agent.load_checkpoint(args.model_path)
    agent.epsilon = 0.0  # Ensure full greedy during evaluation

    # EVAL
    evaluate(env, agent, episodes=args.episodes)

if __name__ == '__main__':
    main()
