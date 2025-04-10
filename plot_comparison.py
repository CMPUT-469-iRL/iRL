import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def load_results(filepath):
    try:
        # Use weights_only=False to handle the unpickling error in PyTorch 2.6
        checkpoint = torch.load(filepath, weights_only=False)
        return {
            'rewards': checkpoint['rewards_history'],
            'losses': checkpoint['losses_history'],
            'eval_rewards': checkpoint['eval_rewards']
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        # Return empty data as fallback
        return {
            'rewards': [],
            'losses': [],
            'eval_rewards': []
        }

def plot_comparison(results_dict, eval_interval=20, window_size=10):
    plt.figure(figsize=(15, 5))
    colors = ['blue', 'red', 'green']
    labels = ['Corridor Length 5', 'Corridor Length 70', 'Corridor Length 80']
    
    # Plot episode rewards with smoothing
    plt.subplot(131)
    for (length, results), color, label in zip(results_dict.items(), colors, labels):
        rewards = results['rewards']
        if len(rewards) > 0:
            # Apply moving average smoothing
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_rewards, color=color, label=label, alpha=0.8)
    plt.title('Episode Rewards (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot training loss
    plt.subplot(132)
    for (length, results), color, label in zip(results_dict.items(), colors, labels):
        losses = results['losses']
        if len(losses) > 0:
            # Apply moving average smoothing
            smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_losses, color=color, label=label, alpha=0.8)
    plt.title('Training Loss (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot evaluation rewards
    plt.subplot(133)
    for (length, results), color, label in zip(results_dict.items(), colors, labels):
        eval_rewards = results['eval_rewards']
        if len(eval_rewards) > 0 and len(results['rewards']) > 0:
            eval_episodes = np.arange(eval_interval, len(results['rewards']) + 1, eval_interval)
            plt.plot(eval_episodes, eval_rewards, color=color, label=label, marker='o', alpha=0.8)
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('corridor_length_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Check if we have multiple model files with different corridor lengths
    results_dict = {}
    
    # Try to load existing files
    corridor_lengths = [5, 70, 80]
    for length in corridor_lengths:
        model_path = f"save_models/stream_q_model_{length}.pt"
        if os.path.exists(model_path):
            print(f"Loading model for corridor length {length}...")
            results_dict[length] = load_results(model_path)
    
    # If no specific models found, try the default model
    if not results_dict:
        default_model = "save_models/stream_q_model.pt"
        if os.path.exists(default_model):
            print("Loading default model...")
            results_dict[5] = load_results(default_model)
    
    # Plot comparison if we have any results
    if results_dict:
        print(f"Plotting comparison for {len(results_dict)} models")
        plot_comparison(results_dict)
    else:
        print("No model files found. Please run training first.") 