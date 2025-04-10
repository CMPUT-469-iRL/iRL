import os
import glob
import re
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

########################
# Optional: Choose a style
########################
# Try these other options:
# plt.style.use('ggplot')
# plt.style.use('fivethirtyeight')
# plt.style.use('seaborn')
plt.style.use('ggplot')  # Example style

def load_results(filepath):
    try:
        checkpoint = torch.load(filepath, weights_only=False)
        return {
            'rewards': checkpoint.get('rewards_history', []),
            'losses': checkpoint.get('losses_history', []),
            'eval_rewards': checkpoint.get('eval_rewards', []),
            'episodes': len(checkpoint.get('rewards_history', [])),
            'hyperparams': checkpoint.get('hyperparams', {})
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {
            'rewards': [],
            'losses': [],
            'eval_rewards': [],
            'episodes': 0,
            'hyperparams': {}
        }

def extract_corridor_length(filepath):
    match = re.search(r'model_(\d+)\.pt', filepath)
    if match:
        return int(match.group(1))
    return 0

def find_model_files():
    results_dict = {}
    model_files = glob.glob('save_models/stream_q_model_*.pt')
    
    for file in model_files:
        corridor_length = extract_corridor_length(file)
        if corridor_length > 0:
            print(f"Found model for corridor length {corridor_length}: {file}")
            results_dict[corridor_length] = load_results(file)
    
    if not results_dict:
        default_model = "save_models/stream_q_model.pt"
        if os.path.exists(default_model):
            print("Loading default model (no corridor length in filename)...")
            results_dict[0] = load_results(default_model)
        else:
            print("No model files found. Please run training first.")
    
    return results_dict

def moving_average(data, window_size=5):
    """Helper function for optional smoothing."""
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    results_dict = find_model_files()
    if not results_dict:
        return

    corridor_lengths = sorted(results_dict.keys())
    
    # Create a figure with 3 rows of subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    fig.suptitle("Model Comparison by Corridor Length", fontsize=16, y=0.95)
    
    # ----------------------------------------------------------------------
    # 1. Episode Rewards
    # ----------------------------------------------------------------------
    ax = axs[0]
    for length in corridor_lengths:
        rewards = results_dict[length]['rewards']
        
        # Optional smoothing
        # Increase or decrease window_size to suit your data
        smoothed_rewards = moving_average(rewards, window_size=5)
        
        if len(smoothed_rewards) > 0:
            ax.plot(
                smoothed_rewards, 
                linewidth=2, 
                label=f"Length {length}"
            )
    ax.set_title("Episode Rewards (Smoothed)", fontsize=13)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=10, loc='best')
    
    # ----------------------------------------------------------------------
    # 2. Training Losses
    # ----------------------------------------------------------------------
    ax = axs[1]
    for length in corridor_lengths:
        losses = results_dict[length]['losses']
        if len(losses) > 0:
            # Smooth losses too, if desired
            smoothed_losses = moving_average(losses, window_size=5)
            ax.plot(smoothed_losses, linewidth=2, label=f"Length {length}")
    
    ax.set_title("Training Losses (Log Scale)", fontsize=13)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_yscale('log')  # maintain log scale for clarity
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=10, loc='best')
    
    # ----------------------------------------------------------------------
    # 3. Evaluation Rewards
    # ----------------------------------------------------------------------
    ax = axs[2]
    for length in corridor_lengths:
        eval_rewards = results_dict[length]['eval_rewards']
        total_eps = results_dict[length]['episodes']
        
        if len(eval_rewards) > 0 and total_eps > 0:
            # If you logged eval rewards every N episodes, 
            # find approximate x-values for each eval point
            step = max(total_eps // len(eval_rewards), 1)
            x_vals = np.arange(step, step * len(eval_rewards) + 1, step)
            ax.plot(
                x_vals, 
                eval_rewards, 
                marker='o', 
                linewidth=2, 
                label=f"Length {length}"
            )
    ax.set_title("Evaluation Rewards", fontsize=13)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Eval Reward", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    # Adjust the spacing to make room for the figure title
    plt.subplots_adjust(top=0.90)
    plt.savefig("model_comparison_improved.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
