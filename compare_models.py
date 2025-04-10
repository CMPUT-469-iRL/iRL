import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import glob
import re
from matplotlib.gridspec import GridSpec
from matplotlib import cm

def load_results(filepath):
    """Load results from PyTorch save file with error handling."""
    try:
        # Handle PyTorch loading issues
        checkpoint = torch.load(filepath, weights_only=False, map_location=torch.device('cpu'))
        
        # Check if this is an RTRL model
        is_rtrl = False
        if 'hyperparams' in checkpoint and 'is_rtrl' in checkpoint['hyperparams']:
            is_rtrl = checkpoint['hyperparams']['is_rtrl']
        elif 'rtrl' in filepath:
            is_rtrl = True
        
        return {
            'rewards': checkpoint['rewards_history'],
            'losses': checkpoint['losses_history'],
            'eval_rewards': checkpoint['eval_rewards'],
            'episodes': len(checkpoint['rewards_history']),
            'hyperparams': checkpoint.get('hyperparams', {}),
            'is_rtrl': is_rtrl
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {
            'rewards': [],
            'losses': [],
            'eval_rewards': [],
            'episodes': 0,
            'hyperparams': {},
            'is_rtrl': False
        }

def extract_corridor_length(filepath):
    """Extract corridor length from filename."""
    match = re.search(r'model_(\d+)\.pt', filepath)
    if match:
        return int(match.group(1))
    return 0

def smooth_data(data, window_size=10):
    """Apply smoothing to data with proper handling of endpoints."""
    if len(data) < window_size:
        return data
    
    # Use valid convolution to avoid edge effects
    weights = np.ones(window_size) / window_size
    smoothed = np.convolve(data, weights, mode='valid')
    
    # Pad the beginning to maintain original length
    padding = np.ones(window_size-1) * data[0]
    return np.concatenate([padding, smoothed])

def calculate_success_rate(rewards, threshold=0):
    """Calculate success rate (rewards >= threshold)."""
    if not rewards:
        return 0
    return sum(r >= threshold for r in rewards) / len(rewards)

def plot_model_comparison(std_results_dict, rtrl_results_dict, length, window_size=10):
    """Create comparison plots for a specific corridor length between standard and RTRL models."""
    # Set up the visual style
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Create a figure with custom layout
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Colors for std vs rtrl
    std_color = 'blue'
    rtrl_color = 'darkgreen'
    
    # 1. Episode Rewards Plot (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot standard model
    std_rewards = std_results_dict[length]['rewards'] if length in std_results_dict else []
    if len(std_rewards) > 0:
        smoothed = smooth_data(std_rewards, window_size)
        ep_count = len(std_rewards)
        x = np.arange(ep_count)
        ax1.plot(x, smoothed, color=std_color, linewidth=2.5, 
               label=f"Standard QuasiLSTM ({ep_count} eps)")
        
        # Add shaded area
        if len(std_rewards) > window_size * 2:
            std_rewards_var = np.array([np.std(std_rewards[max(0, i-window_size):min(len(std_rewards), i+window_size+1)]) 
                                  for i in range(len(std_rewards))])
            ax1.fill_between(x, smoothed - std_rewards_var, smoothed + std_rewards_var, 
                           color=std_color, alpha=0.2)
    
    # Plot RTRL model
    rtrl_rewards = rtrl_results_dict[length]['rewards'] if length in rtrl_results_dict else []
    if len(rtrl_rewards) > 0:
        smoothed = smooth_data(rtrl_rewards, window_size)
        ep_count = len(rtrl_rewards)
        x = np.arange(ep_count)
        ax1.plot(x, smoothed, color=rtrl_color, linewidth=2.5, 
               label=f"RTRL QuasiLSTM ({ep_count} eps)")
        
        # Add shaded area
        if len(rtrl_rewards) > window_size * 2:
            rtrl_rewards_var = np.array([np.std(rtrl_rewards[max(0, i-window_size):min(len(rtrl_rewards), i+window_size+1)]) 
                                  for i in range(len(rtrl_rewards))])
            ax1.fill_between(x, smoothed - rtrl_rewards_var, smoothed + rtrl_rewards_var, 
                           color=rtrl_color, alpha=0.2)
    
    ax1.set_title(f'Episode Rewards (Corridor Length {length})', fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Training Loss (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot standard model
    std_losses = std_results_dict[length]['losses'] if length in std_results_dict else []
    if len(std_losses) > window_size:
        # Clean up losses (remove zeros and extreme values)
        std_losses = np.array([max(1e-10, min(loss, 1.0)) for loss in std_losses if loss > 0])
        
        if len(std_losses) > 0:
            smoothed_losses = smooth_data(std_losses, window_size)
            ax2.semilogy(smoothed_losses, color=std_color, linewidth=2.5, 
                       label=f"Standard QuasiLSTM")
    
    # Plot RTRL model
    rtrl_losses = rtrl_results_dict[length]['losses'] if length in rtrl_results_dict else []
    if len(rtrl_losses) > window_size:
        # Clean up losses (remove zeros and extreme values)
        rtrl_losses = np.array([max(1e-10, min(loss, 1.0)) for loss in rtrl_losses if loss > 0])
        
        if len(rtrl_losses) > 0:
            smoothed_losses = smooth_data(rtrl_losses, window_size)
            ax2.semilogy(smoothed_losses, color=rtrl_color, linewidth=2.5, 
                       label=f"RTRL QuasiLSTM")
    
    ax2.set_title('Training Loss (Log Scale)', fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss (log)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    # 3. Learning Efficiency (Bottom Left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Plot standard model improvement
    if len(std_rewards) > window_size:
        # Normalize to show percentage of improvement
        smoothed = smooth_data(std_rewards, window_size)
        first_val = smoothed[0]
        # Avoid division by zero with small offset
        if abs(first_val) < 1e-6:
            first_val = -1.0  # Reasonable default for negative rewards
        normalized = [(r - first_val) / abs(first_val) * 100 for r in smoothed]
        
        ax3.plot(normalized, color=std_color, linewidth=2.5, 
               label=f"Standard QuasiLSTM")
    
    # Plot RTRL model improvement
    if len(rtrl_rewards) > window_size:
        # Normalize to show percentage of improvement
        smoothed = smooth_data(rtrl_rewards, window_size)
        first_val = smoothed[0]
        # Avoid division by zero with small offset
        if abs(first_val) < 1e-6:
            first_val = -1.0  # Reasonable default for negative rewards
        normalized = [(r - first_val) / abs(first_val) * 100 for r in smoothed]
        
        ax3.plot(normalized, color=rtrl_color, linewidth=2.5, 
               label=f"RTRL QuasiLSTM")
    
    ax3.set_title('Learning Efficiency (% Improvement)', fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Improvement %')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # 4. Evaluation Rewards (Bottom Right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot standard model evaluation
    std_eval_rewards = std_results_dict[length]['eval_rewards'] if length in std_results_dict else []
    std_episodes = std_results_dict[length]['episodes'] if length in std_results_dict else 0
    
    if len(std_eval_rewards) > 0:
        # Calculate appropriate eval interval
        if len(std_eval_rewards) <= 1:
            eval_interval = std_episodes
        else:
            eval_interval = std_episodes // len(std_eval_rewards)
        
        eval_episodes = np.arange(eval_interval, std_episodes + 1, eval_interval)
        
        # Ensure arrays match in length
        min_len = min(len(std_eval_rewards), len(eval_episodes))
        std_eval_rewards = std_eval_rewards[:min_len]
        eval_episodes = eval_episodes[:min_len]
        
        ax4.plot(eval_episodes, std_eval_rewards, color=std_color, marker='o', linewidth=2.5,
                label=f"Standard QuasiLSTM")
    
    # Plot RTRL model evaluation
    rtrl_eval_rewards = rtrl_results_dict[length]['eval_rewards'] if length in rtrl_results_dict else []
    rtrl_episodes = rtrl_results_dict[length]['episodes'] if length in rtrl_results_dict else 0
    
    if len(rtrl_eval_rewards) > 0:
        # Calculate appropriate eval interval
        if len(rtrl_eval_rewards) <= 1:
            eval_interval = rtrl_episodes
        else:
            eval_interval = rtrl_episodes // len(rtrl_eval_rewards)
        
        eval_episodes = np.arange(eval_interval, rtrl_episodes + 1, eval_interval)
        
        # Ensure arrays match in length
        min_len = min(len(rtrl_eval_rewards), len(eval_episodes))
        rtrl_eval_rewards = rtrl_eval_rewards[:min_len]
        eval_episodes = eval_episodes[:min_len]
        
        ax4.plot(eval_episodes, rtrl_eval_rewards, color=rtrl_color, marker='o', linewidth=2.5,
                label=f"RTRL QuasiLSTM")
    
    ax4.set_title('Evaluation Rewards', fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Reward')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    # Add overall title
    plt.suptitle(f'QuasiLSTM vs RTRLQuasiLSTM Comparison (Corridor Length {length})', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save high-quality figure
    plt.savefig(f'model_comparison_length_{length}.png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot for corridor length {length}")
    plt.close()

def find_model_files():
    """Find both standard and RTRL model files."""
    std_results_dict = {}
    rtrl_results_dict = {}
    
    # Search for standard model files
    std_files = glob.glob('save_models/stream_q_model_*.pt')
    for file in std_files:
        corridor_length = extract_corridor_length(file)
        if corridor_length > 0 and 'rtrl' not in file:
            print(f"Found standard model for corridor length {corridor_length}: {file}")
            std_results_dict[corridor_length] = load_results(file)
    
    # Search for RTRL model files
    rtrl_files = glob.glob('save_models/rtrl_stream_q_model_*.pt')
    for file in rtrl_files:
        corridor_length = extract_corridor_length(file)
        if corridor_length > 0:
            print(f"Found RTRL model for corridor length {corridor_length}: {file}")
            rtrl_results_dict[corridor_length] = load_results(file)
    
    return std_results_dict, rtrl_results_dict

def run_model_comparison():
    """Run comparison for all available corridor lengths."""
    print("Searching for model files...")
    std_results, rtrl_results = find_model_files()
    
    # Find common corridor lengths
    std_lengths = set(std_results.keys())
    rtrl_lengths = set(rtrl_results.keys())
    all_lengths = std_lengths.union(rtrl_lengths)
    
    if not all_lengths:
        print("No model files found. Please run training first.")
        return
    
    print(f"Found {len(std_lengths)} standard models and {len(rtrl_lengths)} RTRL models")
    print(f"Corridor lengths for standard models: {sorted(std_lengths)}")
    print(f"Corridor lengths for RTRL models: {sorted(rtrl_lengths)}")
    
    # Create comparison plots for each corridor length
    for length in sorted(all_lengths):
        print(f"Generating comparison for corridor length {length}...")
        plot_model_comparison(std_results, rtrl_results, length)
    
    print("Comparison complete!")

if __name__ == "__main__":
    run_model_comparison() 