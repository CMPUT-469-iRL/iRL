import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import glob
import re
from matplotlib.gridspec import GridSpec
from matplotlib import cm  # For color maps

def load_results(filepath):
    """Load results from PyTorch save file with error handling."""
    try:
        # Handle PyTorch loading issues
        checkpoint = torch.load(filepath, weights_only=False, map_location=torch.device('cpu'))
        return {
            'rewards': checkpoint['rewards_history'],
            'losses': checkpoint['losses_history'],
            'eval_rewards': checkpoint['eval_rewards'],
            'episodes': len(checkpoint['rewards_history']),
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

def plot_improved_comparison(results_dict, window_size=10):
    """Create improved visualizations for comparing corridor lengths."""
    # Set up the visual style
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Create a figure with custom layout
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])
    
    # Get sorted corridor lengths and create a color palette
    corridor_lengths = sorted(results_dict.keys())
    viridis = cm.get_cmap('viridis', len(corridor_lengths))
    colors = [viridis(i/len(corridor_lengths)) for i in range(len(corridor_lengths))]
    
    # 1. Episode Rewards Plot (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    for i, length in enumerate(corridor_lengths):
        results = results_dict[length]
        rewards = results['rewards']
        if len(rewards) > 0:
            smoothed = smooth_data(rewards, window_size)
            ep_count = len(rewards)
            x = np.arange(ep_count)
            ax1.plot(x, smoothed, color=colors[i], linewidth=2.5, 
                   label=f"Length {length} ({ep_count} eps)")
            
            # Add shaded area for min/max if enough data
            if len(rewards) > window_size * 2:
                std_rewards = np.array([np.std(rewards[max(0, i-window_size):min(len(rewards), i+window_size+1)]) 
                                      for i in range(len(rewards))])
                ax1.fill_between(x, smoothed - std_rewards, smoothed + std_rewards, 
                               color=colors[i], alpha=0.2)
    
    ax1.set_title('Episode Rewards vs Corridor Length', fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Learning Efficiency Plot (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    # This will show reward improvement over episodes
    for i, length in enumerate(corridor_lengths):
        results = results_dict[length]
        rewards = results['rewards']
        if len(rewards) > window_size:
            # Normalize to show percentage of improvement
            smoothed = smooth_data(rewards, window_size)
            first_val = smoothed[0]
            # Avoid division by zero with small offset
            if abs(first_val) < 1e-6:
                first_val = -1.0  # Reasonable default for negative rewards
            normalized = [(r - first_val) / abs(first_val) * 100 for r in smoothed]
            
            ax2.plot(normalized, color=colors[i], linewidth=2.5, 
                   label=f"Length {length}")
    
    ax2.set_title('Learning Efficiency (% Improvement)', fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Improvement %')
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # 3. Success Rate Comparison (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    lengths = []
    success_rates = []
    final_rewards = []
    
    for length in corridor_lengths:
        results = results_dict[length]
        rewards = results['rewards']
        if rewards:
            # Calculate success rate for the last 30% of episodes
            cutoff = int(len(rewards) * 0.7)
            recent_rewards = rewards[cutoff:]
            success_rate = calculate_success_rate(recent_rewards)
            
            lengths.append(length)
            success_rates.append(success_rate)
            final_rewards.append(np.mean(recent_rewards[-10:]) if len(recent_rewards) >= 10 else np.mean(recent_rewards))
    
    # Bar chart for success rates
    bars = ax3.bar(range(len(lengths)), success_rates, color=colors[:len(lengths)])
    
    # Add the values on top of the bars
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f"{rate:.2f}", ha='center', va='bottom', fontweight='bold')
    
    ax3.set_title('Success Rate (Last 30% of Episodes)', fontweight='bold')
    ax3.set_xlabel('Corridor Length')
    ax3.set_ylabel('Success Rate')
    ax3.set_xticks(range(len(lengths)))
    ax3.set_xticklabels(lengths)
    ax3.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # 4. Evaluation Rewards (Middle Right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    for i, length in enumerate(corridor_lengths):
        results = results_dict[length]
        eval_rewards = results['eval_rewards']
        if len(eval_rewards) > 0:
            # Find the evaluation interval
            total_episodes = results['episodes']
            
            # Calculate appropriate eval interval
            if len(eval_rewards) <= 1:
                eval_interval = total_episodes
            else:
                eval_interval = total_episodes // len(eval_rewards)
            
            eval_episodes = np.arange(eval_interval, total_episodes + 1, eval_interval)
            
            # Ensure arrays match in length
            min_len = min(len(eval_rewards), len(eval_episodes))
            eval_rewards = eval_rewards[:min_len]
            eval_episodes = eval_episodes[:min_len]
            
            ax4.plot(eval_episodes, eval_rewards, color=colors[i], marker='o', linewidth=2.5,
                    label=f"Length {length}")
    
    ax4.set_title('Evaluation Rewards', fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Reward')
    ax4.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    # 5. Performance vs Corridor Length (Bottom Left)
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Create a scatter plot with trend line
    if final_rewards:
        # Scatter plot
        ax5.scatter(lengths, final_rewards, color=colors[:len(lengths)], s=150, zorder=10)
        
        # Add trend line if we have enough points
        if len(lengths) >= 2:
            try:
                z = np.polyfit(lengths, final_rewards, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(lengths), max(lengths), 100)
                ax5.plot(x_trend, p(x_trend), '--', color='gray', 
                        label=f"Trend: {z[0]:.4f}x + {z[1]:.2f}")
            except:
                pass
        
        # Add text labels
        for i, (x, y) in enumerate(zip(lengths, final_rewards)):
            ax5.annotate(f"{y:.2f}", 
                       (x, y), 
                       textcoords="offset points",
                       xytext=(0, 10), 
                       ha='center',
                       fontweight='bold')
    
    ax5.set_title('Final Performance vs Corridor Length', fontweight='bold')
    ax5.set_xlabel('Corridor Length')
    ax5.set_ylabel('Avg Reward (Last 10 Episodes)')
    ax5.grid(True, linestyle='--', alpha=0.6)
    if len(lengths) >= 2:
        ax5.legend(loc='best', frameon=True)
    
    # 6. Training Loss (Bottom Right)
    ax6 = fig.add_subplot(gs[2, 1])
    
    for i, length in enumerate(corridor_lengths):
        results = results_dict[length]
        losses = results['losses']
        if len(losses) > window_size:
            # Clean up losses (remove zeros and extreme values)
            losses = np.array([max(1e-10, min(loss, 1.0)) for loss in losses if loss > 0])
            
            if len(losses) > 0:
                smoothed_losses = smooth_data(losses, window_size)
                ax6.semilogy(smoothed_losses, color=colors[i], linewidth=2.5, 
                           label=f"Length {length}")
    
    ax6.set_title('Training Loss (Log Scale)', fontweight='bold')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Loss (log)')
    ax6.grid(True, linestyle='--', alpha=0.6)
    ax6.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    # Add overall title and adjust layout
    plt.suptitle(f'StreamQ Performance Analysis Across Corridor Lengths', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save high-quality figure
    plt.savefig('streamq_corridor_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved plot to streamq_corridor_comparison.png")
    plt.show()

def find_all_model_files():
    """Find and load all model files with corridor lengths."""
    results_dict = {}
    
    # Search for model files with pattern
    model_files = glob.glob('save_models/stream_q_model_*.pt')
    
    # Extract corridor length from filenames
    for file in model_files:
        corridor_length = extract_corridor_length(file)
        if corridor_length > 0:
            print(f"Found model for corridor length {corridor_length}: {file}")
            results_dict[corridor_length] = load_results(file)
    
    # If no specific models found, try the default model
    if not results_dict:
        default_model = "save_models/stream_q_model.pt"
        if os.path.exists(default_model):
            print("Loading default model...")
            results_dict[0] = load_results(default_model)
    
    return results_dict

if __name__ == "__main__":
    print("Searching for model files...")
    results_dict = find_all_model_files()
    
    if results_dict:
        print(f"Found {len(results_dict)} models with corridor lengths: {sorted(results_dict.keys())}")
        plot_improved_comparison(results_dict)
    else:
        print("No model files found. Please run training first.") 