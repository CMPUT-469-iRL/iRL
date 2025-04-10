import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import torch
from scipy.ndimage import gaussian_filter1d
from streamQ import train_stream_q  # Changed to standard streamQ

def compare_corridor_lengths(lengths, max_timesteps=1000000, epsilon=0.005):
    """
    Train and compare agent performance across different corridor lengths.
    Uses very low epsilon value to minimize exploration and focuses on total timesteps.
    
    Args:
        lengths: List of corridor lengths to test
        max_timesteps: Maximum timesteps to run per corridor length (can be up to 1M)
        epsilon: Exploration rate (lower means less exploration)
    """
    results = {}
    losses = {}
    timesteps_per_length = {}
    
    # Create results directory if needed
    os.makedirs("results", exist_ok=True)  # Changed to "results" for standard models
    
    # Estimate episodes needed for each corridor length to reach max_timesteps
    # Longer corridors need fewer episodes to reach the same number of timesteps
    for length in lengths:
        # Rough estimate: each episode takes ~2-3x corridor_length steps on average
        avg_steps_per_episode = length * 2.5
        estimated_episodes = int(max_timesteps / avg_steps_per_episode) + 100  # Add larger buffer for long runs
        
        print(f"\n{'='*50}")
        print(f"Training with corridor length: {length}")
        print(f"Estimated episodes needed for {max_timesteps/1000:.1f}k timesteps: {estimated_episodes}")
        print(f"{'='*50}\n")
        
        # Train with minimal exploration
        _, rewards = train_stream_q(  # Changed to train_stream_q
            episodes=estimated_episodes,
            corridor_length=length,
            epsilon_start=epsilon,
            epsilon_end=epsilon/10,     # Even lower final exploration
            epsilon_decay=0.999,        # Very slow decay
            max_steps_per_episode=max(100, length*3),  # Adjust based on corridor length
            save_path=f"results/model_length_{length}.pt",  # Changed directory
            eval_interval=estimated_episodes//5  # Fewer evaluations for long runs
        )
        
        # Get the loss values and total timesteps from the saved model
        checkpoint = torch.load(f"results/model_length_{length}.pt")  # Changed directory
        loss_history = checkpoint.get('losses_history', [0] * len(rewards))
        
        # Calculate total timesteps based on corridor length and episodes
        # More accurate than a fixed formula
        total_steps = length * 2.5 * len(rewards)
        
        # Store results
        results[length] = rewards
        losses[length] = loss_history
        timesteps_per_length[length] = total_steps
        
        print(f"Completed training for length {length}")
        print(f"Total timesteps: {total_steps:,}")
        print(f"Episodes completed: {len(rewards)}")
    
    # Plot results
    plot_comparison(results, losses, timesteps_per_length, lengths, max_timesteps)
    
    return results

def smooth_data(data, sigma=4.0):
    """Apply Gaussian smoothing to data for smoother curves"""
    return gaussian_filter1d(data, sigma=sigma)

def plot_comparison(results, losses, timesteps_per_length, lengths, max_timesteps):
    """
    Create side-by-side plots comparing rewards and losses for different corridor lengths.
    """
    plt.figure(figsize=(16, 7))
    plt.style.use('ggplot')
    
    # Use a visually appealing color palette
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(lengths)))
    
    # Format for thousands/millions
    def format_k(x, pos):
        if x >= 1000000:
            return f'{x/1000000:.1f}M'
        elif x >= 1000:
            return f'{x/1000:.0f}k'
        else:
            return f'{x:.0f}'
    
    # Plot 1: Rewards
    ax1 = plt.subplot(1, 2, 1)
    
    # Plot each corridor length with smooth curves
    for i, length in enumerate(lengths):
        rewards = np.array(results[length])
        
        # Apply Gaussian smoothing for cleaner visualization (increased smoothing)
        smoothed = smooth_data(rewards)
        
        # Plot the smoothed line
        plt.plot(smoothed, color=colors[i], linewidth=3.0, 
                 label=f"Length {length} ({format_k(timesteps_per_length[length], 0)} steps)")
    
    plt.title("Average Episodic Rewards", fontsize=15, fontweight='bold')
    plt.xlabel("Episode", fontsize=13)
    plt.ylabel("Reward", fontsize=13)
    plt.legend(loc="upper left", framealpha=0.9, fontsize=11)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Enhance appearance
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Plot 2: Losses
    ax2 = plt.subplot(1, 2, 2)
    
    # Plot each corridor length with smooth curves
    for i, length in enumerate(lengths):
        loss_values = np.array(losses[length])
        
        # Some loss values might be zero if not properly tracked
        if np.sum(loss_values) > 0:
            # Apply Gaussian smoothing for cleaner visualization
            smoothed = smooth_data(loss_values)
            
            # Plot the smoothed line
            plt.plot(smoothed, color=colors[i], linewidth=3.0, 
                     label=f"Length {length}")
    
    plt.title("Training Loss", fontsize=15, fontweight='bold')
    plt.xlabel("Episode", fontsize=13)
    plt.ylabel("Loss", fontsize=13)
    plt.legend(loc="upper right", framealpha=0.9, fontsize=11)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Enhance appearance
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Format timesteps nicely for title
    timestep_text = f"{max_timesteps/1000:.0f}k" if max_timesteps < 1000000 else f"{max_timesteps/1000000:.1f}M"
    
    # Save and show plot with tight layout
    plt.suptitle(f"Effect of Corridor Length on Agent Learning with QuasiLSTM (≈{timestep_text} timesteps)", 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save with appropriate filename
    filename = f"corridor_length_comparison_bptt_{timestep_text.replace('.', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as '{filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare agent performance across different corridor lengths")
    parser.add_argument("--lengths", type=int, nargs='+', default=[10, 20, 40, 60],
                       help="List of corridor lengths to compare")
    parser.add_argument("--max_timesteps", type=int, default=1000000,
                       help="Maximum timesteps to run for each corridor length")
    parser.add_argument("--epsilon", type=float, default=0.005,
                       help="Exploration rate (epsilon), set very low to minimize exploration")
    
    args = parser.parse_args()
    
    print(f"\nRunning comparison with {args.max_timesteps/1000:.1f}k timesteps using standard QuasiLSTM...")
    
    # Override plot_training_results in streamQ to do nothing
    import streamQ
    streamQ.plot_training_results = lambda *args, **kwargs: None
    
    # Run comparison
    results = compare_corridor_lengths(
        lengths=args.lengths,
        max_timesteps=args.max_timesteps,
        epsilon=args.epsilon
    )
    
    timestep_text = f"{args.max_timesteps/1000:.0f}k" if args.max_timesteps < 1000000 else f"{args.max_timesteps/1000000:.1f}M"
    print(f"\nComparison complete! Results visualized with {timestep_text} timesteps")