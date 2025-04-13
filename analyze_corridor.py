import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import simps

def load_saved_models(base_dir="resultsTEMP", algo_types=["BPTT", "RTRL"]):
    """
    Load saved models for different corridor lengths and algorithms.
    
    Returns:
        dict: Nested dictionary with algorithm and corridor length as keys
    """
    results = {}
    
    for algo in algo_types:
        algo_dir = os.path.join(base_dir, algo)
        results[algo] = {}
        
        if not os.path.exists(algo_dir):
            print(f"Warning: Directory not found - {algo_dir}")
            continue
            
        # Find all corridor lengths in this directory
        lengths = []
        for file in os.listdir(algo_dir):
            if file.startswith("model_length_") and file.endswith(".pt"):
                try:
                    length = int(file.split("_")[-1].split(".")[0])
                    lengths.append(length)
                except ValueError:
                    continue
        
        lengths.sort()
        print(f"Found corridor lengths for {algo}: {lengths}")
        
        # Load each model
        for length in lengths:
            model_path = os.path.join(algo_dir, f"model_length_{length}.pt")
            
            try:
                checkpoint = torch.load(model_path)
                rewards = checkpoint.get('rewards_history', [])
                losses = checkpoint.get('losses_history', [0] * len(rewards))
                
                # Calculate approximate timesteps (based on length * 2.5 * episodes)
                total_steps = length * 2.5 * len(rewards)
                
                results[algo][length] = {
                    'rewards': rewards,
                    'losses': losses,
                    'timesteps': total_steps
                }
                
                print(f"Loaded {algo} model for length {length} with {len(rewards)} episodes")
            except Exception as e:
                print(f"Error loading {model_path}: {e}")
    
    return results

def smooth_data(data, sigma=4.0):
    """Apply Gaussian smoothing to data for smoother curves"""
    return gaussian_filter1d(data, sigma=sigma)

def calculate_auc(rewards, sigma=3.5):
    """Calculate area under the curve for a reward sequence"""
    # Apply smoothing for more reliable AUC calculation
    smoothed = smooth_data(np.array(rewards), sigma)
    # Calculate AUC using Simpson's rule
    x = np.arange(len(smoothed))
    auc = simps(smoothed, x)
    # Get final performance (average of last 10%)
    final_window = max(10, int(len(smoothed) * 0.1))
    final_perf = np.mean(smoothed[-final_window:])
    
    return {
        'auc': auc,
        'final_perf': final_perf,
        'smoothed': smoothed
    }

def create_timestep_plots(results, output_dir="output_models_graph"):
    """Create time-step based plots for both rewards and losses"""
    os.makedirs(output_dir, exist_ok=True)
    
    for algo in results:
        if not results[algo]:
            continue
            
        lengths = sorted(results[algo].keys())
        
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
        
        # Plot 1: Rewards by Time Steps
        ax1 = plt.subplot(1, 2, 1)
        
        for i, length in enumerate(lengths):
            rewards = np.array(results[algo][length]['rewards'])
            # Apply Gaussian smoothing
            smoothed = smooth_data(rewards)
            
            # Calculate time steps for each episode
            steps_per_episode = length * 2.5
            time_steps = np.arange(1, len(smoothed) + 1) * steps_per_episode
            
            # Plot the smoothed line
            plt.plot(time_steps, smoothed, color=colors[i], linewidth=3.0, 
                     label=f"Length {length}")
        
        plt.title("Average Episodic Rewards", fontsize=15, fontweight='bold')
        plt.xlabel("Time Steps", fontsize=13)
        plt.ylabel("Reward", fontsize=13)
        plt.legend(loc="upper left", framealpha=0.9, fontsize=11)
        
        # Format x-axis with k notation for thousands
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
        
        # Enhance appearance
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.4)
        
        # Plot 2: Losses by Time Steps
        ax2 = plt.subplot(1, 2, 2)
        
        for i, length in enumerate(lengths):
            loss_values = np.array(results[algo][length]['losses'])
            
            # Some loss values might be zero if not properly tracked
            if np.sum(loss_values) > 0:
                # Apply Gaussian smoothing
                smoothed = smooth_data(loss_values)
                
                # Calculate time steps for each episode
                steps_per_episode = length * 2.5
                time_steps = np.arange(1, len(smoothed) + 1) * steps_per_episode
                
                # Plot the smoothed line
                plt.plot(time_steps, smoothed, color=colors[i], linewidth=3.0, 
                         label=f"Length {length}")
        
        plt.title("Training Loss", fontsize=15, fontweight='bold')
        plt.xlabel("Time Steps", fontsize=13)
        plt.ylabel("Loss", fontsize=13)
        plt.legend(loc="upper right", framealpha=0.9, fontsize=11)
        
        # Format x-axis with k notation for thousands
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
        
        # Enhance appearance
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.4)
        
        # Save and show plot with tight layout
        max_timesteps = max([results[algo][length]['timesteps'] for length in lengths])
        timestep_text = f"{max_timesteps/1000:.0f}k" if max_timesteps < 1000000 else f"{max_timesteps/1000000:.1f}M"
        
        plt.suptitle(f"Learning Performance with {algo} (≈{timestep_text} timesteps)", 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save with appropriate filename
        filename = os.path.join(output_dir, f"timestep_plots_{algo.lower()}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        print(f"Saved time-step plots: {filename}")

def create_auc_plots(results, output_dir="output_models_graph"):
    """Create area under curve plots for both BPTT and RTRL"""
    os.makedirs(output_dir, exist_ok=True)
    
    for algo in results:
        if not results[algo]:
            continue
            
        lengths = sorted(results[algo].keys())
        
        # Calculate AUC for each corridor length
        auc_data = {}
        for length in lengths:
            auc_data[length] = calculate_auc(results[algo][length]['rewards'])
        
        # Create figure for AUC analysis
        plt.figure(figsize=(12, 6))
        plt.style.use('ggplot')
        
        # Color palette
        colors = plt.cm.plasma(np.linspace(0, 0.85, len(lengths)))
        
        # Plot: Average Return vs Time Steps (single full-page plot)
        # Generate synthetic time steps for each length based on corridor length
        for i, length in enumerate(lengths):
            smoothed = auc_data[length]['smoothed']
            # Estimate timesteps: each episode takes ~2.5 * corridor_length steps
            steps_per_episode = length * 2.5
            time_steps = np.arange(1, len(smoothed) + 1) * steps_per_episode
            
            # Plot the time steps vs rewards
            plt.plot(time_steps, smoothed, color=colors[i], linewidth=2.5, 
                     label=f"Corridor Length {length}")
        
        plt.title(f"Average Return vs Time Steps ({algo})", fontsize=16, fontweight='bold')
        plt.xlabel("Time Steps", fontsize=14)
        plt.ylabel("Average Episodic Return", fontsize=14)
        plt.legend(loc="best", framealpha=0.9)
        
        # Format x-axis with k notation for thousands
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
        
        # Enhance appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f"auc_analysis_{algo.lower()}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        print(f"Saved AUC plot: {filename}")

def main():
    """Main function to analyze saved models and create plots"""
    # Create output directory
    output_dir = "output_models_graph"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load saved models
    results = load_saved_models(base_dir="resultsTEMP")
    
    # Create time-step based plots (both rewards and losses)
    create_timestep_plots(results, output_dir)
    
    # Create AUC plots (average return vs time steps)
    create_auc_plots(results, output_dir)
    
    print(f"All plots saved in {output_dir}")

if __name__ == "__main__":
    main()