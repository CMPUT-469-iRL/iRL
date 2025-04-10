import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import glob
import re

def load_result(filepath):
    """Load results from PyTorch save file with error handling."""
    try:
        checkpoint = torch.load(filepath, weights_only=False, map_location=torch.device('cpu'))
        return {
            'rewards': checkpoint['rewards_history'],
            'losses': checkpoint['losses_history'],
            'eval_rewards': checkpoint['eval_rewards'],
            'name': os.path.basename(filepath)
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_corridor_length(filename):
    """Extract corridor length from filename."""
    match = re.search(r'model_(\d+)\.pt', filename)
    if match:
        return int(match.group(1))
    return None

def smooth_data(data, window_size=10):
    """Apply smoothing to data."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def quick_comparison():
    """Find and compare all available model files."""
    # Look for all saved model files
    standard_models = glob.glob('save_models/stream_q_model_*.pt')
    rtrl_models = glob.glob('save_models/rtrl_*model_*.pt')  # More flexible pattern to match rtrl models
    
    if not standard_models and not rtrl_models:
        print("No model files found. Please train models first.")
        return
    
    all_models = standard_models + rtrl_models
    print(f"Found {len(all_models)} model files")
    
    plt.figure(figsize=(12, 8))
    
    # Process each model
    for model_path in all_models:
        try:
            rewards, losses, eval_rewards, model_name = load_result(model_path)
            
            # Get corridor length from filename
            corridor_length = extract_corridor_length(model_path)
            
            # Determine if this is a standard or RTRL model
            is_rtrl = 'rtrl' in model_path.lower()
            
            # Set line style based on model type
            line_style = '--' if is_rtrl else '-'
            color = 'red' if is_rtrl else 'blue'
            alpha = 0.8
            
            # Create a label with model type and corridor length
            model_type = "RTRL QuasiLSTM" if is_rtrl else "Standard QuasiLSTM"
            label = f"{model_type} (Length {corridor_length})"
            
            # Plot rewards
            if len(rewards) > 5:  # Only smooth if we have enough data
                smoothed_rewards = smooth_data(rewards)
                plt.plot(smoothed_rewards, linestyle=line_style, color=color, alpha=alpha, label=label)
            else:
                plt.plot(rewards, linestyle=line_style, color=color, alpha=alpha, label=label)
            
        except Exception as e:
            print(f"Error processing {model_path}: {e}")
    
    plt.title('Model Reward Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('quick_comparison.png')
    print("Comparison plot saved as 'quick_comparison.png'")
    plt.close()

if __name__ == "__main__":
    quick_comparison() 