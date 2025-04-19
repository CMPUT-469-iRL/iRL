import os
import argparse
import torch
import numpy as np
import random
from streamQ import train_stream_q

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_corridor_length_experiments(corridor_lengths=[10, 100, 1000], 
                                   num_seeds=10, 
                                   base_seed=42,
                                   max_time_steps=1000000,
                                   output_dir="finalResults/RTRL"):
    """
    Train models for multiple corridor lengths using different seeds.
    
    Args:
        corridor_lengths: List of corridor lengths to test
        num_seeds: Number of seeds to use for each corridor length
        base_seed: Base seed value to start from
        max_time_steps: Maximum number of time steps for training
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for corridor_length in corridor_lengths:
        print(f"\n{'='*50}")
        print(f"Training for corridor length: {corridor_length}")
        print(f"{'='*50}")
        
        corridor_dir = os.path.join(output_dir, f"corridor_{corridor_length}")
        os.makedirs(corridor_dir, exist_ok=True)
        
        for i in range(num_seeds):
            seed = base_seed + i
            print(f"\nTraining with seed {seed} ({i+1}/{num_seeds})")
            
            # Set the seed for reproducibility
            set_seed(seed)
            
            # Save path for this specific model
            save_path = os.path.join(corridor_dir, f"model_seed_{seed}.pt")
            
            # Train the model - calculate episodes based on max_time_steps
            # We assume average of 100 steps per episode, but will stop by time steps
            estimated_episodes = max_time_steps // 100 + 500  # Add buffer
            
            train_stream_q(
                episodes=estimated_episodes,
                corridor_length=corridor_length,
                save_path=save_path,
                max_time_steps=max_time_steps
            )
            
            # Save the seed value separately for easier reference
            seed_info = {
                "seed": seed,
                "corridor_length": corridor_length,
                "max_time_steps": max_time_steps
            }
            torch.save(seed_info, os.path.join(corridor_dir, f"seed_info_{seed}.pt"))
            
        print(f"Completed training for corridor length {corridor_length}")

def run_hidden_size_experiments(corridor_length=100,
                               hidden_sizes=[32, 64, 128],
                               num_seeds=10,
                               base_seed=42,
                               max_time_steps=1000000,
                               output_dir="finalResults/RTRL_hidden_sizes"):
    """
    Train models with different hidden sizes using multiple seeds.
    
    Args:
        corridor_length: Fixed corridor length for this experiment
        hidden_sizes: List of hidden sizes to test
        num_seeds: Number of seeds to use for each hidden size
        base_seed: Base seed value to start from
        max_time_steps: Maximum number of time steps for training
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for hidden_size in hidden_sizes:
        print(f"\n{'='*50}")
        print(f"Training with hidden size: {hidden_size}")
        print(f"{'='*50}")
        
        hidden_dir = os.path.join(output_dir, f"hidden_size_{hidden_size}")
        os.makedirs(hidden_dir, exist_ok=True)
        
        for i in range(num_seeds):
            seed = base_seed + i
            print(f"\nTraining with seed {seed} ({i+1}/{num_seeds})")
            
            # Set the seed for reproducibility
            set_seed(seed)
            
            # Save path for this specific model
            save_path = os.path.join(hidden_dir, f"model_seed_{seed}.pt")
            
            # Train the model with specific hidden size
            # We estimate episodes based on max_time_steps
            estimated_episodes = max_time_steps // 100 + 500  # Add buffer
            
            train_stream_q(
                episodes=estimated_episodes,
                corridor_length=corridor_length,
                hidden_size=hidden_size,
                save_path=save_path,
                max_time_steps=max_time_steps
            )
            
            # Save the configuration info
            config_info = {
                "seed": seed,
                "corridor_length": corridor_length,
                "hidden_size": hidden_size,
                "max_time_steps": max_time_steps
            }
            torch.save(config_info, os.path.join(hidden_dir, f"config_info_{seed}.pt"))
            
        print(f"Completed training for hidden size {hidden_size}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with multiple seeds and configurations")
    parser.add_argument("--experiment", type=str, choices=["corridor", "hidden", "both"], default="both",
                        help="Which experiment to run: corridor lengths, hidden sizes, or both")
    parser.add_argument("--corridor_lengths", type=int, nargs='+', default=[10, 100, 1000],
                        help="Corridor lengths to test")
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[32, 64, 128],
                        help="Hidden sizes to test")
    parser.add_argument("--fixed_corridor", type=int, default=100,
                        help="Fixed corridor length for hidden size experiments")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of seeds per configuration")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed to start from")
    parser.add_argument("--max_time_steps", type=int, default=1000000,
                        help="Maximum number of time steps for training")
    
    args = parser.parse_args()
    
    if args.experiment in ["corridor", "both"]:
        run_corridor_length_experiments(
            corridor_lengths=args.corridor_lengths,
            num_seeds=args.num_seeds,
            base_seed=args.base_seed,
            max_time_steps=args.max_time_steps
        )
    
    if args.experiment in ["hidden", "both"]:
        run_hidden_size_experiments(
            corridor_length=args.fixed_corridor,
            hidden_sizes=args.hidden_sizes,
            num_seeds=args.num_seeds,
            base_seed=args.base_seed,
            max_time_steps=args.max_time_steps
        )