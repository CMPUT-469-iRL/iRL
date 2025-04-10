import subprocess
import argparse
import os
from plot_comparison import plot_comparison, load_results

def run_experiment(corridor_length, episodes, lr, hidden_size, embedding_size, max_steps):
    """Run an experiment with a specific corridor length."""
    command = [
        "python3", "streamQ.py",
        "--env_type", "tmaze",
        "--episodes", str(episodes),
        "--lr", str(lr),
        "--hidden_size", str(hidden_size),
        "--embedding_size", str(embedding_size),
        "--corridor_length", str(corridor_length),
        "--max_steps", str(max_steps)
    ]
    
    print(f"\n\n{'='*50}")
    print(f"Running experiment with corridor length {corridor_length}")
    print(f"{'='*50}\n")
    
    # Create save directory if it doesn't exist
    os.makedirs("save_models", exist_ok=True)
    
    # Run the command
    subprocess.run(command)

def main():
    parser = argparse.ArgumentParser(description="Run multiple experiments with different corridor lengths")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes per experiment")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of LSTM")
    parser.add_argument("--embedding_size", type=int, default=16, help="Embedding size")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--lengths", type=str, default="5,70,80", 
                       help="Comma-separated list of corridor lengths to run")
    args = parser.parse_args()
    
    # Parse corridor lengths from command line
    try:
        corridor_lengths = [int(x.strip()) for x in args.lengths.split(',')]
    except:
        print("Error parsing corridor lengths. Using defaults.")
        corridor_lengths = [5, 70, 80]
        
    print(f"Running experiments for corridor lengths: {corridor_lengths}")
    
    for corridor_length in corridor_lengths:
        run_experiment(
            corridor_length=corridor_length, 
            episodes=args.episodes,
            lr=args.lr,
            hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            max_steps=args.max_steps
        )
    
    # Plot comparison
    print("\nGenerating comparison plot...")
    
    # Load results
    results_dict = {}
    for length in corridor_lengths:
        model_path = f"save_models/stream_q_model_{length}.pt"
        if os.path.exists(model_path):
            print(f"Loading results for corridor length {length}")
            results_dict[length] = load_results(model_path)
        else:
            print(f"No results found for corridor length {length}")
    
    # Plot comparison
    if results_dict:
        plot_comparison(results_dict)
        print("Plot saved as corridor_length_comparison.png")
    else:
        print("No results found to plot.")

if __name__ == "__main__":
    main() 