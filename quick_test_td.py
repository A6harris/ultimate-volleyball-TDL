import os
import argparse
from standalone_td_learning import train_volleyball_td

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quick test for TD Learning implementation')
    parser.add_argument('--env-path', type=str, required=True, 
                        help='Path to Unity environment executable')
    parser.add_argument('--time-scale', type=float, default=20.0,
                        help='Time scale for Unity simulation (higher = faster)')
    
    args = parser.parse_args()
    
    # Create config directory if it doesn't exist
    os.makedirs('config', exist_ok=True)
    
    # Run a very short training session
    train_volleyball_td(
        env_path=args.env_path,
        config_path='config/volleyball_td_quick.yaml',
        run_id='quick_test',
        output_dir='results',
        num_episodes=20,       # Very few episodes
        max_steps=200,         # Short episodes
        time_scale=args.time_scale
    )
    
    print("\nQuick test completed! Check the results directory for outputs.")
    print("If no errors occurred, the TD learning implementation is working correctly.")
    print("\nTo run a full training session with more episodes, use:")
    print(f"python standalone_td_learning.py --env-path={args.env_path} --config=config/volleyball_td_quick.yaml --time-scale={args.time_scale}")