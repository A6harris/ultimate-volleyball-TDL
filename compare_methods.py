"""
Script to compare TD Learning and PPO training results using TensorBoard data.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from typing import Dict, List, Tuple, Optional

def extract_tensorboard_data(log_dir: str, tag: str) -> Tuple[List[float], List[float]]:
    """
    Extract data for a specific tag from TensorBoard logs
    
    Args:
        log_dir: Path to the TensorBoard log directory
        tag: Tag to extract data for
        
    Returns:
        Tuple of (steps, values)
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    if tag not in ea.Tags()['scalars']:
        print(f"Tag '{tag}' not found in {log_dir}")
        return [], []
    
    events = ea.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    
    return steps, values

def smooth_data(values: List[float], weight: float = 0.85) -> List[float]:
    """
    Apply exponential smoothing to the data
    
    Args:
        values: List of values to smooth
        weight: Smoothing weight (0 = no smoothing, 1 = flat line)
        
    Returns:
        List of smoothed values
    """
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_comparison(
    td_dir: str, 
    ppo_dir: str, 
    tag: str, 
    title: str, 
    output_file: Optional[str] = None,
    smoothing: float = 0.85
):
    """
    Plot a comparison between TD and PPO for a specific metric
    
    Args:
        td_dir: Path to TD TensorBoard logs
        ppo_dir: Path to PPO TensorBoard logs
        tag: Tag to extract and compare
        title: Plot title
        output_file: Path to save the plot (if None, plot is displayed)
        smoothing: Smoothing factor (0 = none, 1 = flat line)
    """
    # Extract data
    td_steps, td_values = extract_tensorboard_data(td_dir, tag)
    ppo_steps, ppo_values = extract_tensorboard_data(ppo_dir, tag)
    
    if not td_values or not ppo_values:
        print(f"Could not extract data for tag '{tag}'")
        return
    
    # Apply smoothing
    if smoothing > 0:
        td_values_smooth = smooth_data(td_values, smoothing)
        ppo_values_smooth = smooth_data(ppo_values, smoothing)
    else:
        td_values_smooth = td_values
        ppo_values_smooth = ppo_values
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot raw data with low alpha
    plt.plot(td_steps, td_values, 'b-', alpha=0.2)
    plt.plot(ppo_steps, ppo_values, 'r-', alpha=0.2)
    
    # Plot smoothed data
    plt.plot(td_steps, td_values_smooth, 'b-', linewidth=2, label='TD Learning')
    plt.plot(ppo_steps, ppo_values_smooth, 'r-', linewidth=2, label='PPO')
    
    plt.title(title)
    plt.xlabel('Training Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {output_file}")
    else:
        plt.show()

def compare_training_methods(
    td_run_id: str,
    ppo_run_id: str,
    results_dir: str = './results',
    output_dir: Optional[str] = None,
    environment: str = 'Volleyball',
    smoothing: float = 0.85
):
    """
    Compare training results between TD Learning and PPO
    
    Args:
        td_run_id: Run ID for TD Learning
        ppo_run_id: Run ID for PPO
        results_dir: Directory containing TensorBoard logs
        output_dir: Directory to save plots (if None, plots are displayed)
        environment: Environment name
        smoothing: Smoothing factor for plots
    """
    td_dir = os.path.join(results_dir, td_run_id)
    ppo_dir = os.path.join(results_dir, ppo_run_id)
    
    if not os.path.exists(td_dir):
        raise ValueError(f"TD run directory not found: {td_dir}")
    if not os.path.exists(ppo_dir):
        raise ValueError(f"PPO run directory not found: {ppo_dir}")
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define metrics to compare
    metrics = [
        {
            "tag": f"{environment}.Environment/Cumulative Reward",
            "title": "Cumulative Reward",
            "filename": "cumulative_reward.png"
        },
        {
            "tag": f"{environment}.Environment/Episode Length",
            "title": "Episode Length",
            "filename": "episode_length.png"
        },
        {
            "tag": f"{environment}.Losses/Value Loss",
            "title": "Value Loss",
            "filename": "value_loss.png"
        },
        {
            "tag": f"{environment}.Policy/Entropy",
            "title": "Policy Entropy",
            "filename": "policy_entropy.png"
        },
        {
            "tag": f"{environment}.Policy/Learning Rate",
            "title": "Learning Rate",
            "filename": "learning_rate.png"
        }
    ]
    
    # TD-specific metrics
    td_metrics = [
        {
            "tag": f"{environment}.Policy/Epsilon",
            "title": "Exploration Rate (Epsilon)",
            "filename": "epsilon.png"
        }
    ]
    
    # Generate comparison plots for common metrics
    for metric in metrics:
        output_file = os.path.join(output_dir, metric["filename"]) if output_dir else None
        plot_comparison(
            td_dir, 
            ppo_dir, 
            metric["tag"], 
            metric["title"], 
            output_file,
            smoothing
        )
    
    # Generate plots for TD-specific metrics
    if output_dir:
        for metric in td_metrics:
            td_steps, td_values = extract_tensorboard_data(td_dir, metric["tag"])
            if not td_values:
                continue
                
            plt.figure(figsize=(12, 6))
            plt.plot(td_steps, td_values, 'b-', alpha=0.2)
            
            # Apply smoothing
            if smoothing > 0:
                td_values_smooth = smooth_data(td_values, smoothing)
                plt.plot(td_steps, td_values_smooth, 'b-', linewidth=2, label='TD Learning')
            
            plt.title(metric["title"])
            plt.xlabel('Training Steps')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
            
            output_file = os.path.join(output_dir, metric["filename"])
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare TD Learning and PPO training results')
    parser.add_argument('--td-run-id', type=str, required=True, help='Run ID for TD Learning')
    parser.add_argument('--ppo-run-id', type=str, required=True, help='Run ID for PPO')
    parser.add_argument('--results-dir', type=str, default='./results', help='Directory containing TensorBoard logs')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save plots (if None, plots are displayed)')
    parser.add_argument('--environment', type=str, default='Volleyball', help='Environment name')
    parser.add_argument('--smoothing', type=float, default=0.85, help='Smoothing factor for plots')
    
    args = parser.parse_args()
    
    compare_training_methods(
        td_run_id=args.td_run_id,
        ppo_run_id=args.ppo_run_id,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        environment=args.environment,
        smoothing=args.smoothing
    )