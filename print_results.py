#!/usr/bin/env python3
"""
Print Detailed Experiment Results
Displays experiment results in the format shown in the terminal images
"""

import json
import numpy as np
import argparse
from pathlib import Path

def print_detailed_results(results_file, max_samples_to_show=None):
    """Print detailed experiment results from JSON file"""
    
    # Load results
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON file: {results_file}")
        return
    
    # Get summary and detailed results
    summary = data.get('summary', {})
    results = data.get('detailed_results', [])
    
    if not results:
        print("❌ No detailed results found in file")
        return
    
    # Print summary first
    print(f"\n📈 EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total samples: {summary.get('total_samples', len(results))}")
    print(f"Total predictions: {summary.get('total_predictions', len(results))}")
    print(f"Average MAE: {summary.get('avg_mae', 0):.6f}")
    print(f"Average MSE: {summary.get('avg_mse', 0):.6f}")
    print(f"Task completion rate: {summary.get('task_completion_rate', 0):.1f}%")
    print(f"Task success count: {summary.get('task_success_count', 0)}")
    print()
    
    # Print detailed results
    print(f"📊 DETAILED EXPERIMENT RESULTS")
    print("=" * 80)
    
    if max_samples_to_show is None:
        max_samples_to_show = len(results)
    
    # Show results for each sample
    current_sample = None
    sample_count = 0
    
    for i, result in enumerate(results):
        if sample_count >= max_samples_to_show:
            break
            
        # Extract trajectory ID from image path or use sample number
        traj_id = result.get('trajectory_id', f'traj{result["sample"]}')
        
        # Start new sample block
        if current_sample != result['sample']:
            if current_sample is not None:
                print()  # Add space between samples
            
            current_sample = result['sample']
            sample_count += 1
            print(f"Sample {sample_count}: {traj_id}")
            print(f"Instruction: {result['instruction']}")
            print()
        
        # Calculate metrics for this result
        pred = np.array(result['openvla_prediction'])
        gt = np.array(result['ground_truth'])
        
        mae = np.mean(np.abs(pred - gt))
        mse = np.mean((pred - gt) ** 2)
        
        # Determine task success (MAE < 0.1 as threshold)
        task_success = mae < 0.1
        success_symbol = "✔" if task_success else "❌"
        
        # Extract image name from path
        image_name = result['image_path'].split('/')[-1]
        
        # Get prediction time if available
        pred_time = result.get('prediction_time', 0.0)
        
        print(f"Image {i % 2 + 1}: {image_name}")
        print(f"  Ground Truth: [{', '.join([f'{x:.4f}' for x in gt])}]")
        print(f"  Predicted:    [{', '.join([f'{x:.4f}' for x in pred])}]")
        print(f"  MAE: {mae:.4f}, MSE: {mse:.4f}, Task Success: {success_symbol}")
        if pred_time > 0:
            print(f"  Time: {pred_time:.3f}s")
        print()
    
    if len(results) > max_samples_to_show:
        print(f"... and {len(results) - max_samples_to_show} more samples")
    
    print("=" * 80)
    
    # Additional statistics
    successful_tasks = sum(1 for r in results if np.mean(np.abs(np.array(r['openvla_prediction']) - np.array(r['ground_truth']))) < 0.1)
    success_rate = (successful_tasks / len(results)) * 100
    
    print(f"\n📊 ADDITIONAL STATISTICS")
    print("=" * 40)
    print(f"Task success rate (MAE < 0.1): {success_rate:.1f}% ({successful_tasks}/{len(results)})")
    
    # Per-dimension analysis
    if results:
        predictions = np.array([r['openvla_prediction'] for r in results])
        ground_truths = np.array([r['ground_truth'] for r in results])
        
        per_dim_mae = np.mean(np.abs(predictions - ground_truths), axis=0)
        dimension_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
        
        print(f"\nPer-dimension MAE:")
        for i, (name, mae) in enumerate(zip(dimension_names, per_dim_mae)):
            print(f"  {name:<8}: {mae:.4f}")

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Print detailed experiment results")
    parser.add_argument('results_file', type=str, help="JSON file containing experiment results")
    parser.add_argument('--max_samples', type=int, help="Maximum samples to show (default: all)")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.results_file).exists():
        print(f"❌ File not found: {args.results_file}")
        print("Available JSON files in current directory:")
        for json_file in Path('.').glob('*.json'):
            print(f"  - {json_file}")
        return
    
    print_detailed_results(args.results_file, args.max_samples)

if __name__ == "__main__":
    main()
