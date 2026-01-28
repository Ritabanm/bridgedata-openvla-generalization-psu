#!/usr/bin/env python3
"""
Compare Augmented Neural VLA results with OpenVLA baseline
"""

import json
import numpy as np
import argparse

def load_baseline_results():
    """Load baseline results from OpenVLA evaluation"""
    try:
        with open("working_evaluation_results.json", 'r') as f:
            content = f.read()
            # Try to fix truncated JSON by removing incomplete entries
            if not content.strip().endswith('}'):
                # Find the last complete entry
                lines = content.split('\n')
                for i in range(len(lines)-1, -1, -1):
                    if lines[i].strip().endswith('}') and i > 0:
                        content = '\n'.join(lines[:i+1])
                        if not content.strip().endswith('}'):
                            content += '\n}'
                        break
            return json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  OpenVLA baseline results not found or corrupted: {e}")
        return None

def load_augmented_results():
    """Load augmented neural VLA results"""
    try:
        with open("augmented_neural_vla_evaluation_results.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  Augmented Neural VLA results not found (augmented_neural_vla_evaluation_results.json)")
        return None

def print_comparison(baseline_results, augmented_results):
    """Print side-by-side comparison"""
    print("🏆 OPENVLA BASELINE vs AUGMENTED NEURAL VLA COMPARISON")
    print("=" * 70)
    
    if baseline_results and 'summary' in baseline_results:
        baseline = baseline_results['summary']
        print("\n📊 OpenVLA Baseline Results:")
        print(f"   MAE:  {baseline['avg_mae']:.4f} ± {baseline['std_mae']:.4f}")
        print(f"   MSE:  {baseline['avg_mse']:.4f} ± {baseline['std_mse']:.4f}")
        print(f"   Task Completion Rate: {baseline['task_completion_rate']:.1f}%")
        print(f"   Total Predictions: {baseline['total_predictions']}")
    
    if augmented_results:
        aug = augmented_results
        print(f"\n🚀 Augmented Neural VLA Results:")
        print(f"   Baseline MAE:  {aug['baseline']['avg_mae']:.4f} ± {aug['baseline']['std_mae']:.4f}")
        print(f"   Enhanced MAE:  {aug['enhanced']['avg_mae']:.4f} ± {aug['enhanced']['std_mae']:.4f}")
        print(f"   Baseline MSE:  {aug['baseline']['avg_mse']:.4f} ± {aug['baseline']['std_mse']:.4f}")
        print(f"   Enhanced MSE:  {aug['enhanced']['avg_mse']:.4f} ± {aug['enhanced']['std_mse']:.4f}")
        print(f"   Baseline Success Rate: {aug['baseline']['success_rate']:.1f}%")
        print(f"   Enhanced Success Rate: {aug['enhanced']['success_rate']:.1f}%")
        print(f"   Samples Evaluated: {aug['samples_evaluated']}")
        print(f"   Used Augmentation: {aug['use_augmentation']}")
        
        print(f"\n🎯 Augmented Neural VLA Improvements:")
        print(f"   MAE Improvement: {aug['improvements']['mae_improvement']:.4f}")
        print(f"   MSE Improvement: {aug['improvements']['mse_improvement']:.4f}")
        print(f"   MAE Improvement %: {aug['improvements']['mae_improvement_pct']:.2f}%")
        print(f"   Success Rate Improvement: {aug['improvements']['success_rate_improvement']:.1f}%")
    
    # Direct comparison if both available
    if baseline_results and augmented_results and 'summary' in baseline_results:
        baseline = baseline_results['summary']
        aug = augmented_results
        
        print(f"\n⚖️  Direct Comparison:")
        print(f"   OpenVLA Baseline MAE:    {baseline['avg_mae']:.4f}")
        print(f"   Augmented VLA Enhanced: {aug['enhanced']['avg_mae']:.4f}")
        
        if baseline['avg_mae'] > 0:
            direct_improvement = (baseline['avg_mae'] - aug['enhanced']['avg_mae']) / baseline['avg_mae'] * 100
            print(f"   Direct Improvement:      {direct_improvement:.2f}%")
        
        print(f"   OpenVLA Baseline Success: {baseline['task_completion_rate']:.1f}%")
        print(f"   Augmented VLA Success:     {aug['enhanced']['success_rate']:.1f}%")
        
        success_improvement = aug['enhanced']['success_rate'] - baseline['task_completion_rate']
        print(f"   Success Improvement:       {success_improvement:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Compare OpenVLA baseline with Augmented Neural VLA')
    parser.add_argument('--baseline', type=str, default='working_evaluation_results.json', 
                       help='Path to baseline results file')
    parser.add_argument('--augmented', type=str, default='augmented_neural_vla_evaluation_results.json',
                       help='Path to augmented results file')
    
    args = parser.parse_args()
    
    # Load results
    baseline_results = load_baseline_results()
    augmented_results = load_augmented_results()
    
    if not baseline_results and not augmented_results:
        print("❌ No result files found!")
        print("   Make sure to run both:")
        print("   1. OpenVLA baseline: python openvla-baseline.py")
        print("   2. Augmented VLA: python augmented_neural_vla.py --evaluate")
        return
    
    # Print comparison
    print_comparison(baseline_results, augmented_results)
    
    print(f"\n📁 Files used:")
    if baseline_results:
        print(f"   Baseline: {args.baseline}")
    if augmented_results:
        print(f"   Augmented: {args.augmented}")

if __name__ == "__main__":
    main()
