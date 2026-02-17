#!/usr/bin/env python3
"""
Augmented Neural VLA predictions from enhanced run
Includes both baseline and enhanced predictions for comparison
"""

import numpy as np
import argparse

# Enhanced predictions from Augmented Neural VLA
# Each entry contains: enhanced, baseline, ground_truth
HARDCODED_PREDICTIONS = [{'predicted': [-0.016524696722626686, -0.01545649953186512, 0.028252970427274704, 0.029322370886802673, -0.015385311096906662, 0.25418347120285034, 1.1563936471939087], 'baseline_predicted': [-0.0083, 0.0177, 0.038, 0.042, -0.0611, 0.0999, 0.0], 'ground_truth': [-0.0291, 0.0698, 0.0007, -0.0083, 0.0017, -0.6787, 0.0609], 'sample': 1, 'timestep': 0}, {'predicted': [-0.01601656712591648, -0.0059630777686834335, 0.017300423234701157, 0.04179627448320389, -0.022566869854927063, -0.20318609476089478, 1.191200613975525], 'baseline_predicted': [-0.0074, 0.0119, 0.04, 0.0439, -0.0926, 0.0999, 0.0], 'ground_truth': [-0.0321, 0.0641, -0.0044, 0.0002, 0.0026, -0.6839, 0.9952], 'sample': 1, 'timestep': 1}, {'predicted': [0.00267385714687407, -0.005286495666950941, -0.009743956848978996, 0.0070177847519516945, 0.001667574979364872, 0.35449931025505066, 0.6977600455284119], 'baseline_predicted': [-0.0029, -0.005, -0.0153, 0.0158, 0.0133, -0.0017, 0.9961], 'ground_truth': [-0.003, -0.0366, 0.0037, -0.0035, 0.0072, 0.3568, 0.9934], 'sample': 2, 'timestep': 0}, {'predicted': [-0.003144026268273592, -0.008200408890843391, 0.0008291956037282944, 0.02100050449371338, -0.00586249865591526, -0.22436906397342682, 0.7474969625473022], 'baseline_predicted': [0.0025, 0.0015, -0.0057, 0.0145, 0.0059, 0.0112, 0.0], 'ground_truth': [-0.0021, -0.0359, -0.0042, -0.0047, 0.0044, 0.3398, 0.9956], 'sample': 2, 'timestep': 1}, {'predicted': [-0.007266503758728504, -0.018520396202802658, -0.005311693996191025, 0.0077225202694535255, 0.000341854989528656, 0.329276978969574, 0.730478048324585], 'baseline_predicted': [-0.0132, -0.0173, -0.0106, 0.0177, 0.0106, 0.0128, 0.9961], 'ground_truth': [-0.0059, 0.0033, 0.001, -0.0062, -0.0001, 0.4447, 1.0], 'sample': 3, 'timestep': 0}, {'predicted': [0.0024681114591658115, -0.0005087885656394064, -0.007715742103755474, -0.002866793656721711, 0.029171153903007507, 0.42260876297950745, 0.6814108490943909], 'baseline_predicted': [-0.0022, -0.0011, -0.0156, 0.0037, 0.0441, -0.0001, 0.9961], 'ground_truth': [-0.0011, 0.0052, -0.0267, -0.0026, 0.008, 0.4295, 1.0], 'sample': 3, 'timestep': 1}, {'predicted': [-0.003081837436184287, -0.006526961922645569, 0.004553989041596651, 0.01959596388041973, -0.0007773507386445999, -0.25510239601135254, 0.7333161234855652], 'baseline_predicted': [0.0023, 0.0018, -0.0008, 0.0145, 0.0086, 0.0112, 0.0], 'ground_truth': [-0.0243, -0.0485, -0.0007, -0.0077, -0.0003, 0.7667, 0.9955], 'sample': 4, 'timestep': 0}, {'predicted': [0.002673448994755745, -0.0005611327942460775, 0.00010804552584886551, 0.02683238498866558, -0.0023309739772230387, -0.04293181747198105, 0.5365215539932251], 'baseline_predicted': [0.0023, 0.0018, -0.0023, 0.0145, -0.0015, -0.0081, 0.0], 'ground_truth': [-0.0284, -0.0369, -0.0021, -0.0025, 0.0, 0.7629, 0.9945], 'sample': 4, 'timestep': 1}, {'predicted': [0.0001222172286361456, 0.022279253229498863, 0.009555881842970848, 0.013256305828690529, -0.02527151256799698, -0.3492080867290497, 0.7004721164703369], 'baseline_predicted': [0.0034, 0.0122, 0.0218, 0.0037, -0.0537, 0.0128, 0.0], 'ground_truth': [-0.0076, 0.0858, 0.0006, 0.0033, -0.0, -0.6684, 1.0], 'sample': 5, 'timestep': 0}]

def get_hardcoded_data():
    """Get enhanced predictions and ground truths"""
    enhanced_predictions = [p['predicted'] for p in HARDCODED_PREDICTIONS]
    baseline_predictions = [p['baseline_predicted'] for p in HARDCODED_PREDICTIONS]
    ground_truths = [p['ground_truth'] for p in HARDCODED_PREDICTIONS]
    return enhanced_predictions, baseline_predictions, ground_truths

def get_enhanced_only():
    """Get only enhanced predictions (like original baseline format)"""
    enhanced_predictions = [p['predicted'] for p in HARDCODED_PREDICTIONS]
    ground_truths = [p['ground_truth'] for p in HARDCODED_PREDICTIONS]
    return enhanced_predictions, ground_truths

def get_baseline_only():
    """Get only baseline predictions for direct comparison"""
    baseline_predictions = [p['baseline_predicted'] for p in HARDCODED_PREDICTIONS]
    ground_truths = [p['ground_truth'] for p in HARDCODED_PREDICTIONS]
    return baseline_predictions, ground_truths

def compare_performance():
    """Compare enhanced vs baseline performance"""
    enhanced_preds, baseline_preds, gts = get_hardcoded_data()
    
    enhanced_maes = []
    baseline_maes = []
    
    for enhanced_pred, baseline_pred, gt in zip(enhanced_preds, baseline_preds, gts):
        enhanced_mae = np.mean(np.abs(np.array(enhanced_pred) - np.array(gt)))
        baseline_mae = np.mean(np.abs(np.array(baseline_pred) - np.array(gt)))
        
        enhanced_maes.append(enhanced_mae)
        baseline_maes.append(baseline_mae)
    
    print("🏆 Performance Comparison:")
    print(f"   Baseline MAE: {np.mean(baseline_maes):.4f} ± {np.std(baseline_maes):.4f}")
    print(f"   Enhanced MAE: {np.mean(enhanced_maes):.4f} ± {np.std(enhanced_maes):.4f}")
    
    improvement = (np.mean(baseline_maes) - np.mean(enhanced_maes)) / np.mean(baseline_maes) * 100
    print(f"   Improvement: {improvement:.2f}%")
    
    return {
        'baseline_mae': np.mean(baseline_maes),
        'enhanced_mae': np.mean(enhanced_maes),
        'improvement_pct': improvement
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Augmented Neural VLA Results')
    parser.add_argument('--info', action='store_true', help='Show dataset info')
    parser.add_argument('--compare', action='store_true', help='Compare enhanced vs baseline')
    parser.add_argument('--enhanced-only', action='store_true', help='Show enhanced predictions only')
    parser.add_argument('--baseline-only', action='store_true', help='Show baseline predictions only')
    args = parser.parse_args()
    
    if args.compare:
        compare_performance()
        return
    
    enhanced_preds, baseline_preds, gts = get_hardcoded_data()
    
    if args.info:
        print(f"Total samples: {len(enhanced_preds)}")
        print(f"Action dimension: {len(enhanced_preds[0]) if enhanced_preds else 0}")
        
        # Show comparison
        enhanced_maes = []
        baseline_maes = []
        for enhanced_pred, baseline_pred, gt in zip(enhanced_preds, baseline_preds, gts):
            enhanced_maes.append(np.mean(np.abs(np.array(enhanced_pred) - np.array(gt))))
            baseline_maes.append(np.mean(np.abs(np.array(baseline_pred) - np.array(gt))))
        
        print(f"Baseline MAE: {np.mean(baseline_maes):.4f} ± {np.std(baseline_maes):.4f}")
        print(f"Enhanced MAE: {np.mean(enhanced_maes):.4f} ± {np.std(enhanced_maes):.4f}")
        
    elif args.enhanced_only:
        enhanced_preds, gts = get_enhanced_only()
        print(f"Enhanced predictions only: {len(enhanced_preds)} samples")
        
    elif args.baseline_only:
        baseline_preds, gts = get_baseline_only()
        print(f"Baseline predictions only: {len(baseline_preds)} samples")
        
    else:
        # Default: show first few samples
        print("Sample data (first 3 entries):")
        for i, entry in enumerate(HARDCODED_PREDICTIONS[:3]):
            print(f"Sample {entry['sample']}, Timestep {entry['timestep']}:")
            print(f"  Enhanced:   {entry['predicted']}")
            print(f"  Baseline:   {entry['baseline_predicted']}") 
            print(f"  Ground Truth: {entry['ground_truth']}")
            print()

if __name__ == "__main__":
    main()
