#!/usr/bin/env python3
"""
Final evaluation report comparing multimodal augmented neural net vs OpenVLA baseline
Analyzes improvement on held-out episodes and generates comprehensive report
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def load_all_results():
    """Load all evaluation results"""
    print("📂 Loading all evaluation results...")
    
    results = {}
    
    # Load baseline results
    if os.path.exists("baseline_100_samples_results.json"):
        with open("baseline_100_samples_results.json", 'r') as f:
            results['baseline'] = json.load(f)
        print("✅ Loaded baseline results")
    
    # Load multimodal training data
    if os.path.exists("multimodal_training_data.json"):
        with open("multimodal_training_data.json", 'r') as f:
            results['training_data'] = json.load(f)
        print("✅ Loaded training data")
    
    # Load enhancer results
    if os.path.exists("multimodal_enhancer_results.json"):
        with open("multimodal_enhancer_results.json", 'r') as f:
            results['enhancer'] = json.load(f)
        print("✅ Loaded enhancer results")
    
    return results

def analyze_baseline_performance(baseline_data):
    """Analyze baseline OpenVLA performance"""
    print("\n📊 Baseline OpenVLA Performance Analysis")
    print("=" * 50)
    
    summary = baseline_data['summary']
    detailed = baseline_data['detailed_results']
    
    print(f"📈 Overall Baseline Metrics:")
    print(f"   Total Samples: {summary['total_samples']}")
    print(f"   Total Predictions: {summary['total_predictions']}")
    print(f"   MAE: {summary['avg_mae']:.6f} ± {summary['std_mae']:.6f}")
    print(f"   MSE: {summary['avg_mse']:.6f} ± {summary['std_mse']:.6f}")
    print(f"   Task Completion Rate: {summary['task_completion_rate']:.1f}%")
    
    # Per-dimension analysis
    predictions = np.array([r['openvla_prediction'] for r in detailed])
    ground_truths = np.array([r['ground_truth'] for r in detailed])
    
    print(f"\n📊 Per-Dimension Baseline Errors:")
    dim_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
    
    for i, name in enumerate(dim_names):
        mae = np.mean(np.abs(predictions[:, i] - ground_truths[:, i]))
        std = np.std(np.abs(predictions[:, i] - ground_truths[:, i]))
        print(f"   {name}: {mae:.4f} ± {std:.4f}")
    
    return {
        'summary': summary,
        'per_dimension': {
            dim_names[i]: {
                'mae': float(np.mean(np.abs(predictions[:, i] - ground_truths[:, i]))),
                'std': float(np.std(np.abs(predictions[:, i] - ground_truths[:, i])))
            } for i in range(7)
        }
    }

def analyze_training_split(training_data):
    """Analyze training/test split quality"""
    print("\n🔧 Training Data Split Analysis")
    print("=" * 40)
    
    train_pairs = training_data['train_pairs']
    test_pairs = training_data['test_pairs']
    metadata = training_data['metadata']
    
    print(f"📊 Data Split Summary:")
    print(f"   Training pairs: {len(train_pairs)} (from {len(set(p['sample'] for p in train_pairs))} samples)")
    print(f"   Test pairs: {len(test_pairs)} (from {len(set(p['sample'] for p in test_pairs))} samples)")
    
    # Get augmentation factor safely
    aug_factor = metadata.get('augment_factor', 'unknown')
    print(f"   Augmentation factor: {aug_factor}")
    print(f"   Augmented training pairs: {metadata.get('total_train_augmented', 'unknown')}")
    print(f"   Augmented test pairs: {metadata.get('total_test_augmented', 'unknown')}")
    
    # Compare baseline performance between train and test
    train_mae = [p['mae'] for p in train_pairs]
    test_mae = [p['mae'] for p in test_pairs]
    
    print(f"\n📈 Baseline MAE Comparison:")
    print(f"   Training set: {np.mean(train_mae):.4f} ± {np.std(train_mae):.4f}")
    print(f"   Test set: {np.mean(test_mae):.4f} ± {np.std(test_mae):.4f}")
    
    # Check for data leakage
    train_samples = set(p['sample'] for p in train_pairs)
    test_samples = set(p['sample'] for p in test_pairs)
    overlap = train_samples.intersection(test_samples)
    
    if overlap:
        print(f"⚠️  WARNING: Data leakage detected! {len(overlap)} samples appear in both sets")
    else:
        print(f"✅ No data leakage - train/test samples are properly separated")
    
    return {
        'train_size': len(train_pairs),
        'test_size': len(test_pairs),
        'train_samples': len(set(p['sample'] for p in train_pairs)),
        'test_samples': len(set(p['sample'] for p in test_pairs)),
        'augmentation_factor': metadata.get('augment_factor', 5),
        'train_mae_mean': float(np.mean(train_mae)),
        'train_mae_std': float(np.std(train_mae)),
        'test_mae_mean': float(np.mean(test_mae)),
        'test_mae_std': float(np.std(test_mae)),
        'data_leakage': len(overlap) > 0
    }

def analyze_enhancer_performance(enhancer_data):
    """Analyze multimodal enhancer performance"""
    print("\n🚀 Multimodal Enhancer Performance Analysis")
    print("=" * 50)
    
    single_split = enhancer_data['single_split_results']
    cv_results = enhancer_data['cross_validation_results']
    
    print(f"📈 Single Split Results:")
    print(f"   Baseline MAE: {single_split['baseline_mae']:.6f}")
    print(f"   Enhanced MAE: {single_split['enhanced_mae']:.6f}")
    print(f"   MAE Improvement: {single_split['mae_improvement_percent']:.2f}%")
    print(f"   MSE Improvement: {single_split['mse_improvement_percent']:.2f}%")
    
    print(f"\n📊 Per-Dimension Improvements (Single Split):")
    for dim, metrics in single_split['per_dimension'].items():
        improvement = metrics['improvement_percent']
        arrow = "↑" if improvement > 0 else "↓"
        print(f"   {dim}: {metrics['baseline_mae']:.4f} → {metrics['enhanced_mae']:.4f} ({arrow}{improvement:+.1f}%)")
    
    print(f"\n🔄 Cross-Validation Results:")
    cv_improvements = [r['mae_improvement_percent'] for r in cv_results]
    print(f"   Average MAE Improvement: {np.mean(cv_improvements):.2f}% ± {np.std(cv_improvements):.2f}%")
    
    for i, result in enumerate(cv_results):
        print(f"   Fold {i+1}: {result['mae_improvement_percent']:.2f}%")
    
    # Analyze consistency
    positive_improvements = [r for r in cv_improvements if r > 0]
    print(f"\n📊 Consistency Analysis:")
    print(f"   Folds with positive improvement: {len(positive_improvements)}/{len(cv_results)} ({len(positive_improvements)/len(cv_results)*100:.1f}%)")
    print(f"   Best fold improvement: {max(cv_improvements):.2f}%")
    print(f"   Worst fold improvement: {min(cv_improvements):.2f}%")
    
    return {
        'single_split': single_split,
        'cross_validation': {
            'mean_improvement': float(np.mean(cv_improvements)),
            'std_improvement': float(np.std(cv_improvements)),
            'positive_improvement_rate': len(positive_improvements)/len(cv_results),
            'best_improvement': float(max(cv_improvements)),
            'worst_improvement': float(min(cv_improvements)),
            'fold_results': cv_results
        }
    }

def generate_comparison_plots(baseline_analysis, enhancer_analysis):
    """Generate comparison plots"""
    print("\n📊 Generating comparison plots...")
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multimodal Neural Enhancement vs OpenVLA Baseline', fontsize=16, fontweight='bold')
    
    # Plot 1: Overall MAE Comparison
    ax1 = axes[0, 0]
    methods = ['Baseline', 'Enhanced']
    mae_values = [
        baseline_analysis['summary']['avg_mae'],
        enhancer_analysis['single_split']['enhanced_mae']
    ]
    colors = ['#ff7f7f', '#7fbf7f']
    
    bars = ax1.bar(methods, mae_values, color=colors, alpha=0.7)
    ax1.set_ylabel('MAE')
    ax1.set_title('Overall MAE Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mae_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Plot 2: Per-Dimension Improvements
    ax2 = axes[0, 1]
    dim_names = list(enhancer_analysis['single_split']['per_dimension'].keys())
    improvements = [enhancer_analysis['single_split']['per_dimension'][dim]['improvement_percent'] 
                   for dim in dim_names]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax2.bar(dim_names, improvements, color=colors, alpha=0.7)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Per-Dimension MAE Improvements')
    ax2.set_xticklabels(dim_names, rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 3: Cross-Validation Results
    ax3 = axes[1, 0]
    cv_improvements = [r['mae_improvement_percent'] for r in enhancer_analysis['cross_validation']['fold_results']]
    fold_numbers = list(range(1, len(cv_improvements) + 1))
    
    ax3.plot(fold_numbers, cv_improvements, 'o-', linewidth=2, markersize=8)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.axhline(y=np.mean(cv_improvements), color='green', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(cv_improvements):.1f}%')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('MAE Improvement (%)')
    ax3.set_title('Cross-Validation Results')
    ax3.set_xticks(fold_numbers)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training vs Test Baseline Performance
    ax4 = axes[1, 1]
    # This would need training data analysis - placeholder for now
    ax4.text(0.5, 0.5, 'Training/Test Split\nQuality Analysis', 
             ha='center', va='center', fontsize=14, transform=ax4.transAxes)
    ax4.set_title('Data Split Quality')
    
    plt.tight_layout()
    plt.savefig('multimodal_enhancement_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Plots saved to: multimodal_enhancement_comparison.png")

def generate_final_report(baseline_analysis, training_analysis, enhancer_analysis):
    """Generate comprehensive final report"""
    print("\n📝 Generating Final Report...")
    
    report = {
        'experiment_summary': {
            'title': 'Multimodal Augmented Neural Network vs OpenVLA Baseline',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'objective': 'Evaluate improvement of multimodal neural enhancement over OpenVLA baseline on held-out episodes'
        },
        'baseline_performance': baseline_analysis,
        'training_data_analysis': training_analysis,
        'enhancer_performance': enhancer_analysis,
        'key_findings': [],
        'recommendations': []
    }
    
    # Extract key findings
    single_split_improvement = enhancer_analysis['single_split']['mae_improvement_percent']
    cv_mean_improvement = enhancer_analysis['cross_validation']['mean_improvement']
    cv_std_improvement = enhancer_analysis['cross_validation']['std_improvement']
    positive_improvement_rate = enhancer_analysis['cross_validation']['positive_improvement_rate']
    
    findings = []
    
    if single_split_improvement > 0:
        findings.append(f"Single split shows {single_split_improvement:.1f}% MAE improvement over baseline")
    else:
        findings.append(f"Single split shows {abs(single_split_improvement):.1f}% MAE degradation")
    
    findings.append(f"Cross-validation shows average {cv_mean_improvement:.1f}% ± {cv_std_improvement:.1f}% MAE improvement")
    findings.append(f"Enhancer improves performance in {positive_improvement_rate*100:.1f}% of cross-validation folds")
    
    # Find best and worst dimensions
    per_dim = enhancer_analysis['single_split']['per_dimension']
    best_dim = max(per_dim.keys(), key=lambda k: per_dim[k]['improvement_percent'])
    worst_dim = min(per_dim.keys(), key=lambda k: per_dim[k]['improvement_percent'])
    
    findings.append(f"Largest improvement in {best_dim}: {per_dim[best_dim]['improvement_percent']:.1f}%")
    findings.append(f"Worst performance in {worst_dim}: {per_dim[worst_dim]['improvement_percent']:.1f}%")
    
    # Check data quality
    if training_analysis['data_leakage']:
        findings.append("⚠️ WARNING: Data leakage detected in train/test split")
    else:
        findings.append("✅ Proper train/test separation with no data leakage")
    
    report['key_findings'] = findings
    
    # Generate recommendations
    recommendations = []
    
    if cv_mean_improvement > 10:
        recommendations.append("Multimodal enhancement shows significant improvement and should be adopted")
    elif cv_mean_improvement > 0:
        recommendations.append("Multimodal enhancement shows modest improvement, consider further optimization")
    else:
        recommendations.append("Enhancement does not improve performance, requires architectural changes")
    
    if cv_std_improvement > 15:
        recommendations.append("High variance in cross-validation suggests need for more training data")
    
    if positive_improvement_rate < 0.8:
        recommendations.append("Inconsistent performance across folds indicates overfitting concerns")
    
    # Dimension-specific recommendations
    if per_dim['Yaw']['improvement_percent'] > 20:
        recommendations.append("Enhancer particularly effective for yaw rotation correction")
    
    if per_dim['Gripper']['improvement_percent'] < 0:
        recommendations.append("Gripper control needs specialized enhancement strategy")
    
    report['recommendations'] = recommendations
    
    # Save report
    with open('final_evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n🎯 FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"📈 Performance Improvement:")
    print(f"   Single Split: {single_split_improvement:+.1f}%")
    print(f"   Cross-Validation: {cv_mean_improvement:+.1f}% ± {cv_std_improvement:.1f}%")
    print(f"   Consistency: {positive_improvement_rate*100:.1f}% folds with positive improvement")
    
    print(f"\n🏆 Best Improvements:")
    print(f"   {best_dim}: {per_dim[best_dim]['improvement_percent']:+.1f}%")
    print(f"   Overall: Achieved {'significant' if cv_mean_improvement > 10 else 'modest' if cv_mean_improvement > 0 else 'no'} improvement")
    
    print(f"\n📋 Key Findings:")
    for i, finding in enumerate(findings, 1):
        print(f"   {i}. {finding}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n💾 Report saved to: final_evaluation_report.json")
    
    return report

def main():
    """Main evaluation function"""
    print("🚀 Final Evaluation: Multimodal Neural Enhancement vs OpenVLA Baseline")
    print("=" * 70)
    
    # Load all results
    all_results = load_all_results()
    
    if not all_results.get('baseline') or not all_results.get('enhancer'):
        print("❌ Missing required result files. Run baseline and enhancer training first.")
        return
    
    # Analyze each component
    baseline_analysis = analyze_baseline_performance(all_results['baseline'])
    training_analysis = analyze_training_split(all_results['training_data'])
    enhancer_analysis = analyze_enhancer_performance(all_results['enhancer'])
    
    # Generate plots
    generate_comparison_plots(baseline_analysis, enhancer_analysis)
    
    # Generate final report
    final_report = generate_final_report(baseline_analysis, training_analysis, enhancer_analysis)
    
    print(f"\n🎉 Final evaluation complete!")
    print(f"   Report: final_evaluation_report.json")
    print(f"   Plots: multimodal_enhancement_comparison.png")

if __name__ == "__main__":
    main()
