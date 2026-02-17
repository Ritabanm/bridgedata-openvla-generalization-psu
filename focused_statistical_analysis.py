"""
Focused Statistical Analysis: OpenVLA Baseline vs Multimodal Enhancer
Compares only the two models that have individual sample data available
"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime

class FocusedStatisticalAnalyzer:
    def __init__(self):
        self.results = {}
        
    def load_results(self):
        """Load results for OpenVLA baseline and Multimodal enhancer"""
        print("🔬 Loading focused results for OpenVLA vs Multimodal comparison...")
        
        # Load OpenVLA baseline
        try:
            with open('sota_openvla_replication_20260205_085916_results.json', 'r') as f:
                self.results['openvla_baseline'] = json.load(f)
            print("✅ Loaded OpenVLA baseline results")
        except Exception as e:
            print(f"❌ Error loading OpenVLA baseline: {e}")
            return False
        
        # Load Multimodal enhancer
        try:
            with open('multimodal_enhancer_results.json', 'r') as f:
                self.results['multimodal_enhancer'] = json.load(f)
            print("✅ Loaded Multimodal enhancer results")
        except Exception as e:
            print(f"❌ Error loading Multimodal enhancer: {e}")
            return False
        
        return True
    
    def extract_samples(self):
        """Extract individual MAE samples from both models"""
        samples = {}
        
        # Extract OpenVLA baseline samples
        openvla_samples = []
        if 'detailed_rollouts' in self.results['openvla_baseline']:
            for rollout in self.results['openvla_baseline']['detailed_rollouts']:
                if 'predictions' in rollout and 'ground_truths' in rollout:
                    for pred, gt in zip(rollout['predictions'], rollout['ground_truths']):
                        pred = np.array(pred)
                        gt = np.array(gt)
                        mae = np.mean(np.abs(pred - gt))
                        openvla_samples.append(mae)
        samples['openvla_baseline'] = np.array(openvla_samples)
        
        # Extract Multimodal enhancer samples - ONLY the first 40 (original test samples)
        multimodal_samples = []
        if 'single_split_evaluation' in self.results['multimodal_enhancer']:
            if 'detailed_predictions' in self.results['multimodal_enhancer']['single_split_evaluation']:
                # Take only the first 40 predictions to match the original test samples
                detailed_preds = self.results['multimodal_enhancer']['single_split_evaluation']['detailed_predictions'][:40]
                for pred in detailed_preds:
                    if 'ground_truth' in pred and 'enhanced_prediction' in pred:
                        gt = np.array(pred['ground_truth'])
                        enhanced = np.array(pred['enhanced_prediction'])
                        mae = np.mean(np.abs(enhanced - gt))
                        multimodal_samples.append(mae)
        samples['multimodal_enhancer'] = np.array(multimodal_samples)
        
        return samples
    
    def perform_statistical_tests(self, samples):
        """Perform comprehensive statistical tests"""
        baseline_samples = samples['openvla_baseline']
        multimodal_samples = samples['multimodal_enhancer']
        
        # Ensure equal length for paired tests
        min_len = min(len(baseline_samples), len(multimodal_samples))
        baseline_samples = baseline_samples[:min_len]
        multimodal_samples = multimodal_samples[:min_len]
        
        print(f"\n🧪 Performing statistical tests with {min_len} paired samples...")
        
        results = {}
        
        # Descriptive statistics
        results['descriptive_stats'] = {
            'openvla_baseline': {
                'mean': float(np.mean(baseline_samples)),
                'std': float(np.std(baseline_samples)),
                'median': float(np.median(baseline_samples)),
                'min': float(np.min(baseline_samples)),
                'max': float(np.max(baseline_samples)),
                'n': len(baseline_samples)
            },
            'multimodal_enhancer': {
                'mean': float(np.mean(multimodal_samples)),
                'std': float(np.std(multimodal_samples)),
                'median': float(np.median(multimodal_samples)),
                'min': float(np.min(multimodal_samples)),
                'max': float(np.max(multimodal_samples)),
                'n': len(multimodal_samples)
            }
        }
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(baseline_samples, multimodal_samples)
        results['paired_t_test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_05': p_value < 0.05,
            'significant_01': p_value < 0.01,
            'significant_001': p_value < 0.001
        }
        
        # Wilcoxon signed-rank test (non-parametric)
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(baseline_samples, multimodal_samples)
            results['wilcoxon_test'] = {
                'statistic': float(wilcoxon_stat),
                'p_value': float(wilcoxon_p),
                'significant_05': wilcoxon_p < 0.05
            }
        except Exception as e:
            print(f"⚠️  Wilcoxon test failed: {e}")
            results['wilcoxon_test'] = None
        
        # Effect size (Cohen's d)
        diff = baseline_samples - multimodal_samples
        pooled_std = np.sqrt(((len(baseline_samples) - 1) * np.var(baseline_samples) + 
                             (len(multimodal_samples) - 1) * np.var(multimodal_samples)) / 
                            (len(baseline_samples) + len(multimodal_samples) - 2))
        cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0
        
        results['effect_size'] = {
            'cohens_d': float(cohens_d),
            'interpretation': self._interpret_cohens_d(abs(cohens_d))
        }
        
        # Confidence intervals
        mean_diff = np.mean(diff)
        se_diff = stats.sem(diff)
        ci_lower, ci_upper = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=se_diff)
        
        results['confidence_interval'] = {
            'mean_difference': float(mean_diff),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'improvement_percent': float((mean_diff / np.mean(baseline_samples)) * 100) if np.mean(baseline_samples) > 0 else 0
        }
        
        return results
    
    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def create_visualizations(self, samples, test_results):
        """Create comprehensive visualizations"""
        plt.figure(figsize=(15, 10))
        
        baseline_samples = samples['openvla_baseline']
        multimodal_samples = samples['multimodal_enhancer']
        
        # 1. Box plot comparison
        plt.subplot(2, 3, 1)
        plt.boxplot([baseline_samples, multimodal_samples], tick_labels=['OpenVLA Baseline', 'Multimodal Enhancer'])
        plt.ylabel('MAE')
        plt.title('MAE Distribution Comparison')
        plt.grid(True, alpha=0.3)
        
        # 2. Histogram
        plt.subplot(2, 3, 2)
        plt.hist(baseline_samples, alpha=0.7, label='OpenVLA Baseline', bins=20)
        plt.hist(multimodal_samples, alpha=0.7, label='Multimodal Enhancer', bins=20)
        plt.xlabel('MAE')
        plt.ylabel('Frequency')
        plt.title('MAE Distribution Histogram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Paired differences
        plt.subplot(2, 3, 3)
        min_len = min(len(baseline_samples), len(multimodal_samples))
        differences = baseline_samples[:min_len] - multimodal_samples[:min_len]
        plt.hist(differences, alpha=0.7, bins=20)
        plt.axvline(x=0, color='red', linestyle='--', label='No Difference')
        plt.axvline(x=np.mean(differences), color='green', linestyle='-', label=f'Mean Difference: {np.mean(differences):.4f}')
        plt.xlabel('MAE Difference (Baseline - Enhanced)')
        plt.ylabel('Frequency')
        plt.title('Paired Differences Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Scatter plot of paired samples
        plt.subplot(2, 3, 4)
        min_len = min(len(baseline_samples), len(multimodal_samples))
        plt.scatter(baseline_samples[:min_len], multimodal_samples[:min_len], alpha=0.6)
        plt.plot([0, max(baseline_samples[:min_len])], [0, max(baseline_samples[:min_len])], 'r--', label='y=x (no improvement)')
        plt.xlabel('OpenVLA Baseline MAE')
        plt.ylabel('Multimodal Enhancer MAE')
        plt.title('Paired Samples Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Summary statistics bar chart
        plt.subplot(2, 3, 5)
        models = ['OpenVLA\nBaseline', 'Multimodal\nEnhancer']
        means = [np.mean(baseline_samples), np.mean(multimodal_samples)]
        stds = [np.std(baseline_samples), np.std(multimodal_samples)]
        
        bars = plt.bar(models, means, yerr=stds, capsize=10, alpha=0.7)
        plt.ylabel('MAE')
        plt.title('Mean MAE Comparison with Error Bars')
        
        # Color bars based on performance
        if means[0] < means[1]:
            bars[0].set_color('green')  # Lower MAE is better
        else:
            bars[1].set_color('green')
        
        plt.grid(True, alpha=0.3)
        
        # 6. Statistical test results summary
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        summary_text = f"Statistical Test Results:\n\n"
        summary_text += f"Paired t-test:\n"
        summary_text += f"  t = {test_results['paired_t_test']['t_statistic']:.3f}\n"
        summary_text += f"  p = {test_results['paired_t_test']['p_value']:.4f}\n"
        summary_text += f"  Significant (α=0.05): {'Yes' if test_results['paired_t_test']['significant_05'] else 'No'}\n\n"
        
        summary_text += f"Effect Size:\n"
        summary_text += f"  Cohen's d = {test_results['effect_size']['cohens_d']:.3f}\n"
        summary_text += f"  Interpretation: {test_results['effect_size']['interpretation']}\n\n"
        
        summary_text += f"Mean Difference:\n"
        summary_text += f"  {test_results['confidence_interval']['mean_difference']:.4f}\n"
        summary_text += f"  95% CI: [{test_results['confidence_interval']['ci_lower']:.4f}, {test_results['confidence_interval']['ci_upper']:.4f}]\n"
        summary_text += f"  Improvement: {test_results['confidence_interval']['improvement_percent']:.1f}%"
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('focused_statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Saved visualization to 'focused_statistical_analysis.png'")
    
    def run_analysis(self):
        """Run the complete focused analysis"""
        print("🔬 Starting Focused Statistical Analysis: OpenVLA vs Multimodal")
        print("=" * 70)
        
        # Load results
        if not self.load_results():
            return None
        
        # Extract samples
        samples = self.extract_samples()
        print(f"📊 Extracted samples:")
        print(f"   OpenVLA Baseline: {len(samples['openvla_baseline'])} samples")
        print(f"   Multimodal Enhancer: {len(samples['multimodal_enhancer'])} samples")
        
        # Perform statistical tests
        test_results = self.perform_statistical_tests(samples)
        
        # Create visualizations
        self.create_visualizations(samples, test_results)
        
        # Save results
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'comparison': 'OpenVLA Baseline vs Multimodal Enhancer',
                'sample_sizes': {
                    'openvla_baseline': len(samples['openvla_baseline']),
                    'multimodal_enhancer': len(samples['multimodal_enhancer'])
                }
            },
            'descriptive_stats': convert_numpy(test_results['descriptive_stats']),
            'statistical_tests': convert_numpy({
                'paired_t_test': test_results['paired_t_test'],
                'wilcoxon_test': test_results['wilcoxon_test'],
                'effect_size': test_results['effect_size'],
                'confidence_interval': test_results['confidence_interval']
            })
        }
        
        with open('focused_statistical_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("💾 Saved detailed results to 'focused_statistical_analysis.json'")
        
        return results
    
    def print_summary(self, results):
        """Print comprehensive summary"""
        print("\n📈 FOCUSED STATISTICAL ANALYSIS SUMMARY")
        print("=" * 70)
        
        print("\n🎯 DESCRIPTIVE STATISTICS:")
        baseline_stats = results['descriptive_stats']['openvla_baseline']
        multimodal_stats = results['descriptive_stats']['multimodal_enhancer']
        
        print(f"OpenVLA Baseline:")
        print(f"  Mean MAE: {baseline_stats['mean']:.4f} ± {baseline_stats['std']:.4f}")
        print(f"  Median: {baseline_stats['median']:.4f}")
        print(f"  Range: [{baseline_stats['min']:.4f}, {baseline_stats['max']:.4f}]")
        print(f"  N: {baseline_stats['n']}")
        
        print(f"\nMultimodal Enhancer:")
        print(f"  Mean MAE: {multimodal_stats['mean']:.4f} ± {multimodal_stats['std']:.4f}")
        print(f"  Median: {multimodal_stats['median']:.4f}")
        print(f"  Range: [{multimodal_stats['min']:.4f}, {multimodal_stats['max']:.4f}]")
        print(f"  N: {multimodal_stats['n']}")
        
        print("\n🧪 STATISTICAL TESTS:")
        t_test = results['statistical_tests']['paired_t_test']
        print(f"Paired t-test:")
        print(f"  t-statistic: {t_test['t_statistic']:.3f}")
        print(f"  p-value: {t_test['p_value']:.4f}")
        print(f"  Significant at α=0.05: {'✅ Yes' if t_test['significant_05'] else '❌ No'}")
        print(f"  Significant at α=0.01: {'✅ Yes' if t_test['significant_01'] else '❌ No'}")
        print(f"  Significant at α=0.001: {'✅ Yes' if t_test['significant_001'] else '❌ No'}")
        
        if results['statistical_tests']['wilcoxon_test']:
            wilcoxon = results['statistical_tests']['wilcoxon_test']
            print(f"\nWilcoxon Signed-Rank Test:")
            print(f"  Statistic: {wilcoxon['statistic']:.1f}")
            print(f"  p-value: {wilcoxon['p_value']:.4f}")
            print(f"  Significant at α=0.05: {'✅ Yes' if wilcoxon['significant_05'] else '❌ No'}")
        
        print(f"\n📏 EFFECT SIZE:")
        effect = results['statistical_tests']['effect_size']
        print(f"  Cohen's d: {effect['cohens_d']:.3f}")
        print(f"  Interpretation: {effect['interpretation']}")
        
        print(f"\n📊 CONFIDENCE INTERVAL:")
        ci = results['statistical_tests']['confidence_interval']
        print(f"  Mean Difference: {ci['mean_difference']:.4f}")
        print(f"  95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        print(f"  Improvement: {ci['improvement_percent']:.1f}%")
        
        print("\n💡 CONCLUSION:")
        if t_test['significant_05']:
            if ci['improvement_percent'] > 0:
                print("  ✅ Multimodal Enhancer shows STATISTICALLY SIGNIFICANT IMPROVEMENT")
                print(f"     over OpenVLA baseline (p={t_test['p_value']:.4f}, {ci['improvement_percent']:.1f}% improvement)")
            else:
                print("  ⚠️  Multimodal Enhancer shows STATISTICALLY SIGNIFICANT WORSE PERFORMANCE")
                print(f"     compared to OpenVLA baseline (p={t_test['p_value']:.4f}, {abs(ci['improvement_percent']):.1f}% worse)")
        else:
            print("  ❌ No statistically significant difference found")
            print(f"     between Multimodal Enhancer and OpenVLA baseline (p={t_test['p_value']:.4f})")
        
        print(f"\n📈 Effect size interpretation: {effect['interpretation']}")

def main():
    """Main execution function"""
    analyzer = FocusedStatisticalAnalyzer()
    results = analyzer.run_analysis()
    
    if results:
        analyzer.print_summary(results)
        print(f"\n📊 Files generated:")
        print(f"   - focused_statistical_analysis.json (detailed results)")
        print(f"   - focused_statistical_analysis.png (visualizations)")

if __name__ == "__main__":
    main()
