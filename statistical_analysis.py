"""

Focused Statistical Analysis: 100-Sample OpenVLA Baseline vs Multimodal Enhancer
Compares the same skill/task evaluation with 100 samples each

"""

import os
import json
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime

class SameSkillStatisticalAnalyzer:
    def __init__(self):
        self.results = {}
        
    def load_results(self):
        """Load results for 500-sample OpenVLA baseline and Multimodal enhancer"""
        print("🔬 Loading same-skill results: 500-sample OpenVLA vs Multimodal...")
        
        has_500_data = False
        
        # Load real 500-sample OpenVLA baseline first
        if os.path.exists('baseline_500_samples_results.json'):
            try:
                with open('baseline_500_samples_results.json', 'r') as f:
                    self.results['openvla_500'] = json.load(f)
                print("✅ Loaded 500-sample OpenVLA baseline results (real data)")
                has_500_data = True
            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"⚠️  Error loading 500-sample file: {e}")
                print("🔄 Falling back to other 500-sample data...")
        
        if not has_500_data:
            # Try other 500-sample files if main one doesn't exist
            import glob
            baseline_500_files = glob.glob("baseline_500_samples_results_*.json")
            if baseline_500_files:
                baseline_500_files.sort()
                latest_file = baseline_500_files[-1]
                try:
                    with open(latest_file, 'r') as f:
                        self.results['openvla_500'] = json.load(f)
                    print(f"✅ Loaded 500-sample OpenVLA baseline results ({latest_file})")
                    has_500_data = True
                except (json.JSONDecodeError, KeyError, Exception) as e:
                    print(f"⚠️  Error loading 500-sample file {latest_file}: {e}")
        
        if not has_500_data:
            # Fallback to 100 samples if 500 not available
            if os.path.exists('baseline_100_samples_results.json'):
                try:
                    with open('baseline_100_samples_results.json', 'r') as f:
                        self.results['openvla_100'] = json.load(f)
                    print("✅ Loaded 100-sample OpenVLA baseline results (fallback)")
                except (json.JSONDecodeError, KeyError, Exception) as e:
                    print(f"❌ Error loading 100-sample baseline: {e}")
                    return False
            else:
                print(f"❌ Error loading OpenVLA baseline: No baseline results found")
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
        
        # Determine which baseline data we have (500 or 100 samples)
        baseline_key = 'openvla_500' if 'openvla_500' in self.results else 'openvla_100'
        baseline_label = 'openvla_500' if baseline_key == 'openvla_500' else 'openvla_100'
        
        # Extract OpenVLA baseline predictions
        openvla_samples = []
        if 'detailed_results' in self.results[baseline_key]:
            for result in self.results[baseline_key]['detailed_results']:
                if 'openvla_prediction' in result and 'ground_truth' in result:
                    pred = np.array(result['openvla_prediction'])
                    gt = np.array(result['ground_truth'])
                    mae = np.mean(np.abs(pred - gt))
                    openvla_samples.append(mae)
        samples[baseline_label] = np.array(openvla_samples)
        
        # Extract Multimodal enhancer samples - match the same task/instruction
        multimodal_samples = []
        if 'single_split_evaluation' in self.results['multimodal_enhancer']:
            if 'detailed_predictions' in self.results['multimodal_enhancer']['single_split_evaluation']:
                # Filter for the same task: "pick up the object and place it in the bowl"
                detailed_preds = self.results['multimodal_enhancer']['single_split_evaluation']['detailed_predictions']
                for pred in detailed_preds:
                    # Check if this matches the same task (you may need to adjust this filtering)
                    if 'ground_truth' in pred and 'enhanced_prediction' in pred:
                        gt = np.array(pred['ground_truth'])
                        enhanced = np.array(pred['enhanced_prediction'])
                        mae = np.mean(np.abs(enhanced - gt))
                        multimodal_samples.append(mae)
                        # Match the baseline sample size for fair comparison
                        if len(multimodal_samples) >= len(openvla_samples):
                            break
        samples['multimodal_enhancer'] = np.array(multimodal_samples[:len(openvla_samples)])
        
        return samples
    
    def perform_statistical_tests(self, samples):
        """Perform comprehensive statistical tests"""
        # Determine which baseline data we have
        baseline_key = 'openvla_500' if 'openvla_500' in samples else 'openvla_100'
        
        baseline_samples = samples[baseline_key]
        multimodal_samples = samples['multimodal_enhancer']
        
        # Ensure equal length for paired tests
        min_len = min(len(baseline_samples), len(multimodal_samples))
        baseline_samples = baseline_samples[:min_len]
        multimodal_samples = multimodal_samples[:min_len]
        
        print(f"\n🧪 Performing statistical tests with {min_len} paired samples...")
        
        results = {}
        
        # Descriptive statistics
        results['descriptive_stats'] = {
            baseline_key: {
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
        
        # Statistical power analysis
        alpha = 0.05
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        effect_size = abs(cohens_d)
        if effect_size > 0:
            z_power = effect_size * np.sqrt(len(diff)/2) - z_alpha
            power = norm.cdf(z_power)
            n_needed = 2 * ((z_alpha + norm.ppf(0.8)) / effect_size)**2
        else:
            power = 0
            n_needed = float('inf')
        
        results['power_analysis'] = {
            'current_power': float(power),
            'samples_needed_80_power': float(n_needed),
            'effect_size': float(effect_size)
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
        plt.figure(figsize=(16, 12))
        
        # Determine which baseline data we have
        baseline_key = 'openvla_500' if 'openvla_500' in samples else 'openvla_100'
        baseline_label = 'OpenVLA 500' if baseline_key == 'openvla_500' else 'OpenVLA'
        
        baseline_samples = samples[baseline_key]
        multimodal_samples = samples['multimodal_enhancer']
        
        # 1. Box plot comparison
        plt.subplot(2, 4, 1)
        plt.boxplot([baseline_samples, multimodal_samples], tick_labels=[baseline_label, 'Multimodal'])
        plt.ylabel('MAE')
        plt.title('MAE Distribution Comparison')
        plt.grid(True, alpha=0.3)
        
        # 2. Histogram
        plt.subplot(2, 4, 2)
        plt.hist(baseline_samples, alpha=0.7, label=baseline_label, bins=30)
        plt.hist(multimodal_samples, alpha=0.7, label='Multimodal', bins=30)
        plt.xlabel('MAE')
        plt.ylabel('Frequency')
        plt.title('MAE Distribution Histogram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Paired differences
        plt.subplot(2, 4, 3)
        min_len = min(len(baseline_samples), len(multimodal_samples))
        differences = baseline_samples[:min_len] - multimodal_samples[:min_len]
        plt.hist(differences, alpha=0.7, bins=30)
        plt.axvline(x=0, color='red', linestyle='--', label='No Difference')
        plt.axvline(x=np.mean(differences), color='green', linestyle='-', label=f'Mean: {np.mean(differences):.4f}')
        plt.xlabel('MAE Difference (OpenVLA - Multimodal)')
        plt.ylabel('Frequency')
        plt.title('Paired Differences')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Scatter plot of paired samples
        plt.subplot(2, 4, 4)
        plt.scatter(baseline_samples[:min_len], multimodal_samples[:min_len], alpha=0.6)
        max_val = max(baseline_samples[:min_len].max(), multimodal_samples[:min_len].max())
        plt.plot([0, max_val], [0, max_val], 'r--', label='y=x (no improvement)')
        plt.xlabel(f'{baseline_label} MAE')
        plt.ylabel('Multimodal MAE')
        plt.title('Paired Samples')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Summary statistics bar chart
        plt.subplot(2, 4, 5)
        models = [baseline_label, 'Multimodal\nEnhancer']
        means = [np.mean(baseline_samples), np.mean(multimodal_samples)]
        stds = [np.std(baseline_samples), np.std(multimodal_samples)]
        
        bars = plt.bar(models, means, yerr=stds, capsize=10, alpha=0.7)
        plt.ylabel('MAE')
        plt.title('Mean MAE ± Std')
        
        # Color bars based on performance
        if means[0] < means[1]:
            bars[0].set_color('green')  # Lower MAE is better
            bars[1].set_color('red')
        else:
            bars[1].set_color('green')
            bars[0].set_color('red')
        
        plt.grid(True, alpha=0.3)
        
        # 6. Statistical test results summary
        plt.subplot(2, 4, 6)
        plt.axis('off')
        
        summary_text = f"Statistical Results:\n\n"
        summary_text += f"Paired t-test:\n"
        summary_text += f"  t = {test_results['paired_t_test']['t_statistic']:.3f}\n"
        summary_text += f"  p = {test_results['paired_t_test']['p_value']:.4f}\n"
        summary_text += f"  Significant: {'Yes' if test_results['paired_t_test']['significant_05'] else 'No'}\n\n"
        
        summary_text += f"Effect Size:\n"
        summary_text += f"  Cohen's d = {test_results['effect_size']['cohens_d']:.3f}\n"
        summary_text += f"  {test_results['effect_size']['interpretation']}\n\n"
        
        summary_text += f"Power Analysis:\n"
        summary_text += f"  Power: {test_results['power_analysis']['current_power']:.3f}\n"
        summary_text += f"  N for 80%: {test_results['power_analysis']['samples_needed_80_power']:.0f}"
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        # 7. Cumulative distribution
        plt.subplot(2, 4, 7)
        sorted_baseline = np.sort(baseline_samples)
        sorted_multimodal = np.sort(multimodal_samples)
        plt.plot(sorted_baseline, np.arange(1, len(sorted_baseline) + 1) / len(sorted_baseline), 
                label=baseline_label, linewidth=2)
        plt.plot(sorted_multimodal, np.arange(1, len(sorted_multimodal) + 1) / len(sorted_multimodal), 
                label='Multimodal', linewidth=2)
        plt.xlabel('MAE')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Sample size vs power curve
        plt.subplot(2, 4, 8)
        effect_size = test_results['power_analysis']['effect_size']
        if effect_size > 0:
            sample_sizes = np.arange(10, 201, 10)
            powers = []
            for n in sample_sizes:
                z_power = effect_size * np.sqrt(n/2) - norm.ppf(1 - 0.05/2)
                power = norm.cdf(z_power)
                powers.append(max(0, min(1, power)))
            
            plt.plot(sample_sizes, powers, 'b-', linewidth=2)
            plt.axhline(y=0.8, color='r', linestyle='--', label='80% Power')
            plt.axvline(x=len(baseline_samples), color='g', linestyle='--', label=f'Current N={len(baseline_samples)}')
            plt.xlabel('Sample Size')
            plt.ylabel('Statistical Power')
            plt.title('Power Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('same_skill_statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Saved visualization to 'same_skill_statistical_analysis.png'")
    
    def run_analysis(self):
        """Run the complete same-skill analysis"""
        print("🔬 Starting Same-Skill Statistical Analysis: 500-sample OpenVLA vs Multimodal")
        print("=" * 80)
        
        # Load results
        if not self.load_results():
            return None
        
        # Extract samples
        samples = self.extract_samples()
        
        # Determine which baseline key we have
        baseline_key = 'openvla_500' if 'openvla_500' in samples else 'openvla_100'
        baseline_name = 'OpenVLA 500-sample' if baseline_key == 'openvla_500' else 'OpenVLA 100-sample'
        
        print(f"📊 Extracted samples:")
        print(f"   {baseline_name}: {len(samples[baseline_key])} samples")
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
                'comparison': '500-sample OpenVLA vs Multimodal Enhancer (Same Skill)',
                'sample_sizes': {
                    baseline_key: len(samples[baseline_key]),
                    'multimodal_enhancer': len(samples['multimodal_enhancer'])
                }
            },
            'descriptive_stats': convert_numpy(test_results['descriptive_stats']),
            'statistical_tests': convert_numpy({
                'paired_t_test': test_results['paired_t_test'],
                'wilcoxon_test': test_results['wilcoxon_test'],
                'effect_size': test_results['effect_size'],
                'confidence_interval': test_results['confidence_interval'],
                'power_analysis': test_results['power_analysis']
            })
        }
        
        with open('same_skill_statistical_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("💾 Saved detailed results to 'same_skill_statistical_analysis.json'")
        
        return results
    
    def print_summary(self, results):
        """Print comprehensive summary"""
        print("\n📈 SAME-SKILL STATISTICAL ANALYSIS SUMMARY")
        print("=" * 80)
        
        print("\n🎯 DESCRIPTIVE STATISTICS:")
        
        # Determine which baseline data we have
        baseline_key = 'openvla_500' if 'openvla_500' in results['descriptive_stats'] else 'openvla_100'
        baseline_label = 'OpenVLA 500-sample' if baseline_key == 'openvla_500' else 'OpenVLA 100-sample'
        
        baseline_stats = results['descriptive_stats'][baseline_key]
        multimodal_stats = results['descriptive_stats']['multimodal_enhancer']
        
        print(f"{baseline_label}:")
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
        
        print(f"\n⚡ POWER ANALYSIS:")
        power = results['statistical_tests']['power_analysis']
        print(f"  Current statistical power: {power['current_power']:.3f}")
        print(f"  Samples needed for 80% power: {power['samples_needed_80_power']:.0f}")
        print(f"  Effect size: {power['effect_size']:.3f}")
        
        print("\n💡 CONCLUSION:")
        if t_test['significant_05']:
            if ci['improvement_percent'] > 0:
                print("  ✅ Multimodal Enhancer shows STATISTICALLY SIGNIFICANT IMPROVEMENT")
                print(f"     over {baseline_label} (p={t_test['p_value']:.4f}, {ci['improvement_percent']:.1f}% improvement)")
            else:
                print("  ⚠️  Multimodal Enhancer shows STATISTICALLY SIGNIFICANT WORSE PERFORMANCE")
                print(f"     compared to {baseline_label} (p={t_test['p_value']:.4f}, {abs(ci['improvement_percent']):.1f}% worse)")
        else:
            print("  ❌ No statistically significant difference found")
            print(f"     between Multimodal Enhancer and {baseline_label} (p={t_test['p_value']:.4f})")
        
        if power['current_power'] < 0.8:
            print(f"\n⚠️  LOW STATISTICAL POWER WARNING:")
            print(f"     Current power: {power['current_power']:.1%} (recommended: ≥80%)")
            print(f"     Consider collecting {power['samples_needed_80_power']:.0f} samples for adequate power")
        
        print(f"\n📈 Effect size interpretation: {effect['interpretation']}")

def main():
    """Main execution function"""
    analyzer = SameSkillStatisticalAnalyzer()
    results = analyzer.run_analysis()
    
    if results:
        analyzer.print_summary(results)
        print(f"\n📊 Files generated:")
        print(f"   - same_skill_statistical_analysis.json (detailed results)")
        print(f"   - same_skill_statistical_analysis.png (comprehensive visualizations)")

if __name__ == "__main__":
    main()
