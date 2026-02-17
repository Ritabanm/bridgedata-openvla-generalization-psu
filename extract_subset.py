#!/usr/bin/env python3
"""
Extract Subset from Existing Baseline Data
Creates a subset of samples 31-57 from existing baseline results
"""

import json
import argparse
import numpy as np

def extract_subset(input_file, output_file, start_sample, end_sample):
    """Extract a subset of samples from baseline results"""
    print(f"📁 Loading baseline data from: {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading baseline data: {e}")
        return None
    
    # Get detailed results
    detailed_results = data.get('detailed_results', [])
    print(f"📊 Total samples available: {len(detailed_results)}")
    
    # Extract subset (convert to 0-based indexing)
    subset_results = detailed_results[start_sample-1:end_sample]
    print(f"✅ Extracted samples {start_sample} to {end_sample}: {len(subset_results)} samples")
    
    # Calculate subset statistics
    if subset_results:
        mae_values = [r.get('mae', 0) for r in subset_results if 'mae' in r]
        if mae_values:
            subset_mae = np.mean(mae_values)
            subset_std = np.std(mae_values)
        else:
            # Calculate MAE from predictions if not available
            mae_values = []
            for r in subset_results:
                if 'openvla_prediction' in r and 'ground_truth' in r:
                    pred = np.array(r['openvla_prediction'])
                    gt = np.array(r['ground_truth'])
                    mae = np.mean(np.abs(pred - gt))
                    mae_values.append(mae)
            
            if mae_values:
                subset_mae = np.mean(mae_values)
                subset_std = np.std(mae_values)
            else:
                subset_mae = 0
                subset_std = 0
    else:
        subset_mae = 0
        subset_std = 0
    
    # Create subset data structure
    subset_data = {
        'summary': {
            'total_samples': len(subset_results),
            'sample_range': f"{start_sample}-{end_sample}",
            'mae': subset_mae,
            'std_mae': subset_std,
            'source_file': input_file
        },
        'detailed_results': subset_results
    }
    
    # Save subset
    with open(output_file, 'w') as f:
        json.dump(subset_data, f, indent=2)
    
    print(f"💾 Subset saved to: {output_file}")
    print(f"📊 Subset MAE: {subset_mae:.6f} ± {subset_std:.6f}")
    
    return subset_data

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Extract subset from baseline data")
    parser.add_argument('--input_file', type=str, default="baseline_100_samples_results.json", 
                        help="Input baseline file")
    parser.add_argument('--output_file', type=str, default="openvla_baseline_31_57.json",
                        help="Output subset file")
    parser.add_argument('--sample_range', type=str, default="31-57",
                        help="Sample range (e.g., '31-57')")
    
    args = parser.parse_args()
    
    # Parse sample range
    try:
        start, end = map(int, args.sample_range.split('-'))
    except ValueError:
        print("❌ Invalid sample range format. Use '31-57' format.")
        return
    
    # Extract subset
    result = extract_subset(args.input_file, args.output_file, start, end)
    
    if result:
        print("🎉 Subset extraction completed successfully!")
    else:
        print("❌ Subset extraction failed!")

if __name__ == "__main__":
    main()
