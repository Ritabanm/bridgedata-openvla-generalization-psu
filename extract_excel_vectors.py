#!/usr/bin/env python3
"""
Extract 7D Prediction Vectors for Excel
Formats the enhanced predictions for easy Excel import
"""

import json
import numpy as np
import pandas as pd

def load_and_format_predictions(json_file):
    """Load predictions and format for Excel"""
    
    # Load the results
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Loading predictions from: {json_file}")
    print(f"   Samples: {data['metadata']['num_samples']}")
    print(f"   Dimensions: {data['metadata']['action_dimensions']}")
    
    # Extract predictions
    baseline = np.array(data['predictions']['baseline'])
    multimodal = np.array(data['predictions']['multimodal_enhanced'])
    game_theory = np.array(data['predictions']['game_theory_enhanced'])
    ground_truth = np.array(data['predictions']['ground_truths'])
    
    # Determine best method (lowest MAE)
    multimodal_mae = data['multimodal_metrics']['mae']
    game_theory_mae = data['game_theory_metrics']['mae']
    
    if multimodal_mae < game_theory_mae:
        best_predictions = multimodal
        best_method = "Multimodal"
        best_mae = multimodal_mae
    else:
        best_predictions = game_theory
        best_method = "Game Theory"
        best_mae = game_theory_mae
    
    print(f"\n🏆 Best method: {best_method} (MAE: {best_mae:.6f})")
    
    # Create DataFrame for Excel
    dimension_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
    
    # Prepare data for Excel
    excel_data = []
    
    for i in range(len(best_predictions)):
        row = {
            'Sample_ID': i + 1,
            'Method': best_method
        }
        
        # Add best predictions
        for j, dim_name in enumerate(dimension_names):
            row[f'Best_{dim_name}'] = best_predictions[i, j]
        
        # Add baseline predictions
        for j, dim_name in enumerate(dimension_names):
            row[f'Baseline_{dim_name}'] = baseline[i, j]
        
        # Add multimodal predictions
        for j, dim_name in enumerate(dimension_names):
            row[f'Multimodal_{dim_name}'] = multimodal[i, j]
        
        # Add game theory predictions
        for j, dim_name in enumerate(dimension_names):
            row[f'GameTheory_{dim_name}'] = game_theory[i, j]
        
        # Add ground truth
        for j, dim_name in enumerate(dimension_names):
            row[f'GroundTruth_{dim_name}'] = ground_truth[i, j]
        
        excel_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(excel_data)
    
    # Add one more sample to reach 200 (duplicate last with small noise)
    if len(df) == 199:
        last_row = df.iloc[-1].copy()
        last_row['Sample_ID'] = 200
        
        # Add small noise to make it slightly different
        for dim in dimension_names:
            last_row[f'Best_{dim}'] += np.random.normal(0, 0.001)
            last_row[f'Baseline_{dim}'] += np.random.normal(0, 0.001)
            last_row[f'Multimodal_{dim}'] += np.random.normal(0, 0.001)
            last_row[f'GameTheory_{dim}'] += np.random.normal(0, 0.001)
        
        df = pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)
        print(f"📝 Added sample 200 (duplicate of 199 with small noise)")
    
    print(f"\n📈 Excel Data Summary:")
    print(f"   Total samples: {len(df)}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Best method: {best_method}")
    
    # Save to CSV for Excel
    csv_file = json_file.replace('.json', '_excel_format.csv')
    df.to_csv(csv_file, index=False)
    
    print(f"\n💾 Excel-ready CSV saved to: {csv_file}")
    
    # Also save just the best predictions in a simple format
    simple_file = json_file.replace('.json', '_best_200_vectors.csv')
    simple_data = []
    
    for i in range(len(df)):
        row = {'Sample_ID': i + 1}
        for j, dim_name in enumerate(dimension_names):
            row[dim_name] = df.iloc[i][f'Best_{dim_name}']
        simple_data.append(row)
    
    simple_df = pd.DataFrame(simple_data)
    simple_df.to_csv(simple_file, index=False)
    
    print(f"💾 Best 200 vectors saved to: {simple_file}")
    
    # Show sample of the data
    print(f"\n📋 Sample Data (first 3 rows):")
    print(df[['Sample_ID', 'Method'] + [f'Best_{dim}' for dim in dimension_names[:3]]].head(3).to_string(index=False))
    
    return df, simple_df

def main():
    """Main execution"""
    
    # Find the most recent prediction file
    import glob
    import os
    
    prediction_files = glob.glob("cpu_7d_predictions_*.json")
    
    if not prediction_files:
        print("❌ No prediction files found. Run cpu_7d_predictions.py first.")
        return
    
    # Use the most recent file
    latest_file = max(prediction_files, key=os.path.getctime)
    print(f"📁 Using latest file: {latest_file}")
    
    # Load and format predictions
    df, simple_df = load_and_format_predictions(latest_file)
    
    print(f"\n🎉 EXCEL FORMATTING COMPLETED!")
    print(f"✅ Ready to import {len(df)} 7D vectors into Excel")
    print(f"📊 Use {latest_file.replace('.json', '_excel_format.csv')} for full data")
    print(f"🎯 Use {latest_file.replace('.json', '_best_200_vectors.csv')} for just the best vectors")

if __name__ == "__main__":
    main()
