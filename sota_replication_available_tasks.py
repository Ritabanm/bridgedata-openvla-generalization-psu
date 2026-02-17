#!/usr/bin/env python3
"""
Modified SOTA Replication - Only runs tasks with available data
Based on your actual BridgeData V2 dataset structure
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import sys
sys.path.append('.')

from openvla_baseline import OpenVLABaseline

class SOTAAvailableTasksReplicator:
    """SOTA replication that only evaluates tasks with available data"""
    
    def __init__(self, max_trials=10, max_timesteps=2):
        self.max_trials = max_trials
        self.max_timesteps = max_timesteps
        self.global_metrics = {
            'total_rollouts': 0,
            'successful_rollouts': 0,
            'total_mae': 0.0,
            'total_mse': 0.0,
            'total_predictions': 0,
            'total_time': 0.0,
            'task_results': {}
        }
        
        # Only tasks that actually have data in your dataset
        self.available_tasks = [
            "pick_up_object",
            "sweep_to_dustpan"
        ]
        
        print("🎯 SOTA REPLICATION - AVAILABLE TASKS ONLY")
        print("=" * 60)
        print(f"📊 Found {len(self.available_tasks)} tasks with data:")
        for task in self.available_tasks:
            print(f"   ✅ {task}")
        print()
    
    def load_real_bridgedata_samples(self, task_name, max_trials=10):
        """Load REAL BridgeData V2 samples for available tasks"""
        print(f"📊 Loading real BridgeData V2 samples for {task_name}...")
        
        samples = []
        data_dir = Path("data/scripted_raw")
        
        # Task patterns that match your actual data
        task_patterns = {
            "pick_up_object": ["pnp"],
            "sweep_to_dustpan": ["sweep"]
        }
        
        patterns = task_patterns.get(task_name, [task_name])
        
        for root, dirs, files in os.walk(data_dir):
            if 'policy_out.pkl' in files and len(samples) < max_trials:
                traj_dir = Path(root)
                img_dir = traj_dir / "images0"
                
                path_str = str(root).lower()
                if not any(pattern in path_str for pattern in patterns):
                    continue
                
                if not img_dir.exists():
                    continue
                    
                img_files = list(img_dir.glob("im_*.jpg"))
                if len(img_files) < 2:
                    continue
                
                try:
                    with open(traj_dir / "policy_out.pkl", 'rb') as f:
                        episode_data = pickle.load(f)
                    
                    # Get first few timesteps
                    for i in range(min(self.max_timesteps, len(img_files))):
                        img_path = img_dir / f"im_{i}.jpg"
                        if img_path.exists():
                            # Get ground truth action from episode data
                            if i < len(episode_data):
                                gt_action = episode_data[i]
                                samples.append({
                                    'image_path': str(img_path),
                                    'instruction': f"perform {task_name}",
                                    'ground_truth': gt_action,
                                    'sample': len(samples) + 1,
                                    'timestep': i + 1
                                })
                                
                            if len(samples) >= max_trials * self.max_timesteps:
                                break
                                
                except Exception as e:
                    continue
                    
                if len(samples) >= max_trials * self.max_timesteps:
                    break
        
        print(f"✅ Loaded {len(samples)} samples for {task_name}")
        return samples
    
    def evaluate_task_completion(self, baseline, task_name):
        """Evaluate task completion for available tasks"""
        print(f"🎯 Running {task_name}")
        
        samples = self.load_real_bridgedata_samples(task_name, self.max_trials)
        
        if not samples:
            print(f"⚠️  No samples found for {task_name}. Skipping...")
            return None
        
        successful_rollouts = 0
        total_mae = 0.0
        total_mse = 0.0
        predictions_made = 0
        
        print(f"   Processing {len(samples)} timesteps ({self.max_timesteps} per sample)...")
        
        for i, sample in enumerate(samples):
            try:
                # Load image
                image_path = sample['image_path']
                if not os.path.exists(image_path):
                    continue
                
                # Get ground truth
                gt_action = np.array(sample['ground_truth'])
                if len(gt_action.shape) > 1:
                    gt_action = gt_action.flatten()
                
                # Ensure 7-dimensional
                if len(gt_action) > 7:
                    gt_action = gt_action[:7]
                elif len(gt_action) < 7:
                    gt_action = np.pad(gt_action, (0, 7 - len(gt_action)))
                
                # Predict with OpenVLA
                start_time = time.time()
                pred_action = baseline.predict_action(image_path, sample['instruction'])
                prediction_time = time.time() - start_time
                
                # Ensure prediction is 7-dimensional
                pred_action = np.array(pred_action)
                if len(pred_action) > 7:
                    pred_action = pred_action[:7]
                elif len(pred_action) < 7:
                    pred_action = np.pad(pred_action, (0, 7 - len(pred_action)))
                
                # Calculate metrics
                mae = np.mean(np.abs(pred_action - gt_action))
                mse = np.mean((pred_action - gt_action) ** 2)
                
                total_mae += mae
                total_mse += mse
                predictions_made += 1
                
                # Determine success (MAE < 0.1 threshold)
                if mae < 0.1:
                    successful_rollouts += 1
                
                # Print detailed prediction (matching your desired format)
                print(f"   Sample {sample['sample']}, Timestep {sample['timestep']}:")
                print(f"   Ground Truth: [{', '.join([f'{x:.4f}' for x in gt_action])}]")
                print(f"   Predicted: [{', '.join([f'{x:.4f}' for x in pred_action])}]")
                print(f"   Time: {prediction_time:.4f}s")
                print(f"   MAE: {mae:.4f}, MSE: {mse:.4f}")
                print(f"   Task Success: {'✅' if mae < 0.1 else '❌'}")
                print()
                
            except Exception as e:
                print(f"   Error processing sample {i}: {e}")
                continue
        
        # Calculate task metrics
        num_samples = len(samples) // self.max_timesteps
        success_rate = (successful_rollouts / len(samples)) * 100 if samples else 0
        avg_mae = total_mae / predictions_made if predictions_made > 0 else 0
        avg_mse = total_mse / predictions_made if predictions_made > 0 else 0
        
        task_result = {
            'task_name': task_name,
            'samples_found': num_samples,
            'success_rate': success_rate,
            'avg_mae': avg_mae,
            'avg_mse': avg_mse,
            'timesteps_processed': len(samples),
            'successful_rollouts': successful_rollouts
        }
        
        print(f"✅ {task_name} Results:")
        print(f"   Samples Found: {num_samples}")
        print(f"   Success Rate: {success_rate:.1f}% ({successful_rollouts}/{len(samples)})")
        print(f"   Average MAE: {avg_mae:.4f}")
        print(f"   Average MSE: {avg_mse:.4f}")
        print(f"   Timesteps Processed: {len(samples)} ({self.max_timesteps} per sample)")
        print()
        
        return task_result
    
    def run_sota_evaluation(self):
        """Run SOTA evaluation for available tasks only"""
        print(f"📂 Data Source: data/scripted_raw/ (REAL BridgeData V2)")
        print(f"📊 Evaluation Format: {len(self.available_tasks)} Available Tasks × {self.max_trials} Samples × {self.max_timesteps} Timesteps")
        print("=" * 80)
        
        start_time = time.time()
        
        # Initialize OpenVLA baseline
        baseline = OpenVLABaseline()
        
        # Run evaluation for each available task
        for task_name in self.available_tasks:
            print(f"📋 TASK {self.available_tasks.index(task_name) + 1}/{len(self.available_tasks)}: {task_name.upper()}")
            print("-" * 60)
            
            task_result = self.evaluate_task_completion(baseline, task_name)
            
            if task_result:
                self.global_metrics['task_results'][task_name] = task_result
                self.global_metrics['total_rollouts'] += len(task_result) * self.max_timesteps
                self.global_metrics['successful_rollouts'] += task_result['successful_rollouts']
                self.global_metrics['total_mae'] += task_result['avg_mae'] * task_result['timesteps_processed']
                self.global_metrics['total_mse'] += task_result['avg_mse'] * task_result['timesteps_processed']
                self.global_metrics['total_predictions'] += task_result['timesteps_processed']
        
        self.global_metrics['total_time'] = time.time() - start_time
        
        # Print final results
        self.print_final_results()
        
        # Save results
        self.save_results()
        
        return self.global_metrics
    
    def print_final_results(self):
        """Print final SOTA results"""
        metrics = self.global_metrics
        
        overall_success_rate = (metrics['successful_rollouts'] / metrics['total_predictions']) * 100 if metrics['total_predictions'] > 0 else 0
        overall_avg_mae = metrics['total_mae'] / metrics['total_predictions'] if metrics['total_predictions'] > 0 else 0
        overall_avg_mse = metrics['total_mse'] / metrics['total_predictions'] if metrics['total_predictions'] > 0 else 0
        
        print(f"🏆 SOTA REPLICATION COMPLETE ({len(self.available_tasks)}×{self.max_trials}×{self.max_timesteps} REAL DATA)")
        print("=" * 80)
        print(f"📊 Overall Results:")
        print(f"   Total Rollouts: {metrics['total_rollouts']}")
        print(f"   Successful Rollouts: {metrics['successful_rollouts']}")
        print(f"   Overall Success Rate: {overall_success_rate:.1f}%")
        print(f"   Average MAE: {overall_avg_mae:.4f}")
        print(f"   Average MSE: {overall_avg_mse:.4f}")
        print(f"   Total Predictions: {metrics['total_predictions']}")
        print(f"   Total Time: {metrics['total_time']:.1f}s ({metrics['total_time']/60:.1f}m)")
        print(f"   Data Source: REAL BridgeData V2 from data/scripted_raw/")
        print(f"   Evaluation Format: {len(self.available_tasks)} Available Tasks × {self.max_trials} Samples × {self.max_timesteps} Timesteps")
        print("=" * 80)
    
    def save_results(self):
        """Save results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Detailed results
        detailed_file = f"sota_available_tasks_{timestamp}_results.json"
        with open(detailed_file, 'w') as f:
            json.dump(self.global_metrics, f, indent=2)
        
        # Summary
        summary_file = f"sota_available_tasks_{timestamp}_summary.json"
        summary = {
            'total_tasks': len(self.available_tasks),
            'total_rollouts': self.global_metrics['total_rollouts'],
            'successful_rollouts': self.global_metrics['successful_rollouts'],
            'overall_success_rate': (self.global_metrics['successful_rollouts'] / self.global_metrics['total_predictions']) * 100 if self.global_metrics['total_predictions'] > 0 else 0,
            'average_mae': self.global_metrics['total_mae'] / self.global_metrics['total_predictions'] if self.global_metrics['total_predictions'] > 0 else 0,
            'average_mse': self.global_metrics['total_mse'] / self.global_metrics['total_predictions'] if self.global_metrics['total_predictions'] > 0 else 0,
            'total_time': self.global_metrics['total_time'],
            'tasks_evaluated': self.available_tasks,
            'task_results': self.global_metrics['task_results']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 SOTA Results saved:")
        print(f"   📊 Detailed results: {detailed_file}")
        print(f"   📋 Summary: {summary_file}")

def main():
    """Main function to run SOTA replication for available tasks"""
    replicator = SOTAAvailableTasksReplicator(
        max_trials=10,
        max_timesteps=2
    )
    
    metrics = replicator.run_sota_evaluation()
    
    print(f"\n🎉 SOTA replication completed!")
    print(f"📈 Overall Success Rate: {(metrics['successful_rollouts']/metrics['total_predictions'])*100:.1f}%")
    print(f"📊 Average MAE: {metrics['total_mae']/metrics['total_predictions']:.4f}")
    print(f"📊 Evaluation Format: {len(replicator.available_tasks)}×{replicator.max_trials}×{replicator.max_timesteps}")

if __name__ == "__main__":
    import pickle  # Add this import
    main()
