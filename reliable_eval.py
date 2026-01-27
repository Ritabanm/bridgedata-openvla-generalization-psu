#!/usr/bin/env python3
"""
Reliable OpenVLA Evaluation Framework
Fixed version addressing hardcoded issues and scaling problems
"""

import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import json
import argparse

# Set environment for stability
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class ReliableOpenVLAEvaluator:
    """Reliable evaluator with proper error handling and configuration"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.processor = None
        self.model = None
        self.device = "cpu"
        
    def _default_config(self):
        return {
            'data_paths': [
                "data/scripted_raw",
                "bridge_data_v2", 
                "data/bridgedata"
            ],
            'max_samples': 20,
            'max_timesteps_per_sample': 5,
            'model_name': "openvla/openvla-7b",
            'output_file': "reliable_evaluation_results.json",
            'verbose': True
        }
    
    def load_model(self):
        """Load OpenVLA model with error handling"""
        print("🔄 Loading OpenVLA model...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config['model_name'], 
                trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config['model_name'],
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device)
            
            print(f"✅ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def find_data_directory(self):
        """Dynamically find BridgeData directory"""
        for data_path in self.config['data_paths']:
            path = Path(data_path)
            if path.exists() and any(path.rglob("policy_out.pkl")):
                print(f"📁 Found data directory: {path}")
                return path
        raise FileNotFoundError("No valid BridgeData directory found")
    
    def load_bridgedata_samples(self):
        """Load BridgeData samples with robust error handling"""
        print(f"📊 Loading BridgeData samples (max {self.config['max_samples']})...")
        
        data_dir = self.find_data_directory()
        samples = []
        
        # Find all trajectories with policy_out.pkl
        trajectory_dirs = []
        for pkl_file in data_dir.rglob("policy_out.pkl"):
            traj_dir = pkl_file.parent
            img_dir = traj_dir / "images0"
            if img_dir.exists() and len(list(img_dir.glob("im_*.jpg"))) >= 3:
                trajectory_dirs.append(traj_dir)
        
        print(f"🔍 Found {len(trajectory_dirs)} valid trajectories")
        
        # Sample trajectories
        import random
        random.shuffle(trajectory_dirs)
        selected_dirs = trajectory_dirs[:self.config['max_samples']]
        
        for traj_dir in tqdm(selected_dirs, desc="Loading samples"):
            try:
                sample = self._load_single_sample(traj_dir)
                if sample:
                    samples.append(sample)
                    if len(samples) >= self.config['max_samples']:
                        break
            except Exception as e:
                if self.config['verbose']:
                    print(f"⚠️  Error loading {traj_dir}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(samples)} samples")
        return samples
    
    def _load_single_sample(self, traj_dir):
        """Load a single trajectory sample"""
        # Load ground truth actions
        pkl_file = traj_dir / "policy_out.pkl"
        with open(pkl_file, 'rb') as f:
            actions_data = pickle.load(f)
        
        # Extract actions array
        gt_actions = self._extract_actions(actions_data)
        if gt_actions is None:
            return None
        
        # Load images
        img_dir = traj_dir / "images0"
        img_files = sorted(img_dir.glob("im_*.jpg"))
        
        if len(img_files) < 3:
            return None
        
        # Determine instruction from path
        instruction = self._infer_instruction(traj_dir)
        
        # Select timesteps
        max_timesteps = min(
            self.config['max_timesteps_per_sample'],
            len(img_files),
            len(gt_actions) if len(gt_actions.shape) > 1 else len(gt_actions)
        )
        
        selected_images = img_files[:max_timesteps]
        
        return {
            'path': str(traj_dir),
            'instruction': instruction,
            'images': selected_images,
            'gt_actions': gt_actions,
            'trajectory_length': len(gt_actions) if len(gt_actions.shape) > 1 else 1
        }
    
    def _extract_actions(self, actions_data):
        """Extract actions array from various data formats"""
        try:
            if isinstance(actions_data, dict):
                # Try common keys
                for key in ['actions', 'action', 'qpos', 'commands']:
                    if key in actions_data:
                        actions = actions_data[key]
                        break
                else:
                    # Use first array-like value
                    for value in actions_data.values():
                        if isinstance(value, (np.ndarray, list, tuple)):
                            actions = value
                            break
                    else:
                        return None
            else:
                actions = actions_data
            
            # Convert to numpy array
            actions = np.array(actions)
            
            # Handle different dimensions
            if len(actions.shape) == 1:
                # 1D array of actions
                if len(actions) == 0:
                    return None
                # If single action, reshape to 2D
                if len(actions) <= 10:  # Likely a single action
                    actions = actions.reshape(1, -1)
            elif len(actions.shape) == 2:
                # 2D array [timesteps, action_dim]
                pass
            else:
                # Higher dimensional, take first two dimensions
                actions = actions.reshape(actions.shape[0], -1)
            
            return actions
            
        except Exception as e:
            if self.config['verbose']:
                print(f"Error extracting actions: {e}")
            return None
    
    def _infer_instruction(self, traj_dir):
        """Infer instruction from trajectory path"""
        path_str = str(traj_dir).lower()
        
        if 'pnp' in path_str:
            return "pick up the object and place it in the bowl"
        elif 'grab' in path_str:
            return "grab the block and move it to the target"
        elif 'move' in path_str:
            return "move the item to the desired location"
        elif 'place' in path_str:
            return "place the object at the target location"
        else:
            return "complete the manipulation task"
    
    def predict_action(self, image_path, instruction):
        """Predict action with error handling"""
        try:
            image = Image.open(image_path).convert("RGB")
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            
            inputs = self.processor(prompt, image).to(self.device, dtype=torch.float32)
            
            with torch.inference_mode():
                action = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            
            if hasattr(action, "cpu"):
                return action.cpu().numpy()
            else:
                return np.array(action)
                
        except Exception as e:
            if self.config['verbose']:
                print(f"❌ Prediction error for {image_path}: {e}")
            return None
    
    def normalize_actions(self, actions):
        """Normalize actions to handle scaling issues"""
        actions = np.array(actions)
        
        # Handle potential scaling issues
        # Check if actions are in reasonable range
        max_abs = np.max(np.abs(actions))
        if max_abs > 10:
            # Likely needs normalization
            actions = actions / max_abs
        
        # Check for very small values that might cause MAPE issues
        actions = np.where(np.abs(actions) < 1e-6, 1e-6, actions)
        
        return actions
    
    def calculate_metrics(self, pred_action, gt_action):
        """Calculate metrics with proper error handling"""
        # Ensure both are numpy arrays
        pred_action = np.array(pred_action).flatten()
        gt_action = np.array(gt_action).flatten()
        
        # Normalize to handle scaling issues
        pred_action = self.normalize_actions(pred_action)
        gt_action = self.normalize_actions(gt_action)
        
        # Ensure same length
        min_len = min(len(pred_action), len(gt_action))
        pred_action = pred_action[:min_len]
        gt_action = gt_action[:min_len]
        
        # Calculate metrics
        mae = np.mean(np.abs(pred_action - gt_action))
        mse = np.mean((pred_action - gt_action) ** 2)
        
        # Calculate MAPE with handling for small values
        mask = np.abs(gt_action) > 1e-6
        if np.any(mask):
            mape = np.mean(np.abs((gt_action[mask] - pred_action[mask]) / gt_action[mask])) * 100
        else:
            mape = np.inf
        
        return mae, mse, mape
    
    def evaluate_samples(self, samples):
        """Evaluate all samples with progress tracking"""
        print(f"\n🚀 Starting Evaluation")
        print(f"   Samples: {len(samples)}")
        print(f"   Model: {self.config['model_name']}")
        
        all_results = []
        total_start_time = time.time()
        
        for sample_idx, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
            instruction = sample['instruction']
            gt_actions = sample['gt_actions']
            images = sample['images']
            
            print(f"\n📸 Sample {sample_idx + 1}: {Path(sample['path']).name}")
            print(f"   Instruction: {instruction}")
            print(f"   Images: {len(images)}, GT actions shape: {gt_actions.shape}")
            
            for timestep, img_path in enumerate(images):
                # Get ground truth action
                if len(gt_actions.shape) > 1:
                    gt_action = gt_actions[timestep]
                else:
                    gt_action = gt_actions
                
                # Predict action
                start_time = time.time()
                pred_action = self.predict_action(str(img_path), instruction)
                pred_time = time.time() - start_time
                
                if pred_action is None:
                    print(f"   ❌ Prediction failed for {img_path.name}")
                    continue
                
                # Calculate metrics
                mae, mse, mape = self.calculate_metrics(pred_action, gt_action)
                
                result = {
                    'sample_idx': sample_idx,
                    'timestep': timestep,
                    'image_name': img_path.name,
                    'instruction': instruction,
                    'predicted': pred_action.tolist(),
                    'ground_truth': gt_action.tolist(),
                    'mae': float(mae),
                    'mse': float(mse),
                    'mape': float(mape),
                    'prediction_time': pred_time
                }
                
                all_results.append(result)
                
                print(f"   🖼️  {img_path.name}: MAE={mae:.4f}, MSE={mse:.4f}, MAPE={mape:.1f}%")
        
        total_time = time.time() - total_start_time
        
        # Calculate summary statistics
        if all_results:
            summary = self._calculate_summary(all_results, total_time)
        else:
            summary = {'error': 'No successful predictions'}
        
        return {
            'config': self.config,
            'summary': summary,
            'detailed_results': all_results
        }
    
    def _calculate_summary(self, results, total_time):
        """Calculate summary statistics"""
        mae_values = [r['mae'] for r in results]
        mse_values = [r['mse'] for r in results]
        mape_values = [r['mape'] for r in results if r['mape'] != np.inf]
        
        summary = {
            'total_predictions': len(results),
            'total_time_seconds': total_time,
            'avg_prediction_time': np.mean([r['prediction_time'] for r in results]),
            'mae': {
                'mean': float(np.mean(mae_values)),
                'std': float(np.std(mae_values)),
                'min': float(np.min(mae_values)),
                'max': float(np.max(mae_values))
            },
            'mse': {
                'mean': float(np.mean(mse_values)),
                'std': float(np.std(mse_values)),
                'min': float(np.min(mse_values)),
                'max': float(np.max(mse_values))
            }
        }
        
        if mape_values:
            summary['mape'] = {
                'mean': float(np.mean(mape_values)),
                'std': float(np.std(mape_values)),
                'min': float(np.min(mape_values)),
                'max': float(np.max(mape_values))
            }
        else:
            summary['mape'] = {'mean': np.inf, 'std': 0, 'min': np.inf, 'max': np.inf}
        
        return summary
    
    def print_results(self, results):
        """Print evaluation results"""
        if 'error' in results['summary']:
            print(f"❌ {results['summary']['error']}")
            return
        
        summary = results['summary']
        
        print("\n" + "="*60)
        print("🎯 RELIABLE OPENVLA EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📊 Performance Metrics:")
        print(f"   MAE:  {summary['mae']['mean']:.4f} ± {summary['mae']['std']:.4f}")
        print(f"   MSE:  {summary['mse']['mean']:.4f} ± {summary['mse']['std']:.4f}")
        if summary['mape']['mean'] != np.inf:
            print(f"   MAPE: {summary['mape']['mean']:.1f}% ± {summary['mape']['std']:.1f}%")
        
        print(f"\n🔢 Evaluation Details:")
        print(f"   Total predictions: {summary['total_predictions']}")
        print(f"   Total time: {summary['total_time_seconds']:.1f} seconds")
        print(f"   Avg prediction time: {summary['avg_prediction_time']:.1f}s")
        
        print(f"\n🎯 Performance Interpretation:")
        mae_mean = summary['mae']['mean']
        if mae_mean < 0.05:
            print("   ✅ Excellent performance (MAE < 0.05)")
        elif mae_mean < 0.1:
            print("   ✅ Good performance (MAE < 0.1)")
        elif mae_mean < 0.2:
            print("   ⚠️  Moderate performance (MAE < 0.2)")
        else:
            print("   ❌ Poor performance (MAE > 0.2)")
    
    def save_results(self, results):
        """Save results to file"""
        output_file = self.config['output_file']
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")

def main():
    """Main evaluation function with argument parsing"""
    parser = argparse.ArgumentParser(description='Reliable OpenVLA Evaluation')
    parser.add_argument('--samples', type=int, default=20, help='Number of samples to evaluate')
    parser.add_argument('--timesteps', type=int, default=5, help='Max timesteps per sample')
    parser.add_argument('--output', type=str, default='reliable_evaluation_results.json', help='Output file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create configuration
    config = ReliableOpenVLAEvaluator()._default_config()
    config.update({
        'max_samples': args.samples,
        'max_timesteps_per_sample': args.timesteps,
        'output_file': args.output,
        'verbose': args.verbose
    })
    
    print("🚀 Reliable OpenVLA Evaluation Framework")
    print("=" * 50)
    print(f"Configuration: {config}")
    
    # Initialize evaluator
    evaluator = ReliableOpenVLAEvaluator(config)
    
    # Load model
    if not evaluator.load_model():
        print("❌ Failed to load model. Exiting.")
        return
    
    # Load samples
    try:
        samples = evaluator.load_bridgedata_samples()
        if not samples:
            print("❌ No samples loaded. Exiting.")
            return
    except Exception as e:
        print(f"❌ Failed to load samples: {e}")
        return
    
    # Run evaluation
    results = evaluator.evaluate_samples(samples)
    
    # Print and save results
    evaluator.print_results(results)
    evaluator.save_results(results)
    
    print(f"\n✅ Reliable evaluation complete!")

if __name__ == "__main__":
    main()
