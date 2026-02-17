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

# Set environment for stability (same as fast test)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def evaluate_task_completion(pred_action, gt_action, instruction):
    """
    Evaluate task completion based on action components and instruction type
    Returns True if task would likely be completed successfully
    """
    pred_action = np.array(pred_action).flatten()
    gt_action = np.array(gt_action).flatten()
    
    # Ensure both are 7D
    if len(pred_action) != 7 or len(gt_action) != 7:
        return False
    
    # Extract action components: [x, y, z, roll, pitch, yaw, gripper]
    pred_pos = pred_action[:3]  # Position
    pred_rot = pred_action[3:6]  # Rotation
    pred_grip = pred_action[6]   # Gripper
    
    gt_pos = gt_action[:3]
    gt_rot = gt_action[3:6]
    gt_grip = gt_action[6]
    
    # Position accuracy (within 5cm for task completion)
    pos_error = np.linalg.norm(pred_pos - gt_pos)
    pos_success = pos_error < 0.05  # 5cm threshold
    
    # Rotation accuracy (within 15 degrees)
    rot_error = np.linalg.norm(pred_rot - gt_rot)
    rot_success = rot_error < 0.26  # ~15 degrees in radians
    
    # Gripper state accuracy
    # For picking tasks, gripper should be closed (>0.5)
    # For placing tasks, gripper should be open (<0.5)
    if 'pick' in instruction.lower():
        grip_success = pred_grip > 0.5 and gt_grip > 0.5
    elif 'place' in instruction.lower():
        grip_success = pred_grip < 0.5 and gt_grip < 0.5
    else:
        # General gripper accuracy
        grip_success = abs(pred_grip - gt_grip) < 0.3
    
    # Overall task success requires all components to be successful
    # For simpler tasks, position + gripper might be sufficient
    if 'pick' in instruction.lower() or 'place' in instruction.lower():
        return pos_success and grip_success
    else:
        return pos_success and rot_success and grip_success

def load_working_model():
    """Load model using the working approach"""
    print("🔄 Loading OpenVLA model...")
    try:
        processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to("cpu")
        
        print("✅ Model loaded successfully")
        return processor, model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

def load_sample_data(max_samples=3):
    """Load sample data for evaluation"""
    print(f"📊 Loading sample data (max {max_samples})...")
    
    samples = []
    data_dir = Path("data/scripted_raw")
    
    for root, dirs, files in os.walk(data_dir):
        if 'policy_out.pkl' in files and len(samples) < max_samples:
            traj_dir = Path(root)
            img_dir = traj_dir / "images0"
            
            if not img_dir.exists():
                continue
                
            img_files = list(img_dir.glob("im_*.jpg"))
            if len(img_files) < 2:
                continue
            
            try:
                with open(traj_dir / "policy_out.pkl", 'rb') as f:
                    actions = pickle.load(f)
                
                # Handle different action formats
                if isinstance(actions, dict):
                    action_data = actions.get('actions', actions.get('action', list(actions.values())[0]))
                    if isinstance(action_data, (list, tuple)) and len(action_data) > 0:
                        gt_actions = np.array(action_data)
                    elif isinstance(action_data, np.ndarray):
                        gt_actions = action_data
                    else:
                        continue
                elif isinstance(actions, (list, tuple)):
                    gt_actions = np.array(actions)
                elif isinstance(actions, np.ndarray):
                    gt_actions = actions
                else:
                    continue
                
                print(f"✅ Found trajectory: {traj_dir.name}")
                print(f"   Actions shape: {gt_actions.shape}")
                
                # Determine instruction
                path_str = str(root).lower()
                if 'pnp' in path_str:
                    instruction = "pick up the object and place it in the bowl"
                elif 'grab' in path_str:
                    instruction = "grab the block and move it to the target"
                elif 'move' in path_str:
                    instruction = "move the item to the desired location"
                else:
                    instruction = "complete the manipulation task"
                
                samples.append({
                    'path': str(traj_dir),
                    'instruction': instruction,
                    'images': img_files[:2],  # Use first 2 images
                    'gt_actions': gt_actions
                })
                
            except Exception as e:
                print(f"⚠️  Error loading {root}: {e}")
                continue
    
    print(f"✅ Loaded {len(samples)} samples")
    return samples

def predict_action_working(processor, model, image_path, instruction):
    """Predict action using working approach"""
    try:
        image = Image.open(image_path).convert("RGB")
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        
        inputs = processor(prompt, image).to("cpu", dtype=torch.float32)
        
        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        
        if hasattr(action, "cpu"):
            return action.cpu().numpy()
        else:
            return np.array(action)
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def run_evaluation():
    """Run the evaluation"""
    print("🚀 Working Zero-Shot Evaluation")
    print("=" * 40)
    
    # Load model
    processor, model = load_working_model()
    if processor is None or model is None:
        return
    
    # Load samples
    samples = load_sample_data(max_samples=500)  # Increased for statistical power
    if not samples:
        print("❌ No samples loaded")
        return
    
    print(f"\n🎯 Running evaluation...")
    
    all_mae = []
    all_mse = []
    results = []
    
    for sample_idx, sample in enumerate(samples):
        instruction = sample['instruction']
        gt_actions = sample['gt_actions']
        images = sample['images']
        
        print(f"\n📸 Sample {sample_idx + 1}: {sample['path'].split('/')[-1]}")
        print(f"   Instruction: {instruction}")
        
        for timestep, img_path in enumerate(images):
            if timestep >= len(gt_actions):
                continue
                
            # Get ground truth action
            if len(gt_actions.shape) > 1:
                gt_action = gt_actions[timestep]
            else:
                # If 1D array with 49 elements, use timestep as index
                if timestep < len(gt_actions):
                    gt_action = gt_actions[timestep]
                else:
                    gt_action = gt_actions[0]  # fallback to first
            
            # Debug: print type and content
            print(f"   Debug: gt_action type={type(gt_action)}, shape={getattr(gt_action, 'shape', 'N/A')}")
            
            # Handle dictionary case
            if isinstance(gt_action, dict):
                print(f"   Debug: gt_action keys={list(gt_action.keys())}")
                # Try to extract numeric values from dict
                for key in ['actions', 'action', 'qpos', 'commands', 'joint_positions', 'state']:
                    if key in gt_action:
                        gt_action = gt_action[key]
                        print(f"   Debug: extracted using key '{key}'")
                        break
                else:
                    # If no known keys, try to get first numeric value
                    for value in gt_action.values():
                        if isinstance(value, (int, float, np.ndarray, list, tuple)) and not isinstance(value, dict):
                            gt_action = value
                            print(f"   Debug: extracted first numeric value")
                            break
            
            # Convert to numpy array if it's not already
            gt_action = np.array(gt_action).flatten()
            print(f"   Debug: after conversion shape={gt_action.shape}")
            
            # If we have more than 7 elements, take first 7 (likely position/orientation)
            if len(gt_action) > 7:
                gt_action = gt_action[:7]
                print(f"   Debug: truncated to first 7 elements")
            
            # Ensure exactly 7-dimensional
            if len(gt_action) != 7:
                if len(gt_action) > 7:
                    gt_action = gt_action[:7]
                else:
                    gt_action = np.pad(gt_action, (0, 7 - len(gt_action)))
            
            print(f"   Debug: final shape={gt_action.shape}")
            print(f"   🖼️  Image {timestep + 1}: {img_path.name}")
            print(f"   ✅ Ground Truth: [{', '.join([f'{float(x):.4f}' for x in gt_action])}]")
            
            # Predict action
            start_time = time.time()
            pred_action = predict_action_working(processor, model, str(img_path), instruction)
            pred_time = time.time() - start_time
            
            if pred_action is None:
                print(f"   ❌ Prediction failed")
                continue
            
            # Ensure 7-dimensional prediction
            if len(pred_action) != 7: 
                if len(pred_action) >= 7:
                    pred_action = pred_action[:7]
                else:
                    pred_action = np.pad(pred_action, (0, 7 - len(pred_action)))
            
            print(f"   🎯 Predicted:   [{', '.join([f'{x:.4f}' for x in pred_action])}]")
            print(f"   ⏱️  Time: {pred_time:.1f}s")
            
            # Calculate metrics
            mae = np.mean(np.abs(pred_action - gt_action))
            mse = np.mean((pred_action - gt_action) ** 2)
            
            # Task completion success rate based on action components
            success = evaluate_task_completion(pred_action, gt_action, instruction)
            
            print(f"   📊 MAE: {mae:.4f}, MSE: {mse:.4f}, Task Success: {'✅' if success else '❌'}")
            
            all_mae.append(mae)
            all_mse.append(mse)
            
            results.append({
                'sample': sample_idx,
                'timestep': timestep,
                'instruction': instruction,
                'predicted': pred_action.tolist(),
                'ground_truth': gt_action.tolist(),
                'mae': mae,
                'mse': mse,
                'task_success': success,
                'prediction_time': pred_time
            })
    
    # Summary
    if all_mae:
        print(f"\n" + "="*60)
        print("📈 EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\n🎯 Overall Performance:")
        print(f"   MAE:  {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
        print(f"   MSE:  {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
        
        # Calculate task completion success rate
        task_successes = [r['task_success'] for r in results]
        task_success_rate = np.mean(task_successes) * 100
        print(f"   Task Completion Rate: {task_success_rate:.1f}% ({sum(task_successes)}/{len(task_successes)})")
        
        print(f"\n⏱️  Timing:")
        print(f"   Total predictions: {len(all_mae)}")
        print(f"   Avg prediction time: {np.mean([r['prediction_time'] for r in results]):.1f}s")
        
        # Save results
        import json
        summary = {
            'summary': {
                'avg_mae': float(np.mean(all_mae)),
                'avg_mse': float(np.mean(all_mse)),
                'std_mae': float(np.std(all_mae)),
                'std_mse': float(np.std(all_mse)),
                'task_completion_rate': float(task_success_rate),
                'task_success_count': int(sum(task_successes)),
                'total_predictions': len(all_mae)
            },
            'detailed_results': results
        }
        
        print(f"\n✅ Evaluation complete!")
        
        # Convert numpy types to Python types for JSON serialization
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
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Convert the entire summary structure
        json_compatible_summary = convert_numpy(summary)
        
        # Save results to file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"baseline_500_samples_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(json_compatible_summary, f, indent=2)
        
        print(f"💾 Saved results to {filename}")
        print(f"📊 Total predictions: {len(all_mae)}")
        print(f"📈 Average MAE: {np.mean(all_mae):.4f}")
        print(f"🎯 Task completion rate: {task_success_rate:.1%}")

if __name__ == "__main__":
    run_evaluation()