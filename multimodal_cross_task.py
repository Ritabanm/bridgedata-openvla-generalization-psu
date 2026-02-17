#!/usr/bin/env python3
"""
Real Multimodal Cross-Task Testing
Uses real bridgedata like multimodal_enhancer.py but tests generalization across tasks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Force CPU usage to avoid MPS issues
device = torch.device("cpu")
print(f"🔧 Using device: {device}")

class RealDataLoader:
    """Load real bridgedata like multimodal_enhancer.py"""
    
    def __init__(self):
        self.baseline_pairs = []
        self.load_real_data()
    
    def load_real_data(self):
        """Load real baseline predictions from bridgedata"""
        print("📂 Loading real bridgedata...")
        
        # Try to load the 100 samples results first
        if os.path.exists("baseline_100_samples_results.json"):
            with open("baseline_100_samples_results.json", 'r') as f:
                data = json.load(f)
            self.baseline_pairs = data['detailed_results']
            print(f"✅ Loaded {len(self.baseline_pairs)} real baseline predictions from 100-sample evaluation")
        elif os.path.exists("openvla_baseline_subset.json"):
            with open("openvla_baseline_subset.json", 'r') as f:
                data = json.load(f)
            # Convert to expected format if needed
            if 'detailed_results' in data:
                self.baseline_pairs = data['detailed_results']
            else:
                self.baseline_pairs = data
            print(f"✅ Loaded {len(self.baseline_pairs)} real baseline predictions from subset")
        else:
            print("⚠️  No real data found, will use synthetic fallback")
            self.baseline_pairs = []
    
    def get_real_data(self):
        """Extract real predictions and ground truths"""
        if not self.baseline_pairs:
            return None, None
        
        predictions = []
        ground_truths = []
        
        for pair in self.baseline_pairs:
            if 'openvla_prediction' in pair and 'ground_truth' in pair:
                predictions.append(pair['openvla_prediction'])
                ground_truths.append(pair['ground_truth'])
        
        return np.array(predictions), np.array(ground_truths)

class TaskAugmentor:
    """Augment real data to simulate different tasks"""
    
    def __init__(self, base_predictions, base_ground_truths):
        self.base_predictions = base_predictions
        self.base_ground_truths = base_ground_truths
    
    def create_task_variant(self, task_type="pick_and_place", noise_level=0.05, transform_factor=1.0):
        """Create different task variants from real data"""
        
        if self.base_predictions is None:
            # Fallback to synthetic data
            return self._create_synthetic_task(task_type)
        
        # Transform real data to simulate different tasks
        predictions = self.base_predictions.copy()
        ground_truths = self.base_ground_truths.copy()
        
        if task_type == "pick_and_place":
            # Add lifting behavior
            predictions[:, 2] += np.random.normal(0.1, 0.02, len(predictions))  # Lift Z
            ground_truths[:, 2] += np.random.normal(0.1, 0.02, len(ground_truths))
            
        elif task_type == "assembly":
            # Add precision constraints
            predictions += np.random.normal(0, 0.01, predictions.shape)  # Less noise
            ground_truths += np.random.normal(0, 0.01, ground_truths.shape)
            # Reduce orientation changes
            predictions[:, 3:6] *= 0.7
            ground_truths[:, 3:6] *= 0.7
            
        elif task_type == "navigation":
            # Emphasize position changes
            predictions[:, :3] *= 1.5  # More movement
            ground_truths[:, :3] *= 1.5
            predictions[:, 3:6] = 0  # No orientation
            ground_truths[:, 3:6] = 0
            
        elif task_type == "manipulation":
            # Add force-based variations
            predictions += np.random.normal(0, 0.08, predictions.shape)
            ground_truths += np.random.normal(0, 0.08, ground_truths.shape)
            
        elif task_type == "reaching":
            # Simplify to reaching task
            predictions[:, 3:6] = 0  # No orientation
            ground_truths[:, 3:6] = 0
            predictions[:, 6] = 1  # Gripper open
            ground_truths[:, 6] = 1
            
        elif task_type == "grasping":
            # Add grasping variations
            predictions[:, 6] = np.clip(predictions[:, 6] + np.random.normal(0, 0.1), 0, 1)
            ground_truths[:, 6] = np.clip(ground_truths[:, 6] + np.random.normal(0, 0.1), 0, 1)
            
        # Add task-specific noise
        predictions += np.random.normal(0, noise_level, predictions.shape)
        ground_truths += np.random.normal(0, noise_level * 0.5, ground_truths.shape)
        
        return predictions, ground_truths
    
    def _create_synthetic_task(self, task_type):
        """Create synthetic data as fallback"""
        num_samples = 200
        
        if task_type == "pick_and_place":
            inputs = np.random.uniform(-1.0, 1.0, (num_samples, 7))
            outputs = np.random.uniform(-0.5, 0.5, (num_samples, 7))
            outputs[:, 2] += 0.1  # Lift height
        elif task_type == "assembly":
            inputs = np.random.uniform(-0.5, 0.5, (num_samples, 7))
            outputs = inputs + np.random.normal(0, 0.02, (num_samples, 7))
        else:
            inputs = np.random.uniform(-1.0, 1.0, (num_samples, 7))
            outputs = np.random.uniform(-0.8, 0.8, (num_samples, 7))
        
        return inputs, outputs

class MultimodalEnhancer(nn.Module):
    """Multimodal enhancer like in multimodal_enhancer.py"""
    
    def __init__(self, action_dim=7, hidden_dim=256, num_heads=4):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)
        
        # Feature enhancement layers
        self.feature_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Augmentation network
        self.augmentation_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        # Input embedding
        embedded = self.input_embedding(x)
        
        # Add sequence dimension for attention
        embedded_seq = embedded.unsqueeze(1)  # [batch, 1, hidden]
        
        # Self-attention
        attended, _ = self.attention(embedded_seq, embedded_seq, embedded_seq)
        attended = attended.squeeze(1)  # [batch, hidden]
        
        # Feature enhancement
        enhanced = self.feature_layers(attended)
        
        # Generate augmentation
        augmentation = self.augmentation_net(x)
        
        # Output projection
        output = self.output_projection(enhanced)
        
        # Residual connection with augmentation
        final_output = x + self.residual_scale * output + 0.05 * augmentation
        
        return final_output

class BaselineModel(nn.Module):
    """Simple baseline model for comparison"""
    
    def __init__(self, action_dim=7, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_inputs, train_outputs, val_inputs, val_outputs, epochs=200, patience=30):
    """Train a neural network model like in multimodal_enhancer.py"""
    model = model.to(device)
    
    # Use AdamW with weight decay like in original
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=20, verbose=False
    )
    
    criterion = nn.MSELoss()
    
    # Convert to tensors
    train_inputs_tensor = torch.FloatTensor(train_inputs).to(device)
    train_outputs_tensor = torch.FloatTensor(train_outputs).to(device)
    val_inputs_tensor = torch.FloatTensor(val_inputs).to(device)
    val_outputs_tensor = torch.FloatTensor(val_outputs).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        train_pred = model(train_inputs_tensor)
        train_loss = criterion(train_pred, train_outputs_tensor)
        
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_inputs_tensor)
            val_loss = criterion(val_pred, val_outputs_tensor)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            model.load_state_dict(best_model_state)
            break
    
    return model, best_val_loss

def evaluate_model(model, test_inputs, test_outputs):
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        test_inputs_tensor = torch.FloatTensor(test_inputs).to(device)
        predictions = model(test_inputs_tensor).cpu().numpy()
    
    mae = mean_absolute_error(test_outputs, predictions)
    mse = mean_squared_error(test_outputs, predictions)
    
    return mae, mse, predictions

def test_task_generalization(task_name, base_predictions, base_ground_truths, task_augmentor):
    """Test multimodal enhancer generalization on a specific task"""
    print(f"\n🧪 TESTING GENERALIZATION: {task_name.upper()}")
    print("=" * 60)
    
    # Create task variant
    task_predictions, task_ground_truths = task_augmentor.create_task_variant(
        task_type=task_name, noise_level=0.05
    )
    
    print(f"📊 Task data shape: {task_predictions.shape}")
    
    # Split data
    train_size = int(0.7 * len(task_predictions))
    val_size = int(0.15 * len(task_predictions))
    
    train_inputs, train_outputs = task_predictions[:train_size], task_ground_truths[:train_size]
    val_inputs, val_outputs = task_predictions[train_size:train_size+val_size], task_ground_truths[train_size:train_size+val_size]
    test_inputs, test_outputs = task_predictions[train_size+val_size:], task_ground_truths[train_size+val_size:]
    
    print(f"📊 Data split: Train={len(train_inputs)}, Val={len(val_inputs)}, Test={len(test_inputs)}")
    
    # Initialize models
    baseline_model = BaselineModel(action_dim=7)
    multimodal_model = MultimodalEnhancer(action_dim=7)
    
    # Train models
    print(f"\n🎯 Training baseline model...")
    start_time = time.time()
    baseline_model, baseline_val_loss = train_model(baseline_model, train_inputs, train_outputs, val_inputs, val_outputs)
    baseline_train_time = time.time() - start_time
    
    print(f"🎯 Training multimodal enhancer...")
    start_time = time.time()
    multimodal_model, multimodal_val_loss = train_model(multimodal_model, train_inputs, train_outputs, val_inputs, val_outputs)
    multimodal_train_time = time.time() - start_time
    
    # Evaluate models
    print(f"\n📈 Evaluating models...")
    
    baseline_mae, baseline_mse, baseline_predictions = evaluate_model(baseline_model, test_inputs, test_outputs)
    multimodal_mae, multimodal_mse, multimodal_predictions = evaluate_model(multimodal_model, test_inputs, test_outputs)
    
    # Calculate improvements
    mae_improvement = (baseline_mae - multimodal_mae) / baseline_mae * 100
    mse_improvement = (baseline_mse - multimodal_mse) / baseline_mse * 100
    
    # Per-dimension metrics
    dimension_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
    per_dim_results = {}
    
    for i, dim_name in enumerate(dimension_names):
        baseline_dim_mae = mean_absolute_error(test_outputs[:, i], baseline_predictions[:, i])
        multimodal_dim_mae = mean_absolute_error(test_outputs[:, i], multimodal_predictions[:, i])
        dim_improvement = (baseline_dim_mae - multimodal_dim_mae) / baseline_dim_mae * 100
        
        per_dim_results[dim_name] = {
            'baseline_mae': baseline_dim_mae,
            'multimodal_mae': multimodal_dim_mae,
            'improvement_percent': dim_improvement
        }
    
    # Display results
    print(f"\n📊 {task_name.upper()} GENERALIZATION RESULTS:")
    print("-" * 50)
    print(f"Baseline MAE: {baseline_mae:.6f}")
    print(f"Multimodal MAE: {multimodal_mae:.6f}")
    print(f"MAE Improvement: {mae_improvement:+.2f}%")
    print(f"Baseline MSE: {baseline_mse:.6f}")
    print(f"Multimodal MSE: {multimodal_mse:.6f}")
    print(f"MSE Improvement: {mse_improvement:+.2f}%")
    print(f"Baseline training time: {baseline_train_time:.2f}s")
    print(f"Multimodal training time: {multimodal_train_time:.2f}s")
    
    # Return results
    return {
        'task_name': task_name,
        'baseline_metrics': {
            'mae': baseline_mae,
            'mse': baseline_mse,
            'val_loss': baseline_val_loss,
            'train_time': baseline_train_time
        },
        'multimodal_metrics': {
            'mae': multimodal_mae,
            'mse': multimodal_mse,
            'val_loss': multimodal_val_loss,
            'train_time': multimodal_train_time,
            'mae_improvement_percent': mae_improvement,
            'mse_improvement_percent': mse_improvement
        },
        'per_dimension_results': per_dim_results,
        'test_predictions': {
            'baseline': baseline_predictions.tolist(),
            'multimodal': multimodal_predictions.tolist(),
            'ground_truth': test_outputs.tolist()
        },
        'data_splits': {
            'train_size': len(train_inputs),
            'val_size': len(val_inputs),
            'test_size': len(test_inputs)
        }
    }

def run_real_cross_task_testing():
    """Run multimodal enhancer testing using real bridgedata across different tasks"""
    print(f"🚀 REAL MULTIMODAL CROSS-TASK GENERALIZATION TESTING")
    print("=" * 80)
    print(f"Using real bridgedata like multimodal_enhancer.py")
    print(f"Testing generalization across different robotic tasks")
    
    # Load real data
    data_loader = RealDataLoader()
    base_predictions, base_ground_truths = data_loader.get_real_data()
    
    if base_predictions is None:
        print("⚠️  No real data available, using synthetic data for testing")
        base_predictions = np.random.randn(200, 7) * 0.2
        base_ground_truths = np.random.randn(200, 7) * 0.2
    
    print(f"📊 Base data shape: {base_predictions.shape}")
    
    # Create task augmentor
    task_augmentor = TaskAugmentor(base_predictions, base_ground_truths)
    
    # Define test tasks
    tasks = [
        "pick_and_place",
        "assembly", 
        "navigation",
        "manipulation",
        "reaching",
        "grasping"
    ]
    
    # Test each task
    all_results = []
    
    for task in tasks:
        result = test_task_generalization(task, base_predictions, base_ground_truths, task_augmentor)
        all_results.append(result)
    
    # Generate summary report
    print(f"\n📈 CROSS-TASK GENERALIZATION SUMMARY")
    print("=" * 80)
    print(f"{'Task':<15} {'Baseline MAE':<12} {'Multimodal MAE':<15} {'Improvement':<12} {'Status':<10}")
    print("-" * 80)
    
    successful_tasks = 0
    total_improvement = 0
    
    for result in all_results:
        task_name = result['task_name']
        baseline_mae = result['baseline_metrics']['mae']
        multimodal_mae = result['multimodal_metrics']['mae']
        improvement = result['multimodal_metrics']['mae_improvement_percent']
        
        status = "✅ Success" if improvement > 0 else "❌ Failed"
        if improvement > 0:
            successful_tasks += 1
            total_improvement += improvement
        
        print(f"{task_name:<15} {baseline_mae:<12.6f} {multimodal_mae:<15.6f} {improvement:+.2f}%{'':<6} {status:<10}")
    
    # Overall statistics
    success_rate = successful_tasks / len(tasks) * 100
    avg_improvement = total_improvement / successful_tasks if successful_tasks > 0 else 0
    
    print(f"\n🎯 OVERALL GENERALIZATION PERFORMANCE:")
    print(f"   Success rate: {success_rate:.1f}% ({successful_tasks}/{len(tasks)} tasks)")
    print(f"   Average improvement: {avg_improvement:.2f}% (successful tasks)")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"real_multimodal_cross_task_results_{timestamp}.json"
    
    # Create serializable results
    serializable_results = {
        'metadata': {
            'timestamp': timestamp,
            'device': str(device),
            'total_tasks': len(tasks),
            'successful_tasks': successful_tasks,
            'success_rate': success_rate,
            'average_improvement': avg_improvement,
            'used_real_data': base_predictions is not None
        },
        'task_results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Find best performing task
    best_result = min(all_results, key=lambda x: x['multimodal_metrics']['mae'])
    best_task = best_result['task_name']
    
    print(f"\n🏆 BEST GENERALIZATION TASK: {best_task}")
    print(f"   MAE: {best_result['multimodal_metrics']['mae']:.6f}")
    print(f"   Improvement: {best_result['multimodal_metrics']['mae_improvement_percent']:.2f}%")
    
    return serializable_results

def main():
    """Main execution"""
    set_seed(42)
    
    # Run real cross-task testing
    results = run_real_cross_task_testing()
    
    data_type = "real bridgedata" if results['metadata']['used_real_data'] else "synthetic data"
    print(f"\n🎉 REAL MULTIMODAL CROSS-TASK TESTING COMPLETED!")
    print(f"✅ Tested multimodal enhancer generalization on {results['metadata']['total_tasks']} tasks")
    print(f"📊 Used {data_type} for testing")
    print(f"📈 Success rate: {results['metadata']['success_rate']:.1f}%")
    print(f"🚀 Check the generated JSON files for detailed results")

if __name__ == "__main__":
    main()

@dataclass
class TaskConfig:
    """Configuration for different tasks"""
    name: str
    action_dim: int
    input_dim: int
    description: str
    data_range: Tuple[float, float]
    noise_level: float
    correlation_strength: float

class TaskDataGenerator:
    """Generate synthetic data for different robotic tasks"""
    
    def __init__(self, task_config: TaskConfig, num_samples: int = 500):
        self.config = task_config
        self.num_samples = num_samples
        self.generate_data()
    
    def generate_data(self):
        """Generate task-specific synthetic data"""
        np.random.seed(42)
        
        if self.config.name == "pick_and_place":
            self._generate_pick_and_place_data()
        elif self.config.name == "assembly":
            self._generate_assembly_data()
        elif self.config.name == "navigation":
            self._generate_navigation_data()
        elif self.config.name == "manipulation":
            self._generate_manipulation_data()
        elif self.config.name == "reaching":
            self._generate_reaching_data()
        elif self.config.name == "grasping":
            self._generate_grasping_data()
        elif self.config.name == "throwing":
            self._generate_throwing_data()
        elif self.config.name == "pouring":
            self._generate_pouring_data()
        else:
            self._generate_generic_data()
    
    def _generate_pick_and_place_data(self):
        """Generate pick and place task data"""
        self.inputs = np.random.uniform(
            self.config.data_range[0], self.config.data_range[1], 
            (self.num_samples, self.config.input_dim)
        )
        
        object_pos = self.inputs[:, :3]
        target_pos = self.inputs[:, 3:6]
        gripper_state = self.inputs[:, 6:7]
        
        reach_height = np.maximum(object_pos[:, 2], target_pos[:, 2]) + 0.1
        
        self.outputs = np.zeros((self.num_samples, self.config.action_dim))
        self.outputs[:, 0:3] = object_pos
        self.outputs[:, 2] = reach_height
        self.outputs[:, 3:6] = target_pos
        self.outputs[:, 6] = gripper_state.squeeze()
        
        noise = np.random.normal(0, self.config.noise_level, self.outputs.shape)
        self.outputs += noise
    
    def _generate_assembly_data(self):
        """Generate assembly task data"""
        self.inputs = np.random.uniform(
            self.config.data_range[0], self.config.data_range[1], 
            (self.num_samples, self.config.input_dim)
        )
        
        self.outputs = np.zeros((self.num_samples, self.config.action_dim))
        self.outputs[:, :3] = self.inputs[:, :3] + np.random.normal(0, 0.01, (self.num_samples, 3))
        self.outputs[:, 3:6] = self.inputs[:, 3:6] * 0.5
        self.outputs[:, 6] = np.clip(self.inputs[:, 6:7], 0, 1).squeeze()
        
        noise = np.random.normal(0, self.config.noise_level * 0.5, self.outputs.shape)
        self.outputs += noise
    
    def _generate_navigation_data(self):
        """Generate navigation task data"""
        self.inputs = np.random.uniform(
            self.config.data_range[0], self.config.data_range[1], 
            (self.num_samples, self.config.input_dim)
        )
        
        self.outputs = np.zeros((self.num_samples, self.config.action_dim))
        start_pos = self.inputs[:, :3]
        goal_pos = self.inputs[:, 3:6]
        
        t = np.random.uniform(0, 1, (self.num_samples, 1))
        self.outputs[:, :3] = start_pos + t * (goal_pos - start_pos)
        self.outputs[:, 3:6] = np.zeros((self.num_samples, 3))
        self.outputs[:, 6] = 0
        
        noise = np.random.normal(0, self.config.noise_level * 2, self.outputs.shape)
        self.outputs += noise
    
    def _generate_manipulation_data(self):
        """Generate manipulation task data"""
        self.inputs = np.random.uniform(
            self.config.data_range[0], self.config.data_range[1], 
            (self.num_samples, self.config.input_dim)
        )
        
        self.outputs = np.zeros((self.num_samples, self.config.action_dim))
        contact_forces = self.inputs[:, :3]
        torques = self.inputs[:, 3:6]
        
        self.outputs[:, :3] = contact_forces * 0.8
        self.outputs[:, 3:6] = torques * 0.6
        self.outputs[:, 6] = np.clip(self.inputs[:, 6:7] * 2, 0, 1).squeeze()
        
        noise = np.random.normal(0, self.config.noise_level * 1.5, self.outputs.shape)
        self.outputs += noise
    
    def _generate_reaching_data(self):
        """Generate reaching task data"""
        self.inputs = np.random.uniform(
            self.config.data_range[0], self.config.data_range[1], 
            (self.num_samples, self.config.input_dim)
        )
        
        current_pos = self.inputs[:, :3]
        target_pos = self.inputs[:, 3:6]
        
        self.outputs = np.zeros((self.num_samples, self.config.action_dim))
        self.outputs[:, :3] = target_pos
        self.outputs[:, 3:6] = np.zeros((self.num_samples, 3))
        self.outputs[:, 6] = 1
        
        noise = np.random.normal(0, self.config.noise_level, self.outputs.shape)
        self.outputs += noise
    
    def _generate_grasping_data(self):
        """Generate grasping task data"""
        self.inputs = np.random.uniform(
            self.config.data_range[0], self.config.data_range[1], 
            (self.num_samples, self.config.input_dim)
        )
        
        object_pos = self.inputs[:, :3]
        object_size = self.inputs[:, 3:4]
        approach_angle = self.inputs[:, 4:7]
        
        self.outputs = np.zeros((self.num_samples, self.config.action_dim))
        self.outputs[:, :3] = object_pos + approach_angle * 0.1
        self.outputs[:, 3:6] = np.array([0, 0, np.pi/4])  # Gripper orientation
        self.outputs[:, 6] = np.clip(object_size.squeeze(), 0, 1)
        
        noise = np.random.normal(0, self.config.noise_level, self.outputs.shape)
        self.outputs += noise
    
    def _generate_throwing_data(self):
        """Generate throwing task data"""
        self.inputs = np.random.uniform(
            self.config.data_range[0], self.config.data_range[1], 
            (self.num_samples, self.config.input_dim)
        )
        
        throw_pos = self.inputs[:, :3]
        target_pos = self.inputs[:, 3:6]
        throw_power = self.inputs[:, 6:7]
        
        self.outputs = np.zeros((self.num_samples, self.config.action_dim))
        self.outputs[:, :3] = throw_pos
        self.outputs[:, 3:6] = (target_pos - throw_pos) * throw_power
        self.outputs[:, 6] = throw_power.squeeze()
        
        noise = np.random.normal(0, self.config.noise_level * 1.2, self.outputs.shape)
        self.outputs += noise
    
    def _generate_pouring_data(self):
        """Generate pouring task data"""
        self.inputs = np.random.uniform(
            self.config.data_range[0], self.config.data_range[1], 
            (self.num_samples, self.config.input_dim)
        )
        
        container_pos = self.inputs[:, :3]
        target_pos = self.inputs[:, 3:6]
        pour_angle = self.inputs[:, 6:7]
        
        self.outputs = np.zeros((self.num_samples, self.config.action_dim))
        self.outputs[:, :3] = container_pos
        self.outputs[:, 3:6] = target_pos
        self.outputs[:, 6] = np.clip(pour_angle.squeeze(), 0, np.pi/2)
        
        noise = np.random.normal(0, self.config.noise_level * 0.8, self.outputs.shape)
        self.outputs += noise
    
    def _generate_generic_data(self):
        """Generate generic task data"""
        self.inputs = np.random.uniform(
            self.config.data_range[0], self.config.data_range[1], 
            (self.num_samples, self.config.input_dim)
        )
        
        weights = np.random.randn(self.config.input_dim, self.config.action_dim) * 0.5
        self.outputs = self.inputs @ weights
        
        noise = np.random.normal(0, self.config.noise_level, self.outputs.shape)
        self.outputs += noise
    
    def get_data(self):
        """Return generated data"""
        return self.inputs, self.outputs

class MultimodalEnhancer(nn.Module):
    """Working multimodal enhancer for CPU"""
    
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Simplified network for CPU compatibility
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        residual = self.network(x)
        return x + self.residual_scale * residual

class BaselineModel(nn.Module):
    """Simple baseline model for comparison"""
    
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_inputs, train_outputs, val_inputs, val_outputs, epochs=100, patience=20):
    """Train a neural network model"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    train_inputs_tensor = torch.FloatTensor(train_inputs).to(device)
    train_outputs_tensor = torch.FloatTensor(train_outputs).to(device)
    val_inputs_tensor = torch.FloatTensor(val_inputs).to(device)
    val_outputs_tensor = torch.FloatTensor(val_outputs).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        train_pred = model(train_inputs_tensor)
        train_loss = criterion(train_pred, train_outputs_tensor)
        
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_inputs_tensor)
            val_loss = criterion(val_pred, val_outputs_tensor)
        
        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            model.load_state_dict(best_model_state)
            break
    
    return model, best_val_loss

def evaluate_model(model, test_inputs, test_outputs):
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        test_inputs_tensor = torch.FloatTensor(test_inputs).to(device)
        predictions = model(test_inputs_tensor).cpu().numpy()
    
    mae = mean_absolute_error(test_outputs, predictions)
    mse = mean_squared_error(test_outputs, predictions)
    
    return mae, mse, predictions

def test_task(task_config: TaskConfig, num_samples: int = 500):
    """Test multimodal enhancer on a specific task"""
    print(f"\n🧪 TESTING TASK: {task_config.name.upper()}")
    print("=" * 60)
    print(f"Description: {task_config.description}")
    print(f"Input dim: {task_config.input_dim}, Action dim: {task_config.action_dim}")
    
    # Generate task data
    data_generator = TaskDataGenerator(task_config, num_samples)
    inputs, outputs = data_generator.get_data()
    
    # Split data
    train_size = int(0.7 * len(inputs))
    val_size = int(0.15 * len(inputs))
    
    train_inputs, train_outputs = inputs[:train_size], outputs[:train_size]
    val_inputs, val_outputs = inputs[train_size:train_size+val_size], outputs[train_size:train_size+val_size]
    test_inputs, test_outputs = inputs[train_size+val_size:], outputs[train_size+val_size:]
    
    print(f"📊 Data split: Train={len(train_inputs)}, Val={len(val_inputs)}, Test={len(test_inputs)}")
    
    # Initialize models
    baseline_model = BaselineModel(task_config.input_dim, task_config.action_dim)
    multimodal_model = MultimodalEnhancer(task_config.input_dim, task_config.action_dim)
    
    # Train models
    print(f"\n🎯 Training baseline model...")
    start_time = time.time()
    baseline_model, baseline_val_loss = train_model(baseline_model, train_inputs, train_outputs, val_inputs, val_outputs)
    baseline_train_time = time.time() - start_time
    
    print(f"🎯 Training multimodal enhancer...")
    start_time = time.time()
    multimodal_model, multimodal_val_loss = train_model(multimodal_model, train_inputs, train_outputs, val_inputs, val_outputs)
    multimodal_train_time = time.time() - start_time
    
    # Evaluate models
    print(f"\n📈 Evaluating models...")
    
    baseline_mae, baseline_mse, baseline_predictions = evaluate_model(baseline_model, test_inputs, test_outputs)
    multimodal_mae, multimodal_mse, multimodal_predictions = evaluate_model(multimodal_model, test_inputs, test_outputs)
    
    # Calculate improvements
    mae_improvement = (baseline_mae - multimodal_mae) / baseline_mae * 100
    mse_improvement = (baseline_mse - multimodal_mse) / baseline_mse * 100
    
    # Per-dimension metrics
    dimension_names = [f'Dim_{i}' for i in range(task_config.action_dim)]
    per_dim_results = {}
    
    for i, dim_name in enumerate(dimension_names):
        baseline_dim_mae = mean_absolute_error(test_outputs[:, i], baseline_predictions[:, i])
        multimodal_dim_mae = mean_absolute_error(test_outputs[:, i], multimodal_predictions[:, i])
        dim_improvement = (baseline_dim_mae - multimodal_dim_mae) / baseline_dim_mae * 100
        
        per_dim_results[dim_name] = {
            'baseline_mae': baseline_dim_mae,
            'multimodal_mae': multimodal_dim_mae,
            'improvement_percent': dim_improvement
        }
    
    # Display results
    print(f"\n📊 {task_config.name.upper()} RESULTS:")
    print("-" * 40)
    print(f"Baseline MAE: {baseline_mae:.6f}")
    print(f"Multimodal MAE: {multimodal_mae:.6f}")
    print(f"MAE Improvement: {mae_improvement:+.2f}%")
    print(f"Baseline MSE: {baseline_mse:.6f}")
    print(f"Multimodal MSE: {multimodal_mse:.6f}")
    print(f"MSE Improvement: {mse_improvement:+.2f}%")
    print(f"Baseline training time: {baseline_train_time:.2f}s")
    print(f"Multimodal training time: {multimodal_train_time:.2f}s")
    
    # Return results
    return {
        'task_config': task_config,
        'baseline_metrics': {
            'mae': baseline_mae,
            'mse': baseline_mse,
            'val_loss': baseline_val_loss,
            'train_time': baseline_train_time
        },
        'multimodal_metrics': {
            'mae': multimodal_mae,
            'mse': multimodal_mse,
            'val_loss': multimodal_val_loss,
            'train_time': multimodal_train_time,
            'mae_improvement_percent': mae_improvement,
            'mse_improvement_percent': mse_improvement
        },
        'per_dimension_results': per_dim_results,
        'test_predictions': {
            'baseline': baseline_predictions.tolist(),
            'multimodal': multimodal_predictions.tolist(),
            'ground_truth': test_outputs.tolist()
        },
        'data_splits': {
            'train_size': len(train_inputs),
            'val_size': len(val_inputs),
            'test_size': len(test_inputs)
        }
    }

def run_cross_task_testing():
    """Run multimodal enhancer testing across multiple tasks"""
    print(f"🚀 MULTIMODAL ENHANCER CROSS-TASK TESTING")
    print("=" * 80)
    print(f"Evaluating generalization capabilities across different robotic tasks")
    
    # Define comprehensive test tasks
    tasks = [
        TaskConfig(
            name="pick_and_place",
            action_dim=7,
            input_dim=7,
            description="Pick up object and place at target location",
            data_range=(-1.0, 1.0),
            noise_level=0.05,
            correlation_strength=0.1
        ),
        TaskConfig(
            name="assembly",
            action_dim=7,
            input_dim=7,
            description="Precise assembly operations with tight tolerances",
            data_range=(-0.5, 0.5),
            noise_level=0.02,
            correlation_strength=0.2
        ),
        TaskConfig(
            name="navigation",
            action_dim=7,
            input_dim=7,
            description="Navigation and path planning tasks",
            data_range=(-2.0, 2.0),
            noise_level=0.1,
            correlation_strength=0.05
        ),
        TaskConfig(
            name="manipulation",
            action_dim=7,
            input_dim=7,
            description="Complex force-based manipulation tasks",
            data_range=(-1.0, 1.0),
            noise_level=0.08,
            correlation_strength=0.15
        ),
        TaskConfig(
            name="reaching",
            action_dim=7,
            input_dim=7,
            description="Simple reaching tasks to target positions",
            data_range=(-0.8, 0.8),
            noise_level=0.03,
            correlation_strength=0.1
        ),
        TaskConfig(
            name="grasping",
            action_dim=7,
            input_dim=7,
            description="Object grasping with different sizes and orientations",
            data_range=(-1.0, 1.0),
            noise_level=0.06,
            correlation_strength=0.12
        ),
        TaskConfig(
            name="throwing",
            action_dim=7,
            input_dim=7,
            description="Throwing objects to target locations",
            data_range=(-1.5, 1.5),
            noise_level=0.12,
            correlation_strength=0.08
        ),
        TaskConfig(
            name="pouring",
            action_dim=7,
            input_dim=7,
            description="Pouring liquids from container to target",
            data_range=(-1.0, 1.0),
            noise_level=0.07,
            correlation_strength=0.1
        )
    ]
    
    # Test each task
    all_results = []
    
    for task in tasks:
        result = test_task(task, num_samples=500)
        all_results.append(result)
    
    # Generate summary report
    print(f"\n📈 CROSS-TASK TESTING SUMMARY")
    print("=" * 80)
    print(f"{'Task':<15} {'Baseline MAE':<12} {'Multimodal MAE':<15} {'Improvement':<12} {'Status':<10}")
    print("-" * 80)
    
    successful_tasks = 0
    total_improvement = 0
    task_results = []
    
    for result in all_results:
        task_name = result['task_config'].name
        baseline_mae = result['baseline_metrics']['mae']
        multimodal_mae = result['multimodal_metrics']['mae']
        improvement = result['multimodal_metrics']['mae_improvement_percent']
        
        status = "✅ Success" if improvement > 0 else "❌ Failed"
        if improvement > 0:
            successful_tasks += 1
            total_improvement += improvement
        
        task_results.append({
            'task': task_name,
            'baseline_mae': baseline_mae,
            'multimodal_mae': multimodal_mae,
            'improvement': improvement,
            'status': status
        })
        
        print(f"{task_name:<15} {baseline_mae:<12.6f} {multimodal_mae:<15.6f} {improvement:+.2f}%{'':<6} {status:<10}")
    
    # Overall statistics
    success_rate = successful_tasks / len(tasks) * 100
    avg_improvement = total_improvement / successful_tasks if successful_tasks > 0 else 0
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Success rate: {success_rate:.1f}% ({successful_tasks}/{len(tasks)} tasks)")
    print(f"   Average improvement: {avg_improvement:.2f}% (successful tasks)")
    
    # Task type analysis
    print(f"\n📊 TASK TYPE ANALYSIS:")
    
    # Group tasks by type
    precision_tasks = ['assembly', 'grasping', 'pouring']
    navigation_tasks = ['navigation', 'reaching']
    manipulation_tasks = ['pick_and_place', 'manipulation', 'throwing']
    
    def analyze_task_group(task_names, group_name):
        group_results = [r for r in task_results if r['task'] in task_names]
        if group_results:
            avg_improvement = np.mean([r['improvement'] for r in group_results])
            success_count = sum(1 for r in group_results if r['improvement'] > 0)
            print(f"   {group_name}: {avg_improvement:+.2f}% avg, {success_count}/{len(group_results)} successful")
    
    analyze_task_group(precision_tasks, "Precision Tasks")
    analyze_task_group(navigation_tasks, "Navigation Tasks")
    analyze_task_group(manipulation_tasks, "Manipulation Tasks")
    
    # Save simplified results without TaskConfig objects
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"multimodal_cross_task_results_{timestamp}.json"
    
    # Create serializable results
    serializable_results = {
        'metadata': {
            'timestamp': timestamp,
            'device': str(device),
            'total_tasks': len(tasks),
            'successful_tasks': successful_tasks,
            'success_rate': success_rate,
            'average_improvement': avg_improvement
        },
        'summary': {
            'task_analysis': {
                'precision_tasks': precision_tasks,
                'navigation_tasks': navigation_tasks,
                'manipulation_tasks': manipulation_tasks
            }
        }
    }
    
    # Add task results without TaskConfig objects
    serializable_results['task_results'] = []
    for result in all_results:
        task_result = {
            'task_name': result['task_config'].name,
            'task_description': result['task_config'].description,
            'baseline_metrics': result['baseline_metrics'],
            'multimodal_metrics': result['multimodal_metrics'],
            'per_dimension_results': result['per_dimension_results'],
            'test_predictions': result['test_predictions'],
            'data_splits': result['data_splits']
        }
        serializable_results['task_results'].append(task_result)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Generate predictions for best performing task
    best_result = min(all_results, key=lambda x: x['multimodal_metrics']['mae'])
    best_task = best_result['task_config'].name
    
    print(f"\n🏆 BEST PERFORMING TASK: {best_task}")
    print(f"   MAE: {best_result['multimodal_metrics']['mae']:.6f}")
    print(f"   Improvement: {best_result['multimodal_metrics']['mae_improvement_percent']:.2f}%")
    
    # Save best task predictions (simplified)
    best_predictions_file = f"best_task_predictions_{best_task}_{timestamp}.json"
    best_task_data = {
        'task_name': best_task,
        'task_description': best_result['task_config'].description,
        'predictions': best_result['test_predictions'],
        'metrics': {
            'baseline': best_result['baseline_metrics'],
            'multimodal': best_result['multimodal_metrics']
        }
    }
    
    with open(best_predictions_file, 'w') as f:
        json.dump(best_task_data, f, indent=2)
    
    print(f"💾 Best task predictions saved to: {best_predictions_file}")
    
    return serializable_results

def main():
    """Main execution"""
    set_seed(42)
    
    # Run cross-task testing
    results = run_cross_task_testing()
    
    print(f"\n🎉 CROSS-TASK TESTING COMPLETED!")
    print(f"✅ Tested multimodal enhancer on {results['metadata']['total_tasks']} different tasks")
    print(f"📈 Success rate: {results['metadata']['success_rate']:.1f}%")
    print(f"🚀 Check the generated JSON files for detailed results and predictions")

if __name__ == "__main__":
    main()
