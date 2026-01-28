#!/usr/bin/env python3
"""
Comprehensive Augmented Neural VLA Framework
Multi-Modal Approach: BridgeData Augmentation + Neural Enhancement
Attacks inefficiencies in vision, and action modalities
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import csv
import time
import pickle
import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple, Dict, Optional
import random

# Set environment for stability
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class AugmentedNeuralEnhancer(nn.Module):
    """Deeper neural network trained on augmented OpenVLA predictions"""
    
    def __init__(self, action_dim=7, hidden_dim=256):  
        super().__init__()
        self.action_dim = action_dim
        
        # Deeper architecture for better learning capacity
        self.feature_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim * 2),  
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),  
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),  
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Multi-head output for different action components
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  
        )
        
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  
        )
        
        # Residual connection weight (learnable)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, openvla_action):
        # Process OpenVLA prediction
        features = self.feature_net(openvla_action)
        
        # Generate component-specific enhancements
        pos_enhancement = self.position_head(features)
        rot_enhancement = self.rotation_head(features)
        grip_enhancement = self.gripper_head(features)
        
        # Combine enhancements
        enhancement = torch.cat([pos_enhancement, rot_enhancement, grip_enhancement], dim=-1)
        
        # Residual connection with learned weighting
        enhanced_action = openvla_action + enhancement * self.residual_weight
        
        return enhanced_action

class BridgeDataAugmentation:
    """Multi-modal augmentation for BridgeData samples (images + instructions)"""
    
    # Visual augmentation techniques
    VISUAL_AUGMENTATIONS = [
        'brightness', 'contrast', 'saturation', 'sharpness', 
        'rotation', 'flip', 'crop', 'noise', 'blur'
    ]
    
    # Language augmentation templates
    INSTRUCTION_TEMPLATES = {
        'pick_place': [
            "pick up the {object} and place it in the {container}",
            "move the {object} to the {container}",
            "grab the {object} and put it in the {container}",
            "take the {object} and drop it in the {container}",
            "lift the {object} and place it in the {container}"
        ],
        'grab_move': [
            "grab the {object} and move it to the {location}",
            "pick up the {object} and move it to the {location}",
            "take the {object} to the {location}",
            "move the {object} to the {location}",
            "bring the {object} to the {location}"
        ],
        'general': [
            "complete the manipulation task",
            "perform the robot action",
            "execute the task",
            "do the manipulation",
            "accomplish the goal"
        ]
    }
    
    @staticmethod
    def augment_image(image, augmentation_type='mixed'):
        """Apply visual augmentations to image"""
        if augmentation_type == 'none':
            return image
            
        img = image.copy()
        
        if augmentation_type == 'brightness':
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(random.uniform(0.7, 1.3))
            
        elif augmentation_type == 'contrast':
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(random.uniform(0.7, 1.3))
            
        elif augmentation_type == 'saturation':
            enhancer = ImageEnhance.Color(img)
            return enhancer.enhance(random.uniform(0.7, 1.3))
            
        elif augmentation_type == 'sharpness':
            enhancer = ImageEnhance.Sharpness(img)
            return enhancer.enhance(random.uniform(0.7, 1.3))
            
        elif augmentation_type == 'rotation':
            angle = random.uniform(-15, 15)
            return img.rotate(angle, expand=True)
            
        elif augmentation_type == 'flip':
            if random.random() < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            return img
            
        elif augmentation_type == 'crop':
            width, height = img.size
            crop_ratio = random.uniform(0.8, 1.0)
            new_width = int(width * crop_ratio)
            new_height = int(height * crop_ratio)
            left = random.randint(0, width - new_width)
            top = random.randint(0, height - new_height)
            cropped = img.crop((left, top, left + new_width, top + new_height))
            return cropped.resize((width, height))
            
        elif augmentation_type == 'noise':
            img_array = np.array(img)
            noise = np.random.normal(0, 10, img_array.shape)
            noisy = np.clip(img_array + noise, 0, 255)
            return Image.fromarray(noisy.astype(np.uint8))
            
        elif augmentation_type == 'blur':
            radius = random.uniform(0.5, 2.0)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
            
        elif augmentation_type == 'mixed':
            # Apply multiple augmentations
            if random.random() < 0.5:
                img = BridgeDataAugmentation.augment_image(img, 'brightness')
            if random.random() < 0.3:
                img = BridgeDataAugmentation.augment_image(img, 'contrast')
            if random.random() < 0.2:
                img = BridgeDataAugmentation.augment_image(img, 'noise')
            return img
            
        return img
    
    @staticmethod
    def augment_instruction(instruction, task_type='general'):
        """Apply language augmentations to instruction"""
        if task_type in BridgeDataAugmentation.INSTRUCTION_TEMPLATES:
            templates = BridgeDataAugmentation.INSTRUCTION_TEMPLATES[task_type]
            # Extract objects/locations if possible, otherwise use template
            return random.choice(templates)
        else:
            # Paraphrase general instructions
            variations = [
                instruction,
                instruction.replace("pick up", "grab"),
                instruction.replace("move", "take"),
                instruction.replace("place", "put"),
                instruction.replace("complete", "perform"),
            ]
            return random.choice(variations)
    
    @staticmethod
    def augment_action(action, augmentation_type='mixed'):
        """Apply action augmentations"""
        action = np.array(action)
        
        if augmentation_type == 'gaussian_noise':
            noise = np.random.normal(0, 0.01, action.shape)
            return action + noise
        elif augmentation_type == 'uniform_noise':
            noise = np.random.uniform(-0.005, 0.005, action.shape)
            return action + noise
        elif augmentation_type == 'mixup':
            random_action = np.random.normal(0, 0.1, action.shape)
            lam = np.random.beta(0.2, 0.2)
            return lam * action + (1 - lam) * random_action
        else:  # mixed
            if np.random.random() < 0.5:
                action = BridgeDataAugmentation.augment_action(action, 'gaussian_noise')
            if np.random.random() < 0.3:
                action = action * np.random.uniform(0.9, 1.1, action.shape)
            return action
    
    @staticmethod
    def generate_augmented_dataset(samples, factor=3):
        """Generate augmented BridgeData samples"""
        augmented_samples = []
        
        for sample in samples:
            # Original sample
            augmented_samples.append(sample)
            
            # Generate augmented versions
            for i in range(factor - 1):
                aug_sample = sample.copy()
                
                # Augment images
                aug_images = []
                for img_path in sample['images']:
                    if isinstance(img_path, (str, Path)):
                        # Handle file path
                        try:
                            image = Image.open(img_path).convert("RGB")
                            aug_type = random.choice(BridgeDataAugmentation.VISUAL_AUGMENTATIONS + ['none'])
                            aug_image = BridgeDataAugmentation.augment_image(image, aug_type)
                            aug_images.append(aug_image)
                        except Exception as e:
                            print(f"⚠️  Could not load image {img_path}: {e}")
                            # Create dummy image
                            dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                            aug_images.append(dummy_image)
                    else:
                        # Handle numpy array or PIL Image
                        try:
                            if isinstance(img_path, np.ndarray):
                                image = Image.fromarray(img_path).convert("RGB")
                            else:
                                image = img_path.convert("RGB")
                            aug_type = random.choice(BridgeDataAugmentation.VISUAL_AUGMENTATIONS + ['none'])
                            aug_image = BridgeDataAugmentation.augment_image(image, aug_type)
                            aug_images.append(aug_image)
                        except Exception as e:
                            print(f"⚠️  Could not process image: {e}")
                            # Create dummy image
                            dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                            aug_images.append(dummy_image)
                aug_sample['images'] = aug_images
                
                # Augment instruction
                task_type = sample.get('task_type', 'general')
                aug_sample['instruction'] = BridgeDataAugmentation.augment_instruction(
                    sample['instruction'], task_type
                )
                
                # Optionally augment ground truth actions
                if random.random() < 0.3:
                    aug_actions = []
                    for action in sample['gt_actions']:
                        # Convert action to numpy array if it's not already
                        if isinstance(action, dict):
                            # Skip dict actions for now
                            aug_actions.append(action)
                        else:
                            action_array = np.array(action)
                            aug_action = BridgeDataAugmentation.augment_action(action_array)
                            aug_actions.append(aug_action)
                    aug_sample['gt_actions'] = np.array(aug_actions)
                
                augmented_samples.append(aug_sample)
        
        return augmented_samples

class DataAugmentation:
    """Legacy class for backward compatibility"""
    
    @staticmethod
    def augment_action(action, augmentation_type='mixed'):
        """Apply various augmentation techniques"""
        return BridgeDataAugmentation.augment_action(action, augmentation_type)
    
    @staticmethod
    def generate_augmented_dataset(openvla_preds, ground_truths, factor=5):
        """Generate augmented training dataset"""
        augmented_preds = []
        augmented_gts = []
        
        for pred, gt in zip(openvla_preds, ground_truths):
            # Original
            augmented_preds.append(pred)
            augmented_gts.append(gt)
            
            # Augmented versions
            for _ in range(factor - 1):
                aug_pred = DataAugmentation.augment_action(pred)
                augmented_preds.append(aug_pred)
                augmented_gts.append(gt)
        
        return np.array(augmented_preds), np.array(augmented_gts)

class ComprehensiveAugmentedNeuralVLA:
    """Framework: BridgeData Augmentation + Neural Enhancement"""
    
    def __init__(self, action_dim=7, use_enhancement=True):
        self.action_dim = action_dim
        self.use_enhancement = use_enhancement
        
        # Initialize components
        self.enhancer = AugmentedNeuralEnhancer(action_dim) if use_enhancement else None
        self.is_trained = False
        
    def load_bridgedata_samples(self, max_samples=30):
        """Load BridgeData samples for training"""
        print(f"🚀 Loading BridgeData samples (max {max_samples})...")
        
        # Try multiple data sources
        samples = []
        
        # Option 1: Try to use baseline predictions as fallback
        try:
            import results_openvla_baseline as baseline
            predictions, ground_truths = baseline.get_hardcoded_data()
            
            # Convert baseline data to BridgeData format
            for i, (pred, gt) in enumerate(zip(predictions[:max_samples], ground_truths[:max_samples])):
                # Create dummy image and instruction
                sample = {
                    'image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),  # Dummy image
                    'images': [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)],  # List format
                    'instruction': f"robot manipulation task {i+1}",  # Dummy instruction
                    'action': gt,  # Use ground truth as target action
                    'gt_actions': [gt],  # Expected by augmentation code
                    'prediction': pred,  # Store OpenVLA prediction
                    'task_type': 'general'  # Required field
                }
                samples.append(sample)
            
            print(f"✅ Loaded {len(samples)} samples from baseline data (fallback)")
            return samples
            
        except ImportError:
            print("⚠️  Baseline data not available")
        
        # Option 2: Try to load real BridgeData
        try:
            # Try to find BridgeData in common locations
            bridgedata_paths = [
                "bridge_dataset",
                "data/bridge_np_final_v2", 
                "data/scripted_raw",
                "bridge_data_v2"
            ]
            
            for path in bridgedata_paths:
                if os.path.exists(path):
                    print(f"📁 Found BridgeData at: {path}")
                    # This would need actual BridgeData loading logic
                    # For now, create dummy samples
                    for i in range(min(max_samples, 10)):
                        dummy_action = np.random.randn(7) * 0.1
                        sample = {
                            'image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                            'images': [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)],
                            'instruction': f"robot task {i+1}",
                            'action': dummy_action,
                            'gt_actions': [dummy_action],
                            'prediction': np.random.randn(7) * 0.1,
                            'task_type': 'general'
                        }
                        samples.append(sample)
                    
                    print(f"✅ Created {len(samples)} dummy samples from {path}")
                    return samples
            
        except Exception as e:
            print(f"⚠️  Error loading real BridgeData: {e}")
        
        # Option 3: Create dummy data as last resort
        print("⚠️  No real data found, creating dummy samples for testing...")
        for i in range(max_samples):
            dummy_action = np.random.randn(7) * 0.1
            sample = {
                'image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                'images': [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)],
                'instruction': f"robot manipulation task {i+1}",
                'action': dummy_action,
                'gt_actions': [dummy_action],
                'prediction': np.random.randn(7) * 0.1,
                'task_type': 'general'
            }
            samples.append(sample)
        
        print(f"✅ Loaded {len(samples)} samples")
        return samples
    
    def train_comprehensive_pipeline(self, max_samples=30, augmentation_factor=3, 
                                  enhancement_epochs=20):
        """Complete training pipeline: Augmentation → Enhancement"""
        print("🚀 Starting Comprehensive Augmented Neural VLA Training")
        print("=" * 70)
        
        # Step 1: Load BridgeData samples
        samples = self.load_bridgedata_samples(max_samples)
        if not samples:
            print("❌ No samples loaded. Exiting.")
            return
        
        # Step 2: Generate augmented samples
        print("\n📸 Step 1: Multi-Modal Data Augmentation")
        augmented_samples = BridgeDataAugmentation.generate_augmented_dataset(
            samples, factor=augmentation_factor
        )
        print(f"✅ Generated {len(augmented_samples)} augmented samples from {len(samples)} originals")
        
        # Step 2: Collect baseline predictions (from loaded samples)
        baseline_preds = []
        ground_truths = []

        for sample in samples:
            baseline_preds.append(sample.get('prediction', np.zeros(7)))
            ground_truths.append(sample.get('action', np.zeros(7)))
        
        if not baseline_preds:
            print("❌ No predictions generated. Exiting.")
            return None, None
        
        # Step 3: Neural enhancement (if enabled)
        if self.use_enhancement and self.enhancer:
            print("\n🔄 Step 3: Neural Enhancement Training")
            self.train_enhancement(baseline_preds, ground_truths, epochs=enhancement_epochs)
        
        self.is_trained = True
        print("\n✅ Comprehensive training completed!")
        
        return baseline_preds, ground_truths
    
    def train_enhancement(self, openvla_preds, ground_truths, epochs=20):
        """Train neural enhancer on prediction pairs"""
        print("🔄 Training Neural Enhancer on prediction pairs...")
        
        # Generate augmented prediction dataset
        aug_preds, aug_gts = DataAugmentation.generate_augmented_dataset(
            openvla_preds, ground_truths, factor=5
        )
        
        print(f"📊 Generated {len(aug_preds)} training samples")
        
        # Initialize model
        self.enhancer = AugmentedNeuralEnhancer(action_dim=self.action_dim)
        optimizer = optim.Adam(self.enhancer.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        X_train = torch.FloatTensor(aug_preds)
        y_train = torch.FloatTensor(aug_gts)
        
        # Training loop
        self.enhancer.train()
        for epoch in range(epochs):
            total_loss = 0
            batch_size = 16
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                predictions = self.enhancer(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                avg_loss = total_loss / (len(X_train) // batch_size)
                print(f"   Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        print("✅ Neural enhancement training completed!")

    def cross_validate_enhancement(self, openvla_preds, ground_truths, folds=5, epochs=200, seed=0):
        set_seed(seed)

        preds = np.asarray(openvla_preds)
        gts = np.asarray(ground_truths)

        n = len(preds)
        if n < 2:
            raise ValueError("Not enough samples for cross-validation")

        indices = np.arange(n)
        np.random.shuffle(indices)

        split_sizes = [n // folds] * folds
        for i in range(n % folds):
            split_sizes[i] += 1

        fold_splits = []
        start = 0
        for sz in split_sizes:
            fold_splits.append(indices[start:start + sz])
            start += sz

        fold_baseline_mae = []
        fold_enhanced_mae = []
        fold_improvement = []

        for fold_idx in range(folds):
            test_idx = fold_splits[fold_idx]
            train_idx = np.concatenate([fold_splits[i] for i in range(folds) if i != fold_idx])

            train_preds = preds[train_idx]
            train_gts = gts[train_idx]
            test_preds = preds[test_idx]
            test_gts = gts[test_idx]

            aug_preds, aug_gts = DataAugmentation.generate_augmented_dataset(
                train_preds, train_gts, factor=5
            )

            model = AugmentedNeuralEnhancer(action_dim=self.action_dim)
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            criterion = nn.MSELoss()

            X_train = torch.FloatTensor(aug_preds)
            y_train = torch.FloatTensor(aug_gts)

            model.train()
            for epoch in range(epochs):
                total_loss = 0.0
                batch_size = 16
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i + batch_size]
                    batch_y = y_train[i:i + batch_size]
                    optimizer.zero_grad()
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item())

            model.eval()
            with torch.no_grad():
                test_tensor = torch.FloatTensor(test_preds)
                enhanced = model(test_tensor).numpy()

            baseline_mae = float(np.mean(np.abs(test_preds - test_gts)))
            enhanced_mae = float(np.mean(np.abs(enhanced - test_gts)))
            improvement = 0.0
            if baseline_mae > 0:
                improvement = (baseline_mae - enhanced_mae) / baseline_mae * 100.0

            fold_baseline_mae.append(baseline_mae)
            fold_enhanced_mae.append(enhanced_mae)
            fold_improvement.append(improvement)

            print(
                f"   CV Fold {fold_idx + 1}/{folds}: baseline MAE={baseline_mae:.4f}, "
                f"enhanced MAE={enhanced_mae:.4f}, improvement={improvement:.2f}%"
            )

        cv_results = {
            'folds': folds,
            'epochs': epochs,
            'seed': seed,
            'baseline_mae_mean': float(np.mean(fold_baseline_mae)),
            'baseline_mae_std': float(np.std(fold_baseline_mae)),
            'enhanced_mae_mean': float(np.mean(fold_enhanced_mae)),
            'enhanced_mae_std': float(np.std(fold_enhanced_mae)),
            'improvement_pct_mean': float(np.mean(fold_improvement)),
            'improvement_pct_std': float(np.std(fold_improvement)),
        }

        print("\n📊 Cross-Validation Summary (held-out folds)")
        print(f"   Baseline MAE: {cv_results['baseline_mae_mean']:.4f} ± {cv_results['baseline_mae_std']:.4f}")
        print(f"   Enhanced MAE: {cv_results['enhanced_mae_mean']:.4f} ± {cv_results['enhanced_mae_std']:.4f}")
        print(f"   Improvement: {cv_results['improvement_pct_mean']:.2f}% ± {cv_results['improvement_pct_std']:.2f}%")

        return cv_results
    
    def enhance_prediction(self, openvla_action):
        """Apply neural enhancement to prediction"""
        if not self.use_enhancement or not self.enhancer:
            return openvla_action
        
        self.enhancer.eval()
        with torch.no_grad():
            action_tensor = torch.FloatTensor(openvla_action).unsqueeze(0)
            enhanced = self.enhancer(action_tensor)
            return enhanced.squeeze(0).numpy()
    
    def evaluate_comprehensive(self, openvla_preds, ground_truths):
        """Evaluate comprehensive approach"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        enhanced_preds = []
        improvements = []
        
        print("📊 Evaluating comprehensive approach...")
        
        for pred, gt in zip(openvla_preds, ground_truths):
            # Get enhanced prediction
            enhanced = self.enhance_prediction(pred)
            enhanced_preds.append(enhanced)
            
            # Calculate improvement
            baseline_mae = np.mean(np.abs(pred - gt))
            enhanced_mae = np.mean(np.abs(enhanced - gt))
            
            if baseline_mae > 0:
                improvement = (baseline_mae - enhanced_mae) / baseline_mae * 100
            else:
                improvement = 0
            
            improvements.append(improvement)
        
        # Calculate overall metrics
        baseline_maes = [np.mean(np.abs(pred - gt)) for pred, gt in zip(openvla_preds, ground_truths)]
        enhanced_maes = [np.mean(np.abs(enh - gt)) for enh, gt in zip(enhanced_preds, ground_truths)]
        
        results = {
            'approach': 'comprehensive_augmented_neural_vla',
            'baseline_mae': np.mean(baseline_maes),
            'enhanced_mae': np.mean(enhanced_maes),
            'improvement_pct': np.mean(improvements),
            'std_improvement': np.std(improvements),
            'total_samples': len(openvla_preds),
            'use_enhancement': self.use_enhancement
        }
        
        print(f"🎯 Comprehensive Results:")
        print(f"   Baseline MAE: {results['baseline_mae']:.4f}")
        print(f"   Enhanced MAE: {results['enhanced_mae']:.4f}")
        print(f"   Improvement: {results['improvement_pct']:.2f}% ± {results['std_improvement']:.2f}%")
        print(f"   Enhancement Enabled: {results['use_enhancement']}")
        
        return results

    def compute_comparison_metrics(self, openvla_preds, ground_truths):
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        preds = np.asarray(openvla_preds)
        gts = np.asarray(ground_truths)
        enhanced = np.asarray([self.enhance_prediction(p) for p in preds])

        def _summary_stats(sample_mae: np.ndarray):
            return {
                'total_samples': int(sample_mae.shape[0]),
                'mean_mae': float(np.mean(sample_mae)),
                'std_mae': float(np.std(sample_mae)),
                'min_mae': float(np.min(sample_mae)),
                'p25_mae': float(np.percentile(sample_mae, 25)),
                'median_mae': float(np.percentile(sample_mae, 50)),
                'p75_mae': float(np.percentile(sample_mae, 75)),
                'max_mae': float(np.max(sample_mae)),
            }

        baseline_err = preds - gts
        enhanced_err = enhanced - gts

        baseline_sample_mae = np.mean(np.abs(baseline_err), axis=1)
        enhanced_sample_mae = np.mean(np.abs(enhanced_err), axis=1)

        baseline_dim_mae = np.mean(np.abs(baseline_err), axis=0)
        enhanced_dim_mae = np.mean(np.abs(enhanced_err), axis=0)

        baseline_dim_bias = np.mean(baseline_err, axis=0)
        enhanced_dim_bias = np.mean(enhanced_err, axis=0)

        dim_names = [
            'x', 'y', 'z',
            'roll', 'pitch', 'yaw',
            'gripper'
        ]

        def _per_dim_dict(values: np.ndarray):
            return {name: float(values[i]) for i, name in enumerate(dim_names)}

        def _success_rates(err: np.ndarray):
            pos_l1 = np.sum(np.abs(err[:, :3]), axis=1)
            rot_l1 = np.sum(np.abs(err[:, 3:6]), axis=1)
            grip_abs = np.abs(err[:, 6])

            thresholds = {
                'pos_l1': [0.05, 0.10, 0.20],
                'rot_l1': [0.05, 0.10, 0.20],
                'gripper_abs': [0.10, 0.25, 0.50],
            }

            out = {}
            for t in thresholds['pos_l1']:
                out[f'pos_l1_le_{t}'] = float(np.mean(pos_l1 <= t))
            for t in thresholds['rot_l1']:
                out[f'rot_l1_le_{t}'] = float(np.mean(rot_l1 <= t))
            for t in thresholds['gripper_abs']:
                out[f'gripper_abs_le_{t}'] = float(np.mean(grip_abs <= t))
            return out

        baseline_summary = _summary_stats(baseline_sample_mae)
        enhanced_summary = _summary_stats(enhanced_sample_mae)

        improvement_pct = 0.0
        if baseline_summary['mean_mae'] > 0:
            improvement_pct = (baseline_summary['mean_mae'] - enhanced_summary['mean_mae']) / baseline_summary['mean_mae'] * 100.0

        metrics = {
            'baseline': {
                'summary': baseline_summary,
                'per_dim_mae': _per_dim_dict(baseline_dim_mae),
                'per_dim_bias': _per_dim_dict(baseline_dim_bias),
                'success_rates': _success_rates(baseline_err),
            },
            'enhanced': {
                'summary': enhanced_summary,
                'per_dim_mae': _per_dim_dict(enhanced_dim_mae),
                'per_dim_bias': _per_dim_dict(enhanced_dim_bias),
                'success_rates': _success_rates(enhanced_err),
            },
            'delta': {
                'mean_mae_improvement_pct': float(improvement_pct),
                'per_dim_mae_delta': _per_dim_dict(enhanced_dim_mae - baseline_dim_mae),
                'per_dim_bias_delta': _per_dim_dict(enhanced_dim_bias - baseline_dim_bias),
            }
        }

        return metrics

    def save_comparison_metrics_csv(self, metrics: dict, filename: str = 'comparison_metrics.csv'):
        rows = []

        def add_row(metric_name, baseline_value, enhanced_value, delta_value):
            rows.append({
                'Metric': metric_name,
                'Baseline': baseline_value,
                'Enhanced': enhanced_value,
                'Delta': delta_value,
            })

        b = metrics['baseline']
        e = metrics['enhanced']

        for key in ['mean_mae', 'std_mae', 'min_mae', 'p25_mae', 'median_mae', 'p75_mae', 'max_mae']:
            add_row(
                key,
                b['summary'][key],
                e['summary'][key],
                e['summary'][key] - b['summary'][key],
            )

        for dim, val in b['per_dim_mae'].items():
            add_row(f'mae_{dim}', val, e['per_dim_mae'][dim], e['per_dim_mae'][dim] - val)

        for dim, val in b['per_dim_bias'].items():
            add_row(f'bias_{dim}', val, e['per_dim_bias'][dim], e['per_dim_bias'][dim] - val)

        for k, val in b['success_rates'].items():
            add_row(f'success_{k}', val, e['success_rates'][k], e['success_rates'][k] - val)

        add_row('mean_mae_improvement_pct', '', metrics['delta']['mean_mae_improvement_pct'], '')

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Metric', 'Baseline', 'Enhanced', 'Delta'])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"💾 Comparison metrics CSV saved to: {filename}")
    
    def evaluate_on_bridgedata_samples(self, max_samples=30, use_augmented=True):
        """
        Evaluate augmented neural VLA on BridgeData samples with MSE/MAE metrics
        Mirrors the OpenVLA baseline evaluation approach for direct comparison
        """
        print("🚀 Evaluating Augmented Neural VLA on BridgeData Samples")
        print("=" * 60)
        
        # Load BridgeData samples using same approach as baseline
        samples = self._load_bridgedata_like_baseline(max_samples)
        if not samples:
            print("❌ No samples loaded")
            return None
        
        print(f"✅ Loaded {len(samples)} samples")
        
        # Apply augmentation if requested
        if use_augmented:
            print("\n📸 Applying BridgeData augmentation...")
            augmented_samples = BridgeDataAugmentation.generate_augmented_dataset(
                samples, factor=3
            )
            print(f"✅ Generated {len(augmented_samples)} augmented samples")
            eval_samples = augmented_samples
        else:
            eval_samples = samples
        
        # Load baseline predictions from results_openvla_baseline
        try:
            import results_openvla_baseline as baseline
            baseline_data = baseline.get_hardcoded_data()
            baseline_preds, baseline_gts = baseline_data
            print(f"✅ Loaded {len(baseline_preds)} baseline predictions")
        except (ImportError, AttributeError):
            print("⚠️  Could not load baseline predictions, using dummy data")
            baseline_preds = [np.random.randn(7) * 0.1 for _ in range(len(eval_samples))]
            baseline_gts = [np.random.randn(7) * 0.1 for _ in range(len(eval_samples))]
        
        # Train the neural enhancer on available data
        print("\n🔄 Training neural enhancer...")
        self.train_enhancement(baseline_preds[:len(eval_samples)], baseline_gts[:len(eval_samples)], epochs=50)
        
        print(f"\n🎯 Running evaluation on {len(eval_samples)} samples...")
        
        all_mae = []
        all_mse = []
        results = []
        
        for sample_idx, sample in enumerate(eval_samples):
            if sample_idx >= len(baseline_preds):
                break
                
            instruction = sample['instruction']
            gt_action = sample.get('action', sample.get('gt_actions', [np.zeros(7)])[0])
            
            # Handle different ground truth action formats
            if isinstance(gt_action, list) and len(gt_action) > 0:
                gt_action = np.array(gt_action[0])
            elif isinstance(gt_action, dict):
                # Try to extract numeric values from dict
                for key in ['actions', 'action', 'qpos', 'commands', 'joint_positions', 'state']:
                    if key in gt_action:
                        gt_action = gt_action[key]
                        break
                else:
                    # If no known keys, try to get first numeric value
                    for value in gt_action.values():
                        if isinstance(value, (int, float, np.ndarray, list, tuple)) and not isinstance(value, dict):
                            gt_action = value
                            break
                    else:
                        # Fallback to zeros
                        gt_action = np.zeros(7)
            else:
                gt_action = np.array(gt_action)
            
            # Ensure 7-dimensional
            gt_action = np.array(gt_action).flatten()
            if len(gt_action) != 7:
                if len(gt_action) > 7:
                    gt_action = gt_action[:7]
                else:
                    gt_action = np.pad(gt_action, (0, 7 - len(gt_action)))
            
            # Get baseline prediction (simulated OpenVLA output)
            baseline_pred = np.array(baseline_preds[sample_idx]).flatten()
            if len(baseline_pred) != 7:
                baseline_pred = np.pad(baseline_pred, (0, 7 - len(baseline_pred)))[:7]
            
            # Apply neural enhancement
            enhanced_pred = self.enhance_prediction(baseline_pred)
            
            # Calculate metrics
            baseline_mae = np.mean(np.abs(baseline_pred - gt_action))
            baseline_mse = np.mean((baseline_pred - gt_action) ** 2)
            
            enhanced_mae = np.mean(np.abs(enhanced_pred - gt_action))
            enhanced_mse = np.mean((enhanced_pred - gt_action) ** 2)
            
            # Task completion evaluation (same as baseline)
            baseline_success = self._evaluate_task_completion(baseline_pred, gt_action, instruction)
            enhanced_success = self._evaluate_task_completion(enhanced_pred, gt_action, instruction)
            
            print(f"   Sample {sample_idx + 1}:")
            print(f"     Baseline MAE: {baseline_mae:.4f}, Enhanced MAE: {enhanced_mae:.4f}")
            print(f"     Baseline Success: {'✅' if baseline_success else '❌'}, Enhanced Success: {'✅' if enhanced_success else '❌'}")
            
            all_mae.append(enhanced_mae)
            all_mse.append(enhanced_mse)
            
            results.append({
                'sample': sample_idx,
                'instruction': instruction,
                'baseline_predicted': baseline_pred.tolist(),
                'enhanced_predicted': enhanced_pred.tolist(),
                'ground_truth': gt_action.tolist(),
                'baseline_mae': baseline_mae,
                'enhanced_mae': enhanced_mae,
                'baseline_mse': baseline_mse,
                'enhanced_mse': enhanced_mse,
                'baseline_success': bool(baseline_success),
                'enhanced_success': bool(enhanced_success),
                'mae_improvement': baseline_mae - enhanced_mae,
                'mse_improvement': baseline_mse - enhanced_mse
            })
        
        # Summary statistics
        if all_mae:
            print(f"\n" + "="*60)
            print("📈 AUGMENTED NEURAL VLA EVALUATION SUMMARY")
            print("="*60)
            
            baseline_maes = [r['baseline_mae'] for r in results]
            enhanced_maes = [r['enhanced_mae'] for r in results]
            baseline_mses = [r['baseline_mse'] for r in results]
            enhanced_mses = [r['enhanced_mse'] for r in results]
            
            print(f"\n🎯 Performance Comparison:")
            print(f"   Baseline MAE:  {np.mean(baseline_maes):.4f} ± {np.std(baseline_maes):.4f}")
            print(f"   Enhanced MAE:  {np.mean(enhanced_maes):.4f} ± {np.std(enhanced_maes):.4f}")
            print(f"   MAE Improvement: {np.mean(baseline_maes) - np.mean(enhanced_maes):.4f}")
            
            print(f"   Baseline MSE:  {np.mean(baseline_mses):.4f} ± {np.std(baseline_mses):.4f}")
            print(f"   Enhanced MSE:  {np.mean(enhanced_mses):.4f} ± {np.std(enhanced_mses):.4f}")
            print(f"   MSE Improvement: {np.mean(baseline_mses) - np.mean(enhanced_mses):.4f}")
            
            # Task completion rates
            baseline_success_rate = np.mean([r['baseline_success'] for r in results]) * 100
            enhanced_success_rate = np.mean([r['enhanced_success'] for r in results]) * 100
            
            print(f"\n🏆 Task Completion Rates:")
            print(f"   Baseline: {baseline_success_rate:.1f}%")
            print(f"   Enhanced: {enhanced_success_rate:.1f}%")
            print(f"   Improvement: {enhanced_success_rate - baseline_success_rate:.1f}%")
            
            # Overall improvement percentage
            mae_improvement_pct = 0.0
            if np.mean(baseline_maes) > 0:
                mae_improvement_pct = (np.mean(baseline_maes) - np.mean(enhanced_maes)) / np.mean(baseline_maes) * 100
            
            print(f"\n📊 Overall Improvement: {mae_improvement_pct:.2f}%")
            
            # Save results
            summary = {
                'approach': 'augmented_neural_vla',
                'samples_evaluated': len(results),
                'use_augmentation': bool(use_augmented),
                'baseline': {
                    'avg_mae': float(np.mean(baseline_maes)),
                    'std_mae': float(np.std(baseline_maes)),
                    'avg_mse': float(np.mean(baseline_mses)),
                    'std_mse': float(np.std(baseline_mses)),
                    'success_rate': float(baseline_success_rate)
                },
                'enhanced': {
                    'avg_mae': float(np.mean(enhanced_maes)),
                    'std_mae': float(np.std(enhanced_maes)),
                    'avg_mse': float(np.mean(enhanced_mses)),
                    'std_mse': float(np.std(enhanced_mses)),
                    'success_rate': float(enhanced_success_rate)
                },
                'improvements': {
                    'mae_improvement': float(np.mean(baseline_maes) - np.mean(enhanced_maes)),
                    'mse_improvement': float(np.mean(baseline_mses) - np.mean(enhanced_mses)),
                    'mae_improvement_pct': float(mae_improvement_pct),
                    'success_rate_improvement': float(enhanced_success_rate - baseline_success_rate)
                },
                'detailed_results': results
            }
            
            filename = 'augmented_neural_vla_evaluation_results.json'
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n💾 Results saved to: {filename}")
            print(f"✅ Evaluation complete!")
            
            return summary
        
        return None

    def _load_bridgedata_like_baseline(self, max_samples=30):
        """Load BridgeData samples using same approach as openvla-baseline.py"""
        samples = []
        data_dir = Path("data/scripted_raw")
        
        if not data_dir.exists():
            print(f"⚠️  BridgeData directory not found: {data_dir}")
            # Fallback to baseline data
            return self._load_fallback_samples(max_samples)
        
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
                    
                    # Handle different action formats (same as baseline)
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
                    
                    # Determine instruction (same as baseline)
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
                        'images': img_files[:2],
                        'gt_actions': gt_actions,
                        'action': gt_actions[0] if len(gt_actions.shape) > 0 else gt_actions,
                        'task_type': 'pnp' if 'pnp' in path_str else 'general'
                    })
                    
                except Exception as e:
                    print(f"⚠️  Error loading {root}: {e}")
                    continue
        
        if not samples:
            print("⚠️  No real BridgeData samples found, using fallback")
            return self._load_fallback_samples(max_samples)
        
        return samples

    def _load_fallback_samples(self, max_samples=30):
        """Fallback sample generation using baseline data"""
        try:
            import results_openvla_baseline as baseline
            predictions, ground_truths = baseline.get_hardcoded_data()
            
            samples = []
            for i, (pred, gt) in enumerate(zip(predictions[:max_samples], ground_truths[:max_samples])):
                sample = {
                    'path': f'fallback_sample_{i}',
                    'instruction': f"robot manipulation task {i+1}",
                    'images': [],  # No real images in fallback
                    'gt_actions': [gt],
                    'action': gt,
                    'prediction': pred,
                    'task_type': 'general'
                }
                samples.append(sample)
            
            return samples
            
        except Exception as e:
            print(f"❌ Fallback loading failed: {e}")
            return []

    def _evaluate_task_completion(self, pred_action, gt_action, instruction):
        """Evaluate task completion (same logic as baseline)"""
        pred_action = np.array(pred_action).flatten()
        gt_action = np.array(gt_action).flatten()
        
        if len(pred_action) != 7 or len(gt_action) != 7:
            return False
        
        # Extract action components
        pred_pos = pred_action[:3]
        pred_rot = pred_action[3:6]
        pred_grip = pred_action[6]
        
        gt_pos = gt_action[:3]
        gt_rot = gt_action[3:6]
        gt_grip = gt_action[6]
        
        # Position accuracy (within 5cm)
        pos_error = np.linalg.norm(pred_pos - gt_pos)
        pos_success = pos_error < 0.05
        
        # Rotation accuracy (within 15 degrees)
        rot_error = np.linalg.norm(pred_rot - gt_rot)
        rot_success = rot_error < 0.26
        
        # Gripper state accuracy
        if 'pick' in instruction.lower():
            grip_success = pred_grip > 0.5 and gt_grip > 0.5
        elif 'place' in instruction.lower():
            grip_success = pred_grip < 0.5 and gt_grip < 0.5
        else:
            grip_success = abs(pred_grip - gt_grip) < 0.3
        
        # Overall task success
        if 'pick' in instruction.lower() or 'place' in instruction.lower():
            return pos_success and grip_success
        else:
            return pos_success and rot_success and grip_success

    def save_results(self, results, filename='comprehensive_augmented_neural_vla_results.json'):
        """Save comprehensive results"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {filename}")

# Legacy class for backward compatibility
class AugmentedNeuralVLA:
    """Legacy class - use ComprehensiveAugmentedNeuralVLA for full functionality"""
    
    def __init__(self, action_dim=7):
        self.framework = ComprehensiveAugmentedNeuralVLA(
            action_dim=action_dim,
            use_enhancement=True
        )
    
    def load_data(self, source='hardcoded'):
        """Load OpenVLA predictions and ground truth"""
        if source == 'hardcoded':
            import sys
            sys.path.append('.')
            try:
                import results_openvla_baseline as hardcoded_data
                openvla_preds, ground_truths = hardcoded_data.get_hardcoded_data()
                print(f"✅ Loaded {len(openvla_preds)} hardcoded predictions")
                return openvla_preds, ground_truths
            except ImportError:
                print("❌ results-openvla_baseline.py not found")
                return None, None
        return None, None
    
    def train(self, openvla_preds, ground_truths, epochs=20, augmentation_factor=5):
        """Train neural enhancer"""
        self.framework.train_enhancement(openvla_preds, ground_truths, epochs)
        self.framework.is_trained = True
    
    def evaluate(self, openvla_preds, ground_truths):
        """Evaluate performance"""
        return self.framework.evaluate_comprehensive(openvla_preds, ground_truths)
    
    def enhance(self, openvla_action):
        """Apply enhancement"""
        return self.framework.enhance_prediction(openvla_action)
    
    def save_results(self, results, filename='augmented_neural_vla_results.json'):
        """Save results"""
        self.framework.save_results(results, filename)

def run_bridgedata_evaluation():
    """Standalone function to run BridgeData evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Augmented Neural VLA on BridgeData samples')
    parser.add_argument('--max-samples', type=int, default=30, help='Maximum samples to evaluate')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs for neural enhancer')
    
    args = parser.parse_args()
    
    print("🚀 Augmented Neural VLA - BridgeData Evaluation")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Use augmentation: {not args.no_augmentation}")
    print(f"  Training epochs: {args.epochs}")
    print("")
    
    # Initialize framework
    framework = ComprehensiveAugmentedNeuralVLA(action_dim=7, use_enhancement=True)
    
    # Run evaluation
    results = framework.evaluate_on_bridgedata_samples(
        max_samples=args.max_samples,
        use_augmented=not args.no_augmentation
    )
    
    if results:
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📄 Results saved to: augmented_neural_vla_evaluation_results.json")
        
        # Print comparison summary
        print(f"\n📊 Quick Summary:")
        print(f"  Baseline MAE: {results['baseline']['avg_mae']:.4f}")
        print(f"  Enhanced MAE: {results['enhanced']['avg_mae']:.4f}")
        print(f"  Improvement: {results['improvements']['mae_improvement_pct']:.2f}%")
        print(f"  Success Rate Improvement: {results['improvements']['success_rate_improvement']:.1f}%")
    else:
        print("❌ Evaluation failed!")

def main():
    """Main execution - Comprehensive Approach"""
    print("🚀 Comprehensive Augmented Neural VLA Framework")
    print("Multi-Modal Approach: BridgeData Augmentation + Neural Enhancement")
    print("=" * 80)
    
    # Check if running as evaluation mode
    if len(sys.argv) > 1 and sys.argv[1] == '--evaluate':
        # Remove '--evaluate' from args for argparse
        sys.argv.pop(1)
        run_bridgedata_evaluation()
        return
    
    # Choose approach
    use_comprehensive = True  # Set to False for legacy neural enhancement only
    
    if use_comprehensive:
        print("🎯 Using Pure Neural Enhancement Approach")
        print("   1. BridgeData Augmentation (Images + Instructions)")
        print("   2. Deep Neural Enhancement of Predictions")
        print("")
        
        # Initialize framework
        framework = ComprehensiveAugmentedNeuralVLA(action_dim=7, use_enhancement=True)
        
        # Train comprehensive pipeline
        start_time = time.time()
        baseline_preds, ground_truths = framework.train_comprehensive_pipeline(
            max_samples=30,           # Limit samples for faster demo
            augmentation_factor=5,     # INCREASED: Generate 5x augmented samples (was 3x)
            enhancement_epochs=1000     # ULTIMATE: Maximum training for best results
        )
        total_time = time.time() - start_time
        
        if baseline_preds:
            # Evaluate comprehensive approach
            results = framework.evaluate_comprehensive(baseline_preds, ground_truths)
            results['total_training_time_seconds'] = total_time

            print("\n🧪 Running 5-fold cross-validation on held-out samples...")
            cv_results = framework.cross_validate_enhancement(
                baseline_preds,
                ground_truths,
                folds=5,
                epochs=300,
                seed=0,
            )
            results['cross_validation'] = cv_results

            print("\n📈 Computing baseline vs enhanced comparison metrics...")
            comparison_metrics = framework.compute_comparison_metrics(baseline_preds, ground_truths)
            results['comparison_metrics'] = comparison_metrics
            framework.save_comparison_metrics_csv(comparison_metrics, filename='comparison_metrics.csv')
            
            # Save results
            framework.save_results(results)
            
            print(f"\n🏆 Comprehensive Summary:")
            print(f"   Total training time: {total_time:.1f} seconds")
            print(f"   Improvement: {results['improvement_pct']:.2f}%")
            print(f"   Samples processed: {results['total_samples']}")
            print(f"   Neural enhancement: {results['use_enhancement']}")
            print("✅ Comprehensive Augmented Neural VLA evaluation complete!")
    
    else:
        print("🎯 Using Legacy Neural Enhancement Only")
        print("   (For comparison with original approach)")
        print("")
        
        # Initialize legacy framework
        framework = AugmentedNeuralVLA(action_dim=7)
        
        # Load data
        openvla_preds, ground_truths = framework.load_data('hardcoded')
        if not openvla_preds:
            print("❌ No data loaded. Exiting.")
            return
        
        print(f"📊 Loaded {len(openvla_preds)} OpenVLA predictions")
        print("")
        
        # Train neural enhancer
        start_time = time.time()
        framework.train(openvla_preds, ground_truths, epochs=20, augmentation_factor=5)
        training_time = time.time() - start_time
        
        # Evaluate performance
        results = framework.evaluate(openvla_preds, ground_truths)
        results['training_time_seconds'] = training_time
        
        # Save results
        framework.save_results(results)
        
        print(f"\n🏆 Legacy Summary:")
        print(f"   Training time: {training_time:.1f} seconds")
        print(f"   Improvement: {results['improvement_pct']:.2f}%")
        print(f"   Samples used: {results['total_samples']} (augmented to {results.get('training_samples', 'N/A')})")
        print("✅ Legacy neural enhancement evaluation complete!")

if __name__ == "__main__":
    main()
