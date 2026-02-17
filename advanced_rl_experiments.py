#!/usr/bin/env python3
"""
Advanced RL and Unsupervised Learning Experiments for OpenVLA Enhancement
Testing cutting-edge approaches to potentially beat current best methods (35% improvement)
Improved version with statistical rigor and better implementations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import gymnasium as gym
from gymnasium import spaces
import warnings
import nashpy as nash
from scipy import stats
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔧 Using device: {device}")

def validate_data(predictions, ground_truths):
    """Validate data quality and alignment"""
    print("🔍 Data Validation:")
    
    # Check shapes
    assert predictions.shape == ground_truths.shape, f"Shape mismatch: {predictions.shape} vs {ground_truths.shape}"
    print(f"   ✅ Shapes match: {predictions.shape}")
    
    # Check for NaN/inf
    assert not np.isnan(predictions).any(), "Predictions contain NaN"
    assert not np.isnan(ground_truths).any(), "Ground truths contain NaN"
    assert not np.isinf(predictions).any(), "Predictions contain inf"
    assert not np.isinf(ground_truths).any(), "Ground truths contain inf"
    print(f"   ✅ No NaN/inf values")
    
    # Check value ranges
    pred_min, pred_max = predictions.min(), predictions.max()
    gt_min, gt_max = ground_truths.min(), ground_truths.max()
    print(f"   📊 Prediction range: [{pred_min:.3f}, {pred_max:.3f}]")
    print(f"   📊 Ground truth range: [{gt_min:.3f}, {gt_max:.3f}]")
    
    # Check MAE distribution
    mae_per_sample = np.mean(np.abs(predictions - ground_truths), axis=1)
    print(f"   📈 MAE per sample: {mae_per_sample.mean():.4f} ± {mae_per_sample.std():.4f}")
    print(f"   📈 MAE range: [{mae_per_sample.min():.4f}, {mae_per_sample.max():.4f}]")
    
    return True

def paired_t_test(baseline_mae_list, enhanced_mae_list, method_name):
    """Perform paired t-test for statistical significance"""
    if len(baseline_mae_list) != len(enhanced_mae_list):
        print(f"⚠️  Cannot perform t-test for {method_name}: unequal sample sizes")
        return None, None
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_mae_list, enhanced_mae_list)
    
    # Calculate effect size (Cohen's d)
    diff = np.array(baseline_mae_list) - np.array(enhanced_mae_list)
    pooled_std = np.sqrt((np.var(baseline_mae_list) + np.var(enhanced_mae_list)) / 2)
    cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0
    
    print(f"\n📊 Statistical Test - {method_name}:")
    print(f"   Paired t-test: t={t_stat:.4f}, p={p_value:.6f}")
    print(f"   Effect size (Cohen's d): {cohens_d:.4f}")
    
    if p_value < 0.001:
        print(f"   🌟 HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.01:
        print(f"   ✅ SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        print(f"   📈 MARGINALLY SIGNIFICANT (p < 0.05)")
    else:
        print(f"   ❌ NOT SIGNIFICANT (p ≥ 0.05)")
    
    return p_value, cohens_d

def bootstrap_ci(baseline_mae, enhanced_mae, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence intervals for improvement"""
    improvements = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(len(baseline_mae), size=len(baseline_mae), replace=True)
        baseline_sample = baseline_mae[indices]
        enhanced_sample = enhanced_mae[indices]
        
        # Calculate improvement
        baseline_mae_sample = np.mean(baseline_sample)
        enhanced_mae_sample = np.mean(enhanced_sample)
        improvement = (baseline_mae_sample - enhanced_mae_sample) / baseline_mae_sample * 100
        improvements.append(improvement)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    ci_lower = np.percentile(improvements, lower_percentile)
    ci_upper = np.percentile(improvements, upper_percentile)
    
    return ci_lower, ci_upper, np.mean(improvements), np.std(improvements)

@dataclass
class ExperimentResult:
    """Result format for advanced experiments"""
    method_name: str
    predictions: List[np.ndarray]
    ground_truths: List[np.ndarray]
    mae: float
    mse: float
    improvement_pct: float
    training_time: float
    prediction_time: float
    metadata: Dict[str, Any]

class BaselineData:
    """Load and manage baseline OpenVLA predictions"""
    
    def __init__(self, baseline_file="baseline_500_samples_results.json"):
        self.baseline_file = baseline_file
        self.predictions = []
        self.ground_truths = []
        self.metadata = {}
        self.load_data()
    
    def load_data(self):
        """Load baseline prediction data"""
        try:
            with open(self.baseline_file, 'r') as f:
                data = json.load(f)
            
            # Handle both old structure (with summary/detailed_results) and new structure (direct array)
            if isinstance(data, dict):
                self.metadata = data.get('summary', {})
                results = data.get('detailed_results', [])
            else:
                # New structure: direct array of results
                self.metadata = {'total_samples': len(data)}
                results = data
            
            for pred in results:
                if 'openvla_prediction' in pred and 'ground_truth' in pred:
                    self.predictions.append(np.array(pred['openvla_prediction']))
                    self.ground_truths.append(np.array(pred['ground_truth']))
            
            print(f"✅ Loaded {len(self.predictions)} baseline predictions")
            
        except Exception as e:
            print(f"❌ Error loading baseline data: {e}")
    
    def get_arrays(self):
        """Get numpy arrays of predictions and ground truths"""
        return np.array(self.predictions), np.array(self.ground_truths)

# ============================================================================
# 1. REINFORCEMENT LEARNING APPROACHES
# ============================================================================

class LatentActionCorrectionEnvironment(gym.Env):
    """RL Environment for learning action corrections in latent space"""
    
    def __init__(self, latent_predictions, ground_truths, decoder):
        super().__init__()
        self.latent_predictions = latent_predictions
        self.ground_truths = ground_truths
        self.decoder = decoder
        self.current_idx = 0
        
        # Action space: residual correction in latent space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(latent_predictions.shape[1],), dtype=np.float32
        )
        
        # Observation space: latent prediction
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(latent_predictions.shape[1],), dtype=np.float32
        )
    
    def reset(self, seed=None):
        self.current_idx = np.random.randint(len(self.latent_predictions))
        return self.latent_predictions[self.current_idx], {}
    
    def step(self, action):
        # Apply latent correction
        corrected_latent = self.latent_predictions[self.current_idx] + action
        
        # Decode to action space
        with torch.no_grad():
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            decoded_correction = self.decoder(action_tensor).squeeze().numpy()
        
        # Get original prediction and apply decoded correction
        # Note: This is a simplified version - in practice we'd need the original predictions
        original_pred = np.zeros(7)  # Placeholder - would need original predictions
        corrected_action = original_pred + decoded_correction
        gt = self.ground_truths[self.current_idx]
        
        # Reward: negative MAE (higher is better)
        mae = np.mean(np.abs(corrected_action - gt))
        reward = -mae
        
        # Episode ends after one correction
        done = True
        info = {'mae': mae, 'corrected_action': corrected_action}
        
        return corrected_latent, reward, done, False, info

class ImprovedPPOActionCorrectionEnvironment(gym.Env):
    """Enhanced RL Environment with multi-step episodes and dense rewards"""
    
    def __init__(self, predictions, ground_truths, episode_length=5, max_correction=1.0):
        super().__init__()
        self.predictions = predictions
        self.ground_truths = ground_truths
        self.episode_length = episode_length
        self.max_correction = max_correction
        self.current_step = 0
        self.episode_predictions = []
        self.episode_corrections = []
        self.cumulative_reward = 0
        
        # Action space: residual correction for 7D action (larger bounds)
        self.action_space = spaces.Box(
            low=-max_correction, high=max_correction,  # Larger action space
            shape=(7,), dtype=np.float32
        )
        
        # Observation space: original prediction + step info + cumulative correction
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(15,), dtype=np.float32  # 7D prediction + 7D cumulative correction + 1D step
        )
    
    def reset(self, seed=None):
        # Start new episode with random sequence
        start_idx = np.random.randint(len(self.predictions) - self.episode_length + 1)
        self.episode_predictions = self.predictions[start_idx:start_idx + self.episode_length]
        self.episode_ground_truths = self.ground_truths[start_idx:start_idx + self.episode_length]
        self.current_step = 0
        self.episode_corrections = []
        self.cumulative_reward = 0
        
        obs = np.concatenate([
            self.episode_predictions[0], 
            np.zeros(7),  # cumulative correction starts at zero
            [self.current_step / self.episode_length]
        ])
        return obs, {}
    
    def step(self, action):
        # Apply cumulative correction
        cumulative_correction = np.sum(self.episode_corrections, axis=0) if self.episode_corrections else np.zeros(7)
        new_correction = action
        total_correction = cumulative_correction + new_correction
        
        # Apply correction to original prediction
        corrected_action = self.episode_predictions[self.current_step] + total_correction
        gt = self.episode_ground_truths[self.current_step]
        
        # Dense reward: progress toward ground truth
        current_mae = np.mean(np.abs(corrected_action - gt))
        baseline_mae = np.mean(np.abs(self.episode_predictions[self.current_step] - gt))
        
        # Enhanced reward function with better shaping
        improvement_reward = (baseline_mae - current_mae) * 15.0  # Increased improvement reward
        proximity_reward = max(0, (0.2 - current_mae)) * 5.0  # Reward for being very close to target
        step_penalty = -0.05  # Smaller penalty to encourage longer exploration
        
        # Progressive shaping: more reward for consistent improvement
        if len(self.episode_corrections) > 0:
            prev_correction = self.episode_corrections[-1]
            correction_magnitude = np.linalg.norm(new_correction)
            prev_magnitude = np.linalg.norm(prev_correction)
            
            # Reward smooth corrections (penalize erratic changes)
            smoothness_reward = -abs(correction_magnitude - prev_magnitude) * 2.0
        else:
            smoothness_reward = 0
        
        # Final step bonus with scaling
        if self.current_step == self.episode_length - 1:
            improvement_ratio = (baseline_mae - current_mae) / (baseline_mae + 1e-8)
            final_bonus = max(0, improvement_ratio * 10.0)  # Scaled by relative improvement
        else:
            final_bonus = 0
        
        # Exploration bonus for early episodes
        exploration_bonus = 0.1 * np.random.normal(0, 0.1) if self.current_step < 2 else 0
        
        reward = improvement_reward + proximity_reward + step_penalty + final_bonus + smoothness_reward + exploration_bonus
        self.cumulative_reward += reward
        
        self.episode_corrections.append(new_correction)
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        
        # Next observation
        if not done:
            obs = np.concatenate([
                self.episode_predictions[self.current_step], 
                total_correction,
                [self.current_step / self.episode_length]
            ])
        else:
            obs = np.concatenate([corrected_action, total_correction, [1.0]])
        
        info = {
            'mae': current_mae, 
            'baseline_mae': baseline_mae,
            'improvement': baseline_mae - current_mae,
            'corrected_action': corrected_action,
            'cumulative_correction': total_correction,
            'step': self.current_step,
            'cumulative_reward': self.cumulative_reward
        }
        
        return obs, reward, done, False, info

class PPOActionCorrector(nn.Module):
    """Enhanced PPO with improved hyperparameters for better convergence"""
    
    def __init__(self, state_dim=15, action_dim=7, hidden_dim=256, lr=5e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_ratio=0.15, entropy_coef=0.005, value_coef=0.75):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Enhanced Actor-Critic networks with dropout and layer normalization
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim * 2)  # mean and log_std
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Use AdamW with weight decay for better generalization
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        
        # Learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=20
        )
        
        self.is_fitted = False
    
    def get_action(self, state):
        """Get action from policy"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        output = self.actor(state)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        
        return action.squeeze().detach().numpy(), log_prob.item()
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def fit(self, predictions, ground_truths, epochs=300):
        """Train using enhanced PPO with multi-step episodes and improved training"""
        print(f"🎯 Training Enhanced PPO Action Corrector ({epochs} epochs)...")
        
        env = ImprovedPPOActionCorrectionEnvironment(predictions, ground_truths)
        
        # Training metrics tracking
        best_avg_reward = float('-inf')
        patience_counter = 0
        max_patience = 50
        
        for epoch in range(epochs):
            # Collect trajectories
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            episode_rewards = []
            
            for _ in range(12):  # More episodes per epoch for better learning
                state, _ = env.reset()
                episode_states, episode_actions, episode_rewards_ep = [], [], []
                episode_log_probs, episode_values = [], []
                ep_reward = 0
                
                for step in range(8):  # Longer episodes for more learning
                    action, log_prob = self.get_action(state)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    value = self.critic(state_tensor).item()
                    
                    next_state, reward, done, _, info = env.step(action)
                    
                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_rewards_ep.append(reward)
                    episode_log_probs.append(log_prob)
                    episode_values.append(value)
                    
                    state = next_state
                    ep_reward += reward
                    
                    if done:
                        break
                
                # Add episode data
                states.extend(episode_states)
                actions.extend(episode_actions)
                rewards.extend(episode_rewards_ep)
                log_probs.extend(episode_log_probs)
                values.extend(episode_values)
                dones.extend([False] * (len(episode_rewards_ep) - 1) + [True])
                episode_rewards.append(ep_reward)
            
            # Compute next values for GAE
            next_values = values[1:] + [0]  # Terminal state has value 0
            
            # Compute advantages using GAE
            advantages = self.compute_gae(rewards, values, next_values, dones)
            
            # Convert to tensors
            states = torch.FloatTensor(np.array(states))
            actions = torch.FloatTensor(np.array(actions))
            rewards = torch.FloatTensor(np.array(rewards))
            old_log_probs = torch.FloatTensor(log_probs)
            old_values = torch.FloatTensor(values)
            advantages = torch.FloatTensor(np.array(advantages))
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update with improved loss and early stopping
            total_loss_epoch = 0
            for ppo_epoch in range(12):  # More PPO epochs for better learning
                # Get current policy outputs
                actor_output = self.actor(states)
                mean, log_std = actor_output.chunk(2, dim=-1)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(actions).sum(-1)
                
                critic_values = self.critic(states).squeeze()
                
                # Compute ratios
                ratios = torch.exp(new_log_probs - old_log_probs)
                
                # PPO loss with tuned clipping
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss with returns normalization
                returns = advantages + old_values
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                critic_loss = F.mse_loss(critic_values, returns)
                
                # Entropy bonus for exploration (adaptive)
                entropy = dist.entropy().sum(-1).mean()
                adaptive_entropy_coef = self.entropy_coef * max(0.1, 1.0 - epoch / epochs)  # Decay entropy
                
                # Combined loss with tuned coefficients
                total_loss = actor_loss + self.value_coef * critic_loss - adaptive_entropy_coef * entropy
                total_loss_epoch += total_loss.item()
                
                # Update with gradient clipping
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                self.optimizer.step()
            
            # Learning rate scheduling based on performance
            avg_reward = np.mean(episode_rewards)
            if hasattr(self, 'scheduler'):
                self.scheduler.step(-avg_reward)  # Negative because we want to maximize reward
            
            # Early stopping check
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                patience_counter = 0
                # Save best model state
                best_model_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
            
            # Improved logging
            if epoch % 10 == 0:
                avg_advantage = advantages.mean().item()
                current_lr = self.optimizer.param_groups[0]['lr']
                avg_ep_reward = np.mean(episode_rewards)
                print(f"  Epoch {epoch}: Avg Reward = {avg_reward:.4f}, Avg Episode Reward = {avg_ep_reward:.4f}, Avg Advantage = {avg_advantage:.4f}, LR = {current_lr:.6f}")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"  Early stopping at epoch {epoch} (patience: {patience_counter}/{max_patience})")
                # Restore best model
                if best_model_state:
                    self.load_state_dict(best_model_state)
                break
        
        self.is_fitted = True
    
    def predict(self, predictions):
        """Predict corrections for batch of predictions"""
        corrections = []
        
        for pred in predictions:
            # Create state with correct dimensions (15D: 7 prediction + 7 cumulative correction + 1 step)
            state = np.concatenate([pred, np.zeros(7), [0.0]])
            correction, _ = self.get_action(state)
            corrected = pred + correction
            corrections.append(corrected)
        
        return np.array(corrections)

# ============================================================================
# 2. UNSUPERVISED LEARNING APPROACHES
# ============================================================================

class ContrastiveActionEncoder(nn.Module):
    """Contrastive learning for action representations"""
    
    def __init__(self, action_dim=7, latent_dim=16, temperature=0.1):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.temperature = temperature
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.is_fitted = False
    
    def forward(self, x):
        features = self.encoder(x)
        projections = F.normalize(self.projection_head(features), dim=1)
        return features, projections
    
    def contrastive_loss(self, projections, labels):
        """Compute contrastive loss with numerical stability"""
        projections = F.normalize(projections, dim=1, eps=1e-8)
        similarity_matrix = torch.matmul(projections, projections.T)
        
        # Create positive and negative masks
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal
        mask = mask - torch.eye(mask.size(0)).to(mask.device)
        
        # Numerical stability: clamp similarity values
        similarity_matrix = torch.clamp(similarity_matrix, -10, 10)
        
        # Compute loss with temperature scaling
        exp_sim = torch.exp(similarity_matrix / self.temperature)
        
        # Avoid division by zero
        sum_exp_sim = exp_sim.sum(1, keepdim=True)
        sum_exp_sim = torch.clamp(sum_exp_sim, min=1e-8)
        
        log_prob = similarity_matrix / self.temperature - torch.log(sum_exp_sim)
        
        # Compute masked loss
        masked_log_prob = log_prob * mask
        
        # Avoid division by zero in mask
        mask_sum = torch.clamp(mask.sum(1), min=1e-8)
        loss = - (masked_log_prob.sum(1) / mask_sum).mean()
        
        # Check for NaN
        if torch.isnan(loss):
            return torch.tensor(0.0, device=projections.device, requires_grad=True)
        
        return loss
    
    def fit(self, actions, epochs=100):
        """Train contrastive encoder with improved pseudo-labels"""
        print(f"🎯 Training Improved Contrastive Action Encoder ({epochs} epochs)...")
        
        # Create better pseudo-labels using action magnitude and direction
        # Group actions by similarity in magnitude and direction
        action_magnitudes = np.linalg.norm(actions, axis=1)
        action_directions = actions / (action_magnitudes[:, np.newaxis] + 1e-8)
        
        # Cluster based on both magnitude and direction
        features_for_clustering = np.concatenate([
            actions,
            action_magnitudes[:, np.newaxis],
            action_directions
        ], axis=1)
        
        # Use more clusters for finer-grained learning
        kmeans = KMeans(n_clusters=min(12, len(actions) // 10), random_state=42)
        pseudo_labels = kmeans.fit_predict(features_for_clustering)
        
        # Additional temporal consistency: nearby samples should have similar labels
        temporal_labels = pseudo_labels.copy()
        for i in range(1, len(pseudo_labels)):
            if pseudo_labels[i] != pseudo_labels[i-1]:
                # Check if actions are similar enough to merge labels
                similarity = np.dot(actions[i], actions[i-1]) / (
                    np.linalg.norm(actions[i]) * np.linalg.norm(actions[i-1]) + 1e-8
                )
                if similarity > 0.8:  # High similarity threshold
                    temporal_labels[i] = temporal_labels[i-1]
        
        actions_tensor = torch.FloatTensor(actions)
        labels_tensor = torch.LongTensor(temporal_labels)
        
        for epoch in range(epochs):
            # Shuffle data
            perm = torch.randperm(len(actions_tensor))
            actions_shuffled = actions_tensor[perm]
            labels_shuffled = labels_tensor[perm]
            
            # Forward pass
            _, projections = self.forward(actions_shuffled)
            
            # Compute loss
            loss = self.contrastive_loss(projections, labels_shuffled)
            
            # Skip update if loss is NaN
            if torch.isnan(loss):
                print(f"  Epoch {epoch}: NaN loss detected, skipping update")
                continue
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
        
        self.is_fitted = True
    
    def encode(self, actions):
        """Encode actions to latent space"""
        with torch.no_grad():
            actions_tensor = torch.FloatTensor(actions)
            features, _ = self.forward(actions_tensor)
            
            # Check for NaN in features and replace with zeros
            if torch.isnan(features).any():
                print("⚠️ NaN detected in encoded features, replacing with zeros")
                features = torch.nan_to_num(features, nan=0.0)
            
            return features.numpy()

class ImprovedClusterAwareCorrector:
    """Improved Cluster-based residual correction with regularization"""
    
    def __init__(self, n_clusters=5, hidden_dim=64, min_samples_per_cluster=20):
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.min_samples_per_cluster = min_samples_per_cluster
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.correctors = {}
        self.cluster_stats = {}
        self.is_fitted = False
    
    def fit(self, predictions, ground_truths):
        """Train cluster-specific correctors with regularization"""
        print(f"🎯 Training Improved Cluster-Aware Corrector ({self.n_clusters} clusters)...")
        
        # Standardize predictions
        pred_scaled = self.scaler.fit_transform(predictions)
        
        # Cluster predictions
        cluster_labels = self.kmeans.fit_predict(pred_scaled)
        
        # Train separate corrector for each cluster
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            n_samples = mask.sum()
            
            if n_samples < self.min_samples_per_cluster:
                print(f"  ⚠️  Cluster {cluster_id}: {n_samples} samples (< {self.min_samples_per_cluster}), skipping")
                continue
            
            cluster_pred = predictions[mask]
            cluster_gt = ground_truths[mask]
            
            # Regularized neural corrector for this cluster
            corrector = nn.Sequential(
                nn.Linear(7, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),  # Dropout for regularization
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim, 7)
            )
            
            optimizer = optim.Adam(corrector.parameters(), lr=1e-3, weight_decay=1e-4)  # L2 regularization
            criterion = nn.MSELoss()
            
            # Train with early stopping
            pred_tensor = torch.FloatTensor(cluster_pred)
            gt_tensor = torch.FloatTensor(cluster_gt)
            
            best_loss = float('inf')
            patience_counter = 0
            max_patience = 50
            
            for epoch in range(1000):
                optimizer.zero_grad()
                corrected = corrector(pred_tensor)
                
                # MSE loss + L1 regularization for sparsity
                mse_loss = criterion(corrected, gt_tensor)
                l1_loss = 0.001 * sum(p.abs().sum() for p in corrector.parameters())
                total_loss = mse_loss + l1_loss
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(corrector.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Early stopping
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    patience_counter = 0
                    # Save best model
                    best_state = {k: v.clone() for k, v in corrector.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= max_patience:
                    break
            
            # Load best model
            corrector.load_state_dict(best_state)
            
            self.correctors[cluster_id] = corrector
            
            # Store cluster statistics
            self.cluster_stats[cluster_id] = {
                'n_samples': n_samples,
                'final_loss': best_loss,
                'avg_mae': np.mean(np.abs(cluster_pred - cluster_gt))
            }
            
            print(f"  ✅ Cluster {cluster_id}: {n_samples} samples, final loss: {best_loss:.6f}, avg MAE: {self.cluster_stats[cluster_id]['avg_mae']:.4f}")
        
        self.is_fitted = True
    
    def predict(self, predictions):
        """Predict using cluster-specific correctors with fallback"""
        pred_scaled = self.scaler.transform(predictions)
        cluster_labels = self.kmeans.predict(pred_scaled)
        
        corrections = []
        fallback_used = 0
        
        for i, (pred, cluster_id) in enumerate(zip(predictions, cluster_labels)):
            if cluster_id in self.correctors:
                with torch.no_grad():
                    pred_tensor = torch.FloatTensor(pred).unsqueeze(0)
                    corrected = self.correctors[cluster_id](pred_tensor)
                    corrections.append(corrected.squeeze().numpy())
            else:
                # Fallback: use simple linear correction based on cluster statistics
                if cluster_id in self.cluster_stats:
                    # Simple residual correction based on cluster average error
                    avg_error = self.cluster_stats[cluster_id]['avg_mae']
                    correction = pred * 0.1  # Small correction factor
                    corrections.append(pred - correction)
                else:
                    # Ultimate fallback: return original prediction
                    corrections.append(pred)
                fallback_used += 1
        
        if fallback_used > 0:
            print(f"  ⚠️  Used fallback for {fallback_used}/{len(predictions)} predictions")
        
        return np.array(corrections)

# ============================================================================
# 3. HYBRID APPROACHES
# ============================================================================

class ContrastivePPOCorrector:
    """Combine contrastive learning with PPO"""
    
    def __init__(self, latent_dim=16, hidden_dim=64):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Contrastive encoder
        self.encoder = ContrastiveActionEncoder(latent_dim=latent_dim)
        
        # Decoder to map latent corrections back to 7D action space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7)
        )
        
        # PPO in latent space
        self.ppo = PPOActionCorrector(
            state_dim=latent_dim, 
            action_dim=latent_dim,  # Output latent corrections
            hidden_dim=hidden_dim
        )
        
        self.optimizer = optim.Adam(self.decoder.parameters(), lr=1e-3)
        self.is_fitted = False
    
    def fit(self, predictions, ground_truths, epochs=100):
        """Train hybrid model"""
        print(f"🎯 Training Contrastive-PPO Hybrid ({epochs} epochs)...")
        
        # Step 1: Train contrastive encoder
        self.encoder.fit(predictions, epochs=50)
        
        # Step 2: Encode predictions to latent space
        latent_predictions = self.encoder.encode(predictions)
        
        # Step 3: Train PPO in latent space
        self.ppo.fit(latent_predictions, ground_truths, epochs)
        
        # Step 4: Train decoder to map latent corrections back to action space
        pred_tensor = torch.FloatTensor(predictions)
        gt_tensor = torch.FloatTensor(ground_truths)
        
        # Get latent corrections from PPO
        latent_corrections = []
        for i in range(len(predictions)):
            latent_pred = latent_predictions[i:i+1]
            correction, _ = self.ppo.get_action(latent_pred[0])
            latent_corrections.append(correction)
        
        latent_corrections = torch.FloatTensor(np.array(latent_corrections))
        
        # Train decoder
        for epoch in range(200):
            self.optimizer.zero_grad()
            decoded_corrections = self.decoder(latent_corrections)
            corrected_actions = pred_tensor + decoded_corrections
            loss = F.mse_loss(corrected_actions, gt_tensor)
            loss.backward()
            self.optimizer.step()
        
        self.is_fitted = True
    
    def predict(self, predictions):
        """Predict using hybrid model"""
        # Encode to latent space
        latent_predictions = self.encoder.encode(predictions)
        
        # Get latent corrections from PPO
        latent_corrections = []
        for i in range(len(predictions)):
            latent_pred = latent_predictions[i:i+1]
            correction, _ = self.ppo.get_action(latent_pred[0])
            latent_corrections.append(correction)
        
        latent_corrections = torch.FloatTensor(np.array(latent_corrections))
        
        # Decode corrections back to action space
        with torch.no_grad():
            decoded_corrections = self.decoder(latent_corrections)
        
        corrected = predictions + decoded_corrections.numpy()
        return corrected

# ============================================================================
# 4. EXPERIMENT FRAMEWORK
# ============================================================================

class AdvancedExperimentRunner:
    """Runner for advanced enhancement experiments with statistical rigor"""
    
    def __init__(self, baseline_file="baseline_500_samples_results.json"):
        self.baseline_data = BaselineData(baseline_file)
        self.predictions, self.ground_truths = self.baseline_data.get_arrays()
        
        # Validate data
        validate_data(self.predictions, self.ground_truths)
        
        self.baseline_mae = mean_absolute_error(self.predictions, self.ground_truths)
        print(f"📊 Baseline MAE: {self.baseline_mae:.6f}")
    
    def run_experiment(self, method_name: str, enhancer, k_folds=5) -> ExperimentResult:
        """Run a single advanced experiment with cross-validation and statistical testing"""
        print(f"\n🧪 Running advanced experiment: {method_name}")
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        all_predictions = []
        all_ground_truths = []
        fold_maes = []
        baseline_fold_maes = []
        
        training_times = []
        prediction_times = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.predictions)):
            print(f"  Fold {fold + 1}/{k_folds}")
            
            # Split data
            train_pred, val_pred = self.predictions[train_idx], self.predictions[val_idx]
            train_gt, val_gt = self.ground_truths[train_idx], self.ground_truths[val_idx]
            
            # Calculate baseline MAE for this fold
            baseline_fold_mae = mean_absolute_error(val_pred, val_gt)
            baseline_fold_maes.append(baseline_fold_mae)
            
            # Training
            start_time = time.time()
            
            if hasattr(enhancer, 'fit'):
                enhancer.fit(train_pred, train_gt)
            else:
                # Neural network training
                optimizer = optim.Adam(enhancer.parameters(), lr=1e-3)
                criterion = nn.MSELoss()
                
                train_pred_tensor = torch.FloatTensor(train_pred)
                train_gt_tensor = torch.FloatTensor(train_gt)
                
                enhancer.train()
                for epoch in range(1000):
                    optimizer.zero_grad()
                    enhanced = enhancer(train_pred_tensor)
                    loss = criterion(enhanced, train_gt_tensor)
                    loss.backward()
                    optimizer.step()
            
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Prediction
            start_time = time.time()
            
            if hasattr(enhancer, 'predict'):
                enhanced_pred = enhancer.predict(val_pred)
            else:
                enhancer.eval()
                with torch.no_grad():
                    val_pred_tensor = torch.FloatTensor(val_pred)
                    enhanced_pred = enhancer(val_pred_tensor).numpy()
            
            prediction_time = time.time() - start_time
            prediction_times.append(prediction_time)
            
            # Calculate fold MAE
            fold_mae = mean_absolute_error(enhanced_pred, val_gt)
            fold_maes.append(fold_mae)
            
            all_predictions.extend(enhanced_pred)
            all_ground_truths.extend(val_gt)
        
        # Calculate overall metrics
        all_predictions = np.array(all_predictions)
        all_ground_truths = np.array(all_ground_truths)
        
        mae = mean_absolute_error(all_predictions, all_ground_truths)
        mse = mean_squared_error(all_predictions, all_ground_truths)
        improvement = (self.baseline_mae - mae) / self.baseline_mae * 100
        
        # Statistical testing
        p_value, cohens_d = paired_t_test(baseline_fold_maes, fold_maes, method_name)
        
        # Bootstrap confidence intervals
        ci_lower, ci_upper, bootstrap_mean, bootstrap_std = bootstrap_ci(
            np.array(baseline_fold_maes), np.array(fold_maes)
        )
        
        result = ExperimentResult(
            method_name=method_name,
            predictions=all_predictions.tolist(),
            ground_truths=all_ground_truths.tolist(),
            mae=mae,
            mse=mse,
            improvement_pct=improvement,
            training_time=np.mean(training_times),
            prediction_time=np.mean(prediction_times),
            metadata={
                'baseline_mae': self.baseline_mae,
                'k_folds': k_folds,
                'total_samples': len(all_predictions),
                'fold_maes': fold_maes,
                'baseline_fold_maes': baseline_fold_maes,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'bootstrap_mean': bootstrap_mean,
                'bootstrap_std': bootstrap_std
            }
        )
        
        print(f"✅ {method_name}:")
        print(f"   MAE={mae:.6f}, Improvement={improvement:.2f}%")
        print(f"   95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
        if p_value is not None:
            print(f"   Statistical significance: p={p_value:.6f}, d={cohens_d:.3f}")
        
        return result
    
    def run_all_advanced_experiments(self):
        """Run all improved advanced experiments"""
        experiments = []
        
        print("\n🚀 Starting Advanced Reinforcement Learning Experiments")
        print("=" * 70)
        
        # 1. Enhanced PPO Action Correction
        ppo_corrector = PPOActionCorrector()
        ppo_result = self.run_experiment("Enhanced PPO Action Correction", ppo_corrector)
        experiments.append(ppo_result)
        
        # 2. Improved Contrastive + MLP
        class ImprovedContrastiveMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = ContrastiveActionEncoder()
                self.mlp = nn.Sequential(
                    nn.Linear(16, 64), nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 64), nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 7)
                )
                self.optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
                self.is_fitted = False
            
            def fit(self, predictions, ground_truths, epochs=100):
                self.encoder.fit(predictions, epochs=50)
                latent_pred = self.encoder.encode(predictions)
                
                pred_tensor = torch.FloatTensor(latent_pred)
                gt_tensor = torch.FloatTensor(ground_truths)
                
                # Train with early stopping
                best_loss = float('inf')
                patience = 0
                
                for epoch in range(500):
                    self.optimizer.zero_grad()
                    corrected = self.mlp(pred_tensor)
                    loss = F.mse_loss(corrected, gt_tensor)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        patience = 0
                    else:
                        patience += 1
                        if patience >= 50:
                            break
                
                self.is_fitted = True
            
            def predict(self, predictions):
                latent_pred = self.encoder.encode(predictions)
                with torch.no_grad():
                    pred_tensor = torch.FloatTensor(latent_pred)
                    corrected = self.mlp(pred_tensor)
                return corrected.numpy()
        
        contrastive_mlp = ImprovedContrastiveMLP()
        contrastive_result = self.run_experiment("Improved Contrastive + MLP", contrastive_mlp)
        experiments.append(contrastive_result)
        
        # 3. Improved Cluster-Aware Correction
        cluster_corrector = ImprovedClusterAwareCorrector(n_clusters=5, min_samples_per_cluster=20)
        cluster_result = self.run_experiment("Improved Cluster-Aware Correction", cluster_corrector)
        experiments.append(cluster_result)
        
        # Skip hybrid approach due to complexity
        print("⚠️  Skipping Contrastive-PPO Hybrid due to implementation complexity")
        
        return experiments
    
    def print_advanced_summary(self, experiments):
        """Print summary of advanced experiments with statistical rigor"""
        print(f"\n📈 IMPROVED ADVANCED EXPERIMENT SUMMARY")
        print("=" * 70)
        print(f"{'Method':<30} {'MAE':<12} {'Improvement':<12} {'95% CI':<12} {'p-value':<10}")
        print("-" * 70)
        
        for exp in experiments:
            ci_lower = exp.metadata.get('ci_lower', 0)
            ci_upper = exp.metadata.get('ci_upper', 0)
            p_value = exp.metadata.get('p_value', None)
            p_str = f"{p_value:.4f}" if p_value is not None else "N/A"
            
            print(f"{exp.method_name:<30} {exp.mae:<12.6f} {exp.improvement_pct:<12.2f}% [{ci_lower:<6.2f}%,{ci_upper:<6.2f}%] {p_str:<10}")
        
        # Find best method
        best_exp = max(experiments, key=lambda x: x.improvement_pct)
        print(f"\n🏆 Best advanced method: {best_exp.method_name}")
        print(f"   Improvement: {best_exp.improvement_pct:.2f}%")
        print(f"   MAE: {best_exp.mae:.6f}")
        
        if best_exp.metadata.get('p_value') is not None:
            print(f"   Statistical significance: p={best_exp.metadata['p_value']:.6f}")
            print(f"   Effect size (Cohen's d): {best_exp.metadata.get('cohens_d', 0):.3f}")
        
        # Compare to current best (35.11%)
        current_best = 35.11
        if best_exp.improvement_pct > current_best:
            print(f"🎉 NEW RECORD! Beat previous best ({current_best:.2f}%) by {best_exp.improvement_pct - current_best:.2f}%!")
        else:
            print(f"📊 Current best (Deep Neural): {current_best:.2f}% remains undefeated")
            gap = current_best - best_exp.improvement_pct
            print(f"   Gap: {gap:.2f}%")
        
        # Statistical validity assessment
        print(f"\n🔍 Statistical Validity Assessment:")
        significant_methods = [exp for exp in experiments if exp.metadata.get('p_value', 1) < 0.05]
        print(f"   Statistically significant methods (p < 0.05): {len(significant_methods)}/{len(experiments)}")
        
        for exp in significant_methods:
            print(f"   - {exp.method_name}: p={exp.metadata['p_value']:.6f}, d={exp.metadata.get('cohens_d', 0):.3f}")
        
        return experiments

def evaluate_rl_statistical_significance(baseline_maes, enhanced_maes, method_name, confidence_level=0.95):
    """
    Comprehensive statistical evaluation for RL enhancement methods
    """
    if len(baseline_maes) != len(enhanced_maes):
        print(f"⚠️  Warning: Unequal sample sizes for {method_name}")
        min_len = min(len(baseline_maes), len(enhanced_maes))
        baseline_maes = baseline_maes[:min_len]
        enhanced_maes = enhanced_maes[:min_len]
    
    baseline_maes = np.array(baseline_maes)
    enhanced_maes = np.array(enhanced_maes)
    
    results = {}
    
    # Paired t-test
    t_stat, t_p_value = stats.ttest_rel(baseline_maes, enhanced_maes)
    results['paired_t_test'] = {
        'statistic': t_stat,
        'p_value': t_p_value,
        'significant_05': t_p_value < 0.05,
        'significant_01': t_p_value < 0.01,
        'significant_001': t_p_value < 0.001
    }
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(baseline_maes, enhanced_maes)
        results['wilcoxon_test'] = {
            'statistic': wilcoxon_stat,
            'p_value': wilcoxon_p,
            'significant_05': wilcoxon_p < 0.05,
            'significant_01': wilcoxon_p < 0.01
        }
    except Exception as e:
        print(f"⚠️  Wilcoxon test failed for {method_name}: {e}")
        results['wilcoxon_test'] = None
    
    # Effect size calculations
    diff = baseline_maes - enhanced_maes
    pooled_std = np.sqrt((np.var(baseline_maes) + np.var(enhanced_maes)) / 2)
    cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0
    
    # Glass's delta (uses only control group SD)
    glass_delta = np.mean(diff) / np.std(baseline_maes) if np.std(baseline_maes) > 0 else 0
    
    results['effect_sizes'] = {
        'cohens_d': cohens_d,
        'glass_delta': glass_delta,
        'mean_improvement': np.mean(diff),
        'improvement_std': np.std(diff),
        'cohens_d_interpretation': interpret_cohens_d(cohens_d),
        'glass_delta_interpretation': interpret_cohens_d(glass_delta)
    }
    
    # Confidence intervals for mean difference
    alpha = 1 - confidence_level
    degrees_freedom = len(baseline_maes) - 1
    t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom)
    
    mean_diff = np.mean(diff)
    sem_diff = stats.sem(diff)
    ci_lower = mean_diff - t_critical * sem_diff
    ci_upper = mean_diff + t_critical * sem_diff
    
    results['confidence_interval'] = {
        'level': confidence_level,
        'mean_difference': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'contains_zero': ci_lower <= 0 <= ci_upper,
        'width': ci_upper - ci_lower
    }
    
    # Descriptive statistics
    results['descriptive_stats'] = {
        'baseline': {
            'mean': np.mean(baseline_maes),
            'std': np.std(baseline_maes),
            'median': np.median(baseline_maes),
            'min': np.min(baseline_maes),
            'max': np.max(baseline_maes),
            'n': len(baseline_maes)
        },
        'enhanced': {
            'mean': np.mean(enhanced_maes),
            'std': np.std(enhanced_maes),
            'median': np.median(enhanced_maes),
            'min': np.min(enhanced_maes),
            'max': np.max(enhanced_maes),
            'n': len(enhanced_maes)
        }
    }
    
    # Overall significance assessment
    p_values = [t_p_value]
    if wilcoxon_p is not None:
        p_values.append(wilcoxon_p)
    
    min_p_value = min(p_values)
    results['overall_significance'] = {
        'min_p_value': min_p_value,
        'is_significant_05': min_p_value < 0.05,
        'is_significant_01': min_p_value < 0.01,
        'is_significant_001': min_p_value < 0.001,
        'evidence_strength': interpret_evidence_strength(min_p_value),
        'recommended_test': 'wilcoxon' if results.get('normality_test', {}).get('normal', True) == False else 't_test'
    }
    
    return results

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size with more granular levels"""
    abs_d = abs(d)
    if abs_d < 0.01:
        return "Very small"
    elif abs_d < 0.2:
        return "Small"
    elif abs_d < 0.5:
        return "Medium"
    elif abs_d < 0.8:
        return "Large"
    elif abs_d < 1.2:
        return "Very large"
    else:
        return "Huge"

def interpret_evidence_strength(p):
    """Interpret p-value strength of evidence"""
    if p < 0.001:
        return "Extremely strong evidence"
    elif p < 0.01:
        return "Strong evidence"
    elif p < 0.05:
        return "Moderate evidence"
    elif p < 0.1:
        return "Weak evidence"
    else:
        return "No evidence"

def print_rl_statistical_summary(stat_results, method_name):
    """Print comprehensive statistical summary for RL methods"""
    print(f"\n🧪 STATISTICAL ANALYSIS - {method_name.upper()}")
    print("=" * 60)
    
    # Descriptive stats
    desc = stat_results['descriptive_stats']
    print(f"\n📊 DESCRIPTIVE STATISTICS:")
    print(f"  Baseline:  {desc['baseline']['mean']:.6f} ± {desc['baseline']['std']:.6f} (n={desc['baseline']['n']})")
    print(f"  Enhanced: {desc['enhanced']['mean']:.6f} ± {desc['enhanced']['std']:.6f} (n={desc['enhanced']['n']})")
    
    # Paired t-test
    t_test = stat_results['paired_t_test']
    print(f"\n🎯 PAIRED T-TEST:")
    print(f"  t-statistic: {t_test['statistic']:.4f}")
    print(f"  p-value: {t_test['p_value']:.8f}")
    sig_symbols = {
        (True, True, True): "***",
        (True, True, False): "**", 
        (True, False, False): "*",
        (False, False, False): "ns"
    }
    sig_key = (t_test['significant_001'], t_test['significant_01'], t_test['significant_05'])
    print(f"  Significance: {sig_symbols.get(sig_key, 'ns')} (α=0.05: {'✅' if t_test['significant_05'] else '❌'})")
    
    # Wilcoxon test
    if stat_results['wilcoxon_test']:
        wilcoxon = stat_results['wilcoxon_test']
        print(f"\n📈 WILCOXON SIGNED-RANK TEST:")
        print(f"  statistic: {wilcoxon['statistic']:.4f}")
        print(f"  p-value: {wilcoxon['p_value']:.8f}")
        print(f"  Significance: {'✅' if wilcoxon['significant_05'] else '❌'} (α=0.05)")
    
    # Effect sizes
    effect = stat_results['effect_sizes']
    print(f"\n💪 EFFECT SIZES:")
    print(f"  Cohen's d: {effect['cohens_d']:.4f} ({effect['cohens_d_interpretation']})")
    print(f"  Glass's Δ: {effect['glass_delta']:.4f} ({effect['glass_delta_interpretation']})")
    print(f"  Mean improvement: {effect['mean_improvement']:.6f} ± {effect['improvement_std']:.6f}")
    
    # Confidence intervals
    ci = stat_results['confidence_interval']
    print(f"\n📏 {ci['level']*100:.0f}% CONFIDENCE INTERVAL:")
    print(f"  Mean difference: {ci['mean_difference']:.6f}")
    print(f"  CI: [{ci['ci_lower']:.6f}, {ci['ci_upper']:.6f}]")
    print(f"  Width: {ci['width']:.6f}")
    print(f"  Contains zero: {'Yes' if ci['contains_zero'] else 'No'}")
    
    # Overall assessment
    overall = stat_results['overall_significance']
    print(f"\n🏆 OVERALL ASSESSMENT:")
    print(f"  Evidence: {overall['evidence_strength']}")
    print(f"  Recommended test: {overall['recommended_test']}")
    print(f"  Significant at α=0.05: {'✅ Yes' if overall['is_significant_05'] else '❌ No'}")
    print(f"  Significant at α=0.01: {'✅ Yes' if overall['is_significant_01'] else '❌ No'}")
    
    print("=" * 60)

def main():
    """Main execution"""
    set_seed(42)
    
    parser = argparse.ArgumentParser(description="Advanced VLA Enhancement Experiments")
    parser.add_argument('--baseline_file', type=str, default="baseline_500_samples_results.json",
                        help="Path to baseline predictions file")
    parser.add_argument('--k_folds', type=int, default=5, help="K-fold cross validation")
    
    args = parser.parse_args()
    
    # Initialize advanced experiment runner
    runner = AdvancedExperimentRunner(args.baseline_file)
    
    # Run all advanced experiments
    experiments = runner.run_all_advanced_experiments()
    
    # Print summary
    runner.print_advanced_summary(experiments)
    
    print(f"\n🎉 Advanced experiments completed!")
    print(f"📁 Check results for potential new best methods!")

if __name__ == "__main__":
    main()
