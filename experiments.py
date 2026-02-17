#!/usr/bin/env python3
"""
Experiments Framework - Testing Brainstormed Enhancement Ideas
Various approaches for improving OpenVLA action predictions
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
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import gymnasium as gym
from gymnasium import spaces
import nashpy as nash
from scipy import stats

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

@dataclass
class ExperimentResult:
    """Result format for experiments"""
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
    
    def __init__(self, baseline_file="baseline_100_samples_results.json"):
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
            
            self.metadata = data.get('summary', {})
            
            for pred in data.get('detailed_results', []):
                if 'openvla_prediction' in pred and 'ground_truth' in pred:
                    self.predictions.append(np.array(pred['openvla_prediction']))
                    self.ground_truths.append(np.array(pred['ground_truth']))
            
            print(f"✅ Loaded {len(self.predictions)} baseline predictions")
            
        except Exception as e:
            print(f"❌ Error loading baseline data: {e}")
    
    def get_arrays(self):
        """Get predictions and ground truths as numpy arrays"""
        return np.array(self.predictions), np.array(self.ground_truths)

# ============================================================================
# NEURAL NETWORK APPROACHES
# ============================================================================

class SimpleNeuralEnhancer(nn.Module):
    """Simple neural network for action enhancement"""
    
    def __init__(self, action_dim=7, hidden_dim=64):
        super().__init__()
        self.action_dim = action_dim
        
        # Simple but effective architecture
        self.layers = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
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
        residual = self.layers(x)
        return x + self.residual_scale * residual

class DeepNeuralEnhancer(nn.Module):
    """Deep neural network with more capacity"""
    
    def __init__(self, action_dim=7, hidden_dims=[128, 256, 128]):
        super().__init__()
        self.action_dim = action_dim
        
        layers = []
        prev_dim = action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
        
        # Learnable residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        residual = self.network(x)
        return x + self.residual_scale * residual

class AttentionNeuralEnhancer(nn.Module):
    """Neural enhancer with attention mechanism"""
    
    def __init__(self, action_dim=7, hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim
        
        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        features = self.feature_net(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        residual = self.output_net(attended_features)
        return x + self.residual_scale * residual

# ============================================================================
# CLASSICAL MACHINE LEARNING APPROACHES
# ============================================================================

class LinearCorrectionEnhancer:
    """Linear regression-based correction"""
    
    def __init__(self, method='ridge', alpha=1.0):
        self.method = method
        self.alpha = alpha
        self.model = None
        self.is_fitted = False
    
    def fit(self, predictions, ground_truths):
        """Fit correction model"""
        if self.method == 'ridge':
            self.model = Ridge(alpha=self.alpha)
        else:
            self.model = LinearRegression()
        
        # Learn to predict the residual
        residuals = ground_truths - predictions
        self.model.fit(predictions, residuals)
        self.is_fitted = True
    
    def predict(self, predictions):
        """Apply correction"""
        if not self.is_fitted:
            return predictions
        
        residuals = self.model.predict(predictions)
        return predictions + residuals

class EnsembleEnhancer:
    """Ensemble of different enhancement methods"""
    
    def __init__(self, methods=['ridge', 'rf', 'gp']):
        self.methods = methods
        self.models = {}
        self.weights = None
        self.is_fitted = False
    
    def fit(self, predictions, ground_truths):
        """Fit ensemble models"""
        for method in self.methods:
            if method == 'ridge':
                self.models[method] = Ridge(alpha=1.0)
            elif method == 'rf':
                self.models[method] = RandomForestRegressor(n_estimators=50, random_state=42)
            elif method == 'gp':
                kernel = RBF(length_scale=1.0)
                self.models[method] = GaussianProcessRegressor(kernel=kernel, random_state=42)
            
            # Fit to predict residuals
            residuals = ground_truths - predictions
            self.models[method].fit(predictions, residuals)
        
        # Learn ensemble weights based on validation performance
        self._learn_weights(predictions, ground_truths)
        self.is_fitted = True
    
    def _learn_weights(self, predictions, ground_truths, cv_folds=3):
        """Learn optimal ensemble weights"""
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        method_errors = {method: [] for method in self.methods}
        
        for train_idx, val_idx in kf.split(predictions):
            train_pred, val_pred = predictions[train_idx], predictions[val_idx]
            train_gt, val_gt = ground_truths[train_idx], ground_truths[val_idx]
            
            for method in self.methods:
                model = self.models[method] if method != 'gp' else GaussianProcessRegressor(random_state=42)
                residuals = train_gt - train_pred
                model.fit(train_pred, residuals)
                
                pred_residuals = model.predict(val_pred)
                enhanced = val_pred + pred_residuals
                error = mean_absolute_error(enhanced, val_gt)
                method_errors[method].append(error)
        
        # Compute weights inversely proportional to errors
        avg_errors = {method: np.mean(errors) for method, errors in method_errors.items()}
        total_error = sum(avg_errors.values())
        self.weights = {method: (total_error - error) / total_error for method, error in avg_errors.items()}
    
    def predict(self, predictions):
        """Apply ensemble correction"""
        if not self.is_fitted:
            return predictions
        
        enhanced_predictions = []
        
        for i, pred in enumerate(predictions):
            ensemble_pred = pred.copy()
            weighted_residual = np.zeros_like(pred)
            
            for method, weight in self.weights.items():
                residual = self.models[method].predict(pred.reshape(1, -1))
                weighted_residual += weight * residual[0]
            
            ensemble_pred += weighted_residual
            enhanced_predictions.append(ensemble_pred)
        
        return np.array(enhanced_predictions)

# ============================================================================
# DEEP REINFORCEMENT LEARNING APPROACHES
# ============================================================================

class ActionCorrectionEnv(gym.Env):
    """Custom environment for action correction using RL"""
    
    def __init__(self, predictions, ground_truths, max_steps=10):
        super().__init__()
        self.predictions = predictions
        self.ground_truths = ground_truths
        self.max_steps = max_steps
        self.current_idx = 0
        self.current_step = 0
        
        # Action space: correction deltas for each action dimension
        self.action_dim = predictions.shape[1] if len(predictions.shape) > 1 else 1
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, 
            shape=(self.action_dim,), dtype=np.float32
        )
        
        # Observation space: current prediction and ground truth
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.action_dim * 2,), dtype=np.float32
        )
        
        self.current_prediction = None
        self.current_ground_truth = None
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_idx = np.random.randint(0, len(self.predictions))
        self.current_step = 0
        self.current_prediction = self.predictions[self.current_idx].copy()
        self.current_ground_truth = self.ground_truths[self.current_idx].copy()
        return self._get_observation(), {}
    
    def _get_observation(self):
        return np.concatenate([self.current_prediction, self.current_ground_truth])
    
    def step(self, action):
        # Apply correction
        self.current_prediction += action
        
        # Calculate reward (negative MSE)
        mse = np.mean((self.current_prediction - self.current_ground_truth) ** 2)
        reward = -mse
        
        # Check if done
        self.current_step += 1
        done = self.current_step >= self.max_steps or mse < 1e-6
        
        return self._get_observation(), reward, done, False, {}

class DQNCorrector(nn.Module):
    """Deep Q-Network for action correction"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        self.action_dim = action_dim
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output Q-values for discrete actions
        self.q_network = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, action_dim * 21)  # 21 discrete actions per dimension
        
    def forward(self, state):
        features = self.q_network(state)
        q_values = self.value_head(features)
        return q_values.view(-1, self.action_dim, 21)

class DeepRLCorrector:
    """Deep RL-based action correction using DQN"""
    
    def __init__(self, action_dim=7, learning_rate=1e-3, gamma=0.99, epsilon=0.1):
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.is_fitted = False
        
        # Discrete action space
        self.action_bins = np.linspace(-0.1, 0.1, 21)
        
    def fit(self, predictions, ground_truths, episodes=100):
        """Train RL agent"""
        # Create environment
        env = ActionCorrectionEnv(predictions, ground_truths)
        
        # Initialize models
        state_dim = env.observation_space.shape[0]
        self.model = DQNCorrector(state_dim, self.action_dim)
        self.target_model = DQNCorrector(state_dim, self.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        for episode in range(episodes):
            state, _ = env.reset()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            total_reward = 0
            done = False
            
            while not done:
                # Epsilon-greedy action selection
                if np.random.random() < self.epsilon:
                    # Random action
                    actions = np.random.choice(len(self.action_bins), size=self.action_dim)
                else:
                    # Greedy action
                    with torch.no_grad():
                        q_values = self.model(state_tensor)
                        actions = q_values.argmax(dim=-1).squeeze(0).numpy()
                
                # Convert discrete actions to continuous corrections
                action_corrections = self.action_bins[actions]
                
                # Environment step
                next_state, reward, done, _, _ = env.step(action_corrections)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                
                # Q-learning update
                current_q = self.model(state_tensor).gather(2, torch.tensor(actions).unsqueeze(0).unsqueeze(-1))
                next_q = self.target_model(next_state_tensor).max(dim=-1)[0].unsqueeze(-1)
                target_q = reward + (self.gamma * next_q * (1 - done))
                
                loss = F.mse_loss(current_q, target_q.detach())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                state_tensor = next_state_tensor
                total_reward += reward
            
            # Update target network
            if episode % 10 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            if episode % 20 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.4f}")
        
        self.is_fitted = True
    
    def predict(self, predictions):
        """Apply RL correction"""
        if not self.is_fitted:
            return predictions
        
        corrected_predictions = []
        
        for pred in predictions:
            state = np.concatenate([pred, pred])  # Use pred as both current and target
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.model(state_tensor)
                actions = q_values.argmax(dim=-1).squeeze(0).numpy()
                corrections = self.action_bins[actions]
                
                corrected = pred + corrections
                corrected_predictions.append(corrected)
        
        return np.array(corrected_predictions)

# ============================================================================
# COLLABORATIVE GAME THEORY APPROACHES
# ============================================================================

class GameTheoryCorrector:
    """Game theory-based collaborative correction"""
    
    def __init__(self, n_agents=3, action_dim=7, max_iterations=50):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.max_iterations = max_iterations
        self.agent_models = []
        self.weights = None
        self.is_fitted = False
        
    def _create_agent_payoff_matrix(self, predictions, ground_truths, agent_id):
        """Create payoff matrix for an agent"""
        n_samples = len(predictions)
        
        # Different strategies for each agent
        strategies = [
            'conservative',  # Small corrections
            'aggressive',    # Large corrections
            'adaptive'       # Context-dependent corrections
        ]
        
        payoff_matrix = np.zeros((len(strategies), n_samples))
        
        for i, strategy in enumerate(strategies):
            for j, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                if strategy == 'conservative':
                    correction = 0.01 * np.sign(gt - pred)
                elif strategy == 'aggressive':
                    correction = 0.1 * np.sign(gt - pred)
                else:  # adaptive
                    error = gt - pred
                    correction = 0.05 * np.tanh(error * 10)
                
                corrected = pred + correction
                payoff = -np.mean((corrected - gt) ** 2)  # Negative MSE as payoff
                payoff_matrix[i, j] = payoff
        
        return payoff_matrix
    
    def _find_nash_equilibrium(self, payoff_matrices):
        """Find Nash equilibrium for multi-agent game"""
        # Simplified: use weighted average of best responses
        strategies = ['conservative', 'aggressive', 'adaptive']
        equilibrium_strategies = []
        
        for agent_id, payoff_matrix in enumerate(payoff_matrices):
            # Find best response to other agents' average strategy
            avg_payoffs = np.mean(payoff_matrix, axis=1)
            best_strategy_idx = np.argmax(avg_payoffs)
            equilibrium_strategies.append(strategies[best_strategy_idx])
        
        return equilibrium_strategies
    
    def fit(self, predictions, ground_truths):
        """Fit collaborative game theory model"""
        # Create payoff matrices for each agent
        payoff_matrices = []
        for agent_id in range(self.n_agents):
            payoff_matrix = self._create_agent_payoff_matrix(predictions, ground_truths, agent_id)
            payoff_matrices.append(payoff_matrix)
        
        # Find Nash equilibrium
        self.equilibrium_strategies = self._find_nash_equilibrium(payoff_matrices)
        
        # Train agent models based on equilibrium strategies
        self.agent_models = []
        self.weights = []
        
        for agent_id, strategy in enumerate(self.equilibrium_strategies):
            # Create correction model based on strategy
            if strategy == 'conservative':
                model = LinearCorrectionEnhancer('ridge', alpha=10.0)
            elif strategy == 'aggressive':
                model = DeepNeuralEnhancer(action_dim=self.action_dim, hidden_dims=[256, 512])
            else:  # adaptive
                model = AttentionNeuralEnhancer(action_dim=self.action_dim, hidden_dim=256)
            
            # Train the model
            if hasattr(model, 'fit'):
                model.fit(predictions, ground_truths)
            else:
                # Neural network training
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.MSELoss()
                
                pred_tensor = torch.FloatTensor(predictions)
                gt_tensor = torch.FloatTensor(ground_truths)
                
                model.train()
                for epoch in range(500):  # Shorter training for game theory
                    optimizer.zero_grad()
                    enhanced = model(pred_tensor)
                    loss = criterion(enhanced, gt_tensor)
                    loss.backward()
                    optimizer.step()
            
            self.agent_models.append(model)
            
            # Calculate weight based on strategy performance
            if hasattr(model, 'predict'):
                corrected = model.predict(predictions)
            else:
                model.eval()
                with torch.no_grad():
                    pred_tensor = torch.FloatTensor(predictions)
                    corrected = model(pred_tensor).numpy()
            
            mae = mean_absolute_error(corrected, ground_truths)
            weight = 1.0 / (mae + 1e-6)
            self.weights.append(weight)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        self.is_fitted = True
        print(f"Equilibrium strategies: {self.equilibrium_strategies}")
        print(f"Agent weights: {self.weights}")
    
    def predict(self, predictions):
        """Apply collaborative game theory correction"""
        if not self.is_fitted:
            return predictions
        
        # Get predictions from all agents
        agent_predictions = []
        for model in self.agent_models:
            if hasattr(model, 'predict'):
                pred = model.predict(predictions)
            else:
                model.eval()
                with torch.no_grad():
                    pred_tensor = torch.FloatTensor(predictions)
                    pred = model(pred_tensor).numpy()
            agent_predictions.append(pred)
        
        # Weighted ensemble of agent predictions
        final_prediction = np.zeros_like(predictions)
        for i, pred in enumerate(agent_predictions):
            final_prediction += self.weights[i] * pred
        
        return final_prediction

class CooperativeGameCorrector:
    """Cooperative game theory using Shapley values"""
    
    def __init__(self, action_dim=7, base_models=None):
        self.action_dim = action_dim
        self.base_models = base_models or [
            LinearCorrectionEnhancer('ridge', alpha=1.0),
            DeepNeuralEnhancer(action_dim=action_dim),
            AttentionNeuralEnhancer(action_dim=action_dim)
        ]
        self.shapley_values = None
        self.is_fitted = False
    
    def _calculate_shapley_values(self, predictions, ground_truths):
        """Calculate Shapley values for model contribution"""
        n_models = len(self.base_models)
        shapley_values = np.zeros(n_models)
        
        # Get all model predictions
        model_predictions = []
        for model in self.base_models:
            if hasattr(model, 'predict'):
                pred = model.predict(predictions)
            else:
                model.eval()
                with torch.no_grad():
                    pred_tensor = torch.FloatTensor(predictions)
                    pred = model(pred_tensor).numpy()
            model_predictions.append(pred)
            model.mae = mean_absolute_error(pred, ground_truths)
        
        # Calculate Shapley values (simplified approximation)
        for i in range(n_models):
            # Contribution when model i is included
            with_i = np.mean([model_predictions[j] for j in range(n_models) if j != i], axis=0)
            with_i = (with_i + model_predictions[i]) / 2
            mae_with_i = mean_absolute_error(with_i, ground_truths)
            
            # Contribution when model i is excluded
            without_i = np.mean([model_predictions[j] for j in range(n_models) if j != i], axis=0)
            mae_without_i = mean_absolute_error(without_i, ground_truths)
            
            # Shapley value is the marginal contribution
            shapley_values[i] = (mae_without_i - mae_with_i)
        
        # Normalize to positive weights
        shapley_values = np.maximum(shapley_values, 0)
        if np.sum(shapley_values) > 0:
            shapley_values /= np.sum(shapley_values)
        else:
            shapley_values = np.ones(n_models) / n_models
        
        return shapley_values
    
    def fit(self, predictions, ground_truths):
        """Fit cooperative game theory model"""
        # Train all base models
        for model in self.base_models:
            if hasattr(model, 'fit'):
                model.fit(predictions, ground_truths)
            else:
                # Neural network training
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.MSELoss()
                
                pred_tensor = torch.FloatTensor(predictions)
                gt_tensor = torch.FloatTensor(ground_truths)
                
                model.train()
                for epoch in range(500):  # Shorter training for cooperative game
                    optimizer.zero_grad()
                    enhanced = model(pred_tensor)
                    loss = criterion(enhanced, gt_tensor)
                    loss.backward()
                    optimizer.step()
        
        # Calculate Shapley values
        self.shapley_values = self._calculate_shapley_values(predictions, ground_truths)
        
        self.is_fitted = True
        print(f"Shapley values: {self.shapley_values}")
    
    def predict(self, predictions):
        """Apply cooperative game theory correction"""
        if not self.is_fitted:
            return predictions
        
        # Weighted combination using Shapley values
        final_prediction = np.zeros_like(predictions)
        
        for i, (model, weight) in enumerate(zip(self.base_models, self.shapley_values)):
            if hasattr(model, 'predict'):
                pred = model.predict(predictions)
            else:
                model.eval()
                with torch.no_grad():
                    pred_tensor = torch.FloatTensor(predictions)
                    pred = model(pred_tensor).numpy()
            final_prediction += weight * pred
        
        return final_prediction

# ============================================================================
# ADVANCED ENSEMBLE AND META-LEARNING APPROACHES
# ============================================================================

class StackingEnsembleEnhancer:
    """Stacking ensemble with meta-learner"""
    
    def __init__(self, base_models=None, meta_model=None):
        self.base_models = base_models or [
            LinearCorrectionEnhancer('ridge', alpha=1.0),
            LinearCorrectionEnhancer('linear'),
            RandomForestRegressor(n_estimators=50, random_state=42)
        ]
        self.meta_model = meta_model or Ridge(alpha=0.1)
        self.is_fitted = False
    
    def fit(self, predictions, ground_truths):
        """Fit stacking ensemble"""
        # Train base models
        base_predictions = []
        for model in self.base_models:
            if hasattr(model, 'fit'):
                model.fit(predictions, ground_truths)
                pred = model.predict(predictions)
            else:
                # For sklearn models that don't have fit/predict pattern
                residuals = ground_truths - predictions
                model.fit(predictions, residuals)
                pred = predictions + model.predict(predictions)
            base_predictions.append(pred)
        
        # Train meta-learner on base model predictions
        meta_features = np.column_stack(base_predictions)
        self.meta_model.fit(meta_features, ground_truths)
        self.is_fitted = True
    
    def predict(self, predictions):
        """Apply stacking ensemble"""
        if not self.is_fitted:
            return predictions
        
        # Get base model predictions
        base_predictions = []
        for model in self.base_models:
            if hasattr(model, 'predict'):
                pred = model.predict(predictions)
            else:
                pred = predictions + model.predict(predictions)
            base_predictions.append(pred)
        
        # Meta-learner prediction
        meta_features = np.column_stack(base_predictions)
        return self.meta_model.predict(meta_features)

class BayesianOptimizationEnhancer:
    """Bayesian optimization for hyperparameter tuning"""
    
    def __init__(self, n_calls=20, random_state=42):
        self.n_calls = n_calls
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.is_fitted = False
    
    def _objective_function(self, params, predictions, ground_truths):
        """Objective function for optimization"""
        alpha = params[0]
        
        # Create and evaluate model with given parameters
        model = LinearCorrectionEnhancer('ridge', alpha=alpha)
        
        # Cross-validation
        kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in kf.split(predictions):
            train_pred, val_pred = predictions[train_idx], predictions[val_idx]
            train_gt, val_gt = ground_truths[train_idx], ground_truths[val_idx]
            
            model.fit(train_pred, train_gt)
            pred = model.predict(val_pred)
            scores.append(mean_absolute_error(pred, val_gt))
        
        return np.mean(scores)
    
    def fit(self, predictions, ground_truths):
        """Fit using Bayesian optimization"""
        # Simple grid search as approximation to Bayesian optimization
        alphas = np.logspace(-4, 2, 20)
        best_score = float('inf')
        best_alpha = 1.0
        
        for alpha in alphas:
            score = self._objective_function([alpha], predictions, ground_truths)
            if score < best_score:
                best_score = score
                best_alpha = alpha
        
        self.best_params = {'alpha': best_alpha}
        self.best_model = LinearCorrectionEnhancer('ridge', alpha=best_alpha)
        self.best_model.fit(predictions, ground_truths)
        self.is_fitted = True
        
        print(f"Best parameters: {self.best_params}")
    
    def predict(self, predictions):
        """Apply optimized model"""
        if not self.is_fitted:
            return predictions
        return self.best_model.predict(predictions)

class AdversarialTrainingEnhancer(nn.Module):
    """Adversarial training for robust enhancement"""
    
    def __init__(self, action_dim=7, hidden_dim=128, noise_level=0.01):
        super().__init__()
        self.action_dim = action_dim
        self.noise_level = noise_level
        
        # Generator network
        self.generator = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        residual = self.generator(x)
        return x + self.residual_scale * residual
    
    def fit(self, predictions, ground_truths, epochs=500):
        """Adversarial training"""
        optimizer_g = optim.Adam(self.generator.parameters(), lr=1e-3)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        criterion_adv = nn.BCELoss()
        
        pred_tensor = torch.FloatTensor(predictions)
        gt_tensor = torch.FloatTensor(ground_truths)
        
        for epoch in range(epochs):
            # Train discriminator
            optimizer_d.zero_grad()
            
            # Real samples
            real_labels = torch.ones(pred_tensor.size(0), 1)
            real_output = self.discriminator(gt_tensor)
            d_loss_real = criterion_adv(real_output, real_labels)
            
            # Fake samples
            fake_samples = self(pred_tensor)
            fake_labels = torch.zeros(pred_tensor.size(0), 1)
            fake_output = self.discriminator(fake_samples.detach())
            d_loss_fake = criterion_adv(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            
            # Train generator
            optimizer_g.zero_grad()
            
            # Adversarial loss
            output = self.discriminator(fake_samples)
            g_loss_adv = criterion_adv(output, real_labels)
            
            # Reconstruction loss
            g_loss_rec = criterion(fake_samples, gt_tensor)
            
            g_loss = g_loss_rec + 0.1 * g_loss_adv
            g_loss.backward()
            optimizer_g.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")

class CurriculumLearningEnhancer:
    """Curriculum learning for progressive enhancement"""
    
    def __init__(self, action_dim=7, curriculum_stages=3):
        self.action_dim = action_dim
        self.curriculum_stages = curriculum_stages
        self.models = []
        self.is_fitted = False
    
    def fit(self, predictions, ground_truths):
        """Fit with curriculum learning"""
        # Sort samples by difficulty (MSE)
        difficulties = np.mean((predictions - ground_truths) ** 2, axis=1)
        sorted_indices = np.argsort(difficulties)
        
        # Progressive training stages
        for stage in range(self.curriculum_stages):
            # Select subset of data for this stage
            end_idx = int(len(sorted_indices) * (stage + 1) / self.curriculum_stages)
            stage_indices = sorted_indices[:end_idx]
            
            stage_pred = predictions[stage_indices]
            stage_gt = ground_truths[stage_indices]
            
            # Train model for this stage
            model = DeepNeuralEnhancer(action_dim=self.action_dim, hidden_dims=[128, 128])
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
            pred_tensor = torch.FloatTensor(stage_pred)
            gt_tensor = torch.FloatTensor(stage_gt)
            
            model.train()
            for epoch in range(300):
                optimizer.zero_grad()
                enhanced = model(pred_tensor)
                loss = criterion(enhanced, gt_tensor)
                loss.backward()
                optimizer.step()
            
            self.models.append(model)
            print(f"Stage {stage + 1}/{self.curriculum_stages} completed with {len(stage_indices)} samples")
        
        self.is_fitted = True
    
    def predict(self, predictions):
        """Apply curriculum learning model"""
        if not self.is_fitted:
            return predictions
        
        # Use the final (most advanced) model
        final_model = self.models[-1]
        final_model.eval()
        
        with torch.no_grad():
            pred_tensor = torch.FloatTensor(predictions)
            enhanced = final_model(pred_tensor)
        
        return enhanced.numpy()

class KnowledgeDistillationEnhancer:
    """Knowledge distillation for model compression"""
    
    def __init__(self, action_dim=7, teacher_hidden_dims=[256, 256], student_hidden_dims=[64, 64]):
        self.action_dim = action_dim
        self.teacher_model = DeepNeuralEnhancer(action_dim, teacher_hidden_dims)
        self.student_model = DeepNeuralEnhancer(action_dim, student_hidden_dims)
        self.is_fitted = False
    
    def fit(self, predictions, ground_truths, temperature=3.0, alpha=0.7):
        """Knowledge distillation training"""
        # Train teacher model
        print("Training teacher model...")
        optimizer_t = optim.Adam(self.teacher_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        pred_tensor = torch.FloatTensor(predictions)
        gt_tensor = torch.FloatTensor(ground_truths)
        
        self.teacher_model.train()
        for epoch in range(800):
            optimizer_t.zero_grad()
            enhanced = self.teacher_model(pred_tensor)
            loss = criterion(enhanced, gt_tensor)
            loss.backward()
            optimizer_t.step()
        
        # Train student model with distillation
        print("Training student model with distillation...")
        optimizer_s = optim.Adam(self.student_model.parameters(), lr=1e-3)
        
        self.teacher_model.eval()
        self.student_model.train()
        
        for epoch in range(800):
            optimizer_s.zero_grad()
            
            # Student predictions
            student_output = self.student_model(pred_tensor)
            
            # Teacher predictions (soft targets)
            with torch.no_grad():
                teacher_output = self.teacher_model(pred_tensor)
            
            # Distillation loss
            distillation_loss = F.kl_div(
                F.log_softmax(student_output / temperature, dim=1),
                F.softmax(teacher_output / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # Student loss
            student_loss = criterion(student_output, gt_tensor)
            
            # Combined loss
            total_loss = alpha * student_loss + (1 - alpha) * distillation_loss
            total_loss.backward()
            optimizer_s.step()
        
        self.is_fitted = True
        print("Knowledge distillation completed")
    
    def predict(self, predictions):
        """Apply student model"""
        if not self.is_fitted:
            return predictions
        
        self.student_model.eval()
        with torch.no_grad():
            pred_tensor = torch.FloatTensor(predictions)
            enhanced = self.student_model(pred_tensor)
        
        return enhanced.numpy()

# ============================================================================
# DATA AUGMENTATION APPROACHES
# ============================================================================

class AugmentationBasedEnhancer:
    """Enhancement using data augmentation techniques"""
    
    def __init__(self, augment_factor=5, noise_std=0.01, mixup_alpha=0.2):
        self.augment_factor = augment_factor
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
        self.model = None
        self.is_fitted = False
    
    def augment_data(self, predictions, ground_truths):
        """Create augmented training data"""
        augmented_preds = []
        augmented_gts = []
        
        # Add original data
        augmented_preds.extend(predictions)
        augmented_gts.extend(ground_truths)
        
        # Add augmented data
        for _ in range(self.augment_factor):
            # Noise augmentation
            noise = np.random.normal(0, self.noise_std, predictions.shape)
            noisy_preds = predictions + noise
            augmented_preds.extend(noisy_preds)
            augmented_gts.extend(ground_truths)
            
            # Mixup augmentation
            for i in range(len(predictions)):
                if np.random.random() < 0.5:
                    j = np.random.randint(0, len(predictions))
                    alpha = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    
                    mixed_pred = alpha * predictions[i] + (1 - alpha) * predictions[j]
                    mixed_gt = alpha * ground_truths[i] + (1 - alpha) * ground_truths[j]
                    
                    augmented_preds.append(mixed_pred)
                    augmented_gts.append(mixed_gt)
        
        return np.array(augmented_preds), np.array(augmented_gts)
    
    def fit(self, predictions, ground_truths):
        """Fit model with augmented data"""
        # Create augmented data
        aug_preds, aug_gts = self.augment_data(predictions, ground_truths)
        
        # Train neural network on augmented data
        self.model = DeepNeuralEnhancer(action_dim=7, hidden_dims=[128, 256, 128])
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        aug_preds_tensor = torch.FloatTensor(aug_preds)
        aug_gts_tensor = torch.FloatTensor(aug_gts)
        
        # Training loop
        self.model.train()
        for epoch in range(500):  # Shorter training for augmentation
            optimizer.zero_grad()
            enhanced = self.model(aug_preds_tensor)
            loss = criterion(enhanced, aug_gts_tensor)
            loss.backward()
            optimizer.step()
        
        self.is_fitted = True
    
    def predict(self, predictions):
        """Apply augmentation-trained model"""
        if not self.is_fitted:
            return predictions
        
        self.model.eval()
        with torch.no_grad():
            pred_tensor = torch.FloatTensor(predictions)
            enhanced = self.model(pred_tensor)
        
        return enhanced.numpy()

# ============================================================================
# EXPERIMENT FRAMEWORK
# ============================================================================

class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self, baseline_file="baseline_100_samples_results.json"):
        self.baseline_data = BaselineData(baseline_file)
        self.predictions, self.ground_truths = self.baseline_data.get_arrays()
        self.baseline_mae = mean_absolute_error(self.predictions, self.ground_truths)
        print(f"📊 Baseline MAE: {self.baseline_mae:.6f}")
    
    def run_experiment(self, method_name: str, enhancer, k_folds=5) -> ExperimentResult:
        """Run a single experiment with cross-validation"""
        print(f"\n🧪 Running experiment: {method_name}")
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        all_predictions = []
        all_ground_truths = []
        
        training_times = []
        prediction_times = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.predictions)):
            # Split data
            train_pred, val_pred = self.predictions[train_idx], self.predictions[val_idx]
            train_gt, val_gt = self.ground_truths[train_idx], self.ground_truths[val_idx]
            
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
            
            all_predictions.extend(enhanced_pred)
            all_ground_truths.extend(val_gt)
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_ground_truths = np.array(all_ground_truths)
        
        mae = mean_absolute_error(all_predictions, all_ground_truths)
        mse = mean_squared_error(all_predictions, all_ground_truths)
        improvement = (self.baseline_mae - mae) / self.baseline_mae * 100
        
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
                'total_samples': len(all_predictions)
            }
        )
        
        print(f"✅ {method_name}: MAE={mae:.6f}, Improvement={improvement:.2f}%")
        
        return result
    
    def run_all_experiments(self):
        """Run all brainstormed experiments"""
        experiments = []
        
        # 1. Simple Neural Enhancement
        experiments.append(
            self.run_experiment("Simple Neural", SimpleNeuralEnhancer())
        )
        
        # 2. Deep Neural Enhancement
        experiments.append(
            self.run_experiment("Deep Neural", DeepNeuralEnhancer())
        )
        
        # 3. Attention Neural Enhancement
        experiments.append(
            self.run_experiment("Attention Neural", AttentionNeuralEnhancer())
        )
        
        # 4. Linear Ridge Correction
        experiments.append(
            self.run_experiment("Linear Ridge", LinearCorrectionEnhancer('ridge'))
        )
        
        # 5. Ensemble Enhancement
        experiments.append(
            self.run_experiment("Ensemble", EnsembleEnhancer())
        )
        
        # 6. Augmentation-based Enhancement
        experiments.append(
            self.run_experiment("Augmentation", AugmentationBasedEnhancer())
        )
        
        # 7. Deep RL Correction
        experiments.append(
            self.run_experiment("Deep RL", DeepRLCorrector())
        )
        
        # 8. Game Theory Correction
        experiments.append(
            self.run_experiment("Game Theory", GameTheoryCorrector())
        )
        
        # 9. Cooperative Game Correction
        experiments.append(
            self.run_experiment("Cooperative Game", CooperativeGameCorrector())
        )
        
        # 10. Stacking Ensemble
        experiments.append(
            self.run_experiment("Stacking Ensemble", StackingEnsembleEnhancer())
        )
        
        # 11. Bayesian Optimization
        experiments.append(
            self.run_experiment("Bayesian Opt", BayesianOptimizationEnhancer())
        )
        
        # 12. Adversarial Training
        experiments.append(
            self.run_experiment("Adversarial", AdversarialTrainingEnhancer())
        )
        
        # 13. Curriculum Learning
        experiments.append(
            self.run_experiment("Curriculum", CurriculumLearningEnhancer())
        )
        
        # 14. Knowledge Distillation
        experiments.append(
            self.run_experiment("Knowledge Distill", KnowledgeDistillationEnhancer())
        )
        
        return experiments
    
    def save_results(self, experiments, filename="experiment_results.json"):
        """Save experiment results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename_with_time = f"{timestamp}_{filename}"
        
        # Convert results to serializable format
        results_data = {
            'timestamp': timestamp,
            'baseline_mae': self.baseline_mae,
            'experiments': []
        }
        
        for exp in experiments:
            exp_data = {
                'method_name': exp.method_name,
                'mae': exp.mae,
                'mse': exp.mse,
                'improvement_pct': exp.improvement_pct,
                'training_time': exp.training_time,
                'prediction_time': exp.prediction_time,
                'metadata': exp.metadata
            }
            results_data['experiments'].append(exp_data)
        
        with open(filename_with_time, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename_with_time}")
        
        # Print summary
        print(f"\n📈 EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"{'Method':<20} {'MAE':<12} {'Improvement':<12} {'Time (s)':<10}")
        print("-" * 60)
        
        for exp in experiments:
            print(f"{exp.method_name:<20} {exp.mae:<12.6f} {exp.improvement_pct:<12.2f}% {exp.training_time:<10.3f}")
        
        # Find best method
        best_exp = max(experiments, key=lambda x: x.improvement_pct)
        print(f"\n🏆 Best method: {best_exp.method_name}")
        print(f"   Improvement: {best_exp.improvement_pct:.2f}%")
        print(f"   MAE: {best_exp.mae:.6f}")
        
        return filename_with_time

def evaluate_experiment_statistical_significance(baseline_maes, enhanced_maes, method_name, confidence_level=0.95):
    """
    Evaluate statistical significance for experiment methods
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
    
    # Wilcoxon signed-rank test
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
    
    # Effect size (Cohen's d)
    diff = baseline_maes - enhanced_maes
    pooled_std = np.sqrt((np.var(baseline_maes) + np.var(enhanced_maes)) / 2)
    cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0
    
    results['effect_size'] = {
        'cohens_d': cohens_d,
        'interpretation': interpret_effect_size_cohens_d(cohens_d),
        'mean_improvement': np.mean(diff),
        'improvement_std': np.std(diff)
    }
    
    # Confidence interval
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
        'contains_zero': ci_lower <= 0 <= ci_upper
    }
    
    # Descriptive statistics
    results['descriptive_stats'] = {
        'baseline': {
            'mean': np.mean(baseline_maes),
            'std': np.std(baseline_maes),
            'median': np.median(baseline_maes),
            'n': len(baseline_maes)
        },
        'enhanced': {
            'mean': np.mean(enhanced_maes),
            'std': np.std(enhanced_maes),
            'median': np.median(enhanced_maes),
            'n': len(enhanced_maes)
        }
    }
    
    # Overall significance
    p_values = [t_p_value]
    if wilcoxon_p is not None:
        p_values.append(wilcoxon_p)
    
    min_p_value = min(p_values)
    results['overall_significance'] = {
        'min_p_value': min_p_value,
        'is_significant_05': min_p_value < 0.05,
        'is_significant_01': min_p_value < 0.01,
        'evidence_strength': interpret_evidence_strength_experiment(min_p_value)
    }
    
    return results

def interpret_effect_size_cohens_d(d):
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"

def interpret_evidence_strength_experiment(p):
    """Interpret p-value strength"""
    if p < 0.001:
        return "Very strong evidence"
    elif p < 0.01:
        return "Strong evidence"
    elif p < 0.05:
        return "Moderate evidence"
    elif p < 0.1:
        return "Weak evidence"
    else:
        return "No evidence"

def print_experiment_statistical_summary(stat_results, method_name):
    """Print statistical summary for experiments"""
    print(f"\n🧪 {method_name.upper()} STATISTICAL ANALYSIS")
    print("-" * 50)
    
    # Descriptive stats
    desc = stat_results['descriptive_stats']
    print(f"Baseline:  {desc['baseline']['mean']:.6f} ± {desc['baseline']['std']:.6f}")
    print(f"Enhanced: {desc['enhanced']['mean']:.6f} ± {desc['enhanced']['std']:.6f}")
    
    # T-test
    t_test = stat_results['paired_t_test']
    print(f"Paired t-test: t={t_test['statistic']:.4f}, p={t_test['p_value']:.6f}")
    print(f"Significance: {'✅' if t_test['significant_05'] else '❌'} (α=0.05)")
    
    # Effect size
    effect = stat_results['effect_size']
    print(f"Cohen's d: {effect['cohens_d']:.4f} ({effect['interpretation']})")
    print(f"Mean improvement: {effect['mean_improvement']:.6f}")
    
    # Overall
    overall = stat_results['overall_significance']
    print(f"Evidence: {overall['evidence_strength']}")
    
    print("-" * 50)

def perform_multiple_comparison_correction(p_values, method_names, alpha=0.05):
    """
    Perform multiple comparison correction
    """
    p_values = np.array(p_values)
    method_names = np.array(method_names)
    
    # Bonferroni correction
    bonferroni_corrected = p_values * len(p_values)
    bonferroni_significant = bonferroni_corrected < alpha
    
    # Benjamini-Hochberg FDR correction
    bh_indices = np.argsort(p_values)
    bh_p_values = np.empty_like(p_values)
    bh_p_values[bh_indices] = [p * len(p_values) / (i + 1) for i, p in enumerate(p_values[bh_indices])]
    bh_p_values = np.minimum(bh_p_values, 1.0)
    bh_significant = bh_p_values < alpha
    
    results = {
        'original_p_values': dict(zip(method_names, p_values)),
        'bonferroni': {
            'corrected_p_values': dict(zip(method_names, bonferroni_corrected)),
            'significant': dict(zip(method_names, bonferroni_significant))
        },
        'benjamini_hochberg': {
            'corrected_p_values': dict(zip(method_names, bh_p_values)),
            'significant': dict(zip(method_names, bh_significant))
        }
    }
    
    return results

def main():
    """Main execution"""
    set_seed(42)
    
    parser = argparse.ArgumentParser(description="VLA Enhancement Experiments")
    parser.add_argument('--baseline_file', type=str, default="baseline_100_samples_results.json",
                        help="Path to baseline predictions file")
    parser.add_argument('--k_folds', type=int, default=5, help="K-fold cross validation")
    
    args = parser.parse_args()
    
    # Initialize experiment runner
    runner = ExperimentRunner(args.baseline_file)
    
    # Run all experiments
    experiments = runner.run_all_experiments()
    
    # Save results
    runner.save_results(experiments)
    

if __name__ == "__main__":
    main()
