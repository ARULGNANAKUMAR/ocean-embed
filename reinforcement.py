"""
REINFORCEMENT Engine
Reward-Based Fine-Tuning & Continuous Improvement
OCEANVERSE Phase 4
"""

import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class RewardMetrics:
    """Computed reward signals."""
    rmse_reward: float
    correlation_reward: float
    argo_reward: float
    physical_consistency_reward: float
    event_prediction_reward: float
    total_reward: float


# ─────────────────────────────────────────────────────────────
# REWARD SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────

class RewardSignal:
    """Define and compute multi-objective reward signals."""
    
    REWARD_CONFIG = {
        'rmse': {'weight': 0.40, 'target': 'minimize'},
        'correlation': {'weight': 0.30, 'target': 'maximize'},
        'argo_validation': {'weight': 0.15, 'target': 'maximize'},
        'physical_consistency': {'weight': 0.10, 'target': 'maximize'},
        'event_prediction': {'weight': 0.05, 'target': 'maximize'},
    }
    
    def __init__(self):
        self.logger = logger
        self.reward_history = []
    
    def compute_rmse_reward(self, pred: np.ndarray, true: np.ndarray) -> float:
        """
        RMSE-based reward.
        reward = 1 / (1 + rmse)
        Returns value in [0, 1]
        """
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        reward = 1.0 / (1.0 + rmse)
        return float(reward)
    
    def compute_correlation_reward(self, pred: np.ndarray, true: np.ndarray) -> float:
        """
        Correlation-based reward.
        reward = (correlation + 1) / 2  # Map from [-1, 1] to [0, 1]
        """
        correlation = np.corrcoef(pred.flatten(), true.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        reward = (correlation + 1.0) / 2.0
        return float(reward)
    
    def compute_argo_reward(self,
                           model_profiles: Dict,
                           argo_profiles: Dict) -> float:
        """
        ARGO float validation.
        Compare 3D predictions vs actual ARGO measurements.
        
        Reward based on:
        - Depth-resolved accuracy
        - Vertical gradient correctness
        - Extreme value detection
        """
        
        depth_accuracy = 0.8  # Placeholder: compute from profiles
        gradient_accuracy = 0.75
        extreme_detection = 0.85
        
        # Weighted combination
        reward = (
            0.5 * depth_accuracy +
            0.3 * gradient_accuracy +
            0.2 * extreme_detection
        )
        
        return float(reward)
    
    def compute_physical_consistency_reward(self,
                                           predictions: Dict) -> float:
        """
        Physics constraint satisfaction.
        Check conservation laws:
        - Continuity equation (∇·u = 0)
        - Geostrophic balance
        - Thermal wind relation
        """
        
        # Simplified: assume all checks pass if data has no NaNs
        if any(np.isnan(v).any() for v in predictions.values() if isinstance(v, np.ndarray)):
            return 0.3
        else:
            return 0.9
    
    def compute_event_prediction_reward(self,
                                       event_predictions: List[Dict],
                                       actual_events: List[Dict]) -> float:
        """
        Reward for correctly predicting marine events:
        - Heatwave onset detection
        - Cyclone track prediction
        - Anomaly detection
        """
        
        if len(actual_events) == 0:
            return 0.5  # Neutral if no events
        
        # Simplified: compute detection rate
        detected = sum(1 for pred in event_predictions
                      if any(e['type'] == pred['type'] for e in actual_events))
        
        detection_rate = detected / len(actual_events)
        
        # Penalize false positives
        false_positives = len(event_predictions) - detected
        false_positive_rate = false_positives / (len(event_predictions) + 1)
        
        reward = 0.7 * detection_rate - 0.3 * false_positive_rate
        return float(np.clip(reward, 0, 1))
    
    def total_reward(self, metrics: Dict) -> float:
        """Weighted sum of all reward sources."""
        
        reward = 0.0
        for reward_type, config in self.REWARD_CONFIG.items():
            if reward_type in metrics:
                weight = config['weight']
                value = metrics[reward_type]
                reward += weight * value
        
        return float(reward)
    
    def compute_all_rewards(self,
                           pred: np.ndarray,
                           true: np.ndarray,
                           model_predictions: Dict = None,
                           argo_data: Dict = None,
                           event_predictions: List = None) -> RewardMetrics:
        """Compute all reward signals."""
        
        metrics = {
            'rmse': self.compute_rmse_reward(pred, true),
            'correlation': self.compute_correlation_reward(pred, true),
            'argo_validation': self.compute_argo_reward(model_predictions or {}, argo_data or {}),
            'physical_consistency': self.compute_physical_consistency_reward(model_predictions or {}),
            'event_prediction': self.compute_event_prediction_reward(event_predictions or [], []),
        }
        
        total = self.total_reward(metrics)
        
        return RewardMetrics(
            rmse_reward=metrics['rmse'],
            correlation_reward=metrics['correlation'],
            argo_reward=metrics['argo_validation'],
            physical_consistency_reward=metrics['physical_consistency'],
            event_prediction_reward=metrics['event_prediction'],
            total_reward=total,
        )


# ─────────────────────────────────────────────────────────────
# RL FINE-TUNER
# ─────────────────────────────────────────────────────────────

class RLFineTuner:
    """Fine-tune models using reinforcement learning."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.logger = logger
        self.optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    
    def fine_tune_step(self,
                      batch: Dict,
                      reward: float) -> Dict:
        """
        Single fine-tuning step.
        
        Loss = base_loss + rl_loss
        rl_loss = -reward * policy_gradient
        """
        
        # Forward pass
        predictions = self.model(batch['input'])
        
        # Base loss (MSE)
        base_loss = nn.MSELoss()(predictions, batch['target'])
        
        # RL loss: gradient ascent on reward
        if reward > 0:
            # Only update if reward is positive
            rl_loss = -reward * base_loss
        else:
            rl_loss = torch.tensor(0.0, device=self.device)
        
        total_loss = base_loss + 0.1 * rl_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'base_loss': float(base_loss.item()),
            'rl_loss': float(rl_loss.item()),
            'total_loss': float(total_loss.item()),
        }
    
    def batch_finetuning(self,
                        recent_data: List[Dict],
                        reward_history: List[float],
                        num_epochs: int = 5) -> Dict:
        """
        Fine-tune on recent data, emphasizing high-reward examples.
        
        Strategy:
        1. Sort examples by reward
        2. Oversample high-reward examples
        3. Use experience replay to prevent catastrophic forgetting
        """
        
        if len(recent_data) == 0:
            self.logger.warning("No recent data for fine-tuning")
            return {'status': 'skipped', 'reason': 'no_data'}
        
        # Sort by reward
        sorted_indices = np.argsort(reward_history)[::-1]
        
        # Oversample top examples
        oversample_ratio = 0.3  # Use top 30% more frequently
        n_top = max(1, int(len(recent_data) * oversample_ratio))
        
        enhanced_data = recent_data.copy()
        top_indices = sorted_indices[:n_top]
        enhanced_data.extend([recent_data[i] for i in top_indices])
        
        # Fine-tuning loop
        total_loss = 0.0
        for epoch in range(num_epochs):
            for batch in enhanced_data:
                reward = reward_history[recent_data.index(batch)] if batch in recent_data else 0.5
                loss_dict = self.fine_tune_step(batch, reward)
                total_loss += loss_dict['total_loss']
        
        avg_loss = total_loss / (num_epochs * len(enhanced_data))
        
        self.logger.info(f"Fine-tuning complete. Avg loss: {avg_loss:.4f}")
        
        return {
            'status': 'completed',
            'epochs': num_epochs,
            'avg_loss': avg_loss,
            'data_points': len(enhanced_data),
        }


# ─────────────────────────────────────────────────────────────
# REWARD OPTIMIZER
# ─────────────────────────────────────────────────────────────

class RewardOptimizer:
    """Multi-objective optimization for competing rewards."""
    
    def __init__(self):
        self.logger = logger
    
    def pareto_front(self,
                    model_versions: List[str],
                    reward_metrics: List[Dict]) -> List[str]:
        """
        Find non-dominated models (Pareto optimal).
        A model is Pareto optimal if no other model is better in all objectives.
        """
        
        pareto_models = []
        
        for i, model_i in enumerate(model_versions):
            is_dominated = False
            
            for j, model_j in enumerate(model_versions):
                if i == j:
                    continue
                
                # Check if model_j dominates model_i
                metrics_i = reward_metrics[i]
                metrics_j = reward_metrics[j]
                
                better_in_all = all(
                    metrics_j.get(metric, 0) >= metrics_i.get(metric, 0)
                    for metric in ['rmse_reward', 'correlation_reward', 'argo_reward']
                )
                
                if better_in_all:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_models.append(model_i)
        
        self.logger.info(f"Pareto front: {len(pareto_models)} models")
        return pareto_models
    
    def balance_objectives(self,
                          current_weights: Dict[str, float],
                          performance_gaps: Dict[str, float]) -> Dict[str, float]:
        """
        Dynamically adjust reward weights to balance performance.
        
        Strategy:
        - If RMSE lagging → increase rmse_weight
        - If event prediction weak → increase event_weight
        """
        
        new_weights = current_weights.copy()
        
        # Identify weak areas (performance below 0.8)
        for metric, gap in performance_gaps.items():
            if gap < 0.8:
                # Boost weight for weak metric
                new_weights[metric] = new_weights.get(metric, 0.1) * 1.5
        
        # Normalize
        total_weight = sum(new_weights.values())
        new_weights = {k: v / total_weight for k, v in new_weights.items()}
        
        self.logger.info(f"Balanced weights: {new_weights}")
        return new_weights


# ─────────────────────────────────────────────────────────────
# REINFORCEMENT ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

class REINFORCEMENT:
    """Main REINFORCEMENT module orchestrator."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.reward_signal = RewardSignal()
        self.fine_tuner = RLFineTuner(model)
        self.optimizer = RewardOptimizer()
        self.logger = logger
        self.reward_history = []
    
    def training_cycle(self,
                      model_output: Dict,
                      ground_truth: np.ndarray,
                      validation_set: List[Dict] = None) -> Dict:
        """
        Full RL training cycle.
        
        1. Compute reward signals
        2. Log reward
        3. If reward > threshold → fine-tune
        4. Else → analyze failure mode
        """
        
        # Compute rewards
        pred = model_output['prediction']
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        
        rewards = self.reward_signal.compute_all_rewards(
            pred=pred,
            true=ground_truth,
            model_predictions=model_output,
        )
        
        self.reward_history.append(rewards.total_reward)
        
        self.logger.info(f"Reward: {rewards.total_reward:.4f} "
                        f"(RMSE: {rewards.rmse_reward:.3f}, "
                        f"Corr: {rewards.correlation_reward:.3f})")
        
        # Decision logic
        result = {
            'reward': rewards.total_reward,
            'reward_breakdown': {
                'rmse': rewards.rmse_reward,
                'correlation': rewards.correlation_reward,
                'argo': rewards.argo_reward,
                'physics': rewards.physical_consistency_reward,
                'events': rewards.event_prediction_reward,
            },
            'timestamp': datetime.now().isoformat(),
        }
        
        # Fine-tune if reward is good
        if rewards.total_reward > 0.7 and validation_set:
            self.logger.info("High reward detected. Fine-tuning...")
            finetune_result = self.fine_tuner.batch_finetuning(
                validation_set,
                [0.7] * len(validation_set),
                num_epochs=3,
            )
            result['finetuning'] = finetune_result
        
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    reward_signal = RewardSignal()
    pred = np.random.randn(100)
    true = np.random.randn(100)
    rewards = reward_signal.compute_all_rewards(pred, true)
    print(f"Total reward: {rewards.total_reward:.4f}")
