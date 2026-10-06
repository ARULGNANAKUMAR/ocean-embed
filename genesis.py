"""
GENESIS Module
Automated Dataset Analysis & Architecture Selection
OCEANVERSE Phase 4
"""

import json
import logging
import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class DatasetProfile:
    """Statistical profile of a dataset."""
    shape: Tuple[int, ...]
    dtypes: Dict[str, str]
    missing_pct: Dict[str, float]
    variable_ranges: Dict[str, Tuple[float, float]]
    temporal_freq: str  # 'daily', 'weekly', 'monthly', 'hourly', 'irregular'
    spatial_extent: Dict[str, Tuple[float, float]]  # {lat, lon, depth}
    temporal_extent: Tuple[datetime, datetime]
    num_missing_patterns: int
    skewness: Dict[str, float]
    kurtosis: Dict[str, float]
    autocorrelation_lag1: Dict[str, float]
    cross_correlation: Dict[str, float]


@dataclass
class PreprocessConfig:
    """Preprocessing recommendations."""
    normalization: str  # 'minmax', 'zscore', 'robust'
    interpolation_method: str  # 'linear', 'cubic', 'kriging'
    outlier_handling: str  # 'clip', 'remove', 'flag'
    outlier_threshold_std: float
    augmentation_methods: List[str]  # ['rotation', 'noise', 'mixup']
    augmentation_intensity: float
    downsampling_factor: int
    missing_data_strategy: str  # 'fill', 'remove', 'predict'


@dataclass
class ArchitectureConfig:
    """Configuration for model architecture."""
    architecture_type: str  # 'cnn', 'vit', 'attention', 'physics', 'graph'
    input_channels: int
    embed_dim: int
    num_layers: int
    hidden_dim: int
    dropout: float
    attention_heads: int


@dataclass
class HyperparamConfig:
    """Hyperparameter recommendations."""
    learning_rate: float
    batch_size: int
    embedding_dim: int
    dropout: float
    regularization_l2: float
    num_layers: int
    hidden_dim: int
    epochs: int
    patience_early_stopping: int
    optimizer: str  # 'adam', 'sgd', 'adamw'


# ─────────────────────────────────────────────────────────────
# DATASET ANALYZER
# ─────────────────────────────────────────────────────────────

class DatasetAnalyzer:
    """Inspect dataset properties and compute statistics."""
    
    def __init__(self):
        self.profile: Optional[DatasetProfile] = None
        self.logger = logger
    
    def analyze(self, dataset_path: str) -> DatasetProfile:
        """
        Comprehensive dataset analysis.
        Supports: .nc, .h5, .csv, .xr
        """
        self.logger.info(f"Analyzing dataset: {dataset_path}")
        
        try:
            # Load dataset
            if dataset_path.endswith('.nc'):
                ds = xr.open_dataset(dataset_path)
                data_dict = {v: ds[v].values for v in ds.data_vars}
            elif dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
                data_dict = df.to_dict('series')
            else:
                raise ValueError(f"Unsupported format: {dataset_path}")
            
            # Analyze each variable
            profile = self._compute_statistics(data_dict)
            self.profile = profile
            self.logger.info("Dataset analysis complete")
            return profile
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise
    
    def _compute_statistics(self, data_dict: Dict) -> DatasetProfile:
        """Compute comprehensive statistics."""
        
        # Basic statistics
        shape = tuple(v.shape if hasattr(v, 'shape') else len(v) 
                     for v in data_dict.values())
        dtypes = {k: str(v.dtype) if hasattr(v, 'dtype') else type(v).__name__
                 for k, v in data_dict.items()}
        
        # Missing data analysis
        missing_pct = {}
        for k, v in data_dict.items():
            if hasattr(v, 'mask'):
                missing_pct[k] = (v.mask.sum() / v.size * 100) if hasattr(v.mask, 'sum') else 0
            else:
                missing_pct[k] = (pd.isna(v).sum() / len(v) * 100) if hasattr(v, '__len__') else 0
        
        # Value ranges
        variable_ranges = {}
        for k, v in data_dict.items():
            if hasattr(v, 'min') and hasattr(v, 'max'):
                variable_ranges[k] = (float(np.nanmin(v)), float(np.nanmax(v)))
            else:
                variable_ranges[k] = (0.0, 1.0)
        
        # Distribution characteristics
        skewness = {}
        kurtosis = {}
        for k, v in data_dict.items():
            if hasattr(v, 'flatten'):
                flat = v.flatten()
                skewness[k] = float(pd.Series(flat).skew())
                kurtosis[k] = float(pd.Series(flat).kurtosis())
        
        # Temporal autocorrelation
        autocorr = {}
        for k, v in data_dict.items():
            if hasattr(v, 'flatten'):
                flat = v.flatten()
                if len(flat) > 1:
                    autocorr[k] = float(pd.Series(flat).autocorr(lag=1) or 0.0)
        
        profile = DatasetProfile(
            shape=shape,
            dtypes=dtypes,
            missing_pct=missing_pct,
            variable_ranges=variable_ranges,
            temporal_freq=self._infer_temporal_freq(data_dict),
            spatial_extent=self._estimate_spatial_extent(data_dict),
            temporal_extent=self._estimate_temporal_extent(data_dict),
            num_missing_patterns=self._count_missing_patterns(data_dict),
            skewness=skewness,
            kurtosis=kurtosis,
            autocorrelation_lag1=autocorr,
            cross_correlation=self._compute_cross_correlation(data_dict),
        )
        return profile
    
    def _infer_temporal_freq(self, data_dict: Dict) -> str:
        """Infer temporal resolution from time dimension."""
        # Heuristic: check array size and estimate
        total_size = sum(v.size if hasattr(v, 'size') else 1 for v in data_dict.values())
        if total_size > 100000:
            return 'daily'
        elif total_size > 10000:
            return 'weekly'
        else:
            return 'monthly'
    
    def _estimate_spatial_extent(self, data_dict: Dict) -> Dict:
        """Estimate spatial bounds."""
        return {
            'lat': (5.0, 30.0),  # North Indian Ocean default
            'lon': (45.0, 105.0),
            'depth': (0.0, 6000.0),
        }
    
    def _estimate_temporal_extent(self, data_dict: Dict) -> Tuple:
        """Estimate time range."""
        return (datetime(2020, 1, 1), datetime.now())
    
    def _count_missing_patterns(self, data_dict: Dict) -> int:
        """Count systematic missing data patterns."""
        return 0  # Placeholder
    
    def _compute_cross_correlation(self, data_dict: Dict) -> Dict[str, float]:
        """Compute correlations between variables."""
        result = {}
        vars_list = list(data_dict.keys())
        for i, v1 in enumerate(vars_list[:5]):  # Limit to first 5 variables
            for v2 in vars_list[i+1:5]:
                key = f"{v1}_vs_{v2}"
                result[key] = 0.5  # Placeholder
        return result


# ─────────────────────────────────────────────────────────────
# ARCHITECTURE SELECTOR
# ─────────────────────────────────────────────────────────────

class ArchitectureSelector:
    """Recommend model architecture based on data properties."""
    
    def __init__(self):
        self.logger = logger
    
    def score_architecture(self, profile: DatasetProfile) -> Dict[str, float]:
        """Score each architecture (0-100)."""
        
        scores = {
            'cnn': self._cnn_score(profile),
            'vit': self._vit_score(profile),
            'attention': self._attention_score(profile),
            'physics': self._physics_score(profile),
            'graph': self._graph_score(profile),
        }
        return scores
    
    def _cnn_score(self, profile: DatasetProfile) -> float:
        """CNN works well for spatially local patterns."""
        score = 50.0
        # Boost if good spatial resolution
        if 'lat' in profile.spatial_extent and 'lon' in profile.spatial_extent:
            score += 20.0
        # Boost if not too many missing values
        avg_missing = sum(profile.missing_pct.values()) / len(profile.missing_pct)
        if avg_missing < 10:
            score += 15.0
        return min(100.0, score)
    
    def _vit_score(self, profile: DatasetProfile) -> float:
        """Vision Transformer good for global patterns."""
        score = 45.0
        # Boost if large spatial extent
        score += 15.0
        # Boost if lower autocorrelation (more independent patches)
        avg_autocorr = sum(profile.autocorrelation_lag1.values()) / len(profile.autocorrelation_lag1)
        if avg_autocorr < 0.7:
            score += 20.0
        return min(100.0, score)
    
    def _attention_score(self, profile: DatasetProfile) -> float:
        """Attention good for temporal sequences."""
        score = 55.0
        # Boost if high autocorrelation (temporal dependence)
        avg_autocorr = sum(profile.autocorrelation_lag1.values()) / len(profile.autocorrelation_lag1)
        if avg_autocorr > 0.7:
            score += 25.0
        # Boost if daily or finer temporal resolution
        if profile.temporal_freq in ['daily', 'hourly']:
            score += 10.0
        return min(100.0, score)
    
    def _physics_score(self, profile: DatasetProfile) -> float:
        """Physics module good for dynamical systems."""
        score = 50.0
        # Boost if low missing data
        avg_missing = sum(profile.missing_pct.values()) / len(profile.missing_pct)
        if avg_missing < 5:
            score += 25.0
        # Physics benefit for ocean data (always true for OCEANVERSE)
        score += 15.0
        return min(100.0, score)
    
    def _graph_score(self, profile: DatasetProfile) -> float:
        """Graph networks for connectivity patterns."""
        score = 40.0
        # Boost if high cross-correlation between variables
        avg_cross_corr = sum(profile.cross_correlation.values()) / len(profile.cross_correlation)
        if avg_cross_corr > 0.6:
            score += 30.0
        return min(100.0, score)
    
    def recommend(self, profile: DatasetProfile, top_k: int = 3) -> List[Tuple[str, float]]:
        """Recommend top-k architectures."""
        scores = self.score_architecture(profile)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ─────────────────────────────────────────────────────────────
# HYPERPARAMETER OPTIMIZER
# ─────────────────────────────────────────────────────────────

class HyperparameterOptimizer:
    """Auto-tune hyperparameters based on dataset."""
    
    def __init__(self):
        self.logger = logger
    
    def suggest(self,
                dataset_size: int,
                target_var: str,
                architecture: str) -> HyperparamConfig:
        """Suggest optimal hyperparameters."""
        
        # Scale batch size with dataset
        if dataset_size < 1000:
            batch_size = 8
        elif dataset_size < 10000:
            batch_size = 16
        else:
            batch_size = 32
        
        # Architecture-specific settings
        if architecture == 'cnn':
            embed_dim = 64
            num_layers = 4
            hidden_dim = 128
            attention_heads = 4
        elif architecture == 'vit':
            embed_dim = 128
            num_layers = 6
            hidden_dim = 256
            attention_heads = 8
        elif architecture == 'attention':
            embed_dim = 64
            num_layers = 3
            hidden_dim = 128
            attention_heads = 4
        elif architecture == 'physics':
            embed_dim = 32
            num_layers = 2
            hidden_dim = 64
            attention_heads = 2
        else:  # graph
            embed_dim = 64
            num_layers = 3
            hidden_dim = 128
            attention_heads = 4
        
        # Learning rate based on batch size
        learning_rate = 0.001 / (batch_size / 32.0)
        
        config = HyperparamConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            embedding_dim=embed_dim,
            dropout=0.2,
            regularization_l2=0.001,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            epochs=100,
            patience_early_stopping=10,
            optimizer='adamw',
        )
        
        self.logger.info(f"Suggested hyperparams: {config}")
        return config


# ─────────────────────────────────────────────────────────────
# RETRAINING DECIDER
# ─────────────────────────────────────────────────────────────

class RetrainingDecider:
    """Decide when to retrain vs. fine-tune."""
    
    RETRAIN_THRESHOLDS = {
        'data_change_pct': 30.0,      # If data changes >30%, retrain
        'performance_delta': 0.05,    # If performance drops >5%, retrain
        'days_since_retrain': 90,     # If >90 days since last retrain, retrain
    }
    
    def __init__(self):
        self.logger = logger
    
    def should_retrain(self,
                       dataset_change_pct: float,
                       performance_delta: float,
                       last_retrain_days: int) -> bool:
        """Decide retrain vs. fine-tune."""
        
        if dataset_change_pct > self.RETRAIN_THRESHOLDS['data_change_pct']:
            self.logger.info(f"Retrain: data change {dataset_change_pct}%")
            return True
        
        if performance_delta > self.RETRAIN_THRESHOLDS['performance_delta']:
            self.logger.info(f"Retrain: performance drop {performance_delta:.1%}")
            return True
        
        if last_retrain_days > self.RETRAIN_THRESHOLDS['days_since_retrain']:
            self.logger.info(f"Retrain: {last_retrain_days} days since last retrain")
            return True
        
        self.logger.info("Fine-tune: conditions not met for retrain")
        return False


# ─────────────────────────────────────────────────────────────
# GENESIS ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

class GENESIS:
    """Main GENESIS module orchestrator."""
    
    def __init__(self):
        self.analyzer = DatasetAnalyzer()
        self.arch_selector = ArchitectureSelector()
        self.hyperparam_optimizer = HyperparameterOptimizer()
        self.retrain_decider = RetrainingDecider()
        self.logger = logger
    
    def analyze_and_recommend(self, dataset_path: str) -> Dict:
        """Full analysis pipeline."""
        
        self.logger.info(f"GENESIS: Analyzing {dataset_path}")
        
        # Step 1: Analyze dataset
        profile = self.analyzer.analyze(dataset_path)
        
        # Step 2: Recommend architecture
        top_architectures = self.arch_selector.recommend(profile, top_k=3)
        best_arch = top_architectures[0][0]
        
        # Step 3: Suggest hyperparameters
        dataset_size = int(np.prod(profile.shape))
        hyperparams = self.hyperparam_optimizer.suggest(dataset_size, 'temperature', best_arch)
        
        result = {
            'dataset_profile': asdict(profile),
            'recommended_architectures': [
                {'name': arch, 'score': score}
                for arch, score in top_architectures
            ],
            'best_architecture': best_arch,
            'hyperparameters': asdict(hyperparams),
            'timestamp': datetime.now().isoformat(),
        }
        
        self.logger.info(f"GENESIS recommendation: {best_arch}")
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    genesis = GENESIS()
    # result = genesis.analyze_and_recommend("sample_data.nc")
    # print(json.dumps(result, indent=2))
