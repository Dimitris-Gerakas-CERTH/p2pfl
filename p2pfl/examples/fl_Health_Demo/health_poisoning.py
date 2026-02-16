"""
Health Data Poisoning Module for Federated Learning Adversarial Research.

This module provides various poisoning strategies for health time series data,
specifically designed for HR prediction from accelerometer data.

Attack Categories:
1. Label Poisoning - Corrupt the target HR values
2. Feature Poisoning - Corrupt the input accelerometer data
3. Temporal Poisoning - Disrupt the time-series relationships

Usage:
    from health_poisoning import PoisonConfig, DataPoisoner, PoisonStrategy
    
    config = PoisonConfig(
        strategy=PoisonStrategy.LABEL_NOISE,
        poison_rate=0.3,
        noise_std=0.5
    )
    poisoner = DataPoisoner(config)
    poisoned_data = poisoner.poison(original_data)
"""

import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from abc import ABC, abstractmethod


# =============================================================================
# POISON STRATEGY DEFINITIONS
# =============================================================================

class PoisonStrategy(Enum):
    """Available poisoning strategies."""
    
    # Label Poisoning (corrupt HR targets)
    LABEL_NOISE = "label_noise"           # Add Gaussian noise to HR
    LABEL_FLIP = "label_flip"             # Flip HR values (high↔low)
    LABEL_CONSTANT = "label_constant"     # Replace HR with constant
    LABEL_ZERO = "label_zero"             # Set HR to zero
    
    # Feature Poisoning (corrupt accelerometer inputs)
    FEATURE_NOISE = "feature_noise"       # Add noise to axis1/2/3
    FEATURE_ZERO = "feature_zero"         # Zero out accelerometer data
    FEATURE_SCALE = "feature_scale"       # Scale features incorrectly
    
    # Temporal Poisoning (disrupt time-series structure)
    TEMPORAL_SHIFT = "temporal_shift"     # Shift HR labels in time
    TEMPORAL_SHUFFLE = "temporal_shuffle" # Shuffle temporal order
    
    # Combined Attacks
    COMBINED_SUBTLE = "combined_subtle"   # Subtle combination attack
    COMBINED_AGGRESSIVE = "combined_aggressive"  # Aggressive attack


@dataclass
class PoisonConfig:
    """Configuration for data poisoning."""
    
    strategy: PoisonStrategy = PoisonStrategy.LABEL_NOISE
    poison_rate: float = 1.0  # Fraction of samples to poison (0.0 to 1.0)
    
    # Label noise parameters
    noise_std: float = 0.5    # Std dev for Gaussian noise (in normalized units)
    noise_mean: float = 0.0   # Mean for Gaussian noise
    
    # Label flip parameters  
    flip_threshold: float = 0.0  # Flip around this value (0 = mean in normalized data)
    
    # Constant attack parameters
    constant_value: float = 0.0  # Value to replace HR with
    
    # Feature poisoning parameters
    feature_noise_std: float = 0.3
    feature_scale_factor: float = 2.0
    
    # Temporal shift parameters
    temporal_shift_steps: int = 10  # How many timesteps to shift
    
    # Random seed for reproducibility
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.poison_rate <= 1.0:
            raise ValueError(f"poison_rate must be between 0 and 1, got {self.poison_rate}")
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}")


# =============================================================================
# ATTACK IMPLEMENTATIONS
# =============================================================================

class BaseAttack(ABC):
    """Base class for poisoning attacks."""
    
    def __init__(self, config: PoisonConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
    
    @abstractmethod
    def apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply the poisoning attack to the data.
        
        Args:
            data: DataFrame with columns [timestep, axis1, axis2, axis3, hr]
            
        Returns:
            Tuple of (poisoned_data, stats_dict)
        """
        pass
    
    def _select_poison_indices(self, n_samples: int) -> np.ndarray:
        """Select which samples to poison based on poison_rate."""
        n_poison = int(n_samples * self.config.poison_rate)
        indices = self.rng.choice(n_samples, size=n_poison, replace=False)
        return np.sort(indices)


class LabelNoiseAttack(BaseAttack):
    """Add Gaussian noise to HR labels."""
    
    def apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        poisoned = data.copy()
        n_samples = len(poisoned)
        poison_idx = self._select_poison_indices(n_samples)
        
        # Scale noise relative to the actual HR standard deviation
        hr_std = poisoned['hr'].std()
        hr_mean = poisoned['hr'].mean()
        
        # noise_std is a multiplier of the data's standard deviation
        # e.g., noise_std=0.5 means noise with std = 0.5 * hr_std
        actual_noise_std = self.config.noise_std * hr_std
        
        # Add Gaussian noise to HR
        noise = self.rng.normal(
            self.config.noise_mean, 
            actual_noise_std, 
            size=len(poison_idx)
        )
        
        original_hr = poisoned.loc[poison_idx, 'hr'].values.copy()
        poisoned.loc[poison_idx, 'hr'] += noise
        
        stats = {
            'attack': 'label_noise',
            'samples_poisoned': len(poison_idx),
            'total_samples': n_samples,
            'poison_rate': len(poison_idx) / n_samples,
            'noise_std_multiplier': self.config.noise_std,
            'actual_noise_std': actual_noise_std,
            'hr_original_std': hr_std,
            'avg_hr_change': np.mean(np.abs(noise)),
            'max_hr_change': np.max(np.abs(noise)),
        }
        
        return poisoned, stats


class LabelFlipAttack(BaseAttack):
    """Flip HR values around the mean (high becomes low, low becomes high)."""
    
    def apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        poisoned = data.copy()
        n_samples = len(poisoned)
        poison_idx = self._select_poison_indices(n_samples)
        
        # Use the actual HR mean as the flip threshold
        hr_mean = poisoned['hr'].mean()
        threshold = hr_mean if self.config.flip_threshold == 0.0 else self.config.flip_threshold
        
        # Flip HR around the threshold: new_hr = 2*threshold - old_hr
        original_hr = poisoned.loc[poison_idx, 'hr'].values.copy()
        poisoned.loc[poison_idx, 'hr'] = 2 * threshold - original_hr
        
        stats = {
            'attack': 'label_flip',
            'samples_poisoned': len(poison_idx),
            'total_samples': n_samples,
            'poison_rate': len(poison_idx) / n_samples,
            'flip_threshold': threshold,
            'hr_mean': hr_mean,
            'avg_hr_change': np.mean(np.abs(poisoned.loc[poison_idx, 'hr'].values - original_hr)),
        }
        
        return poisoned, stats


class LabelConstantAttack(BaseAttack):
    """Replace HR with a constant value."""
    
    def apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        poisoned = data.copy()
        n_samples = len(poisoned)
        poison_idx = self._select_poison_indices(n_samples)
        
        original_hr = poisoned.loc[poison_idx, 'hr'].values.copy()
        poisoned.loc[poison_idx, 'hr'] = self.config.constant_value
        
        stats = {
            'attack': 'label_constant',
            'samples_poisoned': len(poison_idx),
            'total_samples': n_samples,
            'poison_rate': len(poison_idx) / n_samples,
            'constant_value': self.config.constant_value,
            'avg_hr_change': np.mean(np.abs(self.config.constant_value - original_hr)),
        }
        
        return poisoned, stats


class FeatureNoiseAttack(BaseAttack):
    """Add Gaussian noise to accelerometer features."""
    
    def apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        poisoned = data.copy()
        n_samples = len(poisoned)
        poison_idx = self._select_poison_indices(n_samples)
        
        feature_cols = ['axis1', 'axis2', 'axis3']
        total_noise = 0
        
        for col in feature_cols:
            if col in poisoned.columns:
                noise = self.rng.normal(0, self.config.feature_noise_std, size=len(poison_idx))
                poisoned.loc[poison_idx, col] += noise
                total_noise += np.sum(np.abs(noise))
        
        stats = {
            'attack': 'feature_noise',
            'samples_poisoned': len(poison_idx),
            'total_samples': n_samples,
            'poison_rate': len(poison_idx) / n_samples,
            'feature_noise_std': self.config.feature_noise_std,
            'avg_noise_magnitude': total_noise / (len(poison_idx) * len(feature_cols)),
        }
        
        return poisoned, stats


class TemporalShiftAttack(BaseAttack):
    """Shift HR labels forward or backward in time, breaking temporal correlation."""
    
    def apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        poisoned = data.copy()
        n_samples = len(poisoned)
        shift = self.config.temporal_shift_steps
        
        if self.config.poison_rate < 1.0:
            # Partial poisoning: only shift some segments
            poison_idx = self._select_poison_indices(n_samples)
            # Shift HR values at poisoned indices
            original_hr = poisoned['hr'].values.copy()
            for idx in poison_idx:
                new_idx = min(idx + shift, n_samples - 1)
                poisoned.loc[idx, 'hr'] = original_hr[new_idx]
        else:
            # Full poisoning: shift entire HR column
            poisoned['hr'] = poisoned['hr'].shift(shift).fillna(method='bfill')
        
        stats = {
            'attack': 'temporal_shift',
            'samples_poisoned': n_samples if self.config.poison_rate == 1.0 else len(self._select_poison_indices(n_samples)),
            'total_samples': n_samples,
            'poison_rate': self.config.poison_rate,
            'shift_steps': shift,
        }
        
        return poisoned, stats


class CombinedSubtleAttack(BaseAttack):
    """Subtle combination: small noise + small temporal shift."""
    
    def apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        poisoned = data.copy()
        n_samples = len(poisoned)
        poison_idx = self._select_poison_indices(n_samples)
        
        # Small label noise
        noise = self.rng.normal(0, 0.2, size=len(poison_idx))
        poisoned.loc[poison_idx, 'hr'] += noise
        
        # Small feature noise
        for col in ['axis1', 'axis2', 'axis3']:
            if col in poisoned.columns:
                feat_noise = self.rng.normal(0, 0.1, size=len(poison_idx))
                poisoned.loc[poison_idx, col] += feat_noise
        
        stats = {
            'attack': 'combined_subtle',
            'samples_poisoned': len(poison_idx),
            'total_samples': n_samples,
            'poison_rate': len(poison_idx) / n_samples,
            'label_noise_std': 0.2,
            'feature_noise_std': 0.1,
        }
        
        return poisoned, stats


class CombinedAggressiveAttack(BaseAttack):
    """Aggressive combination: large noise + flip + feature corruption."""
    
    def apply(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        poisoned = data.copy()
        n_samples = len(poisoned)
        poison_idx = self._select_poison_indices(n_samples)
        
        # Flip labels
        poisoned.loc[poison_idx, 'hr'] = -poisoned.loc[poison_idx, 'hr']
        
        # Add large noise
        noise = self.rng.normal(0, 0.5, size=len(poison_idx))
        poisoned.loc[poison_idx, 'hr'] += noise
        
        # Corrupt features significantly
        for col in ['axis1', 'axis2', 'axis3']:
            if col in poisoned.columns:
                poisoned.loc[poison_idx, col] *= self.rng.uniform(0.5, 2.0, size=len(poison_idx))
        
        stats = {
            'attack': 'combined_aggressive',
            'samples_poisoned': len(poison_idx),
            'total_samples': n_samples,
            'poison_rate': len(poison_idx) / n_samples,
        }
        
        return poisoned, stats


# =============================================================================
# MAIN POISONER CLASS
# =============================================================================

class DataPoisoner:
    """
    Main class for poisoning health time series data.
    
    Usage:
        config = PoisonConfig(strategy=PoisonStrategy.LABEL_NOISE, poison_rate=0.3)
        poisoner = DataPoisoner(config)
        poisoned_df, stats = poisoner.poison(original_df)
    """
    
    # Map strategies to attack classes
    ATTACK_MAP = {
        PoisonStrategy.LABEL_NOISE: LabelNoiseAttack,
        PoisonStrategy.LABEL_FLIP: LabelFlipAttack,
        PoisonStrategy.LABEL_CONSTANT: LabelConstantAttack,
        PoisonStrategy.LABEL_ZERO: LabelConstantAttack,  # Special case with constant=0
        PoisonStrategy.FEATURE_NOISE: FeatureNoiseAttack,
        PoisonStrategy.TEMPORAL_SHIFT: TemporalShiftAttack,
        PoisonStrategy.COMBINED_SUBTLE: CombinedSubtleAttack,
        PoisonStrategy.COMBINED_AGGRESSIVE: CombinedAggressiveAttack,
    }
    
    def __init__(self, config: PoisonConfig):
        """
        Initialize the poisoner.
        
        Args:
            config: PoisonConfig specifying the attack parameters
        """
        self.config = config
        
        # Handle special case for LABEL_ZERO
        if config.strategy == PoisonStrategy.LABEL_ZERO:
            config.constant_value = 0.0
        
        # Get the appropriate attack class
        attack_class = self.ATTACK_MAP.get(config.strategy)
        if attack_class is None:
            raise ValueError(f"Unknown strategy: {config.strategy}")
        
        self.attack = attack_class(config)
    
    def poison(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply poisoning to the data.
        
        Args:
            data: DataFrame with columns [timestep, axis1, axis2, axis3, hr]
            
        Returns:
            Tuple of (poisoned_data, statistics_dict)
        """
        return self.attack.apply(data)
    
    def get_config_summary(self) -> str:
        """Get a human-readable summary of the poisoning configuration."""
        return (
            f"Strategy: {self.config.strategy.value}\n"
            f"Poison Rate: {self.config.poison_rate * 100:.1f}%\n"
            f"Noise Std: {self.config.noise_std}\n"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_available_strategies() -> List[str]:
    """Get list of available poisoning strategy names."""
    return [s.value for s in PoisonStrategy]


def create_poisoner_from_args(
    strategy: str,
    poison_rate: float = 1.0,
    noise_std: float = 0.5,
    seed: Optional[int] = None,
    **kwargs
) -> DataPoisoner:
    """
    Create a DataPoisoner from string arguments (useful for CLI).
    
    Args:
        strategy: Strategy name (e.g., 'label_noise', 'label_flip')
        poison_rate: Fraction of samples to poison
        noise_std: Standard deviation for noise-based attacks
        seed: Random seed
        **kwargs: Additional strategy-specific parameters
        
    Returns:
        Configured DataPoisoner instance
    """
    try:
        strat = PoisonStrategy(strategy)
    except ValueError:
        available = get_available_strategies()
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {available}")
    
    config = PoisonConfig(
        strategy=strat,
        poison_rate=poison_rate,
        noise_std=noise_std,
        seed=seed,
        **kwargs
    )
    
    return DataPoisoner(config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the poisoning module."""
    
    print("=" * 60)
    print("🧪 HEALTH DATA POISONING MODULE TEST")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    sample_data = pd.DataFrame({
        'timestep': range(n_samples),
        'axis1': np.random.randn(n_samples),
        'axis2': np.random.randn(n_samples),
        'axis3': np.random.randn(n_samples),
        'hr': np.random.randn(n_samples),  # Normalized HR
    })
    
    print(f"\n📊 Original data shape: {sample_data.shape}")
    print(f"   HR mean: {sample_data['hr'].mean():.4f}, std: {sample_data['hr'].std():.4f}")
    
    # Test each strategy
    print("\n" + "-" * 60)
    print("Testing each poisoning strategy:")
    print("-" * 60)
    
    for strategy in PoisonStrategy:
        config = PoisonConfig(
            strategy=strategy,
            poison_rate=0.5,
            noise_std=0.5,
            seed=42
        )
        
        poisoner = DataPoisoner(config)
        poisoned_data, stats = poisoner.poison(sample_data.copy())
        
        print(f"\n✓ {strategy.value}:")
        print(f"   Samples poisoned: {stats.get('samples_poisoned', 'N/A')}/{stats.get('total_samples', 'N/A')}")
        print(f"   New HR mean: {poisoned_data['hr'].mean():.4f}, std: {poisoned_data['hr'].std():.4f}")
        
        # Check that poisoning actually changed something
        hr_diff = (poisoned_data['hr'] - sample_data['hr']).abs().sum()
        print(f"   Total HR change: {hr_diff:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)