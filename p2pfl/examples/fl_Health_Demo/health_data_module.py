"""
Health Data Module for Federated Learning HR Prediction.

This module provides:
1. HealthTimeSeriesDataset - PyTorch dataset with sliding window for HR prediction
2. HRPredictorMLP - LightningModule for HR prediction (compatible with p2pfl)
3. Helper functions for loading and partitioning patient data

The task: Given a window of (axis1, axis2, axis3, HR) values, predict the next HR value.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import lightning as L
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanAbsoluteError


# =============================================================================
# DATASET CLASSES
# =============================================================================

class HealthTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for health time series data with sliding window.
    
    Given a sequence of (axis1, axis2, axis3, HR) values, creates windows
    where the input is the last `window_size` timesteps and the target
    is the HR value at the next timestep.
    
    IMPORTANT: Returns dictionaries with "features" and "target" keys
    to match p2pfl's expected batch format.
    
    Input features per timestep: axis1, axis2, axis3, hr (4 features)
    Total input size: window_size * 4
    Output: single HR value (regression)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 10,
        feature_cols: List[str] = ['axis1', 'axis2', 'axis3', 'hr'],
        target_col: str = 'hr',
        normalize: bool = True,
        normalization_stats: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame with columns including feature_cols
            window_size: Number of past timesteps to use as input
            feature_cols: Columns to use as input features
            target_col: Column to predict (must be in feature_cols too)
            normalize: Whether to normalize the data
            normalization_stats: Optional dict of {col: (mean, std)} for normalization
                                If None and normalize=True, computed from data
        """
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.normalize = normalize
        
        # Extract the relevant columns
        self.data = data[feature_cols].values.astype(np.float32)
        
        # Compute or use provided normalization stats
        if normalize:
            if normalization_stats is None:
                # Compute from this data
                self.norm_stats = {
                    col: (data[col].mean(), data[col].std() + 1e-8)
                    for col in feature_cols
                }
            else:
                self.norm_stats = normalization_stats
            
            # Apply normalization
            for i, col in enumerate(feature_cols):
                mean, std = self.norm_stats[col]
                self.data[:, i] = (self.data[:, i] - mean) / std
        else:
            self.norm_stats = None
        
        # Get target column index
        self.target_idx = feature_cols.index(target_col)
        
        # Number of valid windows (need window_size + 1 for input + target)
        self.n_samples = len(self.data) - window_size
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with "features" and "target" keys (p2pfl format)
            - features: shape (window_size * n_features,) flattened
            - target: shape (1,) the next HR value
        """
        # Input: window of features from idx to idx + window_size
        window = self.data[idx:idx + self.window_size]  # (window_size, n_features)
        
        # Flatten to 1D for MLP input
        features = torch.tensor(window.flatten(), dtype=torch.float32)
        
        # Target: HR at the next timestep
        target = torch.tensor(self.data[idx + self.window_size, self.target_idx], dtype=torch.float32)
        
        # Return as dictionary (p2pfl expected format)
        return {
            "features": features,
            "target": target
        }
    
    def get_normalization_stats(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Return normalization statistics for use in other datasets."""
        return self.norm_stats


def load_patient_data(
    data_dir: Path,
    patient_id: int,
    reduced: bool = False,
    reduced_fraction: float = 0.1
) -> pd.DataFrame:
    """
    Load data for a single patient.
    
    Args:
        data_dir: Directory containing cleaned patient CSV files
        patient_id: Patient ID (1-22)
        reduced: Whether to use only a fraction of the data
        reduced_fraction: Fraction of data to use if reduced=True
        
    Returns:
        DataFrame with patient data
    """
    filename = f"patient_{patient_id:02d}_clean.csv"
    filepath = data_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Patient data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    if reduced:
        # Take first X% of the data (maintaining time order)
        n_rows = int(len(df) * reduced_fraction)
        df = df.head(n_rows)
    
    return df


def create_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.
    
    IMPORTANT: For time series, we split sequentially (not randomly)
    to avoid data leakage from future to past.
    
    Args:
        df: DataFrame with time series data
        train_ratio: Fraction of data to use for training
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    return train_df, test_df


# =============================================================================
# MODEL (LightningModule compatible with p2pfl)
# =============================================================================

class HRPredictorMLP(L.LightningModule):
    """
    MLP for HR prediction, compatible with p2pfl's LightningModel.
    
    This follows the same pattern as p2pfl's example MLP class:
    - Inherits from L.LightningModule
    - Implements training_step, test_step, configure_optimizers
    - Expects batch as dictionary with "features" and "target" keys
    
    Architecture:
        Input (window_size * 4 features) 
        → Linear → ReLU
        → Linear → ReLU  
        → Linear → Output (1 HR value)
    """
    
    def __init__(
        self,
        input_size: int = 40,  # window_size * n_features (10 * 4)
        hidden_sizes: Optional[List[int]] = None,
        lr_rate: float = 0.001,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the MLP."""
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        self.lr_rate = lr_rate
        self.metric = MeanAbsoluteError()
        
        # Build layers
        self.layers = torch.nn.ModuleList()
        
        # Input layer
        self.layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(torch.nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(torch.nn.ReLU())
        
        # Output layer (single value for regression)
        self.layers.append(torch.nn.Linear(hidden_sizes[-1], 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch["features"].float()
        y = batch["target"].float()
        
        y_hat = self(x).squeeze(-1)  # Squeeze only the last dimension
        
        # Ensure shapes match
        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
            
        loss = torch.nn.functional.mse_loss(y_hat, y)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        x = batch["features"].float()
        y = batch["target"].float()
        
        y_hat = self(x).squeeze(-1)  # Squeeze only the last dimension
        
        # Ensure shapes match
        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        
        loss = torch.nn.functional.mse_loss(y_hat, y)
        
        # Calculate MAE metric
        metric = self.metric(y_hat, y)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step (same as test)."""
        return self.test_step(batch, batch_idx)


# =============================================================================
# P2PFL INTEGRATION HELPERS
# =============================================================================

def create_p2pfl_dataset_for_patient(
    data_dir: Path,
    patient_id: int,
    window_size: int = 10,
    train_ratio: float = 0.8,
    reduced: bool = False,
    reduced_fraction: float = 0.1
) -> Tuple[HealthTimeSeriesDataset, HealthTimeSeriesDataset, Dict]:
    """
    Create train and test datasets for a single patient.
    
    Args:
        data_dir: Directory containing cleaned patient CSV files
        patient_id: Patient ID (1-22)
        window_size: Sliding window size
        train_ratio: Fraction for training
        reduced: Use reduced dataset
        reduced_fraction: Fraction to use if reduced
        
    Returns:
        Tuple of (train_dataset, test_dataset, norm_stats)
    """
    # Load patient data
    df = load_patient_data(data_dir, patient_id, reduced, reduced_fraction)
    
    # Split into train/test (sequential, not random!)
    train_df, test_df = create_train_test_split(df, train_ratio)
    
    # Create datasets
    train_dataset = HealthTimeSeriesDataset(
        train_df,
        window_size=window_size,
        normalize=True,
        normalization_stats=None
    )
    
    # Use training stats for test normalization
    norm_stats = train_dataset.get_normalization_stats()
    
    test_dataset = HealthTimeSeriesDataset(
        test_df,
        window_size=window_size,
        normalize=True,
        normalization_stats=norm_stats
    )
    
    return train_dataset, test_dataset, norm_stats


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    """Quick test of the data pipeline and model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test health data module")
    parser.add_argument('--data_dir', type=str, default='./cleaned_data',
                       help='Directory with cleaned patient CSVs')
    parser.add_argument('--patient', type=int, default=1,
                       help='Patient ID to test with')
    parser.add_argument('--window_size', type=int, default=10,
                       help='Sliding window size')
    parser.add_argument('--reduced', action='store_true',
                       help='Use reduced dataset')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    print("=" * 60)
    print("🏥 HEALTH DATA MODULE TEST")
    print("=" * 60)
    
    # Load patient data
    print(f"\n📂 Loading patient {args.patient} data...")
    df = load_patient_data(data_dir, args.patient, reduced=args.reduced)
    print(f"   Total samples: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   First 5 rows:\n{df.head()}")
    
    # Create train/test split
    print(f"\n✂️  Creating train/test split (80/20)...")
    train_df, test_df = create_train_test_split(df)
    print(f"   Train samples: {len(train_df):,}")
    print(f"   Test samples: {len(test_df):,}")
    
    # Create datasets
    print(f"\n📊 Creating sliding window datasets (window_size={args.window_size})...")
    train_dataset, test_dataset, norm_stats = create_p2pfl_dataset_for_patient(
        data_dir, args.patient, 
        window_size=args.window_size,
        reduced=args.reduced
    )
    print(f"   Train windows: {len(train_dataset):,}")
    print(f"   Test windows: {len(test_dataset):,}")
    
    # Check a sample (now returns dict)
    print(f"\n🔍 Sample input/output:")
    sample = train_dataset[0]
    print(f"   Sample keys: {list(sample.keys())}")
    print(f"   Features shape: {sample['features'].shape}")
    print(f"   Target shape: {sample['target'].shape}")
    print(f"   Features (first 10 values): {sample['features'][:10].numpy()}")
    print(f"   Target HR: {sample['target'].item():.4f} (normalized)")
    
    # Test model
    print(f"\n🤖 Testing model...")
    input_size = args.window_size * 4  # 4 features per timestep
    model = HRPredictorMLP(input_size=input_size)
    print(f"   Input size: {input_size}")
    print(f"   Model layers: {len(model.layers)}")
    
    # Forward pass test
    batch = {
        "features": sample["features"].unsqueeze(0),  # Add batch dimension
        "target": sample["target"].unsqueeze(0)
    }
    y_pred = model(batch["features"])
    print(f"\n   Forward pass test:")
    print(f"   Input shape: {batch['features'].shape}")
    print(f"   Output shape: {y_pred.shape}")
    print(f"   Predicted HR: {y_pred.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)