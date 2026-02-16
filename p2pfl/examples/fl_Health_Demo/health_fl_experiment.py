#!/usr/bin/env python3
"""
Federated Learning for Health Data - HR Prediction with Adversarial Nodes

This script implements a federated learning experiment where each patient's
data is treated as a separate node. The model predicts heart rate (HR) from
accelerometer data (axis1, axis2, axis3) using a sliding window approach.

Supports adversarial/poisoned nodes for security research.

Based on the p2pfl library structure from the MNIST example.

Usage:
    # Clean experiment
    python health_fl_experiment.py --nodes 5 --rounds 3 --topology full
    
    # With poisoned nodes
    python health_fl_experiment.py --nodes 5 --rounds 3 --poisoned 2 --attack label_noise
    
    # Compare clean vs poisoned
    python health_fl_experiment.py --nodes 5 --rounds 3 --poisoned 2 --compare
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import torch
import pandas as pd

# p2pfl imports
from p2pfl.communication.protocols.memory.memory_communication_protocol import InMemoryCommunicationProtocol
from p2pfl.communication.protocols.grpc.grpc_communication_protocol import GrpcCommunicationProtocol
from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.aggregators.scaffold import Scaffold
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import wait_convergence, wait_to_finish

# Local imports
from health_data_module import (
    load_patient_data,
    create_train_test_split,
    HealthTimeSeriesDataset,
    HRPredictorMLP,
)

# Poisoning imports
from health_poisoning import (
    PoisonConfig,
    PoisonStrategy,
    DataPoisoner,
    get_available_strategies,
    create_poisoner_from_args,
)


# =============================================================================
# SETTINGS
# =============================================================================

def set_standalone_settings(disable_ray: bool = False) -> None:
    """Configure p2pfl settings for standalone execution."""
    Settings.GRPC_TIMEOUT = 0.5
    Settings.HEARTBEAT_PERIOD = 5
    Settings.HEARTBEAT_TIMEOUT = 40
    Settings.GOSSIP_PERIOD = 1
    Settings.TTL = 40
    Settings.GOSSIP_MESSAGES_PER_PERIOD = 9999999999
    Settings.AMOUNT_LAST_MESSAGES_SAVED = 10000
    Settings.GOSSIP_MODELS_PERIOD = 1
    Settings.GOSSIP_MODELS_PER_ROUND = 4
    Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 10
    Settings.TRAIN_SET_SIZE = 4
    Settings.VOTE_TIMEOUT = 60
    Settings.AGGREGATION_TIMEOUT = 60
    Settings.WAIT_HEARTBEATS_CONVERGENCE = 0.2 * Settings.HEARTBEAT_TIMEOUT
    Settings.LOG_LEVEL = "INFO"
    Settings.EXCLUDE_BEAT_LOGS = True
    Settings.DISABLE_RAY = disable_ray
    logger.set_level(Settings.LOG_LEVEL)


# =============================================================================
# DATA PREPARATION
# =============================================================================

@dataclass
class PatientDataset:
    """Container for a patient's train and test datasets."""
    patient_id: int
    p2pfl_dataset: Any  # P2PFLDataset
    n_train_samples: int
    n_test_samples: int
    norm_stats: Dict
    is_poisoned: bool = False
    poison_stats: Optional[Dict] = None


def prepare_patient_datasets(
    data_dir: Path,
    num_patients: int,
    window_size: int = 10,
    train_ratio: float = 0.8,
    reduced: bool = False,
    reduced_fraction: float = 0.1,
    num_poisoned: int = 0,
    poison_config: Optional[PoisonConfig] = None,
) -> List[PatientDataset]:
    """
    Prepare datasets for multiple patients using p2pfl's P2PFLDataset.
    
    Args:
        data_dir: Directory containing cleaned patient CSV files
        num_patients: Number of patients to load (uses patients 1 to num_patients)
        window_size: Sliding window size for time series
        train_ratio: Fraction of data for training
        reduced: Whether to use reduced dataset
        reduced_fraction: Fraction of data to use if reduced
        num_poisoned: Number of nodes to poison (first N nodes will be poisoned)
        poison_config: Configuration for poisoning attacks
        
    Returns:
        List of PatientDataset objects
    """
    from datasets import Dataset, DatasetDict
    from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
    
    datasets = []
    
    print(f"\n📦 Preparing datasets for {num_patients} patients...")
    print(f"   Window size: {window_size}")
    print(f"   Train/Test ratio: {train_ratio}/{1-train_ratio}")
    print(f"   Reduced dataset: {reduced} ({reduced_fraction*100:.0f}% if enabled)")
    
    if num_poisoned > 0 and poison_config:
        print(f"   ☠️  Poisoned nodes: {num_poisoned}/{num_patients}")
        print(f"   ☠️  Attack strategy: {poison_config.strategy.value}")
        print(f"   ☠️  Poison rate: {poison_config.poison_rate*100:.0f}%")
    
    # Create poisoner if needed
    poisoner = DataPoisoner(poison_config) if poison_config and num_poisoned > 0 else None
    
    for patient_id in range(1, num_patients + 1):
        try:
            # Determine if this node should be poisoned
            is_poisoned = patient_id <= num_poisoned and poisoner is not None
            
            # Load patient data
            df = load_patient_data(data_dir, patient_id, reduced, reduced_fraction)
            
            # Apply poisoning BEFORE train/test split (poisons training data)
            poison_stats = None
            if is_poisoned:
                # Store original stats for comparison
                original_hr_mean = df['hr'].mean()
                original_hr_std = df['hr'].std()
                original_hr_min = df['hr'].min()
                original_hr_max = df['hr'].max()
                
                # Apply poisoning
                df, poison_stats = poisoner.poison(df)
                
                # Calculate post-poison stats
                poisoned_hr_mean = df['hr'].mean()
                poisoned_hr_std = df['hr'].std()
                poisoned_hr_min = df['hr'].min()
                poisoned_hr_max = df['hr'].max()
                
                # Add comparison to poison_stats
                poison_stats['original_hr_mean'] = original_hr_mean
                poison_stats['original_hr_std'] = original_hr_std
                poison_stats['poisoned_hr_mean'] = poisoned_hr_mean
                poison_stats['poisoned_hr_std'] = poisoned_hr_std
                poison_stats['hr_mean_change'] = poisoned_hr_mean - original_hr_mean
                poison_stats['hr_std_change'] = poisoned_hr_std - original_hr_std
                
                # Print verification
                print(f"\n   ☠️  POISONING VERIFICATION - Patient {patient_id:02d}:")
                print(f"      Attack: {poison_stats.get('attack', 'unknown')}")
                print(f"      Samples poisoned: {poison_stats.get('samples_poisoned', 0):,}/{poison_stats.get('total_samples', 0):,}")
                print(f"      HR Before: mean={original_hr_mean:.2f}, std={original_hr_std:.2f}, range=[{original_hr_min:.1f}, {original_hr_max:.1f}]")
                print(f"      HR After:  mean={poisoned_hr_mean:.2f}, std={poisoned_hr_std:.2f}, range=[{poisoned_hr_min:.1f}, {poisoned_hr_max:.1f}]")
                print(f"      HR Change: mean={poison_stats['hr_mean_change']:+.2f}, std={poison_stats['hr_std_change']:+.2f}")
            
            # Split into train/test (sequential for time series!)
            train_df, test_df = create_train_test_split(df, train_ratio)
            
            # Create our custom datasets for windowing
            train_torch_dataset = HealthTimeSeriesDataset(
                train_df,
                window_size=window_size,
                normalize=True,
                normalization_stats=None
            )
            
            norm_stats = train_torch_dataset.get_normalization_stats()
            
            test_torch_dataset = HealthTimeSeriesDataset(
                test_df,
                window_size=window_size,
                normalize=True,
                normalization_stats=norm_stats
            )
            
            # Convert to HuggingFace Dataset format
            # We need to materialize the windowed data
            def torch_dataset_to_hf(torch_ds):
                """Convert our torch dataset to HuggingFace format."""
                all_features = []
                all_targets = []
                for i in range(len(torch_ds)):
                    sample = torch_ds[i]
                    all_features.append(sample["features"].numpy().tolist())
                    all_targets.append(float(sample["target"].numpy()))
                return Dataset.from_dict({
                    "features": all_features,
                    "target": all_targets
                })
            
            train_hf = torch_dataset_to_hf(train_torch_dataset)
            test_hf = torch_dataset_to_hf(test_torch_dataset)
            
            # Create DatasetDict and wrap in P2PFLDataset
            dataset_dict = DatasetDict({
                "train": train_hf,
                "test": test_hf
            })
            
            p2pfl_dataset = P2PFLDataset(dataset_dict)
            
            patient_data = PatientDataset(
                patient_id=patient_id,
                p2pfl_dataset=p2pfl_dataset,
                n_train_samples=len(train_torch_dataset),
                n_test_samples=len(test_torch_dataset),
                norm_stats=norm_stats,
                is_poisoned=is_poisoned,
                poison_stats=poison_stats
            )
            
            datasets.append(patient_data)
            
            # Print with poison indicator
            poison_indicator = " ☠️ POISONED" if is_poisoned else ""
            print(f"   ✓ Patient {patient_id:02d}: {len(train_torch_dataset):,} train, {len(test_torch_dataset):,} test samples{poison_indicator}")
            
        except FileNotFoundError as e:
            print(f"   ✗ Patient {patient_id:02d}: File not found - {e}")
        except Exception as e:
            print(f"   ✗ Patient {patient_id:02d}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n   Total: {len(datasets)} patients loaded successfully")
    total_train = sum(d.n_train_samples for d in datasets)
    total_test = sum(d.n_test_samples for d in datasets)
    print(f"   Total samples: {total_train:,} train, {total_test:,} test")
    
    # Print poisoning summary
    poisoned_count = sum(1 for d in datasets if d.is_poisoned)
    if poisoned_count > 0:
        print(f"   ☠️  Poisoned nodes: {poisoned_count}/{len(datasets)}")
    
    return datasets


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_hr_predictor_model(window_size: int = 10) -> LightningModel:
    """
    Create a LightningModel for HR prediction compatible with p2pfl.
    
    Args:
        window_size: Sliding window size (must match dataset)
        
    Returns:
        LightningModel wrapping the HR predictor
    """
    # Input size = window_size * number of features (axis1, axis2, axis3, hr)
    input_size = window_size * 4
    
    # Create Lightning module (follows same pattern as p2pfl's MLP)
    model = HRPredictorMLP(
        input_size=input_size,
        hidden_sizes=[64, 32],
        lr_rate=0.001
    )
    return LightningModel(model)


# =============================================================================
# METRICS COLLECTION
# =============================================================================

def collect_experiment_metrics(num_rounds: int) -> Dict[int, Dict[str, Any]]:
    """
    Collect metrics from the p2pfl logger after experiment completion.
    
    Args:
        num_rounds: Number of rounds in the experiment
        
    Returns:
        Dictionary mapping round_num -> metrics dict
    """
    all_rounds = {}
    
    try:
        global_logs = logger.get_global_logs()
        if not global_logs:
            print("   Warning: No global logs found")
            return all_rounds
        
        exp_logs = global_logs.get('experiment', {})
        if not exp_logs:
            print("   Warning: No experiment logs found")
            return all_rounds
        
        print(f"   Found {len(exp_logs)} nodes in logs")
        
        for round_num in range(num_rounds):
            result = {
                'metrics': {},
                'avg_loss': 0.0,
                'avg_mae': 0.0,
                'min_mae': float('inf'),
                'max_mae': 0.0
            }
            
            total_loss = 0.0
            total_mae = 0.0
            count = 0
            
            for node_name, node_metrics in exp_logs.items():
                test_metrics = node_metrics.get('test_metric', [])
                test_losses = node_metrics.get('test_loss', [])
                
                # Use negative indexing to get the last num_rounds entries
                if len(test_metrics) >= num_rounds and len(test_losses) >= num_rounds:
                    actual_idx = -(num_rounds - round_num)
                    
                    try:
                        _, mae_value = test_metrics[actual_idx]
                        _, loss_value = test_losses[actual_idx]
                        
                        mae_float = float(mae_value)
                        loss_float = float(loss_value)
                        
                        result['metrics'][node_name] = {
                            'test_loss': loss_float,
                            'test_mae': mae_float
                        }
                        
                        total_loss += loss_float
                        total_mae += mae_float
                        count += 1
                        
                        result['min_mae'] = min(result['min_mae'], mae_float)
                        result['max_mae'] = max(result['max_mae'], mae_float)
                        
                    except (IndexError, TypeError):
                        continue
            
            if count > 0:
                result['avg_loss'] = total_loss / count
                result['avg_mae'] = total_mae / count
            if result['min_mae'] == float('inf'):
                result['min_mae'] = 0.0
            
            all_rounds[round_num] = result
    
    except Exception as e:
        print(f"   Error collecting metrics: {e}")
    
    return all_rounds


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def print_results_summary(
    rounds_data: Dict[int, Dict[str, Any]],
    num_nodes: int,
    num_rounds: int,
    topology: str,
    execution_time: float,
    reduced: bool,
    num_poisoned: int = 0,
    poison_config: Optional[PoisonConfig] = None
) -> None:
    """Print a summary of experiment results."""
    
    is_poisoned = num_poisoned > 0 and poison_config is not None
    
    print("\n" + "=" * 70)
    if is_poisoned:
        print("📊 FEDERATED LEARNING RESULTS - HR PREDICTION (POISONED)")
    else:
        print("📊 FEDERATED LEARNING RESULTS - HR PREDICTION (CLEAN)")
    print("=" * 70)
    
    print(f"\n🔧 Configuration:")
    print(f"   Nodes (Patients): {num_nodes}")
    print(f"   Rounds: {num_rounds}")
    print(f"   Topology: {topology}")
    print(f"   Reduced Dataset: {reduced}")
    print(f"   Execution Time: {execution_time:.2f}s")
    
    # Poisoning details
    if is_poisoned:
        print(f"\n☠️  Poisoning Configuration:")
        print(f"   Poisoned Nodes: {num_poisoned}/{num_nodes} ({num_poisoned/num_nodes*100:.1f}%)")
        print(f"   Attack Type: {poison_config.strategy.value}")
        print(f"   Poison Rate: {poison_config.poison_rate*100:.0f}% of samples")
        
        # Show attack-specific parameters
        if poison_config.strategy in [PoisonStrategy.LABEL_NOISE, PoisonStrategy.FEATURE_NOISE, 
                                       PoisonStrategy.COMBINED_SUBTLE, PoisonStrategy.COMBINED_AGGRESSIVE]:
            print(f"   Noise Std: {poison_config.noise_std}")
        if poison_config.strategy == PoisonStrategy.TEMPORAL_SHIFT:
            print(f"   Temporal Shift: {poison_config.temporal_shift_steps} steps")
        if poison_config.strategy == PoisonStrategy.LABEL_CONSTANT:
            print(f"   Constant Value: {poison_config.constant_value}")
    else:
        print(f"\n✅ Clean Experiment (No Poisoning)")
    
    if not rounds_data:
        print("\n⚠️  No results to display")
        return
    
    print(f"\n" + "=" * 70)
    print("📈 ROUND-BY-ROUND PERFORMANCE (MAE = Mean Absolute Error)")
    print("=" * 70)
    
    print(f"\n{'Round':<8} {'Avg Loss':<15} {'Avg MAE':<15} {'Min MAE':<12} {'Max MAE':<12}")
    print("-" * 65)
    
    prev_mae = None
    for round_num in sorted(rounds_data.keys()):
        rd = rounds_data[round_num]
        if rd['metrics']:
            change = ""
            if prev_mae is not None:
                diff = rd['avg_mae'] - prev_mae
                change = f"({diff:+.4f})"
            print(f"{round_num:<8} {rd['avg_loss']:<15.4f} {rd['avg_mae']:<15.4f} {rd['min_mae']:<12.4f} {rd['max_mae']:<12.4f} {change}")
            prev_mae = rd['avg_mae']
    
    # Final summary
    if rounds_data:
        first_round = min(rounds_data.keys())
        last_round = max(rounds_data.keys())
        
        initial_mae = rounds_data[first_round]['avg_mae']
        final_mae = rounds_data[last_round]['avg_mae']
        improvement = initial_mae - final_mae
        
        print(f"\n" + "=" * 70)
        print("🎯 FINAL RESULTS")
        print("=" * 70)
        print(f"\n   Initial MAE: {initial_mae:.4f}")
        print(f"   Final MAE:   {final_mae:.4f}")
        print(f"   Improvement: {improvement:+.4f} ({improvement/initial_mae*100:+.1f}%)" if initial_mae > 0 else "")
        
        # Interpretation
        print(f"\n💡 Interpretation:")
        print(f"   MAE represents the average absolute error in HR prediction")
        print(f"   (in normalized units - lower is better)")
        
        if final_mae < 0.1:
            print(f"   ✅ Excellent prediction accuracy!")
        elif final_mae < 0.3:
            print(f"   🟢 Good prediction accuracy")
        elif final_mae < 0.5:
            print(f"   🟡 Moderate prediction accuracy - consider more rounds")
        else:
            print(f"   🔴 Poor prediction accuracy - may need tuning")
    
    print("\n" + "=" * 70)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_health_fl_experiment(
    data_dir: Path,
    num_nodes: int = 5,
    num_rounds: int = 3,
    epochs_per_round: int = 1,
    window_size: int = 10,
    batch_size: int = 32,
    topology: TopologyType = TopologyType.FULL,
    protocol: str = "memory",
    aggregator: str = "fedavg",
    reduced: bool = False,
    reduced_fraction: float = 0.1,
    num_poisoned: int = 0,
    poison_config: Optional[PoisonConfig] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Run a federated learning experiment for HR prediction.
    
    Args:
        data_dir: Directory containing cleaned patient CSV files
        num_nodes: Number of nodes/patients to use
        num_rounds: Number of federated learning rounds
        epochs_per_round: Training epochs per round
        window_size: Sliding window size for time series
        batch_size: Batch size for training
        topology: Network topology
        protocol: Communication protocol (memory/grpc)
        aggregator: Aggregation strategy (fedavg/scaffold)
        reduced: Use reduced dataset
        reduced_fraction: Fraction of data to use if reduced
        num_poisoned: Number of nodes to poison
        poison_config: Configuration for poisoning attacks
        
    Returns:
        Dictionary with results per round
    """
    # Determine experiment type
    is_poisoned_experiment = num_poisoned > 0 and poison_config is not None
    
    print("\n" + "=" * 70)
    if is_poisoned_experiment:
        print("☠️  FEDERATED LEARNING - HR PREDICTION (POISONED EXPERIMENT)")
    else:
        print("🏥 FEDERATED LEARNING - HEART RATE PREDICTION")
    print("=" * 70)
    print(f"\n   Patients/Nodes: {num_nodes}")
    if is_poisoned_experiment:
        print(f"   ☠️  Poisoned Nodes: {num_poisoned} ({num_poisoned/num_nodes*100:.1f}%)")
        print(f"   ☠️  Attack: {poison_config.strategy.value}")
        print(f"   ☠️  Poison Rate: {poison_config.poison_rate*100:.0f}%")
    print(f"   Rounds: {num_rounds}")
    print(f"   Epochs/Round: {epochs_per_round}")
    print(f"   Topology: {topology.value}")
    print(f"   Reduced Dataset: {reduced}")
    
    start_time = time.time()
    
    # Validate settings
    if num_nodes > Settings.TTL:
        raise ValueError("Number of nodes exceeds TTL setting")
    
    if num_poisoned > num_nodes:
        raise ValueError(f"num_poisoned ({num_poisoned}) cannot exceed num_nodes ({num_nodes})")
    
    # Prepare datasets for all patients
    patient_datasets = prepare_patient_datasets(
        data_dir=data_dir,
        num_patients=num_nodes,
        window_size=window_size,
        train_ratio=0.8,
        reduced=reduced,
        reduced_fraction=reduced_fraction,
        num_poisoned=num_poisoned,
        poison_config=poison_config
    )
    
    if len(patient_datasets) < num_nodes:
        print(f"\n⚠️  Warning: Only {len(patient_datasets)} patients available, requested {num_nodes}")
        num_nodes = len(patient_datasets)
    
    if num_nodes == 0:
        raise ValueError("No patient data available!")
    
    # Create nodes
    print(f"\n🖥️  Creating {num_nodes} nodes...")
    nodes = []
    
    for i, patient_data in enumerate(patient_datasets):
        # Create model
        model = create_hr_predictor_model(window_size=window_size)
        
        # Node address (add poison indicator)
        if protocol == "memory":
            poison_suffix = "-☠️" if patient_data.is_poisoned else ""
            address = f"patient-{patient_data.patient_id:02d}{poison_suffix}"
            protocol_class = InMemoryCommunicationProtocol
        else:
            address = "127.0.0.1"
            protocol_class = GrpcCommunicationProtocol
        
        # Create aggregator
        agg = Scaffold() if aggregator == "scaffold" else None
        
        # Create node with p2pfl_dataset directly
        node = Node(
            model,
            patient_data.p2pfl_dataset,  # Use the P2PFLDataset directly
            learner=LightningLearner,
            protocol=protocol_class,
            address=address,
            simulation=True,
            aggregator=agg,
        )
        node.start()
        nodes.append(node)
        poison_indicator = " ☠️" if patient_data.is_poisoned else ""
        print(f"   ✓ Patient {patient_data.patient_id:02d} node started ({patient_data.n_train_samples:,} samples){poison_indicator}")
        
        # Delay to avoid race conditions in p2pfl threading
        # Increased to 1.0s for stability with more nodes
        time.sleep(1.0)
    
    results = {}
    
    try:
        # Connect nodes
        print(f"\n🔗 Connecting nodes in {topology.value} topology...")
        adjacency_matrix = TopologyFactory.generate_matrix(topology, len(nodes))
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)
        
        # Wait for convergence
        print("⏳ Waiting for network convergence...")
        wait_convergence(nodes, num_nodes - 1, only_direct=False, wait=60)
        
        # Start learning
        print(f"\n🎓 Starting federated learning ({num_rounds} rounds)...")
        nodes[0].set_start_learning(rounds=num_rounds, epochs=epochs_per_round)
        
        # Wait for completion
        wait_to_finish(nodes, timeout=60 * 60)
        
        print("\n✅ Training completed!")
        
        # Collect metrics
        print("\n📊 Collecting metrics...")
        results = collect_experiment_metrics(num_rounds)
        
    finally:
        # Stop nodes
        print("\n🧹 Stopping nodes...")
        for node in nodes:
            node.stop()
    
    execution_time = time.time() - start_time
    
    # Print results
    print_results_summary(
        results,
        num_nodes,
        num_rounds,
        topology.value,
        execution_time,
        reduced,
        num_poisoned=num_poisoned,
        poison_config=poison_config
    )
    
    return results


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated Learning for HR Prediction from Accelerometer Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic run with 5 patients
    python health_fl_experiment.py --nodes 5 --rounds 3
    
    # Quick demo (5%% data - fastest, for presentations)
    python health_fl_experiment.py --nodes 3 --rounds 2 --quick
    
    # Reduced dataset (10%% data - balanced speed/accuracy)
    python health_fl_experiment.py --nodes 3 --rounds 3 --reduced_dataset
    
    # With poisoned nodes (label noise attack)
    python health_fl_experiment.py --nodes 5 --rounds 3 --poisoned 2 --attack label_noise --quick
    
    # Aggressive attack with high poison rate
    python health_fl_experiment.py --nodes 5 --poisoned 2 --attack label_flip --quick
    
    # Different topologies
    python health_fl_experiment.py --nodes 5 --rounds 5 --topology star --reduced_dataset

Dataset size options:
    (none)            - Full dataset (100%%)
    --reduced_dataset - Reduced dataset (10%%, ~6k samples/patient)
    --quick           - Minimal dataset (5%%, ~3k samples/patient) - FASTEST

Available attack strategies:
    label_noise      - Add Gaussian noise to HR targets
    label_flip       - Flip HR values (high becomes low)
    label_constant   - Replace HR with constant value
    feature_noise    - Add noise to accelerometer data
    temporal_shift   - Shift HR labels in time
    combined_subtle  - Subtle combination attack
    combined_aggressive - Aggressive multi-vector attack
        """
    )
    
    # Data settings
    parser.add_argument('--data_dir', type=str, default='./cleaned_data',
                       help='Directory with cleaned patient CSVs')
    parser.add_argument('--reduced_dataset', action='store_true',
                       help='Use reduced dataset (10%%) for faster testing')
    parser.add_argument('--quick', action='store_true',
                       help='Use minimal dataset (5%%) for quick demos/presentations')
    parser.add_argument('--reduced_fraction', type=float, default=0.1,
                       help='Fraction of data to use if reduced (default: 0.1)')
    
    # Experiment settings
    parser.add_argument('--nodes', type=int, default=5,
                       help='Number of nodes/patients (default: 5)')
    parser.add_argument('--rounds', type=int, default=3,
                       help='Number of federated learning rounds (default: 3)')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Epochs per round (default: 1)')
    
    # Model settings
    parser.add_argument('--window_size', type=int, default=10,
                       help='Sliding window size (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    
    # Network settings
    parser.add_argument('--topology', type=str, default='full',
                       choices=[t.value for t in TopologyType],
                       help='Network topology (default: full)')
    parser.add_argument('--protocol', type=str, default='memory',
                       choices=['memory', 'grpc'],
                       help='Communication protocol (default: memory)')
    parser.add_argument('--aggregator', type=str, default='fedavg',
                       choices=['fedavg', 'scaffold'],
                       help='Aggregation strategy (default: fedavg)')
    
    # Poisoning settings
    parser.add_argument('--poisoned', type=int, default=0,
                       help='Number of poisoned nodes (default: 0)')
    parser.add_argument('--attack', type=str, default='label_noise',
                       choices=get_available_strategies(),
                       help='Attack strategy (default: label_noise)')
    parser.add_argument('--poison_rate', type=float, default=1.0,
                       help='Fraction of samples to poison per node (default: 1.0)')
    parser.add_argument('--noise_std', type=float, default=0.5,
                       help='Noise standard deviation for noise attacks (default: 0.5)')
    
    # Other
    parser.add_argument('--disable_ray', action='store_true',
                       help='Disable Ray for parallel processing')
    
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize settings
    set_standalone_settings(disable_ray=args.disable_ray)
    
    # Parse topology
    topology = TopologyType(args.topology)
    
    # Handle dataset size options
    # --quick overrides --reduced_dataset with 5% (1/20th)
    # --reduced_dataset uses 10% (1/10th)
    if args.quick:
        use_reduced = True
        reduced_fraction = 0.05  # 5% = 1/20th of data
    elif args.reduced_dataset:
        use_reduced = True
        reduced_fraction = args.reduced_fraction  # Default 10%
    else:
        use_reduced = False
        reduced_fraction = 1.0
    
    # Create poison config if poisoning is enabled
    poison_config = None
    if args.poisoned > 0:
        poison_config = PoisonConfig(
            strategy=PoisonStrategy(args.attack),
            poison_rate=args.poison_rate,
            noise_std=args.noise_std,
            seed=42  # For reproducibility
        )
    
    # Run experiment
    run_health_fl_experiment(
        data_dir=Path(args.data_dir),
        num_nodes=args.nodes,
        num_rounds=args.rounds,
        epochs_per_round=args.epochs,
        window_size=args.window_size,
        batch_size=args.batch_size,
        topology=topology,
        protocol=args.protocol,
        aggregator=args.aggregator,
        reduced=use_reduced,
        reduced_fraction=reduced_fraction,
        num_poisoned=args.poisoned,
        poison_config=poison_config,
    )