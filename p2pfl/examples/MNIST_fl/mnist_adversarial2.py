#
# Enhanced P2PFL MNIST experiment with adversarial/malicious node support
# Based on the original p2pfl MNIST example
#

"""
P2PFL MNIST experiment with configurable malicious nodes for studying
the impact of data poisoning attacks on federated learning.

Attack Types Supported:
- Label flipping (targeted and random)
- Label permutation
- Random label assignment

Usage:
    python mnist_adversarial.py --nodes 5 --malicious 2 --attack random_flip --rounds 5
    python mnist_adversarial.py --nodes 5 --malicious 2 --attack targeted_flip --rounds 5
    python mnist_adversarial.py --compare --nodes 5 --rounds 3  # Run clean vs poisoned comparison
"""

import argparse
import time
import uuid
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum

import matplotlib.pyplot as plt

from p2pfl.communication.protocols.grpc.grpc_communication_protocol import GrpcCommunicationProtocol
from p2pfl.communication.protocols.memory.memory_communication_protocol import InMemoryCommunicationProtocol
from p2pfl.learning.aggregators.scaffold import Scaffold
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import wait_convergence, wait_to_finish


# =============================================================================
# ATTACK STRATEGIES
# =============================================================================

class AttackType(Enum):
    """Enumeration of supported attack types."""
    NONE = "none"
    RANDOM_FLIP = "random_flip"           # Flip labels to random values
    TARGETED_FLIP = "targeted_flip"       # Flip specific digit pairs
    PERMUTATION = "permutation"           # Cyclic permutation of all labels
    PARTIAL_FLIP = "partial_flip"         # Only flip a percentage of labels


@dataclass
class AttackConfig:
    """Configuration for a poisoning attack."""
    attack_type: AttackType = AttackType.NONE
    poison_rate: float = 1.0              # Fraction of samples to poison (0.0-1.0)
    flip_map: Optional[Dict[int, int]] = None  # For targeted attacks
    num_classes: int = 10                 # Number of classes in dataset
    
    def __post_init__(self):
        if self.flip_map is None and self.attack_type == AttackType.TARGETED_FLIP:
            # Default: swap visually similar digits
            self.flip_map = {
                3: 8, 8: 3,  # 3 and 8 look similar
                4: 9, 9: 4,  # 4 and 9 look similar
                5: 6, 6: 5,  # 5 and 6 look similar
                1: 7, 7: 1,  # 1 and 7 look similar
            }


class PoisonStrategy(ABC):
    """Abstract base class for poisoning strategies."""
    
    @abstractmethod
    def poison_label(self, original_label: int, config: AttackConfig) -> int:
        """Transform a single label according to the attack strategy."""
        pass
    
    def should_poison(self, config: AttackConfig) -> bool:
        """Determine if this sample should be poisoned based on poison_rate."""
        return random.random() < config.poison_rate


class RandomFlipStrategy(PoisonStrategy):
    """Flip labels to completely random values."""
    
    def poison_label(self, original_label: int, config: AttackConfig) -> int:
        if self.should_poison(config):
            # Ensure we pick a DIFFERENT label
            new_label = random.randint(0, config.num_classes - 1)
            while new_label == original_label:
                new_label = random.randint(0, config.num_classes - 1)
            return new_label
        return original_label


class TargetedFlipStrategy(PoisonStrategy):
    """Flip specific label pairs according to a mapping."""
    
    def poison_label(self, original_label: int, config: AttackConfig) -> int:
        if config.flip_map and original_label in config.flip_map:
            if self.should_poison(config):
                return config.flip_map[original_label]
        return original_label


class PermutationStrategy(PoisonStrategy):
    """Cyclic permutation: 0->1, 1->2, ..., 9->0."""
    
    def poison_label(self, original_label: int, config: AttackConfig) -> int:
        if self.should_poison(config):
            return (original_label + 1) % config.num_classes
        return original_label


class PartialFlipStrategy(PoisonStrategy):
    """Flip only odd/even labels or specific subsets."""
    
    def poison_label(self, original_label: int, config: AttackConfig) -> int:
        # Only poison odd numbers
        if original_label % 2 == 1 and self.should_poison(config):
            return random.randint(0, config.num_classes - 1)
        return original_label


# Strategy factory
def get_poison_strategy(attack_type: AttackType) -> PoisonStrategy:
    """Factory function to get the appropriate poisoning strategy."""
    strategies = {
        AttackType.RANDOM_FLIP: RandomFlipStrategy(),
        AttackType.TARGETED_FLIP: TargetedFlipStrategy(),
        AttackType.PERMUTATION: PermutationStrategy(),
        AttackType.PARTIAL_FLIP: PartialFlipStrategy(),
    }
    return strategies.get(attack_type, RandomFlipStrategy())


# =============================================================================
# DATASET POISONING
# =============================================================================

def poison_partition(partition: Any, config: AttackConfig) -> Any:
    """
    Apply poisoning to a dataset partition.
    
    The P2PFLDataset wraps a HuggingFace dataset. We need to access the 
    underlying train/test splits and modify the labels.
    
    Args:
        partition: The P2PFLDataset partition to poison
        config: Attack configuration
        
    Returns:
        Poisoned partition (same object, modified in place)
    """
    if config.attack_type == AttackType.NONE:
        return partition
    
    strategy = get_poison_strategy(config.attack_type)
    
    # Track poisoning statistics
    poison_stats = {'total': 0, 'poisoned': 0, 'original_labels': [], 'new_labels': []}
    
    def corrupt(example: Dict[str, Any]) -> Dict[str, Any]:
        original_label = example["label"]
        new_label = strategy.poison_label(original_label, config)
        poison_stats['total'] += 1
        if new_label != original_label:
            poison_stats['poisoned'] += 1
            if len(poison_stats['original_labels']) < 10:  # Track first 10 for debugging
                poison_stats['original_labels'].append(original_label)
                poison_stats['new_labels'].append(new_label)
        example["label"] = new_label
        return example
    
    poisoned = False
    
    # Debug: Show what we're working with
    print(f"      Partition type: {type(partition).__name__}")
    
    try:
        # P2PFLDataset stores data in _data attribute which is a DatasetDict
        # with 'train' and 'test' splits
        if hasattr(partition, '_data'):
            data = partition._data
            print(f"      _data type: {type(data).__name__}")
            
            # Check if it's a DatasetDict (has 'train' key)
            if hasattr(data, 'keys'):
                print(f"      _data keys: {list(data.keys()) if hasattr(data, 'keys') else 'N/A'}")
            
            # Case 1: DatasetDict with train/test splits
            if hasattr(data, '__getitem__') and 'train' in data:
                print(f"      Found 'train' split with {len(data['train'])} samples")
                partition._data['train'] = data['train'].map(corrupt)
                poisoned = True
                print(f"      ✓ Poisoned train split: {poison_stats['poisoned']}/{poison_stats['total']} samples modified")
            
            # Case 2: Direct Dataset (not a dict)
            elif hasattr(data, 'map') and hasattr(data, '__len__'):
                print(f"      Found direct dataset with {len(data)} samples")
                partition._data = data.map(corrupt)
                poisoned = True
                print(f"      ✓ Poisoned _data directly: {poison_stats['poisoned']}/{poison_stats['total']} samples modified")
        
        # Alternative: try train_data attribute
        if not poisoned and hasattr(partition, 'train_data') and partition.train_data is not None:
            train_data = partition.train_data
            print(f"      train_data type: {type(train_data).__name__}, len: {len(train_data) if hasattr(train_data, '__len__') else 'unknown'}")
            if hasattr(train_data, 'map'):
                partition.train_data = train_data.map(corrupt)
                poisoned = True
                print(f"      ✓ Poisoned train_data: {poison_stats['poisoned']}/{poison_stats['total']} samples modified")
        
        # Alternative: try _train_data attribute  
        if not poisoned and hasattr(partition, '_train_data') and partition._train_data is not None:
            train_data = partition._train_data
            print(f"      _train_data type: {type(train_data).__name__}, len: {len(train_data) if hasattr(train_data, '__len__') else 'unknown'}")
            if hasattr(train_data, 'map'):
                partition._train_data = train_data.map(corrupt)
                poisoned = True
                print(f"      ✓ Poisoned _train_data: {poison_stats['poisoned']}/{poison_stats['total']} samples modified")
        
        if not poisoned:
            # Last resort: dump all attributes to help debug
            print(f"      ⚠ WARNING: Could not find data to poison!")
            attrs = [a for a in dir(partition) if not a.startswith('__')]
            print(f"      Available attributes: {attrs}")
            
            # Try to find any dataset-like attributes
            for attr in attrs:
                try:
                    val = getattr(partition, attr)
                    if hasattr(val, '__len__') and not callable(val):
                        print(f"        {attr}: type={type(val).__name__}, len={len(val)}")
                    elif hasattr(val, 'keys'):
                        print(f"        {attr}: type={type(val).__name__}, keys={list(val.keys())}")
                except Exception:
                    pass
        else:
            # Show sample of label changes
            if poison_stats['original_labels']:
                print(f"      Sample flips: {list(zip(poison_stats['original_labels'][:5], poison_stats['new_labels'][:5]))}")
                
    except Exception as e:
        print(f"      ✗ ERROR poisoning partition: {e}")
        import traceback
        traceback.print_exc()
    
    return partition


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Complete configuration for an experiment run."""
    # Basic settings
    num_nodes: int = 5
    num_rounds: int = 3
    epochs_per_round: int = 1
    
    # Attack settings
    num_malicious: int = 0
    attack_config: AttackConfig = field(default_factory=AttackConfig)
    
    # Technical settings
    protocol: str = "memory"
    framework: str = "pytorch"
    aggregator: str = "fedavg"
    topology: TopologyType = TopologyType.FULL
    reduced_dataset: bool = True  # Use smaller dataset for faster testing
    
    # Output settings
    show_plots: bool = False
    measure_time: bool = True
    
    @property
    def malicious_fraction(self) -> float:
        """Fraction of nodes that are malicious."""
        return self.num_malicious / self.num_nodes if self.num_nodes > 0 else 0.0
    
    def __str__(self) -> str:
        attack_str = "Clean" if self.num_malicious == 0 else f"{self.num_malicious} malicious ({self.attack_config.attack_type.value})"
        return (f"ExperimentConfig(nodes={self.num_nodes}, rounds={self.num_rounds}, "
                f"topology={self.topology.value}, attack={attack_str})")


@dataclass
class ExperimentResults:
    """Results from a single experiment run."""
    config: ExperimentConfig
    execution_time: float = 0.0
    rounds_data: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    @property
    def final_accuracy(self) -> float:
        """Get the final average accuracy."""
        if not self.rounds_data:
            return 0.0
        last_round = max(self.rounds_data.keys())
        return self.rounds_data[last_round].get('avg_accuracy', 0.0)
    
    @property
    def final_loss(self) -> float:
        """Get the final average loss."""
        if not self.rounds_data:
            return float('inf')
        last_round = max(self.rounds_data.keys())
        return self.rounds_data[last_round].get('avg_loss', float('inf'))
    
    @property
    def accuracy_progression(self) -> List[float]:
        """Get accuracy values for each round."""
        return [self.rounds_data[r]['avg_accuracy'] for r in sorted(self.rounds_data.keys())]
    
    @property
    def loss_progression(self) -> List[float]:
        """Get loss values for each round."""
        return [self.rounds_data[r]['avg_loss'] for r in sorted(self.rounds_data.keys())]


# =============================================================================
# SETTINGS & MODEL CREATION
# =============================================================================

def set_standalone_settings(disable_ray: bool = False) -> None:
    """Set settings for standalone execution."""
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


def create_pytorch_model() -> P2PFLModel:
    """Create a PyTorch model."""
    from p2pfl.learning.frameworks.pytorch.lightning_model import MLP, LightningModel
    return LightningModel(MLP())


def create_tensorflow_model() -> P2PFLModel:
    """Create a TensorFlow model."""
    import tensorflow as tf
    from p2pfl.learning.frameworks.tensorflow.keras_model import MLP as MLP_KERAS
    from p2pfl.learning.frameworks.tensorflow.keras_model import KerasModel
    model = MLP_KERAS()
    model(tf.zeros((1, 28, 28, 1)))
    return KerasModel(model)


# =============================================================================
# METRICS COLLECTION
# =============================================================================

def _get_per_node_log_counts() -> Dict[str, int]:
    """Get the current count of test_metric entries per node."""
    try:
        global_logs = logger.get_global_logs()
        if not global_logs:
            return {}
        exp_logs = global_logs.get('experiment', {})
        return {node: len(metrics.get('test_metric', [])) for node, metrics in exp_logs.items()}
    except Exception:
        return {}


def collect_all_metrics_now(num_rounds: int) -> Dict[int, Dict[str, Any]]:
    """
    Collect all metrics from the logger RIGHT NOW and return them.
    This captures the current state before it gets polluted by another experiment.
    
    IMPORTANT: This function reads the LAST num_rounds entries from each node,
    assuming they are from the most recent experiment.
    
    Args:
        num_rounds: Number of rounds to collect
        
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
        
        # Debug: show what we have
        print(f"   Found {len(exp_logs)} nodes in logs")
        for node_name, node_metrics in exp_logs.items():
            test_metrics = node_metrics.get('test_metric', [])
            print(f"      {node_name}: {len(test_metrics)} test_metric entries")
        
        # Collect metrics for each round
        for round_num in range(num_rounds):
            result = {
                'metrics': {},
                'avg_loss': 0.0,
                'avg_accuracy': 0.0,
                'min_accuracy': float('inf'),
                'max_accuracy': 0.0
            }
            
            total_loss = 0.0
            total_acc = 0.0
            count = 0
            
            for node_name, node_metrics in exp_logs.items():
                test_metrics = node_metrics.get('test_metric', [])
                test_losses = node_metrics.get('test_loss', [])
                
                # We need exactly num_rounds entries from the END of the list
                # If a node has fewer entries, it might not have participated in all rounds
                if len(test_metrics) < num_rounds or len(test_losses) < num_rounds:
                    # This node doesn't have enough entries - skip it for this round
                    # or use what's available
                    available = min(len(test_metrics), len(test_losses))
                    if available == 0:
                        continue
                    # Map round_num to available entries
                    # If we have 2 entries and need 3 rounds, only rounds 1,2 are available
                    offset = num_rounds - available
                    if round_num < offset:
                        continue  # This round's data isn't available for this node
                    actual_index = round_num - offset
                else:
                    # Node has enough entries - use negative indexing from the end
                    # For round 0 of 3 rounds: index -3
                    # For round 1 of 3 rounds: index -2
                    # For round 2 of 3 rounds: index -1
                    actual_index = -(num_rounds - round_num)
                
                try:
                    _, acc_value = test_metrics[actual_index]
                    _, loss_value = test_losses[actual_index]
                    
                    acc_pct = float(acc_value) * 100.0
                    loss_float = float(loss_value)
                    
                    result['metrics'][node_name] = {
                        'test_loss': loss_float,
                        'test_metric': acc_value
                    }
                    
                    total_loss += loss_float
                    total_acc += acc_pct
                    count += 1
                    
                    result['min_accuracy'] = min(result['min_accuracy'], acc_pct)
                    result['max_accuracy'] = max(result['max_accuracy'], acc_pct)
                except (IndexError, TypeError) as e:
                    print(f"      Warning: Could not get metrics for {node_name} round {round_num}: {e}")
                    continue
            
            if count > 0:
                result['avg_loss'] = total_loss / count
                result['avg_accuracy'] = total_acc / count
            if result['min_accuracy'] == float('inf'):
                result['min_accuracy'] = 0.0
            
            all_rounds[round_num] = result
            
    except Exception as e:
        print(f"   Error collecting metrics: {e}")
        import traceback
        traceback.print_exc()
    
    return all_rounds


def collect_round_metrics(round_num: int, num_rounds: int, log_offset: int = 0) -> Dict[str, Any]:
    """
    Collect metrics for a specific round from the logger.
    
    DEPRECATED: Use collect_all_metrics_now() instead for accurate results.
    
    Args:
        round_num: The round number (0-indexed) within this experiment
        num_rounds: Total number of rounds in this experiment
        log_offset: Number of rounds to skip in logs (for second experiment)
        
    Returns:
        Dictionary with round metrics
    """
    result = {
        'metrics': {},
        'avg_loss': 0.0,
        'avg_accuracy': 0.0,
        'min_accuracy': float('inf'),
        'max_accuracy': 0.0
    }
    
    try:
        global_logs = logger.get_global_logs()
        if not global_logs:
            return result
        
        exp_logs = global_logs.get('experiment', {})
        
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        
        # Calculate the actual index in the accumulated logs
        # For second experiment, we need to offset by the number of rounds from first experiment
        actual_index = log_offset + round_num
        
        for node_name, node_metrics in exp_logs.items():
            test_metrics = node_metrics.get('test_metric', [])
            test_losses = node_metrics.get('test_loss', [])
            
            if actual_index < len(test_metrics) and actual_index < len(test_losses):
                _, acc_value = test_metrics[actual_index]
                _, loss_value = test_losses[actual_index]
                
                acc_pct = float(acc_value) * 100.0
                loss_float = float(loss_value)
                
                result['metrics'][node_name] = {
                    'test_loss': loss_float,
                    'test_metric': acc_value
                }
                
                total_loss += loss_float
                total_acc += acc_pct
                count += 1
                
                result['min_accuracy'] = min(result['min_accuracy'], acc_pct)
                result['max_accuracy'] = max(result['max_accuracy'], acc_pct)
        
        if count > 0:
            result['avg_loss'] = total_loss / count
            result['avg_accuracy'] = total_acc / count
            
    except Exception as e:
        print(f"Warning: Could not collect metrics for round {round_num}: {e}")
    
    return result


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """
    Run a single federated learning experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        ExperimentResults object with all metrics
    """
    print(f"\n{'='*80}")
    print(f"🚀 STARTING EXPERIMENT")
    print(f"{'='*80}")
    print(f"   {config}")
    if config.num_malicious > 0:
        print(f"   Attack: {config.attack_config.attack_type.value} with poison_rate={config.attack_config.poison_rate}")
        print(f"   Malicious nodes: indices 0-{config.num_malicious-1}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    results = ExperimentResults(config=config)
    
    # Validate settings
    if config.num_nodes > Settings.TTL:
        raise ValueError("TTL must be greater than the number of nodes for line topology")
    
    if config.num_malicious > config.num_nodes:
        raise ValueError("Number of malicious nodes cannot exceed total nodes")
    
    # Setup framework
    if config.framework == "tensorflow":
        from p2pfl.learning.frameworks.tensorflow.keras_learner import KerasLearner
        model_fn = create_tensorflow_model
        learner = KerasLearner
    elif config.framework == "pytorch":
        from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
        model_fn = create_pytorch_model
        learner = LightningLearner
    else:
        raise ValueError(f"Unsupported framework: {config.framework}")
    
    # Load and partition data
    print("📦 Loading MNIST dataset...")
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(
        config.num_nodes * 50 if config.reduced_dataset else config.num_nodes,
        RandomIIDPartitionStrategy,
    )
    
    # Apply poisoning to malicious nodes' partitions
    if config.num_malicious > 0:
        print(f"☠️  Poisoning data for {config.num_malicious} malicious nodes...")
        for i in range(config.num_malicious):
            print(f"   Node {i}: applying {config.attack_config.attack_type.value} attack...")
            partitions[i] = poison_partition(partitions[i], config.attack_config)
        
        # Verify poisoning by sampling some labels
        print(f"\n   🔍 Verifying poisoning effect...")
        try:
            for i in range(min(config.num_malicious, 2)):  # Check first 2 malicious nodes
                partition = partitions[i]
                # Try to get some labels to verify
                sample_labels = []
                if hasattr(partition, 'train_data') and partition.train_data is not None:
                    for j, item in enumerate(partition.train_data):
                        if j >= 20:
                            break
                        sample_labels.append(item.get('label', '?'))
                elif hasattr(partition, '_train_data') and partition._train_data is not None:
                    for j, item in enumerate(partition._train_data):
                        if j >= 20:
                            break
                        sample_labels.append(item.get('label', '?'))
                if sample_labels:
                    print(f"      Node {i} sample labels (first 20): {sample_labels}")
        except Exception as e:
            print(f"      Could not verify labels: {e}")
    
    # Create nodes
    print(f"\n🖥️  Creating {config.num_nodes} nodes...")
    nodes = []
    for i in range(config.num_nodes):
        if config.protocol == "memory":
            address = f"node-{i}"
            protocol_class = InMemoryCommunicationProtocol
        elif config.protocol == "unix":
            address = f"unix:///tmp/p2pfl-{i}.sock"
            protocol_class = GrpcCommunicationProtocol
        else:  # grpc
            address = "127.0.0.1"
            protocol_class = GrpcCommunicationProtocol
        
        node = Node(
            model_fn(),
            partitions[i],
            learner=learner,
            protocol=protocol_class,
            address=address,
            simulation=True,
            aggregator=Scaffold() if config.aggregator == "scaffold" else None,
        )
        node.start()
        nodes.append(node)
        
        node_type = "☠️ MALICIOUS" if i < config.num_malicious else "✅ honest"
        print(f"   Node {i} ({node_type}): started")
    
    try:
        # Connect nodes according to topology
        print(f"\n🔗 Connecting nodes in {config.topology.value} topology...")
        adjacency_matrix = TopologyFactory.generate_matrix(config.topology, len(nodes))
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)
        
        # Wait for network convergence
        print("⏳ Waiting for network convergence...")
        wait_convergence(nodes, config.num_nodes - 1, only_direct=False, wait=60)
        
        # Start training
        print(f"\n🎓 Starting federated learning ({config.num_rounds} rounds, {config.epochs_per_round} epochs/round)...")
        nodes[0].set_start_learning(rounds=config.num_rounds, epochs=config.epochs_per_round)
        
        # Wait for completion
        wait_to_finish(nodes, timeout=60 * 60)
        
        print("\n✅ Training completed!")
        
        # Collect metrics for all rounds IMMEDIATELY before stopping nodes
        # This captures the current state before any cleanup
        print("📊 Collecting metrics...")
        results.rounds_data = collect_all_metrics_now(config.num_rounds)
        
    finally:
        # Cleanup
        print("\n🧹 Stopping nodes...")
        for node in nodes:
            node.stop()
    
    results.execution_time = time.time() - start_time
    return results


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def print_experiment_results(results: ExperimentResults) -> None:
    """Print detailed results for a single experiment."""
    config = results.config
    
    print(f"\n{'='*80}")
    print("📊 EXPERIMENT RESULTS")
    print(f"{'='*80}")
    
    # Configuration
    print(f"\n🔧 Configuration:")
    print(f"   Nodes: {config.num_nodes} ({config.num_malicious} malicious, {config.num_nodes - config.num_malicious} honest)")
    print(f"   Rounds: {config.num_rounds} | Epochs/Round: {config.epochs_per_round}")
    print(f"   Topology: {config.topology.value} | Framework: {config.framework}")
    print(f"   Dataset: {'Reduced (testing)' if config.reduced_dataset else 'Full'}")
    if config.num_malicious > 0:
        print(f"   Attack: {config.attack_config.attack_type.value} (poison_rate={config.attack_config.poison_rate})")
    print(f"   Execution Time: {results.execution_time:.2f}s")
    
    # Round-by-round results
    print(f"\n{'='*80}")
    print("📈 ROUND-BY-ROUND PERFORMANCE")
    print(f"{'='*80}")
    
    print(f"\n{'Round':<8} {'Avg Loss':<15} {'Avg Accuracy':<15} {'Min Acc':<12} {'Max Acc':<12}")
    print("-" * 70)
    
    prev_acc = None
    for round_num in sorted(results.rounds_data.keys()):
        rd = results.rounds_data[round_num]
        if rd['metrics']:
            change = ""
            if prev_acc is not None:
                diff = rd['avg_accuracy'] - prev_acc
                change = f"({diff:+.2f}%)"
            print(f"{round_num:<8} {rd['avg_loss']:<15.4f} {rd['avg_accuracy']:<14.2f}% {rd['min_accuracy']:<11.2f}% {rd['max_accuracy']:<11.2f}% {change}")
            prev_acc = rd['avg_accuracy']
    
    # Final summary
    print(f"\n{'='*80}")
    print("🎯 FINAL RESULTS")
    print(f"{'='*80}")
    print(f"   Final Accuracy: {results.final_accuracy:.2f}%")
    print(f"   Final Loss: {results.final_loss:.4f}")
    
    if len(results.rounds_data) > 1:
        first_acc = results.rounds_data[0]['avg_accuracy']
        improvement = results.final_accuracy - first_acc
        print(f"   Total Improvement: {improvement:+.2f}%")


def print_comparison_results(clean_results: ExperimentResults, poisoned_results: ExperimentResults) -> None:
    """Print a comparison between clean and poisoned experiment results."""
    
    print(f"\n{'='*80}")
    print("🔬 COMPARISON: CLEAN vs POISONED")
    print(f"{'='*80}")
    
    # Header
    print(f"\n{'Metric':<25} {'Clean':<20} {'Poisoned':<20} {'Difference':<15}")
    print("-" * 80)
    
    # Final accuracy
    clean_acc = clean_results.final_accuracy
    poison_acc = poisoned_results.final_accuracy
    acc_diff = poison_acc - clean_acc
    print(f"{'Final Accuracy':<25} {clean_acc:<19.2f}% {poison_acc:<19.2f}% {acc_diff:+.2f}%")
    
    # Final loss
    clean_loss = clean_results.final_loss
    poison_loss = poisoned_results.final_loss
    loss_diff = poison_loss - clean_loss
    print(f"{'Final Loss':<25} {clean_loss:<20.4f} {poison_loss:<20.4f} {loss_diff:+.4f}")
    
    # Execution time
    print(f"{'Execution Time (s)':<25} {clean_results.execution_time:<20.2f} {poisoned_results.execution_time:<20.2f}")
    
    # Round-by-round comparison
    print(f"\n{'='*80}")
    print("📈 ROUND-BY-ROUND COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Round':<8} {'Clean Acc':<15} {'Poisoned Acc':<15} {'Difference':<15} {'Impact %':<12}")
    print("-" * 70)
    
    for round_num in sorted(clean_results.rounds_data.keys()):
        if round_num in poisoned_results.rounds_data:
            c_acc = clean_results.rounds_data[round_num]['avg_accuracy']
            p_acc = poisoned_results.rounds_data[round_num]['avg_accuracy']
            diff = p_acc - c_acc
            impact = (diff / c_acc * 100) if c_acc > 0 else 0
            print(f"{round_num:<8} {c_acc:<14.2f}% {p_acc:<14.2f}% {diff:<+14.2f}% {impact:<+11.1f}%")
    
    # Attack effectiveness summary
    print(f"\n{'='*80}")
    print("💀 ATTACK EFFECTIVENESS ANALYSIS")
    print(f"{'='*80}")
    
    attack_config = poisoned_results.config.attack_config
    num_malicious = poisoned_results.config.num_malicious
    total_nodes = poisoned_results.config.num_nodes
    malicious_pct = (num_malicious / total_nodes) * 100
    
    print(f"\n   Attack Type: {attack_config.attack_type.value}")
    print(f"   Poison Rate: {attack_config.poison_rate * 100:.0f}%")
    print(f"   Malicious Nodes: {num_malicious}/{total_nodes} ({malicious_pct:.1f}%)")
    print(f"   Topology: {poisoned_results.config.topology.value}")
    
    accuracy_drop = clean_acc - poison_acc
    relative_drop = (accuracy_drop / clean_acc * 100) if clean_acc > 0 else 0
    
    print(f"\n   Accuracy Drop: {accuracy_drop:.2f}% (relative: {relative_drop:.1f}%)")
    
    # Verdict
    if accuracy_drop > 10:
        verdict = "🔴 HIGH IMPACT - Attack significantly degraded model performance"
    elif accuracy_drop > 5:
        verdict = "🟠 MODERATE IMPACT - Attack had noticeable effect"
    elif accuracy_drop > 1:
        verdict = "🟡 LOW IMPACT - Model showed some resilience"
    elif accuracy_drop > 0:
        verdict = "🟢 MINIMAL IMPACT - Model is robust to this attack"
    else:
        verdict = "✅ NO IMPACT - Attack was ineffective"
    
    print(f"\n   Verdict: {verdict}")
    
    # Recommendations
    print(f"\n   Recommendations:")
    if accuracy_drop > 5:
        print("     • Consider using Byzantine-robust aggregation (e.g., Krum, Median)")
        print("     • Try Scaffold aggregator for better convergence")
        print("     • Increase the number of honest nodes")
    else:
        print("     • Current setup shows good resilience")
        print("     • Try increasing malicious fraction to stress-test")
        print("     • Experiment with different attack types")


def plot_comparison(clean_results: ExperimentResults, poisoned_results: ExperimentResults, save_path: Optional[str] = None) -> None:
    """Generate comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = list(range(len(clean_results.accuracy_progression)))
    
    # Accuracy plot
    ax1 = axes[0]
    ax1.plot(rounds, clean_results.accuracy_progression, 'g-o', label='Clean', linewidth=2, markersize=8)
    ax1.plot(rounds, poisoned_results.accuracy_progression, 'r-s', label='Poisoned', linewidth=2, markersize=8)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy: Clean vs Poisoned', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(rounds)
    
    # Loss plot
    ax2 = axes[1]
    ax2.plot(rounds, clean_results.loss_progression, 'g-o', label='Clean', linewidth=2, markersize=8)
    ax2.plot(rounds, poisoned_results.loss_progression, 'r-s', label='Poisoned', linewidth=2, markersize=8)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss: Clean vs Poisoned', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(rounds)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {save_path}")
    
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_comparison_experiment(
    num_nodes: int = 5,
    num_malicious: int = 2,
    num_rounds: int = 3,
    epochs: int = 1,
    attack_type: AttackType = AttackType.RANDOM_FLIP,
    poison_rate: float = 1.0,
    topology: TopologyType = TopologyType.FULL,
    framework: str = "pytorch",
    aggregator: str = "fedavg",
    reduced_dataset: bool = True,
    show_plots: bool = True,
    save_plot_path: Optional[str] = None,
) -> Tuple[ExperimentResults, ExperimentResults]:
    """
    Run a comparison experiment between clean and poisoned scenarios.
    
    Returns:
        Tuple of (clean_results, poisoned_results)
    """
    # Clean experiment config
    clean_config = ExperimentConfig(
        num_nodes=num_nodes,
        num_rounds=num_rounds,
        epochs_per_round=epochs,
        num_malicious=0,
        attack_config=AttackConfig(attack_type=AttackType.NONE),
        topology=topology,
        framework=framework,
        aggregator=aggregator,
        reduced_dataset=reduced_dataset,
        show_plots=False,
    )
    
    # Poisoned experiment config
    poisoned_config = ExperimentConfig(
        num_nodes=num_nodes,
        num_rounds=num_rounds,
        epochs_per_round=epochs,
        num_malicious=num_malicious,
        attack_config=AttackConfig(
            attack_type=attack_type,
            poison_rate=poison_rate,
        ),
        topology=topology,
        framework=framework,
        aggregator=aggregator,
        reduced_dataset=reduced_dataset,
        show_plots=False,
    )
    
    # Run clean experiment
    print("\n" + "🟢 "*20)
    print("RUNNING CLEAN EXPERIMENT (Baseline)")
    print("🟢 "*20)
    
    # Record how many log entries exist BEFORE the clean experiment
    pre_clean_counts = _get_per_node_log_counts()
    print(f"📊 Pre-clean log counts: {pre_clean_counts}")
    
    clean_results = run_experiment(clean_config)
    
    # Record counts AFTER clean experiment
    post_clean_counts = _get_per_node_log_counts()
    print(f"📊 Post-clean log counts: {post_clean_counts}")
    
    # IMPORTANT: Deep copy the results immediately to avoid log pollution
    import copy
    clean_results_copy = ExperimentResults(
        config=clean_results.config,
        execution_time=clean_results.execution_time,
        rounds_data=copy.deepcopy(clean_results.rounds_data)
    )
    
    print_experiment_results(clean_results_copy)
    
    # Verify clean results make sense
    print(f"\n📋 Clean experiment final round metrics:")
    if clean_results_copy.rounds_data:
        last_round = max(clean_results_copy.rounds_data.keys())
        for node, metrics in clean_results_copy.rounds_data[last_round]['metrics'].items():
            print(f"   {node}: acc={metrics['test_metric']*100:.2f}%, loss={metrics['test_loss']:.4f}")
    
    # Small delay between experiments
    time.sleep(2)
    
    # Run poisoned experiment  
    print("\n" + "🔴 "*20)
    print("RUNNING POISONED EXPERIMENT")
    print("🔴 "*20)
    
    # Record counts BEFORE poisoned experiment
    pre_poison_counts = _get_per_node_log_counts()
    print(f"📊 Pre-poison log counts: {pre_poison_counts}")
    
    poisoned_results = run_experiment(poisoned_config)
    
    # Record counts AFTER poisoned experiment
    post_poison_counts = _get_per_node_log_counts()
    print(f"📊 Post-poison log counts: {post_poison_counts}")
    
    # Deep copy poisoned results too
    poisoned_results_copy = ExperimentResults(
        config=poisoned_results.config,
        execution_time=poisoned_results.execution_time,
        rounds_data=copy.deepcopy(poisoned_results.rounds_data)
    )
    
    print_experiment_results(poisoned_results_copy)
    
    # Verify poisoned results make sense
    print(f"\n📋 Poisoned experiment final round metrics:")
    if poisoned_results_copy.rounds_data:
        last_round = max(poisoned_results_copy.rounds_data.keys())
        for node, metrics in poisoned_results_copy.rounds_data[last_round]['metrics'].items():
            print(f"   {node}: acc={metrics['test_metric']*100:.2f}%, loss={metrics['test_loss']:.4f}")
    
    # Use the copies for comparison
    clean_results = clean_results_copy
    poisoned_results = poisoned_results_copy
    
    # Print comparison
    print_comparison_results(clean_results, poisoned_results)
    
    # Generate plots
    if show_plots or save_plot_path:
        plot_comparison(clean_results, poisoned_results, save_plot_path)
    
    return clean_results, poisoned_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="P2PFL MNIST experiment with adversarial nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comparison between clean and poisoned (uses reduced dataset by default)
  python mnist_adversarial.py --compare --nodes 5 --malicious 2 --rounds 3
  
  # Run single poisoned experiment
  python mnist_adversarial.py --nodes 5 --malicious 2 --attack random_flip --rounds 5
  
  # Test different attack types
  python mnist_adversarial.py --compare --attack targeted_flip --poison-rate 0.8
  
  # Test with different topologies
  python mnist_adversarial.py --compare --topology star --nodes 6 --malicious 2
  
  # Run with full dataset (slower but more realistic)
  python mnist_adversarial.py --compare --full-dataset --nodes 5 --rounds 3
        """
    )
    
    # Mode
    parser.add_argument("--compare", action="store_true", 
                       help="Run comparison between clean and poisoned experiments")
    
    # Basic settings
    parser.add_argument("--nodes", type=int, default=5, help="Number of nodes")
    parser.add_argument("--rounds", type=int, default=3, help="Number of training rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per round")
    
    # Attack settings
    parser.add_argument("--malicious", type=int, default=2, help="Number of malicious nodes")
    parser.add_argument("--attack", type=str, default="random_flip",
                       choices=["none", "random_flip", "targeted_flip", "permutation", "partial_flip"],
                       help="Type of poisoning attack")
    parser.add_argument("--poison-rate", type=float, default=1.0,
                       help="Fraction of samples to poison (0.0-1.0)")
    
    # Technical settings
    parser.add_argument("--protocol", type=str, default="memory",
                       choices=["grpc", "unix", "memory"], help="Communication protocol")
    parser.add_argument("--framework", type=str, default="pytorch",
                       choices=["pytorch", "tensorflow"], help="ML framework")
    parser.add_argument("--aggregator", type=str, default="fedavg",
                       choices=["fedavg", "scaffold"], help="Aggregation strategy")
    parser.add_argument("--topology", type=str, default="full",
                       choices=[t.value for t in TopologyType], help="Network topology")
    
    # Dataset settings
    parser.add_argument("--reduced-dataset", action="store_true", default=True,
                       help="Use reduced dataset for faster testing (default: True)")
    parser.add_argument("--full-dataset", action="store_true",
                       help="Use full dataset (overrides --reduced-dataset)")
    
    # Output settings
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--save-plot", type=str, default=None, help="Path to save comparison plot")
    parser.add_argument("--disable-ray", action="store_true", help="Disable Ray")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Initialize settings
    set_standalone_settings(disable_ray=args.disable_ray)
    
    # Parse enums
    topology = TopologyType(args.topology)
    attack_type = AttackType(args.attack)
    
    # Determine reduced_dataset setting (--full-dataset overrides --reduced-dataset)
    reduced_dataset = not args.full_dataset
    
    if args.compare:
        # Run comparison experiment
        run_comparison_experiment(
            num_nodes=args.nodes,
            num_malicious=args.malicious,
            num_rounds=args.rounds,
            epochs=args.epochs,
            attack_type=attack_type,
            poison_rate=args.poison_rate,
            topology=topology,
            framework=args.framework,
            aggregator=args.aggregator,
            reduced_dataset=reduced_dataset,
            show_plots=not args.no_plots,
            save_plot_path=args.save_plot,
        )
    else:
        # Run single experiment
        config = ExperimentConfig(
            num_nodes=args.nodes,
            num_rounds=args.rounds,
            epochs_per_round=args.epochs,
            num_malicious=args.malicious,
            attack_config=AttackConfig(
                attack_type=attack_type,
                poison_rate=args.poison_rate,
            ),
            topology=topology,
            protocol=args.protocol,
            framework=args.framework,
            aggregator=args.aggregator,
            reduced_dataset=reduced_dataset,
            show_plots=not args.no_plots,
        )
        
        results = run_experiment(config)
        print_experiment_results(results)