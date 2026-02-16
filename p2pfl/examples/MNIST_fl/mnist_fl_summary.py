#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Example of a P2PFL MNIST experiment, using a MLP model and a MnistFederatedDM."""

import argparse
import time
import uuid
from typing import Dict, List, Any

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


def set_standalone_settings(disable_ray: bool = False) -> None:
    """
    Set settings for testing.

    Important:
        - HEARTBEAT_PERIOD: Too high values can cause late node discovery/fault detection. Too low values can cause high CPU usage.
        - GOSSIP_PERIOD: Too low values can cause high CPU usage.
        - TTL: Low TTLs can cause that some messages are not delivered.

    """
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
    logger.set_level(Settings.LOG_LEVEL)  # Refresh (maybe already initialized)


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2PFL MNIST experiment using the Web Logger.")
    parser.add_argument("--nodes", type=int, help="The number of nodes.", default=2)
    parser.add_argument("--rounds", type=int, help="The number of rounds.", default=2)
    parser.add_argument("--epochs", type=int, help="The number of epochs.", default=1)
    parser.add_argument("--show_metrics", action="store_true", help="Show metrics.", default=True)
    parser.add_argument("--measure_time", action="store_true", help="Measure time.", default=False)
    parser.add_argument("--token", type=str, help="The API token for the Web Logger.", default="")
    parser.add_argument("--protocol", type=str, help="The protocol to use.", default="grpc", choices=["grpc", "unix", "memory"])
    parser.add_argument("--framework", type=str, help="The framework to use.", default="pytorch", choices=["pytorch", "tensorflow", "flax"])
    parser.add_argument("--aggregator", type=str, help="The aggregator to use.", default="fedavg", choices=["fedavg", "scaffold"])
    parser.add_argument("--profiling", action="store_true", help="Enable profiling.", default=False)
    parser.add_argument("--reduced_dataset", action="store_true", help="Use a reduced dataset just for testing.", default=False)
    parser.add_argument("--use_scaffold", action="store_true", help="Use the Scaffold aggregator.", default=False)
    parser.add_argument("--disable_ray", action="store_true", help="Disable Ray.", default=False)
    parser.add_argument(
        "--topology",
        type=str,
        choices=[t.value for t in TopologyType],
        default="line",
        help="The network topology (star, full, line, ring).",
    )
    args = parser.parse_args()
    # parse topology to TopologyType enum
    args.topology = TopologyType(args.topology)

    return args


def print_results_summary(
    results_tracker: Dict[str, Any],
    n: int,
    r: int,
    e: int,
    topology: TopologyType,
    framework: str,
    execution_time: float = 0
) -> None:
    """
    Print a comprehensive summary of federated learning results.
    
    Args:
        results_tracker: Dictionary containing all metrics per round
        n: Number of nodes
        r: Number of rounds
        e: Epochs per round
        topology: Network topology
        framework: ML framework used
        execution_time: Total execution time
    """
    print("\n" + "="*80)
    print("📊 MNIST FEDERATED LEARNING RESULTS SUMMARY")
    print("="*80)
    
    # Configuration summary
    print("\n🔧 Configuration:")
    print(f"  Nodes: {n} | Rounds: {r} | Epochs/Round: {e}")
    print(f"  Topology: {topology.value} | Framework: {framework}")
    if execution_time > 0:
        print(f"  Total Time: {execution_time:.2f}s ({execution_time/60:.2f} min)")
    
    # Check if we have results
    if not results_tracker or 'rounds' not in results_tracker:
        print("\n⚠️  No results to display")
        return
    
    # Create summary table
    print("\n" + "="*80)
    print("📈 ROUND-BY-ROUND PERFORMANCE")
    print("="*80)
    
    rounds_data = results_tracker['rounds']
    
    # Table header
    print(f"\n{'Round':<8} {'Avg Loss':<15} {'Avg Accuracy':<15} {'Min Acc':<15} {'Max Acc':<15} {'Improvement':<12}")
    print("-" * 90)
    
    prev_avg_acc = None
    for round_num, round_data in rounds_data.items():
        if round_data['metrics']:
            avg_loss = round_data['avg_loss']
            avg_acc = round_data['avg_accuracy']
            min_acc = round_data['min_accuracy']
            max_acc = round_data['max_accuracy']
            
            # Calculate improvement
            if prev_avg_acc is not None:
                improvement = avg_acc - prev_avg_acc
                improvement_pct = (improvement / prev_avg_acc) * 100 if prev_avg_acc > 0 else 0
                improvement_str = f"{improvement:+.2f}% ({improvement_pct:+.1f}%)"
            else:
                improvement_str = "Baseline"
            
            print(f"{round_num:<8} {avg_loss:<15.4f} {avg_acc:<14.2f}% {min_acc:<14.2f}% {max_acc:<14.2f}% {improvement_str:<12}")
            prev_avg_acc = avg_acc
    
    # Overall improvement summary
    if len(rounds_data) > 1:
        first_round = list(rounds_data.keys())[0]
        last_round = list(rounds_data.keys())[-1]
        
        initial_acc = rounds_data[first_round]['avg_accuracy']
        final_acc = rounds_data[last_round]['avg_accuracy']
        total_improvement = final_acc - initial_acc
        total_improvement_pct = (total_improvement / initial_acc) * 100 if initial_acc > 0 else 0
        
        initial_loss = rounds_data[first_round]['avg_loss']
        final_loss = rounds_data[last_round]['avg_loss']
        loss_improvement_pct = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
        
        print("\n" + "="*80)
        print("🎯 OVERALL IMPROVEMENT")
        print("="*80)
        print(f"\n  Initial Accuracy (Round {first_round}):  {initial_acc:.2f}%")
        print(f"  Final Accuracy (Round {last_round}):    {final_acc:.2f}%")
        print(f"  Total Improvement:           {total_improvement:+.2f}% ({total_improvement_pct:+.1f}%)")
        print(f"\n  Initial Loss:                {initial_loss:.4f}")
        print(f"  Final Loss:                  {final_loss:.4f}")
        print(f"  Loss Reduction:              {loss_improvement_pct:.1f}%")
    
    # Node-level statistics
    print("\n" + "="*80)
    print("🖥️  NODE-LEVEL STATISTICS (Final Round)")
    print("="*80)
    
    last_round_data = rounds_data[list(rounds_data.keys())[-1]]
    if last_round_data['metrics']:
        print(f"\n{'Node':<20} {'Test Loss':<20} {'Accuracy':<20}")
        print("-" * 60)
        for node_addr, metrics in last_round_data['metrics'].items():
            node_name = node_addr.split(':')[1] if ':' in node_addr else node_addr
            acc = metrics.get('test_metric', 0) * 100  # Convert to percentage
            print(f"Node-{node_name:<15} {metrics.get('test_loss', 0):<20.4f} {acc:<19.2f}%")
    
    # Performance verdict
    print("\n" + "="*80)
    print("💡 PERFORMANCE VERDICT")
    print("="*80)
    
    if len(rounds_data) > 1:
        if final_acc > 95:
            verdict = "🟢 EXCELLENT - Model achieved high accuracy (>95%)"
        elif final_acc > 90:
            verdict = "🟢 VERY GOOD - Model achieved good accuracy (>90%)"
        elif final_acc > 85:
            verdict = "🟡 GOOD - Model achieved acceptable accuracy (>85%)"
        elif total_improvement > 5:
            verdict = "🟠 FAIR - Model is improving but needs more training"
        else:
            verdict = "🔴 POOR - Model needs significant tuning"
        
        print(f"\n  {verdict}")
        
        # Recommendations
        print("\n  Recommendations:")
        if final_acc < 90:
            print("    • Try increasing the number of rounds")
            print("    • Consider increasing epochs per round")
            print("    • Experiment with different aggregation strategies (--aggregator scaffold)")
        elif total_improvement < 1 and r < 5:
            print("    • Model may have plateaued - try different topologies")
            print("    • Consider adjusting learning rate in model")
        else:
            print("    • Model is performing well!")
            print("    • Consider testing with more nodes for scalability")
            print("    • Try different topologies to test robustness")
    
    print("\n" + "="*80)


def track_round_metrics(
    results_tracker: Dict[str, Any],
    round_num: int,
    nodes: List[Any]
) -> None:
    """
    Track metrics for a specific round by collecting from global logs.
    
    Args:
        results_tracker: Dictionary to store results
        round_num: Current round number (0-indexed)
        nodes: List of node objects
    """
    # Initialize round data structure
    if 'rounds' not in results_tracker:
        results_tracker['rounds'] = {}
    
    results_tracker['rounds'][round_num] = {
        'metrics': {},
        'avg_loss': 0.0,
        'avg_accuracy': 0.0,
        'min_accuracy': float('inf'),
        'max_accuracy': 0.0
    }
    
    try:
        # Use global_logs which has all test evaluations
        global_logs = logger.get_global_logs()
        if not global_logs:
            return
        
        # Structure: {'experiment': {'node_addr': {'test_metric': [(round, value), ...], ...}}}
        exp_logs = global_logs.get('experiment', {})
        
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        
        for node_name, node_metrics in exp_logs.items():
            # Get test_metric and test_loss lists
            test_metrics = node_metrics.get('test_metric', [])
            test_losses = node_metrics.get('test_loss', [])
            
            # Each round has one evaluation, so index directly
            # Round 0 is the first evaluation (index 0)
            if round_num < len(test_metrics) and round_num < len(test_losses):
                # test_metrics is a list of (round_num, value) tuples
                _, acc_value = test_metrics[round_num]
                _, loss_value = test_losses[round_num]
                
                acc_pct = float(acc_value) * 100.0  # Convert to percentage
                loss_float = float(loss_value)
                
                results_tracker['rounds'][round_num]['metrics'][node_name] = {
                    'test_loss': loss_float,
                    'test_metric': acc_value
                }
                
                total_loss += loss_float
                total_acc += acc_pct
                count += 1
                
                # Track min/max
                if acc_pct < results_tracker['rounds'][round_num]['min_accuracy']:
                    results_tracker['rounds'][round_num]['min_accuracy'] = acc_pct
                if acc_pct > results_tracker['rounds'][round_num]['max_accuracy']:
                    results_tracker['rounds'][round_num]['max_accuracy'] = acc_pct
        
        # Calculate averages
        if count > 0:
            results_tracker['rounds'][round_num]['avg_loss'] = total_loss / count
            results_tracker['rounds'][round_num]['avg_accuracy'] = total_acc / count
    
    except Exception as e:
        print(f"Warning: Could not collect metrics for round {round_num}: {e}")


def create_tensorflow_model() -> P2PFLModel:
    """Create a TensorFlow model."""
    import tensorflow as tf  # type: ignore

    from p2pfl.learning.frameworks.tensorflow.keras_model import MLP as MLP_KERAS
    from p2pfl.learning.frameworks.tensorflow.keras_model import KerasModel

    model = MLP_KERAS()  # type: ignore[no-untyped-call]
    model(tf.zeros((1, 28, 28, 1)))
    return KerasModel(model)


def create_pytorch_model() -> P2PFLModel:
    """Create a PyTorch model."""
    from p2pfl.learning.frameworks.pytorch.lightning_model import MLP, LightningModel

    return LightningModel(MLP())  # type: ignore[no-untyped-call]


def mnist(
    n: int,
    r: int,
    e: int,
    show_metrics: bool = True,
    measure_time: bool = False,
    protocol: str = "grpc",
    framework: str = "pytorch",
    aggregator: str = "fedavg",
    reduced_dataset: bool = True,
    topology: TopologyType = TopologyType.LINE,
) -> None:
    """
    P2PFL MNIST experiment.

    Args:
        n: The number of nodes.
        r: The number of rounds.
        e: The number of epochs.
        show_metrics: Show metrics.
        measure_time: Measure time.
        protocol: The protocol to use.
        framework: The framework to use.
        aggregator: The aggregator to use.
        reduced_dataset: Use a reduced dataset just for testing.
        topology: The network topology (star, full, line, ring).

    """
    if measure_time:
        start_time = time.time()
    
    # Initialize results tracker
    results_tracker: Dict[str, Any] = {
        'config': {
            'nodes': n,
            'rounds': r,
            'epochs': e,
            'topology': topology.value,
            'framework': framework,
            'reduced_dataset': reduced_dataset
        },
        'rounds': {}
    }

    # Check settings
    if n > Settings.TTL:
        raise ValueError(
            "For in-line topology TTL must be greater than the number of nodes." "Otherwise, some messages will not be delivered."
        )

    # Imports
    if framework == "tensorflow":
        from p2pfl.learning.frameworks.tensorflow.keras_learner import KerasLearner

        model_fn = create_tensorflow_model
        learner = KerasLearner
    elif framework == "pytorch":
        from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner

        model_fn = create_pytorch_model
        learner = LightningLearner  # type: ignore
    else:
        raise ValueError(f"Framework {framework} not added on this example.")

    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(
        n * 50 if reduced_dataset else n,
        RandomIIDPartitionStrategy,  # type: ignore
    )

    # Node Creation
    nodes = []
    for i in range(n):
        address = f"node-{i}" if protocol == "memory" else f"unix:///tmp/p2pfl-{i}.sock" if protocol == "unix" else "127.0.0.1"

        # Nodes
        node = Node(
            model_fn(),
            partitions[i],
            learner=learner,  # type: ignore
            protocol=InMemoryCommunicationProtocol if protocol == "memory" else GrpcCommunicationProtocol,  # type: ignore
            address=address,
            simulation=True,
            aggregator=Scaffold() if aggregator == "scaffold" else None,
        )
        node.start()
        nodes.append(node)

    try:
        adjacency_matrix = TopologyFactory.generate_matrix(topology, len(nodes))
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)

        wait_convergence(nodes, n - 1, only_direct=False, wait=60)  # type: ignore

        if r < 1:
            raise ValueError("Skipping training, amount of round is less than 1")

        # Start Learning
        nodes[0].set_start_learning(rounds=r, epochs=e)

        # Wait and check
        wait_to_finish(nodes, timeout=60 * 60)  # 1 hour
        
        print("\n✅ Training completed!\n")
        
        # Collect metrics for all rounds
        print("📊 Collecting final metrics...")
        
        # Debug: Let's see what's actually in the logs
        local_logs = logger.get_local_logs()
        global_logs = logger.get_global_logs()
        
        print(f"Debug - Local logs keys: {list(local_logs.keys()) if local_logs else 'None'}")
        print(f"Debug - Global logs keys: {list(global_logs.keys()) if global_logs else 'None'}")
        
        if local_logs:
            exp_id = list(local_logs.keys())[0]
            print(f"Debug - Rounds in local logs: {list(local_logs[exp_id].keys())}")
        
        if global_logs:
            exp_id = list(global_logs.keys())[0]
            print(f"Debug - Nodes in global logs: {list(global_logs[exp_id].keys())}")
            # Show structure of one node
            if global_logs[exp_id]:
                node_name = list(global_logs[exp_id].keys())[0]
                node_data = global_logs[exp_id][node_name]
                print(f"Debug - Metrics for {node_name}: {list(node_data.keys())}")
                if 'test_metric' in node_data:
                    print(f"Debug - test_metric has {len(node_data['test_metric'])} entries")
        
        for round_num in range(r):
            track_round_metrics(results_tracker, round_num, nodes)
        
        # Print comprehensive summary
        execution_time = time.time() - start_time if measure_time else 0
        print_results_summary(
            results_tracker,
            n, r, e,
            topology,
            framework,
            execution_time
        )

        # Local Logs - Optional plotting
        if show_metrics:
            print("\n📊 Generating detailed metric plots...")
            local_logs = logger.get_local_logs()
            if local_logs != {}:
                logs_l = list(local_logs.items())[0][1]
                #  Plot experiment metrics
                for round_num, round_metrics in logs_l.items():
                    for node_name, node_metrics in round_metrics.items():
                        for metric, values in node_metrics.items():
                            x, y = zip(*values)
                            plt.plot(x, y, label=metric)
                            # Add a red point to the last data point
                            plt.scatter(x[-1], y[-1], color="red")
                            plt.title(f"Round {round_num} - {node_name}")
                            plt.xlabel("Epoch")
                            plt.ylabel(metric)
                            plt.legend()
                            plt.show()

            # Global Logs
            global_logs = logger.get_global_logs()
            if global_logs != {}:
                logs_g = list(global_logs.items())[0][1]  # Accessing the nested dictionary directly
                # Plot experiment metrics
                for node_name, node_metrics in logs_g.items():
                    for metric, values in node_metrics.items():
                        x, y = zip(*values)
                        plt.plot(x, y, label=metric)
                        # Add a red point to the last data point
                        plt.scatter(x[-1], y[-1], color="red")
                        plt.title(f"{node_name} - {metric}")
                        plt.xlabel("Epoch")
                        plt.ylabel(metric)
                        plt.legend()
                        plt.show()
    except Exception as e:
        raise e
    finally:
        # Stop Nodes
        for node in nodes:
            node.stop()


if __name__ == "__main__":
    # Parse args
    args = __parse_args()

    set_standalone_settings(disable_ray=args.disable_ray)

    if args.profiling:
        import os  # noqa: I001
        import yappi  # type: ignore

        # Start profiler
        yappi.start()

    # Set logger
    if args.token != "":
        logger.connect_web("http://localhost:3000/api/v1", args.token)

    # Launch experiment
    try:
        mnist(
            args.nodes,
            args.rounds,
            args.epochs,
            show_metrics=args.show_metrics,
            measure_time=args.measure_time,
            protocol=args.protocol,
            framework=args.framework,
            aggregator=args.aggregator,
            reduced_dataset=args.reduced_dataset,
            topology=args.topology,
        )
    finally:
        if args.profiling:
            # Stop profiler
            yappi.stop()
            # Save stats
            profile_dir = os.path.join("profile", "mnist", str(uuid.uuid4()))
            os.makedirs(profile_dir, exist_ok=True)
            for thread in yappi.get_thread_stats():
                yappi.get_func_stats(ctx_id=thread.id).save(f"{profile_dir}/{thread.name}-{thread.id}.pstat", type="pstat")