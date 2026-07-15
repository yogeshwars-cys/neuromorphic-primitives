
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from neuromorphic import Network


def create_large_network(n_neurons: int = 1000, connectivity: float = 0.1):
    """Create a large network for benchmarking."""
    net = Network()
    # Use add_neurons for fast construction
    net.add_neurons(list(range(n_neurons)))
    
    # Add sparse connections (batch!)
    connections = []
    for i in range(n_neurons):
        # Sample random postsynaptic neurons
        num_post = int(n_neurons * connectivity)
        post_indices = np.random.choice(n_neurons, num_post, replace=False)
        for j in post_indices:
            if i != j:
                connections.append((i, j, 0.3))
    net.connect_batch(connections)
    
    return net


def test_network_benchmark():
    """Benchmark network simulation performance."""
    n_neurons = 1000
    duration = 100
    net = create_large_network(n_neurons)
    
    def ext_fn(t):
        if t % 10 == 0:
            return list(range(0, n_neurons, 10))
        return {}
    
    import time
    start = time.time()
    net.simulate(duration, external_fn=ext_fn)
    elapsed = time.time() - start
    print(f"\nBenchmark results: {duration} steps, {n_neurons} neurons, {net.W.nnz} synapses")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Throughput: {duration/elapsed:.2f} steps/s, {n_neurons*duration/elapsed:.2f} neuron-steps/s")
    
    return elapsed


if __name__ == "__main__":
    test_network_benchmark()
