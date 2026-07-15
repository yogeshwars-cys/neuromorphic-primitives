
import numpy as np
from neuromorphic import Network


def create_large_network(n_neurons: int = 1000, connectivity: float = 0.1):
    """Create a large network for benchmarking."""
    net = Network()
    for i in range(n_neurons):
        net.add_neuron(i)
    
    # Add sparse connections
    for i in range(n_neurons):
        for j in np.random.choice(n_neurons, int(n_neurons * connectivity), replace=False):
            if i != j:
                net.connect(i, j, 0.3)
    
    return net


def test_network_benchmark(benchmark):
    """Benchmark network simulation performance."""
    n_neurons = 1000
    duration = 100
    net = create_large_network(n_neurons)
    
    def ext_fn(t):
        if t % 10 == 0:
            return list(range(0, n_neurons, 10))
        return {}
    
    def run_simulation():
        net.reset()
        return net.simulate(duration, external_fn=ext_fn)
    
    # Run benchmark
    result = benchmark(run_simulation)
    
    # Check that it worked
    assert result is not None
    assert len(result) == n_neurons


if __name__ == "__main__":
    # Simple manual benchmark
    import time
    n_neurons = 1000
    duration = 100
    print(f"Creating network with {n_neurons} neurons...")
    net = create_large_network(n_neurons)
    print(f"Simulating {duration} steps...")
    start = time.time()
    net.simulate(duration)
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.2f}s, {duration/elapsed:.2f} ticks/s, {n_neurons*duration/elapsed:.2f} neuron-ticks/s")

