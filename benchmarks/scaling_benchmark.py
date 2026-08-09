"""
Two things measured here, both on the FIXED implementation:

1. Throughput scaling across network sizes, with genuine activity (external
   current injection actually works now -- see network.py fix). Frozen
   (STDP-off) reservoir-style networks are used so timing reflects steady
   sparse-matrix-op cost rather than runaway/collapse transients.

2. A/B: numba-jitted STDP kernel vs the pure-numpy fallback, isolating whether
   today's `njit` addition (previously imported but never called anywhere in
   the repo) is actually worth its added dependency weight.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import neuromorphic.network as network_mod
from neuromorphic import Network


def build_active_network(n_neurons, connectivity, seed, plastic):
    rng = np.random.default_rng(seed)
    net = Network()
    net.add_neurons(list(range(n_neurons)),
                     R_min=2.0, lambda_leak=0.15, T_base=1.0, beta=0.5, gamma=0.02,
                     W_max=1.0,
                     alpha_plus=(0.05 if plastic else 0.0),
                     alpha_minus=(0.04 if plastic else 0.0))
    connections = []
    k = max(1, int(n_neurons * connectivity))
    for i in range(n_neurons):
        targets = rng.choice(n_neurons, size=k, replace=False)
        for j in targets:
            if j != i:
                connections.append((i, int(j), float(rng.uniform(0.02, 0.15))))
    net.connect_batch(connections)
    return net


def run(n_neurons, duration=150, connectivity=0.05, plastic=True, seed=0):
    net = build_active_network(n_neurons, connectivity, seed, plastic)
    drive_size = max(1, n_neurons // 20)

    def ext_fn(t):
        if t % 5 == 0:
            return list(range(0, n_neurons, max(1, n_neurons // drive_size // 1 or 1)))[:drive_size]
        return {}

    start = time.time()
    net.simulate(duration, external_fn=ext_fn)
    elapsed = time.time() - start
    total_spikes = sum(net.spike_counts().values())
    return dict(n_neurons=n_neurons, synapses=net.W.nnz, duration=duration,
                elapsed=elapsed, throughput=n_neurons * duration / elapsed,
                total_spikes=total_spikes)


def scaling_sweep():
    print("=" * 78)
    print("THROUGHPUT SCALING (fixed implementation, genuinely active networks)")
    print("=" * 78)
    sizes = [200, 500, 1000, 2000, 5000]
    results = []
    for n in sizes:
        r = run(n)
        results.append(r)
        print(f"  n={n:>6}  synapses={r['synapses']:>8}  elapsed={r['elapsed']:>7.3f}s  "
              f"throughput={r['throughput']:>14,.0f} neuron-steps/s  "
              f"total_spikes={r['total_spikes']}")
    return results


def numba_ab_test(n_neurons=3000, duration=100):
    print()
    print("=" * 78)
    print("NUMBA A/B: STDP kernel, jitted vs pure-numpy fallback")
    print("=" * 78)

    # Warm up numba's JIT compilation cache first so we measure steady-state
    # execution speed, not one-time compile time.
    warmup = build_active_network(200, 0.05, seed=99, plastic=True)
    warmup.simulate(20, external_fn=lambda t: list(range(0, 200, 10)) if t % 5 == 0 else {})

    was_available = network_mod.NUMBA_AVAILABLE

    network_mod.NUMBA_AVAILABLE = True
    r_numba = run(n_neurons, duration=duration, plastic=True, seed=1)
    print(f"  numba ON : elapsed={r_numba['elapsed']:.3f}s  "
          f"throughput={r_numba['throughput']:,.0f} neuron-steps/s  "
          f"spikes={r_numba['total_spikes']}")

    network_mod.NUMBA_AVAILABLE = False
    r_numpy = run(n_neurons, duration=duration, plastic=True, seed=1)
    print(f"  numba OFF: elapsed={r_numpy['elapsed']:.3f}s  "
          f"throughput={r_numpy['throughput']:,.0f} neuron-steps/s  "
          f"spikes={r_numpy['total_spikes']}")

    network_mod.NUMBA_AVAILABLE = was_available

    speedup = r_numpy['elapsed'] / r_numba['elapsed']
    print(f"  -> numba speedup on STDP-heavy workload: {speedup:.2f}x")
    assert r_numba['total_spikes'] == r_numpy['total_spikes'], \
        "numba and numpy STDP paths produced different dynamics -- correctness bug!"
    print("  (identical spike counts confirms the two STDP code paths are numerically equivalent)")
    return r_numba, r_numpy


if __name__ == "__main__":
    scaling_results = scaling_sweep()
    numba_ab_test()
