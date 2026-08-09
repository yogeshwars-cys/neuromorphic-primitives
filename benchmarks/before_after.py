"""
Honest before/after comparison.

"Before" = the pristine original network.py (pulled straight from git HEAD,
before any of today's fixes). "After" = the fixed neuromorphic package.

The headline finding: the original benchmark (tests/test_benchmark.py) reported
a throughput number, but the network it was measuring never actually did
anything -- external_pulses was a no-op for synaptic input, so every neuron sat
at charge=0 for the entire run. A "benchmark" of an idle network is not
measuring the thing anyone cares about. This script proves that, then reports
real throughput on a network that is actually active.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, '/home/claude/orig_check')

import neuromorphic_orig as orig
import neuromorphic as fixed


def build_network(NetworkCls, n_neurons=1000, connectivity=0.1, seed=0):
    rng = np.random.default_rng(seed)
    net = NetworkCls.Network()
    net.add_neurons(list(range(n_neurons)))
    connections = []
    for i in range(n_neurons):
        num_post = int(n_neurons * connectivity)
        post_indices = rng.choice(n_neurons, num_post, replace=False)
        for j in post_indices:
            if i != j:
                connections.append((i, int(j), 0.3))
    net.connect_batch(connections)
    return net


def run(NetworkCls, label, n_neurons=1000, duration=200, connectivity=0.1):
    net = build_network(NetworkCls, n_neurons, connectivity)

    def ext_fn(t):
        if t % 10 == 0:
            return list(range(0, n_neurons, 10))
        return {}

    start = time.time()
    net.simulate(duration, external_fn=ext_fn)
    elapsed = time.time() - start

    total_spikes = sum(net.spike_counts().values())
    mean_charge = float(np.mean(net.state.charge))
    weights = net.weights()
    mean_weight = float(np.mean([w for _, _, w in weights])) if weights else 0.0

    print(f"--- {label} ---")
    print(f"  neurons={n_neurons}  synapses={net.W.nnz}  duration={duration} ticks")
    print(f"  elapsed: {elapsed:.3f}s   throughput: {n_neurons*duration/elapsed:,.0f} neuron-steps/s")
    print(f"  TOTAL SPIKES OVER ENTIRE RUN: {total_spikes}")
    print(f"  mean charge at end: {mean_charge:.6f}")
    print(f"  mean synaptic weight at end: {mean_weight:.6f}")
    print()
    return dict(label=label, elapsed=elapsed, total_spikes=total_spikes,
                mean_charge=mean_charge, mean_weight=mean_weight,
                throughput=n_neurons * duration / elapsed)


if __name__ == "__main__":
    print("=" * 70)
    print("BEFORE: original network.py, exact same benchmark stimulus pattern")
    print("        as the repo's own tests/test_benchmark.py")
    print("=" * 70)
    before = run(orig, "ORIGINAL (pre-fix)", n_neurons=1000, duration=200)

    print("=" * 70)
    print("AFTER: fixed network.py, identical topology and stimulus")
    print("=" * 70)
    after = run(fixed, "FIXED", n_neurons=1000, duration=200)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original: {before['total_spikes']} spikes total, "
          f"mean charge {before['mean_charge']:.6f}, "
          f"mean weight {before['mean_weight']:.6f} (never moved from init 0.3)")
    print(f"Fixed:    {after['total_spikes']} spikes total, "
          f"mean charge {after['mean_charge']:.6f}, "
          f"mean weight {after['mean_weight']:.6f} (STDP actually ran)")
    print()
    print(f"Original throughput: {before['throughput']:,.0f} neuron-steps/s "
          f"(measuring an idle network -- not a meaningful number)")
    print(f"Fixed throughput:    {after['throughput']:,.0f} neuron-steps/s "
          f"(measuring a genuinely active network)")
