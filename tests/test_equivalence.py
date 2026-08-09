"""
Equivalence tests: the scalar `Neuron.tick()` path (kept in neuron.py purely as a
readable reference / backward-compat implementation) must produce bit-for-bit
identical dynamics to the vectorized `Network.tick()` path (NumPy + SciPy sparse),
since the latter is a performance rewrite of the exact same equations.

NOTE: this file used to import a `neuromorphic_original` module that does not
exist anywhere in the repo or its git history -- the whole test suite failed to
even collect (pytest aborts collection entirely on an import error in ANY test
file), so this check, test_benchmark, and test_properties were silently never
running. Fixed to compare against the scalar Neuron class that already lives in
neuron.py as the reference implementation.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from neuromorphic import Neuron, Network


def test_neuron_equivalence():
    """A standalone scalar Neuron must match a 2-neuron vectorized Network."""
    scalar_n = Neuron(1, R_min=2.0, lambda_leak=0.12, T_base=1.0)
    scalar_n.register_synapse(2, 0.5)

    net = Network()
    net.add_neuron(1, R_min=2.0, lambda_leak=0.12, T_base=1.0)
    net.add_neuron(2, R_min=2.0, lambda_leak=0.12, T_base=1.0)
    net.connect(2, 1, 0.5)  # pre=2 -> post=1, matches scalar_n.register_synapse(2, 0.5)

    scalar_spikes = []
    net_spikes = []

    for t in range(100):
        incoming = [2] if (t % 5 == 0) else []
        scalar_spikes.append(scalar_n.tick(t, incoming))

        ext = {1: [2]} if (t % 5 == 0) else {}
        result = net.tick(t, ext)
        net_spikes.append(int(result[1]))

    assert scalar_spikes == net_spikes, f"Spike sequences differ: {scalar_spikes} vs {net_spikes}"

    idx1 = net.id_to_idx[1]
    assert abs(scalar_n.charge - net.state.charge[idx1]) < 1e-9
    assert abs(scalar_n.H - net.state.H[idx1]) < 1e-9
    assert abs(scalar_n.U - net.state.U[idx1]) < 1e-9
    assert scalar_n.last_spike_time == net.state.last_spike_time[idx1]

    w_2_to_1 = [w for pre, post, w in net.weights() if pre == 2 and post == 1][0]
    assert abs(scalar_n.incoming_weights[2] - w_2_to_1) < 1e-9


def test_network_self_consistency():
    """Running the same topology twice from a fresh Network must be deterministic."""
    def build_and_run():
        net = Network()
        net.add_neuron(1)
        net.add_neuron(2)
        net.add_neuron(3)
        net.connect(1, 2, 0.5)
        net.connect(2, 3, 0.5)

        def ext_fn(t):
            return [1] if t % 3 == 0 else {}

        return net.simulate(100, external_fn=ext_fn), net.weights()

    hist_a, weights_a = build_and_run()
    hist_b, weights_b = build_and_run()

    for nid in [1, 2, 3]:
        assert hist_a[nid] == hist_b[nid], f"Neuron {nid} spike history not deterministic"

    assert sorted(weights_a) == sorted(weights_b)


def test_chain_matches_scalar_neurons():
    """
    An externally-driven 'src' -> 1 -> 2 -> 3 chain must match between the scalar
    reference and Network. This specifically exercises the external_pulses fix in
    Network.tick(): 'src' is a registered synapse source (not a real dynamical
    neuron -- it never ticks) driven purely via external_pulses each tick, exactly
    the pattern a sensory/input layer would use.
    """
    scalar_neurons = {nid: Neuron(nid) for nid in [1, 2, 3]}
    scalar_neurons[1].register_synapse('src', 0.6)
    scalar_neurons[2].register_synapse(1, 0.5)
    scalar_neurons[3].register_synapse(2, 0.5)

    net = Network()
    net.add_neuron('src')  # placeholder driver: added so it has a valid W column,
    net.add_neuron(1)      # but never ticked -- purely driven via external_pulses
    net.add_neuron(2)
    net.add_neuron(3)
    net.connect('src', 1, 0.6)
    net.connect(1, 2, 0.5)
    net.connect(2, 3, 0.5)

    scalar_prev_spike = {1: 0, 2: 0}
    for t in range(1, 151):
        drive = (t % 4 == 0)

        s1 = scalar_neurons[1].tick(t, ['src'] if drive else [])
        s2 = scalar_neurons[2].tick(t, [1] if scalar_prev_spike[1] else [])
        s3 = scalar_neurons[3].tick(t, [2] if scalar_prev_spike[2] else [])
        scalar_prev_spike = {1: s1, 2: s2}

        ext_dict = {1: ['src']} if drive else {}
        net_result = net.tick(t, ext_dict)

        assert s1 == net_result[1], f"t={t}: neuron 1 mismatch {s1} vs {net_result[1]}"
        assert s2 == net_result[2], f"t={t}: neuron 2 mismatch {s2} vs {net_result[2]}"
        assert s3 == net_result[3], f"t={t}: neuron 3 mismatch {s3} vs {net_result[3]}"


if __name__ == "__main__":
    test_neuron_equivalence()
    print("Single neuron equivalence test passed!")
    test_network_self_consistency()
    print("Determinism test passed!")
    test_chain_matches_scalar_neurons()
    print("Chain equivalence test passed!")
    print("All equivalence tests passed!")
