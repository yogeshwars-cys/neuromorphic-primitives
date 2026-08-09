
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from neuromorphic_original import Neuron as OldNeuron
from neuromorphic_original import Network as OldNetwork
from neuromorphic import Neuron as NewNeuron
from neuromorphic import Network as NewNetwork


def test_neuron_equivalence():
    """Test that single neuron behavior is identical between old and new."""
    old_n = OldNeuron(1, R_min=2.0, lambda_leak=0.12, T_base=1.0)
    new_n = NewNeuron(1, R_min=2.0, lambda_leak=0.12, T_base=1.0)
    
    old_n.register_synapse(2, 0.5)
    new_n.register_synapse(2, 0.5)
    
    # Run same sequence of ticks on both
    old_spikes = []
    new_spikes = []
    
    for t in range(100):
        incoming = [2] if (t % 5 == 0) else []
        old_s = old_n.tick(t, incoming)
        new_s = new_n.tick(t, incoming)
        old_spikes.append(old_s)
        new_spikes.append(new_s)
    
    # Check results are identical
    assert old_spikes == new_spikes, f"Spike sequences differ: {old_spikes} vs {new_spikes}"
    assert abs(old_n.charge - new_n.charge) < 1e-12
    assert abs(old_n.H - new_n.H) < 1e-12
    assert abs(old_n.U - new_n.U) < 1e-12
    assert old_n.last_spike_time == new_n.last_spike_time
    assert old_n.incoming_weights == new_n.incoming_weights


def test_network_equivalence():
    """Test that network behavior is identical between old and new implementations."""
    # Create identical networks
    old_net = OldNetwork()
    new_net = NewNetwork()
    
    # Add neurons
    old_net.add_neuron(1)
    old_net.add_neuron(2)
    old_net.add_neuron(3)
    new_net.add_neuron(1)
    new_net.add_neuron(2)
    new_net.add_neuron(3)
    
    # Add connections
    old_net.connect(1, 2, 0.5)
    old_net.connect(2, 3, 0.5)
    new_net.connect(1, 2, 0.5)
    new_net.connect(2, 3, 0.5)
    
    # Define external input function
    def ext_fn(t):
        if t % 3 == 0:
            return [1]
        return {}
    
    # Simulate
    old_hist = old_net.simulate(100, external_fn=ext_fn)
    new_hist = new_net.simulate(100, external_fn=ext_fn)
    
    # Check spike histories are identical
    for nid in [1, 2, 3]:
        assert old_hist[nid] == new_hist[nid], f"Neuron {nid} spike history differs"
    
    # Check weights are identical (within floating point tolerance)
    old_weights = set(old_net.weights())
    new_weights = set(new_net.weights())
    assert len(old_weights) == len(new_weights)
    for (pre_old, post_old, w_old), (pre_new, post_new, w_new) in zip(sorted(old_weights), sorted(new_weights)):
        assert pre_old == pre_new
        assert post_old == post_new
        assert abs(w_old - w_new) < 1e-10


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath('..'))
    test_neuron_equivalence()
    print("Single neuron test passed!")
    test_network_equivalence()
    print("Network test passed!")
    print("All equivalence tests passed!")
