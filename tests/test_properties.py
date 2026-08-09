
import numpy as np
import hypothesis as hyp
import hypothesis.strategies as st
from neuromorphic import Network, NeuronConfig


@hyp.given(
    n_neurons=st.integers(min_value=1, max_value=50),
    duration=st.integers(min_value=1, max_value=100),
    hyperparams=st.fixed_dictionaries({
        'R_min': st.floats(min_value=0.1, max_value=10.0),
        'lambda_leak': st.floats(min_value=0.01, max_value=1.0),
        'T_base': st.floats(min_value=0.1, max_value=10.0),
        'beta': st.floats(min_value=0.01, max_value=2.0),
        'gamma': st.floats(min_value=0.001, max_value=0.1),
        'alpha_plus': st.floats(min_value=0.01, max_value=1.0),
        'alpha_minus': st.floats(min_value=0.01, max_value=1.0),
        'tau_stdp': st.floats(min_value=1.0, max_value=100.0),
        'W_max': st.floats(min_value=0.1, max_value=10.0),
    })
)
def test_network_invariants(n_neurons, duration, hyperparams):
    """Test that network invariants hold for all valid inputs."""
    net = Network()
    
    # Add neurons
    for i in range(n_neurons):
        net.add_neuron(i, **hyperparams)
    
    # Add some random connections
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i != j and np.random.rand() < 0.2:
                net.connect(i, j, 0.3)
    
    # Simulate
    def ext_fn(t):
        if t % 5 == 0:
            return list(range(n_neurons))
        return {}
    
    net.simulate(duration, external_fn=ext_fn)
    
    # Check invariants
    # 1. Charge is non-negative, finite, and not NaN
    assert np.all(net.state.charge >= 0)
    assert np.all(np.isfinite(net.state.charge))
    assert not np.any(np.isnan(net.state.charge))
    
    # 2. H is between 0 and 1
    assert np.all(net.state.H >= 0)
    assert np.all(net.state.H <= 1)
    
    # 3. Weights are between 0 and W_max
    for pre, post, w in net.weights():
        assert w >= 0
        assert w <= hyperparams['W_max'] + 1e-12
    
    # 4. U is non-negative
    assert np.all(net.state.U >= 0)

