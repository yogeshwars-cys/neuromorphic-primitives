
import numpy as np
import scipy.sparse as sp
from .neuron import Neuron, NeuronState


class Network:
    """
    A spiking neural network composed of interconnected Neuron primitives.
    Uses structure-of-arrays (NumPy) and sparse matrices (SciPy) for performance.
    """

    def __init__(self):
        # Mapping from neuron_id to internal index
        self.id_to_idx = {}
        self.idx_to_id = []
        
        # State variables
        self.state = NeuronState(0)
        
        # Sparse weight matrix: W[post_idx, pre_idx] = weight
        self.W = sp.csr_matrix((0, 0), dtype=np.float64)
        
        # Ordered list of layers: [ [id, id, ...], [id, id, ...] ]
        self.layers = []
        
        # Spike history: {neuron_id: [spike_at_t0, spike_at_t1, ...]}
        self.spike_history = {}
        
        # Last spike vector (for propagation)
        self.prev_spike_vec = np.zeros(0, dtype=np.int32)

    # ------------------------------------------------------------------
    # Building the network
    # ------------------------------------------------------------------

    def add_neuron(self, neuron_id, **kwargs):
        """
        Add a single neuron to the network.
        kwargs are forwarded to NeuronState hyperparameters.
        """
        if neuron_id in self.id_to_idx:
            raise ValueError(f"Neuron {neuron_id} already exists in network.")
        
        # Grow our structures
        new_size = len(self.idx_to_id) + 1
        self.id_to_idx[neuron_id] = new_size - 1
        self.idx_to_id.append(neuron_id)
        
        # Resize state and weight matrix
        self.state.resize(new_size)
        
        # Apply custom hyperparameters if provided
        idx = new_size - 1
        if 'R_min' in kwargs: self.state.R_min[idx] = kwargs['R_min']
        if 'lambda_leak' in kwargs: self.state.lambda_leak[idx] = kwargs['lambda_leak']
        if 'T_base' in kwargs: self.state.T_base[idx] = kwargs['T_base']
        if 'beta' in kwargs: self.state.beta[idx] = kwargs['beta']
        if 'gamma' in kwargs: self.state.gamma[idx] = kwargs['gamma']
        if 'alpha_plus' in kwargs: self.state.alpha_plus[idx] = kwargs['alpha_plus']
        if 'alpha_minus' in kwargs: self.state.alpha_minus[idx] = kwargs['alpha_minus']
        if 'tau_stdp' in kwargs: self.state.tau_stdp[idx] = kwargs['tau_stdp']
        if 'W_max' in kwargs: self.state.W_max[idx] = kwargs['W_max']
        
        # Resize weight matrix (CSR format)
        if new_size > 1:
            self.W = sp.vstack([self.W, sp.csr_matrix((1, new_size-1), dtype=np.float64)], format='csr')
            self.W = sp.hstack([self.W, sp.csr_matrix((new_size, 1), dtype=np.float64)], format='csr')
        else:
            self.W = sp.csr_matrix((1, 1), dtype=np.float64)
        
        # Resize prev_spike_vec
        self.prev_spike_vec = np.zeros(new_size, dtype=np.int32)
        
        # Initialize spike history
        self.spike_history[neuron_id] = []
        
        # Return a backward-compatibility Neuron wrapper
        neuron = Neuron(neuron_id, **kwargs)
        return neuron

    def add_layer(self, neuron_ids, **kwargs):
        """
        Add a group of neurons as a layer.
        Returns the list of Neuron objects created.
        """
        layer = []
        for nid in neuron_ids:
            layer.append(self.add_neuron(nid, **kwargs))
        self.layers.append(neuron_ids)
        return layer

    def connect(self, pre_id, post_id, weight=0.3):
        """
        Create a directed synapse: pre_id --> post_id.
        """
        if pre_id not in self.id_to_idx:
            raise ValueError(f"Neuron {pre_id} not in network.")
        if post_id not in self.id_to_idx:
            raise ValueError(f"Neuron {post_id} not in network.")
        
        pre_idx = self.id_to_idx[pre_id]
        post_idx = self.id_to_idx[post_id]
        
        # Update sparse matrix
        self.W[post_idx, pre_idx] = weight

    def connect_layers(self, from_layer_ids, to_layer_ids, weight=0.3):
        """
        Fully connect every neuron in from_layer to every neuron in to_layer.
        """
        # Collect indices
        from_indices = [self.id_to_idx[nid] for nid in from_layer_ids]
        to_indices = [self.id_to_idx[nid] for nid in to_layer_ids]
        
        # Create COO matrix for batch insertion
        rows = []
        cols = []
        data = []
        for to_idx in to_indices:
            for from_idx in from_indices:
                rows.append(to_idx)
                cols.append(from_idx)
                data.append(weight)
        
        # Build COO and add to existing W
        if rows:
            coo = sp.coo_matrix((data, (rows, cols)), shape=self.W.shape)
            self.W = self.W.tocsr() + coo.tocsr()
            self.W.eliminate_zeros()

    # ------------------------------------------------------------------
    # Running the simulation
    # ------------------------------------------------------------------

    def tick(self, current_time, external_pulses=None):
        """
        Advance all neurons by one tick (1ms).
        """
        N = len(self.idx_to_id)
        if N == 0:
            return {}
        
        if external_pulses is None:
            external_pulses = {}
        
        # Normalize external_pulses to dict form
        if isinstance(external_pulses, (list, set)):
            external_pulses = {nid: [] for nid in external_pulses}
        
        # 0. Update incoming firing clocks
        for nid, pre_ids in external_pulses.items():
            if nid not in self.id_to_idx:
                continue
            post_idx = self.id_to_idx[nid]
            for pre_id in pre_ids:
                if pre_id not in self.id_to_idx:
                    continue
                pre_idx = self.id_to_idx[pre_id]
                self.state.pre_spike_clocks[post_idx, pre_idx] = current_time
        
        # Also update from previous spike vec
        for pre_idx in np.where(self.prev_spike_vec == 1)[0]:
            for post_idx in range(N):
                if self.W[post_idx, pre_idx] > 0:
                    self.state.pre_spike_clocks[post_idx, pre_idx] = current_time
        
        # 1. Calculate adaptive refractory period
        num_connections = np.array(self.W.getnnz(axis=1))  # per-neuron incoming connections
        num_connections = np.maximum(num_connections, 1)
        R_dynamic = self.state.R_min * (1.0 + 5.0 * (self.state.U / num_connections))
        is_refractory = (current_time - self.state.last_spike_time) < R_dynamic
        
        # 2. Update stress thermostat
        did_spike_last_tick = (current_time - self.state.last_spike_time) == 1.0
        self.state.H = self.state.H + self.state.gamma * (did_spike_last_tick - self.state.H)
        self.state.H = np.clip(self.state.H, 0.0, 1.0)
        
        # 3. Gather synaptic inputs: sparse matrix multiply
        input_energy = self.W @ self.prev_spike_vec
        
        # 4-6. Process each neuron's state
        new_spike_vec = np.zeros(N, dtype=np.int32)
        
        # Process refractory neurons
        refrac_mask = is_refractory
        reduced_leak = self.state.charge[refrac_mask] * self.state.lambda_leak[refrac_mask] * self.state.H[refrac_mask] * 0.5
        self.state.charge[refrac_mask] = np.maximum(0.0, self.state.charge[refrac_mask] + input_energy[refrac_mask] - reduced_leak)
        self.state.U[refrac_mask] *= 0.95
        
        # Process non-refractory neurons
        non_refrac_mask = ~is_refractory
        dynamic_leak = self.state.charge[non_refrac_mask] * self.state.lambda_leak[non_refrac_mask] * self.state.H[non_refrac_mask]
        raw_charge = self.state.charge[non_refrac_mask] + input_energy[non_refrac_mask] - dynamic_leak
        dynamic_threshold = self.state.T_base[non_refrac_mask] + (self.state.beta[non_refrac_mask] * self.state.H[non_refrac_mask])
        
        spiked_mask = raw_charge >= dynamic_threshold
        not_spiked_mask = ~spiked_mask
        
        # Update spiked neurons
        spiked_indices = np.where(non_refrac_mask)[0][spiked_mask]
        self.state.charge[spiked_indices] = np.maximum(0.0, raw_charge[spiked_mask] - dynamic_threshold[spiked_mask])
        self.state.last_spike_time[spiked_indices] = current_time
        self.state.U[spiked_indices] = 0.95 * self.state.U[spiked_indices] + 0.05 * 1.0
        new_spike_vec[spiked_indices] = 1
        
        # Update non-spiked neurons
        not_spiked_indices = np.where(non_refrac_mask)[0][not_spiked_mask]
        self.state.charge[not_spiked_indices] = np.maximum(0.0, raw_charge[not_spiked_mask])
        self.state.U[not_spiked_indices] *= 0.95
        
        # 7. Apply STDP learning to spiking neurons
        for post_idx in spiked_indices:
            self._apply_stdp_learning_single(current_time, post_idx)
        
        # Update spike history
        tick_results = {}
        for idx, nid in enumerate(self.idx_to_id):
            spike = new_spike_vec[idx]
            self.spike_history[nid].append(spike)
            tick_results[nid] = spike
        
        self.prev_spike_vec = new_spike_vec
        return tick_results

    def _apply_stdp_learning_single(self, current_time, post_idx):
        """Apply STDP updates for a single post-synaptic neuron."""
        t_post = current_time
        post_W = self.W.getrow(post_idx)
        pre_indices = post_W.indices
        if len(pre_indices) == 0:
            return
        
        w_old = post_W.data
        t_pre = self.state.pre_spike_clocks[post_idx, pre_indices]
        dt_gap = t_post - t_pre
        
        # Compute dw for all synapses
        alpha_plus = self.state.alpha_plus[post_idx]
        alpha_minus = self.state.alpha_minus[post_idx]
        tau_stdp = self.state.tau_stdp[post_idx]
        H = self.state.H[post_idx]
        W_max = self.state.W_max[post_idx]
        
        dw = np.zeros_like(w_old)
        
        # Causal case (dt_gap > 0)
        causal_mask = dt_gap > 0
        dw[causal_mask] = (alpha_plus / (1.0 + H)) * np.exp(-dt_gap[causal_mask] / tau_stdp)
        
        # Anti-causal case (dt_gap < 0)
        anti_causal_mask = dt_gap < 0
        dw[anti_causal_mask] = -alpha_minus * np.exp(dt_gap[anti_causal_mask] / tau_stdp)
        
        # Zero case (dt_gap == 0)
        zero_mask = dt_gap == 0
        dw[zero_mask] = (alpha_plus / (1.0 + H))
        
        # Update weights
        w_new = np.clip(w_old + dw, 0.0, W_max)
        self.W[post_idx, pre_indices] = w_new

    def simulate(self, duration_ms, external_fn=None):
        """
        Run the network for `duration_ms` ticks.
        """
        for t in range(1, duration_ms + 1):
            ext = external_fn(t) if external_fn else {}
            self.tick(t, ext)
        return self.spike_history

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def spike_counts(self):
        """Return total spike count per neuron."""
        return {nid: sum(hist) for nid, hist in self.spike_history.items()}

    def weights(self):
        """Return all current synaptic weights as (pre, post, weight) tuples."""
        edges = []
        coo = self.W.tocoo()
        for post_idx, pre_idx, w in zip(coo.row, coo.col, coo.data):
            if abs(w) > 1e-15:  # ignore zero weights
                pre_id = self.idx_to_id[pre_idx]
                post_id = self.idx_to_id[post_idx]
                edges.append((pre_id, post_id, round(w, 5)))
        return edges

    def reset(self):
        """Clear spike history without destroying topology or learned weights."""
        for nid in self.spike_history:
            self.spike_history[nid] = []
        self.prev_spike_vec = np.zeros_like(self.prev_spike_vec)
        self.state.charge[:] = 0.0
        self.state.H[:] = 0.2
        self.state.U[:] = 0.0
        self.state.last_spike_time[:] = -9999.0

    def __repr__(self):
        return (f"<Network neurons={len(self.idx_to_id)} "
                f"synapses={self.W.nnz} "
                f"layers={len(self.layers)}>")

