
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
        
        # Sparse weight matrix: W[post_idx, pre_idx] = weight (CSR)
        self.W = sp.csr_matrix((0, 0), dtype=np.float64)
        
        # Ordered list of layers: [ [id, id, ...], [id, id, ...] ]
        self.layers = []
        
        # Spike history: {neuron_id: [spike_at_t0, spike_at_t1, ...]}
        self.spike_history = {}
        
        # Last spike vector (for propagation)
        self.prev_spike_vec = np.zeros(0, dtype=np.int32)
        
        # Backward compatibility: dict of Neuron wrappers
        self.neurons = {}
    
    def __len__(self):
        return len(self.idx_to_id)

    # ------------------------------------------------------------------
    # Building the network
    # ------------------------------------------------------------------

    def add_neuron(self, neuron_id, **kwargs):
        """
        Add a single neuron to the network.
        kwargs are forwarded to NeuronState hyperparameters.
        """
        return self.add_neurons([neuron_id], **kwargs)[0]
    
    def add_neurons(self, neuron_ids, **kwargs):
        """
        Add multiple neurons at once for O(N) construction time.
        kwargs are forwarded to NeuronState hyperparameters.
        """
        # Validate input
        for nid in neuron_ids:
            if nid in self.id_to_idx:
                raise ValueError(f"Neuron {nid} already exists in network.")
        
        # Grow structures
        old_size = len(self.idx_to_id)
        new_size = old_size + len(neuron_ids)
        for i, nid in enumerate(neuron_ids):
            self.id_to_idx[nid] = old_size + i
            self.idx_to_id.append(nid)
        
        # Resize state
        self.state.resize(new_size)
        
        # Apply custom hyperparameters if provided
        for i, nid in enumerate(neuron_ids):
            idx = old_size + i
            if 'R_min' in kwargs: self.state.R_min[idx] = kwargs['R_min']
            if 'lambda_leak' in kwargs: self.state.lambda_leak[idx] = kwargs['lambda_leak']
            if 'T_base' in kwargs: self.state.T_base[idx] = kwargs['T_base']
            if 'beta' in kwargs: self.state.beta[idx] = kwargs['beta']
            if 'gamma' in kwargs: self.state.gamma[idx] = kwargs['gamma']
            if 'alpha_plus' in kwargs: self.state.alpha_plus[idx] = kwargs['alpha_plus']
            if 'alpha_minus' in kwargs: self.state.alpha_minus[idx] = kwargs['alpha_minus']
            if 'tau_stdp' in kwargs: self.state.tau_stdp[idx] = kwargs['tau_stdp']
            if 'W_max' in kwargs: self.state.W_max[idx] = kwargs['W_max']
        
        # Resize weight matrix (efficiently via COO)
        if old_size > 0:
            old_coo = self.W.tocoo()
            self.W = sp.csr_matrix(
                (old_coo.data, (old_coo.row, old_coo.col)),
                shape=(new_size, new_size)
            )
        else:
            self.W = sp.csr_matrix((new_size, new_size), dtype=np.float64)
        
        # Resize prev_spike_vec
        self.prev_spike_vec = np.zeros(new_size, dtype=np.int32)
        
        # Initialize spike history and Neuron wrappers (backward compatibility)
        created_neurons = []
        for nid in neuron_ids:
            self.spike_history[nid] = []
            neuron = Neuron(nid, **kwargs)
            self.neurons[nid] = neuron
            created_neurons.append(neuron)
        
        return created_neurons

    def add_layer(self, neuron_ids, **kwargs):
        """
        Add a group of neurons as a layer.
        Returns the list of Neuron objects created.
        """
        layer = self.add_neurons(neuron_ids, **kwargs)
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
        self.connect_batch([(pre_id, post_id, weight)])

    def connect_batch(self, connections):
        """
        Add multiple connections at once: list of (pre_id, post_id, weight) tuples.
        """
        # Convert ids to indices
        rows = []
        cols = []
        data = []
        for pre_id, post_id, weight in connections:
            pre_idx = self.id_to_idx[pre_id]
            post_idx = self.id_to_idx[post_id]
            rows.append(post_idx)
            cols.append(pre_idx)
            data.append(weight)
        
        # Batch update using COO
        if not rows:
            return
        new_coo = sp.coo_matrix((data, (rows, cols)), shape=self.W.shape)
        self.W = self.W.tocoo() + new_coo
        self.W = self.W.tocsr()
        self.W.eliminate_zeros()
        
        # Also initialize pre_spike_clocks entries to -9999
        new_clocks_coo = sp.coo_matrix(
            (np.full(len(data), -9999.0), (rows, cols)),
            shape=self.W.shape
        )
        self.state.pre_spike_clocks = self.state.pre_spike_clocks.tocoo() + new_clocks_coo
        self.state.pre_spike_clocks = self.state.pre_spike_clocks.tocsr()
        self.state.pre_spike_clocks.eliminate_zeros()

    def connect_layers(self, from_layer_ids, to_layer_ids, weight=0.3):
        """
        Fully connect every neuron in from_layer to every neuron in to_layer.
        """
        # Collect all connections and batch add them
        connections = []
        for pre_id in from_layer_ids:
            for post_id in to_layer_ids:
                connections.append((pre_id, post_id, weight))
        self.connect_batch(connections)

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
        
        # 0. Update incoming firing clocks (vectorized, no Python loops!)
        self._update_pre_spike_clocks(current_time, external_pulses)
        
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
        self._apply_stdp_learning_batch(current_time, spiked_indices)
        
        # Update spike history
        tick_results = {}
        for idx, nid in enumerate(self.idx_to_id):
            spike = new_spike_vec[idx]
            self.spike_history[nid].append(spike)
            tick_results[nid] = spike
        
        self.prev_spike_vec = new_spike_vec
        return tick_results
    
    def _update_pre_spike_clocks(self, current_time, external_pulses):
        """
        Vectorized update of pre_spike_clocks:
          1. From external_pulses
          2. From previous spike vector (sparse matrix-based, no Python loops)
        """
        N = len(self.idx_to_id)
        
        # 1. Update from external_pulses
        if external_pulses:
            ext_rows = []
            ext_cols = []
            for post_id, pre_ids in external_pulses.items():
                if post_id not in self.id_to_idx:
                    continue
                post_idx = self.id_to_idx[post_id]
                for pre_id in pre_ids:
                    if pre_id not in self.id_to_idx:
                        continue
                    pre_idx = self.id_to_idx[pre_id]
                    ext_rows.append(post_idx)
                    ext_cols.append(pre_idx)
            
            if ext_rows:
                ext_updates = sp.csr_matrix(
                    (np.full(len(ext_rows), current_time), (ext_rows, ext_cols)),
                    shape=(N, N)
                )
                # Use maximum to keep the latest time in case of duplicates
                self.state.pre_spike_clocks = self.state.pre_spike_clocks.maximum(ext_updates)
        
        # 2. Update from previous spike vector: W @ spike_vec is O(N + nnz), no Python loops!
        # Get all pre indices that spiked
        spiked_pre_indices = np.where(self.prev_spike_vec == 1)[0]
        if len(spiked_pre_indices) == 0:
            return
        
        # For each spiked pre, update all post neurons connected to it
        # Use CSC for fast column slicing
        W_csc = self.W.tocsc()
        for pre_idx in spiked_pre_indices:
            col = W_csc[:, pre_idx]
            if col.nnz > 0:
                post_indices = col.indices
                # Build update matrix
                update_rows = post_indices
                update_cols = np.full_like(post_indices, pre_idx)
                update_data = np.full_like(post_indices, current_time, dtype=np.float64)
                update = sp.csr_matrix(
                    (update_data, (update_rows, update_cols)),
                    shape=(N, N)
                )
                self.state.pre_spike_clocks = self.state.pre_spike_clocks.maximum(update)

    def _apply_stdp_learning_batch(self, current_time, spiked_post_indices):
        """
        Apply STDP updates to all spiking post neurons in batch.
        """
        for post_idx in spiked_post_indices:
            self._apply_stdp_learning_single(current_time, post_idx)

    def _apply_stdp_learning_single(self, current_time, post_idx):
        """
        Apply STDP updates for a single post-synaptic neuron.
        """
        t_post = current_time
        post_W = self.W.getrow(post_idx)
        pre_indices = post_W.indices
        if len(pre_indices) == 0:
            return
        
        w_old = post_W.data
        # Get pre_spike_clocks for this post neuron's incoming synapses
        pre_spikes_row = self.state.pre_spike_clocks.getrow(post_idx)
        t_pre = pre_spikes_row.data
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
        
        # Update weights in batch
        w_new = np.clip(w_old + dw, 0.0, W_max)
        # Update W with new weights
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

