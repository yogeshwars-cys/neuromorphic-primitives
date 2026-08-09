
import numpy as np
import scipy.sparse as sp
from .neuron import Neuron, NeuronState

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        # no-op fallback so @njit(cache=True) still works without numba installed
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator


@njit(cache=True)
def _stdp_kernel(w_old, dt_gap, alpha_plus, alpha_minus, tau_stdp, H, W_max):
    """
    JIT-compiled elementwise STDP weight update. Numerically identical to the
    numpy-vectorized masked version, but avoids numpy's per-call overhead and
    several temporary array allocations (causal_mask, anti_causal_mask, zero_mask,
    and one intermediate array per masked assignment) -- worthwhile because this
    runs once per active synapse on every tick that has any spiking neurons.

    NOTE: neuron.py has imported `numba.njit` since this project's first commit
    but never actually called it anywhere -- numba sat in requirements.txt as a
    hard dependency doing precisely nothing. This is the first real use of it.
    """
    n = w_old.shape[0]
    w_new = np.empty(n, dtype=np.float64)
    for i in range(n):
        gap = dt_gap[i]
        if gap > 0.0:
            dw = (alpha_plus[i] / (1.0 + H[i])) * np.exp(-gap / tau_stdp[i])
        elif gap < 0.0:
            dw = -alpha_minus[i] * np.exp(gap / tau_stdp[i])
        else:
            dw = alpha_plus[i] / (1.0 + H[i])
        w = w_old[i] + dw
        if w < 0.0:
            w = 0.0
        elif w > W_max[i]:
            w = W_max[i]
        w_new[i] = w
    return w_new


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
            # Respect the destination neuron's configured weight ceiling at creation
            # time. Previously only STDP-updated weights were clipped to W_max; a
            # weight passed straight into connect()/connect_batch() above W_max stayed
            # unbounded forever (violates the model's own invariant, see NeuronConfig).
            w_max = self.state.W_max[post_idx]
            clipped_weight = max(0.0, min(w_max, weight))
            rows.append(post_idx)
            cols.append(pre_idx)
            data.append(clipped_weight)
        
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

    def tick(self, current_time, external_pulses=None, external_current=1.0):
        """
        Advance all neurons by one tick (1ms).

        `external_pulses` (previously a no-op for actual synaptic input -- see fix
        note below) accepts two forms:
          - dict {post_id: [pre_ids]}: declares each pre_id as firing *this* tick
            (same-tick, unlike the one-tick-delayed internal W @ prev_spike_vec
            propagation). If a real synapse pre_id->post_id exists, its weight
            contributes input energy to post_id this tick, mirroring the scalar
            Neuron.tick(current_time, incoming_pulses) contract exactly. The STDP
            timing clock for that pair is also updated (as before).
          - list/set of neuron_ids: directly injects `external_current` units of
            raw input energy into each of those neurons this tick. Useful for
            driving dedicated sensory/input neurons that have no incoming synapses
            of their own (e.g. a reservoir-computing input layer).

        BUG FIX: previously external_pulses only ever updated STDP timing clocks
        and never contributed anything to input_energy in either form. That meant
        a freshly built Network had no way to receive external/sensory drive at
        all -- only neurons that already had internal spiking activity could ever
        propagate anything (via W @ prev_spike_vec), so a network starting from
        rest (charge=0 everywhere, as it always does) could never become active.
        """
        N = len(self.idx_to_id)
        if N == 0:
            return {}
        
        if external_pulses is None:
            external_pulses = {}
        
        # List/set form: direct external current injection into named neurons.
        direct_injection = np.zeros(N, dtype=np.float64)
        if isinstance(external_pulses, (list, set)):
            for nid in external_pulses:
                if nid in self.id_to_idx:
                    direct_injection[self.id_to_idx[nid]] += external_current
            external_pulses = {}
        
        # 0. Update incoming firing clocks (vectorized, no Python loops!)
        self._update_pre_spike_clocks(current_time, external_pulses)
        
        # 0b. Weighted external synaptic input from declared (post, pre) pulse pairs.
        external_input_energy = np.zeros(N, dtype=np.float64)
        if external_pulses:
            ext_post_idxs = []
            ext_pre_idxs = []
            for post_id, pre_ids in external_pulses.items():
                if post_id not in self.id_to_idx:
                    continue
                post_idx = self.id_to_idx[post_id]
                for pre_id in pre_ids:
                    if pre_id not in self.id_to_idx:
                        continue
                    ext_post_idxs.append(post_idx)
                    ext_pre_idxs.append(self.id_to_idx[pre_id])
            if ext_post_idxs:
                ext_post_idxs = np.array(ext_post_idxs)
                ext_pre_idxs = np.array(ext_pre_idxs)
                # Only counts if a real synapse exists, matching scalar semantics
                # (total_input_energy += self.incoming_weights[pre_id]).
                w_vals = np.asarray(self.W[ext_post_idxs, ext_pre_idxs]).flatten()
                np.add.at(external_input_energy, ext_post_idxs, w_vals)
        
        # 1. Calculate adaptive refractory period
        num_connections = np.array(self.W.getnnz(axis=1))  # per-neuron incoming connections
        num_connections = np.maximum(num_connections, 1)
        R_dynamic = self.state.R_min * (1.0 + 5.0 * (self.state.U / num_connections))
        is_refractory = (current_time - self.state.last_spike_time) < R_dynamic
        
        # 2. Update stress thermostat
        did_spike_last_tick = (current_time - self.state.last_spike_time) == 1.0
        self.state.H = self.state.H + self.state.gamma * (did_spike_last_tick - self.state.H)
        self.state.H = np.clip(self.state.H, 0.0, 1.0)
        
        # 3. Gather synaptic inputs: sparse matrix multiply + external contributions
        input_energy = self.W @ self.prev_spike_vec + external_input_energy + direct_injection
        
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
        
        # 2. Update from previous spike vector: fully vectorized, no Python loop over
        # spiking neurons (the old code ran a per-spiked-neuron Python loop that each
        # rebuilt and `.maximum()`-merged a whole (N,N) sparse matrix -- O(active
        # neurons) sparse allocations per tick, the dominant cost at scale).
        spiked_pre_indices = np.where(self.prev_spike_vec == 1)[0]
        if len(spiked_pre_indices) == 0:
            return
        
        W_csc = self.W.tocsc()
        # Column (pre) index of every stored entry, built with a single O(nnz)
        # vectorized repeat over indptr -- equivalent to W_csc.tocoo().col but
        # without a full COO materialization.
        entry_col = np.repeat(np.arange(N), np.diff(W_csc.indptr))
        spiked_col_mask = np.zeros(N, dtype=bool)
        spiked_col_mask[spiked_pre_indices] = True
        entry_mask = spiked_col_mask[entry_col]
        
        if np.any(entry_mask):
            update_rows = W_csc.indices[entry_mask]  # post indices
            update_cols = entry_col[entry_mask]       # pre indices
            update_data = np.full(update_rows.shape[0], current_time, dtype=np.float64)
            update = sp.csr_matrix((update_data, (update_rows, update_cols)), shape=(N, N))
            self.state.pre_spike_clocks = self.state.pre_spike_clocks.maximum(update)

    def _apply_stdp_learning_batch(self, current_time, spiked_post_indices):
        """
        Apply STDP updates to all spiking post neurons in one fully vectorized pass.

        BUG FIX: the previous implementation looped over spiked_post_indices in
        Python (expensive at scale) and, worse, merged updates back into W by
        rebuilding it from COO triplets via `sp.csr_matrix((data, (row, col)), ...)`.
        That constructor SUMS duplicate (row, col) entries instead of overwriting
        them -- if any (row, col) pair ever appeared twice across the merge (e.g.
        from residual duplicate structural entries after prior sparse-matrix algebra),
        the resulting weight would silently balloon above W_max. This was caught by
        test_properties.py's invariant check. The rewrite below writes new weights
        directly into self.W.data at their exact existing positions -- it never
        changes the sparsity structure, so duplicate-summing cannot happen, and it
        needs no per-post-neuron Python loop.
        """
        if len(spiked_post_indices) == 0:
            return
        
        N = self.W.shape[0]
        indptr = self.W.indptr
        
        # Select exactly the W.data entries belonging to spiking post rows, in one
        # vectorized pass (same O(nnz) repeat trick as above, but row-major on W).
        row_mask = np.zeros(N, dtype=bool)
        row_mask[spiked_post_indices] = True
        entry_row = np.repeat(np.arange(N), np.diff(indptr))
        entry_mask = row_mask[entry_row]
        
        if not np.any(entry_mask):
            return
        
        post_indices = entry_row[entry_mask]
        pre_indices = self.W.indices[entry_mask]
        w_old = self.W.data[entry_mask]
        
        # Vectorized paired lookup of t_pre from pre_spike_clocks. Every (post, pre)
        # pair present in W was seeded with -9999.0 in pre_spike_clocks at connect()
        # time (see connect_batch), so every entry we need here is guaranteed to be
        # explicitly stored -- no missing-vs-zero ambiguity.
        clocks_csr = self.state.pre_spike_clocks.tocsr()
        t_pre = np.asarray(clocks_csr[post_indices, pre_indices]).flatten()
        dt_gap = current_time - t_pre
        
        alpha_plus = self.state.alpha_plus[post_indices]
        alpha_minus = self.state.alpha_minus[post_indices]
        tau_stdp = self.state.tau_stdp[post_indices]
        H = self.state.H[post_indices]
        W_max = self.state.W_max[post_indices]
        
        if NUMBA_AVAILABLE:
            w_new = _stdp_kernel(w_old, dt_gap, alpha_plus, alpha_minus, tau_stdp, H, W_max)
        else:
            dw = np.zeros_like(w_old)
            
            causal_mask = dt_gap > 0
            dw[causal_mask] = (alpha_plus[causal_mask] / (1.0 + H[causal_mask])) * \
                np.exp(-dt_gap[causal_mask] / tau_stdp[causal_mask])
            
            anti_causal_mask = dt_gap < 0
            dw[anti_causal_mask] = -alpha_minus[anti_causal_mask] * \
                np.exp(dt_gap[anti_causal_mask] / tau_stdp[anti_causal_mask])
            
            zero_mask = dt_gap == 0
            dw[zero_mask] = alpha_plus[zero_mask] / (1.0 + H[zero_mask])
            
            w_new = np.clip(w_old + dw, 0.0, W_max)
        
        # Write straight back into W.data at the same positions -- structure-preserving.
        self.W.data[entry_mask] = w_new

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
        """
        Return all current synaptic weights as (pre, post, weight) tuples.

        BUG FIX: this used to cosmetically round each weight to 5 decimal places
        before returning it. Standard rounding can push a value sitting exactly at
        the W_max clip boundary slightly ABOVE W_max (e.g. round(1.74477514..., 5)
        == 1.74478 > 1.74477514...), which made a stored, correctly-clipped weight
        LOOK like it violated its own ceiling. The actual stored weight never
        exceeds W_max (np.clip guarantees that); only the display rounding lied
        about it, as caught by test_properties.py. Report full precision and let
        callers round for display if they want to.
        """
        edges = []
        coo = self.W.tocoo()
        for post_idx, pre_idx, w in zip(coo.row, coo.col, coo.data):
            if abs(w) > 1e-15:  # ignore zero weights
                pre_id = self.idx_to_id[pre_idx]
                post_id = self.idx_to_id[post_idx]
                edges.append((pre_id, post_id, float(w)))
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

