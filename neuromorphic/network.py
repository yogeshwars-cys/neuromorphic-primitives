from .neuron import Neuron


class Network:
    """
    A spiking neural network composed of interconnected Neuron primitives.

    Supports layered or arbitrary topologies, external input injection,
    and full spike history recording.
    """

    def __init__(self):
        # {neuron_id: Neuron}
        self.neurons = {}
        # Ordered list of layers: [ [id, id, ...], [id, id, ...] ]
        self.layers = []
        # Spike history: {neuron_id: [spike_at_t0, spike_at_t1, ...]}
        self.spike_history = {}

    # ------------------------------------------------------------------
    # Building the network
    # ------------------------------------------------------------------

    def add_neuron(self, neuron_id, **kwargs):
        """
        Add a single neuron to the network.
        kwargs are forwarded to Neuron.__init__ as hyperparameters.
        """
        neuron = Neuron(neuron_id, **kwargs)
        self.neurons[neuron_id] = neuron
        self.spike_history[neuron_id] = []
        return neuron

    def add_layer(self, neuron_ids, **kwargs):
        """
        Add a group of neurons as a named layer.
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
        if post_id not in self.neurons:
            raise ValueError(f"Neuron {post_id} not in network.")
        if pre_id not in self.neurons:
            raise ValueError(f"Neuron {pre_id} not in network.")
        self.neurons[post_id].register_synapse(pre_id, initial_weight=weight)

    def connect_layers(self, from_layer_ids, to_layer_ids, weight=0.3):
        """
        Fully connect every neuron in from_layer to every neuron in to_layer.
        """
        for pre in from_layer_ids:
            for post in to_layer_ids:
                self.connect(pre, post, weight)

    # ------------------------------------------------------------------
    # Running the simulation
    # ------------------------------------------------------------------

    def tick(self, current_time, external_pulses=None):
        """
        Advance all neurons by one tick (1ms).

        :param current_time: Master clock time in ms.
        :param external_pulses: dict {neuron_id: [pre_ids to inject]}
                                OR list of neuron_ids to pulse externally.
        :return: dict {neuron_id: 1 or 0}
        """
        if external_pulses is None:
            external_pulses = {}

        # Normalise external_pulses to dict form
        if isinstance(external_pulses, (list, set)):
            external_pulses = {nid: [] for nid in external_pulses}

        # Collect spikes from previous tick (needed for propagation)
        prev_spikes = [nid for nid, hist in self.spike_history.items()
                       if hist and hist[-1] == 1]

        tick_results = {}
        for nid, neuron in self.neurons.items():
            # Build incoming pulse list: propagated spikes + any external injection
            incoming = [pid for pid in prev_spikes if pid in neuron.incoming_weights]
            incoming += external_pulses.get(nid, [])
            spiked = neuron.tick(current_time, incoming)
            self.spike_history[nid].append(spiked)
            tick_results[nid] = spiked

        return tick_results

    def simulate(self, duration_ms, external_fn=None):
        """
        Run the network for `duration_ms` ticks.

        :param duration_ms: Number of ms to simulate.
        :param external_fn: Optional callable(t) -> external_pulses dict/list.
                            Use this to inject time-varying inputs.
        :return: Full spike history dict {neuron_id: [0/1 per tick]}
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
        for post_id, neuron in self.neurons.items():
            for pre_id, w in neuron.incoming_weights.items():
                edges.append((pre_id, post_id, round(w, 5)))
        return edges

    def reset(self):
        """Clear spike history without destroying topology or learned weights."""
        for nid in self.spike_history:
            self.spike_history[nid] = []

    def __repr__(self):
        return (f"<Network neurons={len(self.neurons)} "
                f"synapses={sum(n.num_connections for n in self.neurons.values())} "
                f"layers={len(self.layers)}>")
