import math

class Neuron:
    def __init__(self, neuron_id, R_min=2.0, lambda_leak=0.12, T_base=1.0, beta=0.5, gamma=0.015, alpha_plus=0.12, alpha_minus=0.06, tau_stdp=20.0, W_max=2.0):
        """
        An autonomous neuromorphic computing primitive.
        """
        self.id = neuron_id
        
        # --- Physical State Variables ---
        self.charge = 0.0                  # C(t) - Internal membrane potential
        self.H = 0.2                       # H(t) - Homeostatic stress factor (thermostat)
        self.U = 0.0                       # U(t) - Running average spike frequency
        self.last_spike_time = -9999.0     # t_last_spike
        
        # --- Local Synaptic Connections ---
        # {pre_neuron_id: weight_value}
        self.incoming_weights = {}         
        # {pre_neuron_id: last_spike_time} - Needed to calculate exact STDP delta t
        self.pre_spike_clocks = {}         

        # --- Hyperparameters ---
        self.R_min = R_min                 # Minimum refractory wait time
        self.lambda_leak = lambda_leak     # Base rate of charge leak
        self.T_base = T_base               # Absolute minimum threshold
        self.beta = beta                   # Homeostatic threshold penalty scaler
        self.gamma = gamma                 # Adaptation speed for stress thermostat
        self.alpha_plus = alpha_plus       # Maximum positive STDP reinforcement
        self.alpha_minus = alpha_minus     # Maximum negative STDP penalty
        self.tau_stdp = tau_stdp           # Temporal learning window decay rate
        self.W_max = W_max                 # Ceiling for synaptic weight thickness

    @property
    def num_connections(self):
        """
        Represents the local network topology size (N).
        """
        return len(self.incoming_weights)

    def register_synapse(self, pre_id, initial_weight=0.3):
        """
        Physically wires an upstream neighbor to this node.
        """
        self.incoming_weights[pre_id] = initial_weight
        self.pre_spike_clocks[pre_id] = -9999.0

    def tick(self, current_time, incoming_pulses):
        """
        Executes one millisecond (tick) of the neuron's physics loop.
        
        :param current_time: The master system clock time (float/int in ms)
        :param incoming_pulses: A list of node_ids that pulsed this tick (e.g., [1, 4])
        :return: int (1 if this neuron spiked, 0 otherwise)
        """
        # --- 0. Update Incoming Firing Clocks ---
        for pre_id in incoming_pulses:
            if pre_id in self.pre_spike_clocks:
                self.pre_spike_clocks[pre_id] = current_time

        # --- 1. Calculate Adaptive Refractory Period ---
        # Formula: R = R_min * (1 + 5 * (U / N))
        N = max(1, self.num_connections)
        R_dynamic = self.R_min * (1.0 + 5.0 * (self.U / N))
        is_refractory = (current_time - self.last_spike_time) < R_dynamic

        # --- 2. Update Stress Thermostat (Homeostasis) ---
        # Formula: H(t) = H(t-1) + gamma * (S_out_prev - H(t-1))
        # (We pass whether the neuron spiked in the previous millisecond)
        did_spike_last_tick = 1.0 if (current_time - self.last_spike_time == 1.0) else 0.0
        self.H = self.H + self.gamma * (did_spike_last_tick - self.H)
        self.H = max(0.0, min(1.0, self.H)) # Bounded between 0 and 1

        # --- 3. Gather Synaptic Inputs ---
        total_input_energy = 0.0
        for pre_id in incoming_pulses:
            if pre_id in self.incoming_weights:
                total_input_energy += self.incoming_weights[pre_id]

        # --- 4. Refractory Lock Check ---
        if is_refractory:
            # Under a refractory lock, we still accumulate charge but cannot fire,
            # and leak is reduced to mimic closed ion channels.
            reduced_leak = self.charge * self.lambda_leak * self.H * 0.5
            self.charge = max(0.0, self.charge + total_input_energy - reduced_leak)
            self.U = 0.95 * self.U  # Freq trace decays slowly
            return 0

        # --- 5. Energy Accumulation & Dynamic Leak ---
        # Formula: Leak = C * lambda * H
        dynamic_leak = self.charge * self.lambda_leak * self.H
        raw_charge = self.charge + total_input_energy - dynamic_leak

        # --- 6. Dynamic Threshold & Soft Reset ---
        # Formula: T = T_base + beta * H
        dynamic_threshold = self.T_base + (self.beta * self.H)

        if raw_charge >= dynamic_threshold:
            # WE SPIKE!
            spiked = 1
            self.charge = max(0.0, raw_charge - dynamic_threshold)  # Soft reset
            self.last_spike_time = current_time
            self.U = 0.95 * self.U + 0.05 * 1.0  # Update frequency trace
            
            # --- 7. Event-Triggered Synaptic Learning (STDP) ---
            self._apply_stdp_learning(current_time)
        else:
            # NO SPIKE
            spiked = 0
            self.charge = max(0.0, raw_charge)
            self.U = 0.95 * self.U

        return spiked

    def _apply_stdp_learning(self, current_time):
        """
        Executes the localized timing-dependent learning rule on incoming synapses.
        """
        t_post = current_time
        
        for pre_id, w_old in self.incoming_weights.items():
            t_pre = self.pre_spike_clocks[pre_id]
            dt_gap = t_post - t_pre
            
            if dt_gap > 0:
                # CAUSAL REWARD: Sender fired BEFORE receiver spiked
                # Alpha is moderated by current homeostasis stress (H)
                dw = (self.alpha_plus / (1.0 + self.H)) * math.exp(-dt_gap / self.tau_stdp)
                new_w = w_old + dw
            elif dt_gap < 0:
                # LAGGING PUNISHMENT: Sender fired AFTER receiver spiked
                dw = -self.alpha_minus * math.exp(dt_gap / self.tau_stdp)
                new_w = w_old + dw
            else:
                # Perfect timing (delta t == 0) gets full baseline reward
                dw = (self.alpha_plus / (1.0 + self.H))
                new_w = w_old + dw
                
            # Keep weights strictly inside physical bounds
            self.incoming_weights[pre_id] = max(0.0, min(self.W_max, new_w))
