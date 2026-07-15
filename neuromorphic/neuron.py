
import numpy as np
import math

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        return func


class NeuronState:
    """
    Structure-of-arrays container for all neuron state variables.
    """
    def __init__(self, size: int, **kwargs):
        # State variables
        self.charge = np.zeros(size, dtype=np.float64)
        self.H = np.full(size, 0.2, dtype=np.float64)
        self.U = np.zeros(size, dtype=np.float64)
        self.last_spike_time = np.full(size, -9999.0, dtype=np.float64)
        
        # Hyperparameters (per-neuron)
        self.R_min = np.full(size, kwargs.get('R_min', 2.0), dtype=np.float64)
        self.lambda_leak = np.full(size, kwargs.get('lambda_leak', 0.12), dtype=np.float64)
        self.T_base = np.full(size, kwargs.get('T_base', 1.0), dtype=np.float64)
        self.beta = np.full(size, kwargs.get('beta', 0.5), dtype=np.float64)
        self.gamma = np.full(size, kwargs.get('gamma', 0.015), dtype=np.float64)
        self.alpha_plus = np.full(size, kwargs.get('alpha_plus', 0.12), dtype=np.float64)
        self.alpha_minus = np.full(size, kwargs.get('alpha_minus', 0.06), dtype=np.float64)
        self.tau_stdp = np.full(size, kwargs.get('tau_stdp', 20.0), dtype=np.float64)
        self.W_max = np.full(size, kwargs.get('W_max', 2.0), dtype=np.float64)
        
        # Pre-synaptic spike clocks: matrix of shape (N, N)
        self.pre_spike_clocks = np.full((size, size), -9999.0, dtype=np.float64)
    
    def resize(self, new_size: int):
        """Grow all arrays to new_size"""
        if new_size <= len(self.charge):
            return
        
        for attr in ['charge', 'H', 'U', 'last_spike_time', 
                     'R_min', 'lambda_leak', 'T_base', 'beta', 
                     'gamma', 'alpha_plus', 'alpha_minus', 'tau_stdp', 'W_max']:
            arr = getattr(self, attr)
            new_arr = np.zeros(new_size, dtype=arr.dtype)
            new_arr[:len(arr)] = arr
            if attr == 'H':
                new_arr[len(arr):] = 0.2
            elif attr == 'last_spike_time':
                new_arr[len(arr):] = -9999.0
            elif attr in ['R_min', 'lambda_leak', 'T_base', 'beta', 'gamma', 'alpha_plus', 'alpha_minus', 'tau_stdp', 'W_max']:
                # Keep default values for new neurons
                if attr == 'R_min': val = 2.0
                elif attr == 'lambda_leak': val = 0.12
                elif attr == 'T_base': val = 1.0
                elif attr == 'beta': val = 0.5
                elif attr == 'gamma': val = 0.015
                elif attr == 'alpha_plus': val = 0.12
                elif attr == 'alpha_minus': val = 0.06
                elif attr == 'tau_stdp': val = 20.0
                elif attr == 'W_max': val = 2.0
                new_arr[len(arr):] = val
            setattr(self, attr, new_arr)
        
        # Resize pre_spike_clocks matrix
        old_size = self.pre_spike_clocks.shape[0]
        new_pre_spike = np.full((new_size, new_size), -9999.0, dtype=np.float64)
        new_pre_spike[:old_size, :old_size] = self.pre_spike_clocks
        self.pre_spike_clocks = new_pre_spike


class Neuron:
    """
    Backward-compatibility wrapper for old Neuron API.
    Uses NeuronState under the hood.
    """
    def __init__(self, neuron_id, R_min=2.0, lambda_leak=0.12, T_base=1.0, beta=0.5, gamma=0.015, alpha_plus=0.12, alpha_minus=0.06, tau_stdp=20.0, W_max=2.0):
        self.id = neuron_id
        # We'll store individual hyperparams here for backwards compatibility,
        # but the actual state is managed by Network
        self._R_min = R_min
        self._lambda_leak = lambda_leak
        self._T_base = T_base
        self._beta = beta
        self._gamma = gamma
        self._alpha_plus = alpha_plus
        self._alpha_minus = alpha_minus
        self._tau_stdp = tau_stdp
        self._W_max = W_max
        
        # Dummy state for standalone use (not recommended)
        self.charge = 0.0
        self.H = 0.2
        self.U = 0.0
        self.last_spike_time = -9999.0
        self.incoming_weights = {}
        self.pre_spike_clocks = {}
    
    @property
    def num_connections(self):
        return len(self.incoming_weights)
    
    def register_synapse(self, pre_id, initial_weight=0.3):
        self.incoming_weights[pre_id] = initial_weight
        self.pre_spike_clocks[pre_id] = -9999.0
    
    def tick(self, current_time, incoming_pulses):
        # Fallback to original scalar implementation for standalone Neuron use
        # --- 0. Update Incoming Firing Clocks ---
        for pre_id in incoming_pulses:
            if pre_id in self.pre_spike_clocks:
                self.pre_spike_clocks[pre_id] = current_time

        # --- 1. Calculate Adaptive Refractory Period ---
        N = max(1, self.num_connections)
        R_dynamic = self._R_min * (1.0 + 5.0 * (self.U / N))
        is_refractory = (current_time - self.last_spike_time) < R_dynamic

        # --- 2. Update Stress Thermostat (Homeostasis) ---
        did_spike_last_tick = 1.0 if (current_time - self.last_spike_time == 1.0) else 0.0
        self.H = self.H + self._gamma * (did_spike_last_tick - self.H)
        self.H = max(0.0, min(1.0, self.H))

        # --- 3. Gather Synaptic Inputs ---
        total_input_energy = 0.0
        for pre_id in incoming_pulses:
            if pre_id in self.incoming_weights:
                total_input_energy += self.incoming_weights[pre_id]

        # --- 4. Refractory Lock Check ---
        if is_refractory:
            reduced_leak = self.charge * self._lambda_leak * self.H * 0.5
            self.charge = max(0.0, self.charge + total_input_energy - reduced_leak)
            self.U = 0.95 * self.U
            return 0

        # --- 5. Energy Accumulation & Dynamic Leak ---
        dynamic_leak = self.charge * self._lambda_leak * self.H
        raw_charge = self.charge + total_input_energy - dynamic_leak

        # --- 6. Dynamic Threshold & Soft Reset ---
        dynamic_threshold = self._T_base + (self._beta * self.H)

        if raw_charge >= dynamic_threshold:
            spiked = 1
            self.charge = max(0.0, raw_charge - dynamic_threshold)
            self.last_spike_time = current_time
            self.U = 0.95 * self.U + 0.05 * 1.0
            
            # --- 7. Event-Triggered Synaptic Learning (STDP) ---
            self._apply_stdp_learning(current_time)
        else:
            spiked = 0
            self.charge = max(0.0, raw_charge)
            self.U = 0.95 * self.U

        return spiked

    def _apply_stdp_learning(self, current_time):
        t_post = current_time
        
        for pre_id, w_old in self.incoming_weights.items():
            t_pre = self.pre_spike_clocks[pre_id]
            dt_gap = t_post - t_pre
            
            if dt_gap > 0:
                dw = (self._alpha_plus / (1.0 + self.H)) * math.exp(-dt_gap / self._tau_stdp)
                new_w = w_old + dw
            elif dt_gap < 0:
                dw = -self._alpha_minus * math.exp(dt_gap / self._tau_stdp)
                new_w = w_old + dw
            else:
                dw = (self._alpha_plus / (1.0 + self.H))
                new_w = w_old + dw
                
            self.incoming_weights[pre_id] = max(0.0, min(self._W_max, new_w))

