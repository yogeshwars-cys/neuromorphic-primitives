# neuromorphic-primitives

An autonomous neuromorphic computing primitive library in Python.

## Install

```bash
pip install git+https://github.com/yogeshwars-cys/neuromorphic-primitives.git
```

## Usage

```python
from neuromorphic import Neuron

# Create neurons
n1 = Neuron(neuron_id=1)
n2 = Neuron(neuron_id=2)

# Wire n1 -> n2
n2.register_synapse(pre_id=1, initial_weight=0.5)

# Simulate ticks
for t in range(100):
    n1_spiked = n1.tick(t, incoming_pulses=[])
    n2.tick(t, incoming_pulses=[1] if n1_spiked else [])
```

## Neuron Parameters

| Parameter     | Default | Description                              |
|---------------|---------|------------------------------------------|
| `R_min`       | 2.0     | Minimum refractory wait time             |
| `lambda_leak` | 0.12    | Base rate of charge leak                 |
| `T_base`      | 1.0     | Absolute minimum firing threshold        |
| `beta`        | 0.5     | Homeostatic threshold penalty scaler     |
| `gamma`       | 0.015   | Adaptation speed for stress thermostat   |
| `alpha_plus`  | 0.12    | Max positive STDP reinforcement          |
| `alpha_minus` | 0.06    | Max negative STDP penalty                |
| `tau_stdp`    | 20.0    | Temporal learning window decay rate      |
| `W_max`       | 2.0     | Ceiling for synaptic weight              |
