
from dataclasses import dataclass
import yaml


@dataclass
class NeuronConfig:
    """Configuration for neuron hyperparameters."""
    R_min: float = 2.0
    lambda_leak: float = 0.12
    T_base: float = 1.0
    beta: float = 0.5
    gamma: float = 0.015
    alpha_plus: float = 0.12
    alpha_minus: float = 0.06
    tau_stdp: float = 20.0
    W_max: float = 2.0
    
    @classmethod
    def from_yaml(cls, file_path: str) -> "NeuronConfig":
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, file_path: str):
        with open(file_path, 'w') as f:
            yaml.dump(vars(self), f)

