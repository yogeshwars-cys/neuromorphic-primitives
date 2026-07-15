
import numpy as np
from .network import Network


try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SimulationLogger:
    """Lightweight logger for simulation metrics."""
    
    def __init__(self, log_dir: str = None, use_tensorboard: bool = False, use_wandb: bool = False, **wandb_kwargs):
        self.log_dir = log_dir
        self.writer = None
        self.wandb_initialized = False
        
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=log_dir)
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(**wandb_kwargs)
            self.wandb_initialized = True
        
        # Local logs
        self.history = {
            'step': [],
            'mean_charge': [],
            'mean_H': [],
            'mean_U': [],
            'spike_count': [],
            'mean_weight': [],
            'std_weight': []
        }
    
    def log_step(self, step: int, network: Network):
        """Log metrics at a single simulation step."""
        # Collect metrics
        N = len(network.idx_to_id)
        if N == 0:
            return
        
        mean_charge = np.mean(network.state.charge)
        mean_H = np.mean(network.state.H)
        mean_U = np.mean(network.state.U)
        spike_count = sum(network.spike_counts().values())
        
        weights = network.weights()
        if weights:
            weight_values = [w for _, _, w in weights]
            mean_weight = np.mean(weight_values)
            std_weight = np.std(weight_values)
        else:
            mean_weight = 0.0
            std_weight = 0.0
        
        # Save to local history
        self.history['step'].append(step)
        self.history['mean_charge'].append(mean_charge)
        self.history['mean_H'].append(mean_H)
        self.history['mean_U'].append(mean_U)
        self.history['spike_count'].append(spike_count)
        self.history['mean_weight'].append(mean_weight)
        self.history['std_weight'].append(std_weight)
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('neuron/mean_charge', mean_charge, step)
            self.writer.add_scalar('neuron/mean_H', mean_H, step)
            self.writer.add_scalar('neuron/mean_U', mean_U, step)
            self.writer.add_scalar('network/spike_count', spike_count, step)
            self.writer.add_scalar('weights/mean', mean_weight, step)
            self.writer.add_scalar('weights/std', std_weight, step)
        
        # Log to wandb
        if self.wandb_initialized:
            wandb.log({
                'step': step,
                'neuron/mean_charge': mean_charge,
                'neuron/mean_H': mean_H,
                'neuron/mean_U': mean_U,
                'network/spike_count': spike_count,
                'weights/mean': mean_weight,
                'weights/std': std_weight
            }, step=step)
    
    def close(self):
        """Close the logger."""
        if self.writer:
            self.writer.close()
        if self.wandb_initialized:
            wandb.finish()
    
    def plot_history(self, show: bool = True):
        """Plot local history using matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        fig.tight_layout(pad=5.0)
        
        # Mean charge
        axes[0, 0].plot(self.history['step'], self.history['mean_charge'])
        axes[0, 0].set_title('Mean Membrane Charge')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Charge')
        
        # Mean H
        axes[0, 1].plot(self.history['step'], self.history['mean_H'])
        axes[0, 1].set_title('Mean Homeostatic Stress (H)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('H')
        axes[0, 1].set_ylim(0, 1)
        
        # Mean U
        axes[1, 0].plot(self.history['step'], self.history['mean_U'])
        axes[1, 0].set_title('Mean Spike Frequency (U)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('U')
        
        # Spike count
        axes[1, 1].plot(self.history['step'], self.history['spike_count'])
        axes[1, 1].set_title('Total Spike Count per Step')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Spikes')
        
        # Mean weight
        axes[2, 0].plot(self.history['step'], self.history['mean_weight'])
        axes[2, 0].set_title('Mean Synaptic Weight')
        axes[2, 0].set_xlabel('Step')
        axes[2, 0].set_ylabel('Weight')
        
        # Std weight
        axes[2, 1].plot(self.history['step'], self.history['std_weight'])
        axes[2, 1].set_title('Std Synaptic Weight')
        axes[2, 1].set_xlabel('Step')
        axes[2, 1].set_ylabel('Std Dev')
        
        if show:
            plt.show()
        
        return fig

