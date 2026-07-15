
import numpy as np
import scipy.sparse as sp
import h5py
from .network import Network
from .config import NeuronConfig


def save_network(network: Network, file_path: str):
    """Save network state to .npz or .h5 file."""
    if file_path.endswith('.npz'):
        _save_npz(network, file_path)
    elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
        _save_h5(network, file_path)
    else:
        raise ValueError("Unsupported file format. Use .npz or .h5/.hdf5.")


def _save_npz(network: Network, file_path: str):
    np.savez_compressed(
        file_path,
        id_to_idx=network.id_to_idx,
        idx_to_id=network.idx_to_id,
        layers=network.layers,
        spike_history=network.spike_history,
        charge=network.state.charge,
        H=network.state.H,
        U=network.state.U,
        last_spike_time=network.state.last_spike_time,
        R_min=network.state.R_min,
        lambda_leak=network.state.lambda_leak,
        T_base=network.state.T_base,
        beta=network.state.beta,
        gamma=network.state.gamma,
        alpha_plus=network.state.alpha_plus,
        alpha_minus=network.state.alpha_minus,
        tau_stdp=network.state.tau_stdp,
        W_max=network.state.W_max,
        pre_spike_clocks=network.state.pre_spike_clocks,
        W_data=network.W.data,
        W_indices=network.W.indices,
        W_indptr=network.W.indptr,
        W_shape=network.W.shape,
        format_version='1.0'
    )


def _save_h5(network: Network, file_path: str):
    with h5py.File(file_path, 'w') as f:
        f.attrs['format_version'] = '1.0'
        f.create_dataset('idx_to_id', data=np.array(network.idx_to_id, dtype=h5py.string_dtype()))
        f.create_dataset('layers', data=np.array(network.layers, dtype=h5py.string_dtype()))
        # Save spike history
        spike_group = f.create_group('spike_history')
        for nid, hist in network.spike_history.items():
            spike_group.create_dataset(str(nid), data=np.array(hist, dtype=np.int32))
        # Save state
        state_group = f.create_group('state')
        for attr in ['charge', 'H', 'U', 'last_spike_time', 'R_min', 'lambda_leak', 
                     'T_base', 'beta', 'gamma', 'alpha_plus', 'alpha_minus', 'tau_stdp', 'W_max']:
            state_group.create_dataset(attr, data=getattr(network.state, attr))
        state_group.create_dataset('pre_spike_clocks', data=network.state.pre_spike_clocks)
        # Save sparse weight matrix
        w_group = f.create_group('W')
        w_group.create_dataset('data', data=network.W.data)
        w_group.create_dataset('indices', data=network.W.indices)
        w_group.create_dataset('indptr', data=network.W.indptr)
        w_group.attrs['shape'] = network.W.shape


def load_network(file_path: str) -> Network:
    """Load network state from .npz or .h5 file."""
    if file_path.endswith('.npz'):
        return _load_npz(file_path)
    elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
        return _load_h5(file_path)
    else:
        raise ValueError("Unsupported file format. Use .npz or .h5/.hdf5.")


def _load_npz(file_path: str) -> Network:
    data = np.load(file_path, allow_pickle=True)
    net = Network()
    net.idx_to_id = data['idx_to_id'].tolist()
    net.id_to_idx = {nid: idx for idx, nid in enumerate(net.idx_to_id)}
    net.layers = data['layers'].tolist()
    net.spike_history = data['spike_history'].item()
    
    # Restore state
    N = len(net.idx_to_id)
    net.state = type(net.state)(N)
    for attr in ['charge', 'H', 'U', 'last_spike_time', 'R_min', 'lambda_leak', 
                 'T_base', 'beta', 'gamma', 'alpha_plus', 'alpha_minus', 'tau_stdp', 'W_max']:
        setattr(net.state, attr, data[attr])
    net.state.pre_spike_clocks = data['pre_spike_clocks']
    
    # Restore weight matrix
    net.W = sp.csr_matrix((data['W_data'], data['W_indices'], data['W_indptr']), 
                         shape=tuple(data['W_shape']))
    net.prev_spike_vec = np.zeros(N, dtype=np.int32)
    
    return net


def _load_h5(file_path: str) -> Network:
    with h5py.File(file_path, 'r') as f:
        net = Network()
        net.idx_to_id = f['idx_to_id'][:].tolist()
        net.id_to_idx = {nid: idx for idx, nid in enumerate(net.idx_to_id)}
        net.layers = [layer.tolist() for layer in f['layers'][:]]
        net.spike_history = {}
        for nid in f['spike_history']:
            net.spike_history[int(nid) if nid.isdigit() else nid] = f['spike_history'][nid][:].tolist()
        
        N = len(net.idx_to_id)
        net.state = type(net.state)(N)
        for attr in ['charge', 'H', 'U', 'last_spike_time', 'R_min', 'lambda_leak', 
                     'T_base', 'beta', 'gamma', 'alpha_plus', 'alpha_minus', 'tau_stdp', 'W_max']:
            setattr(net.state, attr, f['state'][attr][:])
        net.state.pre_spike_clocks = f['state']['pre_spike_clocks'][:]
        
        w_group = f['W']
        net.W = sp.csr_matrix((w_group['data'][:], w_group['indices'][:], w_group['indptr'][:]), 
                             shape=tuple(w_group.attrs['shape']))
        net.prev_spike_vec = np.zeros(N, dtype=np.int32)
        
        return net

