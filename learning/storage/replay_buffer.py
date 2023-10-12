import torch
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, obs_horizon, buffer_size, device):
        """Initialize a ReplayBuffer object.
        Arguments:
            buffer_size (int): maximum size of buffer
        """
        self.state_buf = torch.zeros(buffer_size, obs_horizon, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0
    
    def insert(self, state_buf):
        """Add new states to memory."""
        
        num_states = state_buf.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.state_buf[self.step:self.buffer_size] = state_buf[:self.buffer_size - self.step]
            self.state_buf[:end_idx - self.buffer_size] = state_buf[self.buffer_size - self.step:]
        else:
            self.state_buf[start_idx:end_idx] = state_buf

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield self.state_buf[sample_idxs, :].to(self.device)
