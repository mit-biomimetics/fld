import torch

class DistributionBuffer:
    def __init__(self, buffer_dim, buffer_size, device) -> None:
        self.distribution_buffer = torch.zeros(buffer_size, buffer_dim, dtype=torch.float, requires_grad=False).to(device)
        self.buffer_size = buffer_size
        self.device = device
        self.step = 0
        self.num_samples = 0

    def insert(self, data):
        num_data = data.shape[0]
        start_idx = self.step
        end_idx = self.step + num_data
        if end_idx > self.buffer_size:
            self.distribution_buffer[self.step:self.buffer_size] = data[:self.buffer_size - self.step]
            self.distribution_buffer[:end_idx - self.buffer_size] = data[self.buffer_size - self.step:]
        else:
            self.distribution_buffer[start_idx:end_idx] = data

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_data) % self.buffer_size

    def get_distribution(self):
        return self.distribution_buffer
