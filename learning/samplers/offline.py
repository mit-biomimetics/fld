from learning.samplers.base import BaseSampler
import torch


class OfflineSampler(BaseSampler):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    
    def load_data(self, load_path):
        self.data = torch.load(load_path)
    
    
    def update(self, x):
        pass
        
        
    def update_curriculum(self):
        pass
    
    
    def sample(self, n_samples):
        sample_ids = torch.randint(0, self.data.size(0), (n_samples,), device=self.device, dtype=torch.long, requires_grad=False)
        return self.data[sample_ids]
        