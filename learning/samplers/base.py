import torch.nn as nn

class BaseSampler(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    
    def update(self):
        raise NotImplementedError
    
    
    def update_curriculum(self):
        raise NotImplementedError


    def sample(self):
        raise NotImplementedError