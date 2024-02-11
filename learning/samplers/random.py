from learning.samplers.base import BaseSampler
import torch
import torch.nn as nn


class RandomSampler(BaseSampler):
    def __init__(self, n_features, min, max, device, curriculum_scale=1.0):
        super().__init__()
        self.n_features = n_features
        self.min = nn.Parameter(min.clone().detach(), requires_grad=False)
        self.max = nn.Parameter(max.clone().detach(), requires_grad=False)
        self.device = device
        self.curriculum_scale = curriculum_scale
        
    
    def update(self, x):
        pass
        
        
    def update_curriculum(self):
        mean = (self.min + self.max) / 2
        std = (self.max - self.min) / 2
        std *= self.curriculum_scale
        self.max.data = mean + std
        self.min.data = mean - std
    
    
    def sample(self, n_samples):
        return (self.max - self.min) * torch.rand(n_samples, self.n_features, device=self.device, dtype=torch.float, requires_grad=False) + self.min
        