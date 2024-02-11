from learning.samplers.base import BaseSampler
from learning.modules.gmm import GaussianMixture
import torch


class GMMSampler(BaseSampler):
    def __init__(self, n_components, n_features, device, covariance_type="full", curriculum_scale=1.0):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.device = device
        self.curriculum_scale = curriculum_scale
        self.gmm = GaussianMixture(n_components, n_features, device=device, covariance_type=covariance_type).eval()
        
    
    def load_gmm(self, load_path):
        self.gmm.load_state_dict(torch.load(load_path)["gmm_state_dict"])

    
    def update(self, x):
        self.gmm.fit(x)
    
    
    def update_curriculum(self):
        self.gmm.set_variances(self.gmm.var * self.curriculum_scale ** 2)
        
    
    def sample(self, n_samples):
        return self.gmm.sample(n_samples)[0]
        