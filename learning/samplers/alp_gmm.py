from learning.samplers.base import BaseSampler
from learning.modules.gmm import GaussianMixture, GaussianMixtures
import torch
import torch.nn as nn
import faiss


class ALPGMMSampler(BaseSampler):
    def __init__(self, init_n_components, min_n_components, max_n_components, n_features, min, max, device, covariance_type="full", curriculum_scale=1.0, random_type="uniform"):
        super().__init__()
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.n_features = n_features
        self.min = nn.Parameter(min.clone().detach(), requires_grad=False)
        self.max = nn.Parameter(max.clone().detach(), requires_grad=False)
        self.device = device
        self.curriculum_scale = curriculum_scale
        self.random_type = random_type
        self.gmms = GaussianMixtures(min_n_components, max_n_components, n_features + 1, device=device, covariance_type=covariance_type).eval()
        self.gmm_idx = nn.Parameter(torch.tensor(-1, device=device, dtype=torch.long), requires_grad=False)
        self.init_random_sampling = nn.Parameter(torch.tensor(True, device=device, dtype=torch.bool), requires_grad=False)
        self.task_performance_buffer = KNNBuffer(n_features, 1, device=device)
        if random_type == "gmm":
            self.gmm = GaussianMixture(init_n_components, n_features, device=device, covariance_type=covariance_type).eval()

    def load_gmm(self, load_path):
        self.gmm.load_state_dict(torch.load(load_path)["gmm_state_dict"])
    
    def compute_alp(self, task, performance):
        old_performance = self.task_performance_buffer.get_k_nearest_neighbors(task, performance)
        alp = (performance - old_performance).abs()
        return alp
    
    
    def update(self, task, performance):
        alp = self.compute_alp(task, performance)
        self.task_performance_buffer.insert(task, performance)
        x = torch.cat((task, alp), dim=-1)
        self.gmms.fit(x)
        self.gmm_idx.data = self.gmms.get_best_gmm_idx(x)
        self.init_random_sampling.data = torch.tensor(False, device=self.device, dtype=torch.bool)
    
    
    def update_curriculum(self):
        if self.init_random_sampling:
            pass
        else:
            self.gmms.candidates[self.gmm_idx].set_variances(self.gmms.candidates[self.gmm_idx].var * self.curriculum_scale ** 2)
        
    
    def sample(self, n_samples, random_ratio=0.2):
        if self.init_random_sampling:
            return (self.max - self.min) * torch.rand(n_samples, self.n_features, device=self.device, dtype=torch.float, requires_grad=False) + self.min
        random = torch.rand(1, device=self.device, dtype=torch.float, requires_grad=False) < random_ratio
        if random:
            if self.random_type == "uniform":
                return (self.max - self.min) * torch.rand(n_samples, self.n_features, device=self.device, dtype=torch.float, requires_grad=False) + self.min
            elif self.random_type == "gmm":
                return self.gmm.sample(n_samples)[0]
        else:
            return self.gmms.sample(n_samples, self.gmm_idx)[:, :-1]

    
    def get_knn_buffer_size(self):
        return self.task_performance_buffer.key_index.ntotal
    
    
    def get_knn_buffer(self):
        keys, values = self.task_performance_buffer.get_samples(10000)
        return keys, values


class KNNBuffer:
    def __init__(self, key_dim, value_dim, device):
        res = faiss.StandardGpuResources()
        key_index = faiss.IndexFlatL2(key_dim)
        self.key_index = faiss.index_cpu_to_gpu(res, 0, key_index)
        self.values = torch.zeros(0, value_dim, device=device, dtype=torch.float, requires_grad=False)
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.device = device
         
    
    def insert(self, keys, values):
        self.key_index.add(keys.cpu().numpy())
        self.values = torch.cat((self.values, values), dim=0)
    
    
    def get_k_nearest_neighbors(self, keys, values, k=1, average=True):
        if self.key_index.ntotal == 0:
            return torch.zeros((keys.size(0), self.value_dim), device=self.device, dtype=torch.float)
        _, k_nearest_neighbors_ids = self.key_index.search(keys.cpu().numpy(), k)
        k_nearest_neighbors_values = self.values[k_nearest_neighbors_ids, :].clone()
        # update the values of the k nearest neighbors
        self.values[k_nearest_neighbors_ids, :] = values.unsqueeze(1).repeat(1, k, 1)
        if average:
            return k_nearest_neighbors_values.mean(dim=1)
        else:
            return k_nearest_neighbors_values
        
        
    def get_samples(self, n_samples):
        if self.key_index.ntotal <= n_samples:
            return torch.tensor(self.key_index.reconstruct_n(0, self.key_index.ntotal), device=self.device, dtype=torch.float, requires_grad=False), self.values
        else:
            sample_ids = torch.randperm(self.key_index.ntotal, device=self.device, dtype=torch.long, requires_grad=False)[:n_samples]
            samples = torch.tensor(self.key_index.reconstruct_batch(sample_ids.cpu().numpy()), device=self.device, dtype=torch.float, requires_grad=False)
            return samples, self.values[sample_ids, :]
