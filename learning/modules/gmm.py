import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture as skGaussianMixture

class GaussianMixtures(nn.Module):
    
    def __init__(self, min_n_components, max_n_components, n_features, device, covariance_type="full"):
        super(GaussianMixtures, self).__init__()
        self.candidates = nn.ModuleList(
            [
                GaussianMixture(
                    n_components,
                    n_features,
                    device,
                    covariance_type=covariance_type,
                    )
                for n_components in range(min_n_components, max_n_components + 1)
                ]
        )
        self.num_gmms = len(self.candidates)
        self.n_features = n_features
        self.device = device
    
    
    def aic(self, x):
        aics = torch.zeros(self.num_gmms, device=self.device, dtype=torch.float, requires_grad=False)
        for i in range(self.num_gmms):
            aics[i] = self.candidates[i].aic(x)
        return aics

    
    def bic(self, x):
        bics = torch.zeros(self.num_gmms, device=self.device, dtype=torch.float, requires_grad=False)
        for i in range(self.num_gmms):
            bics[i] = self.candidates[i].bic(x)
        return bics
            

    def fit(self, x):
        for i in range(self.num_gmms):
            self.candidates[i].fit(x)


    def get_best_gmm_idx(self, x, criterion="bic"):
        if criterion == "bic":
            bics = self.bic(x)
            idx = torch.argmin(bics)
        elif criterion == "aic":
            aics = self.aic(x)
            idx = torch.argmin(aics)
        else:
            raise ValueError("[GMMs] Invalid criterion.")
        return idx


    def sample(self, n, idx):
        alp_means = self.candidates[idx].mu[:, -1]
        k = torch.multinomial(alp_means, n, replacement=True)
        samples = self.candidates[idx].sample_class(n, k)
        return samples


class GaussianMixture(nn.Module):
    def __init__(self, n_components, n_features, device, covariance_type="full"):
        super().__init__()

        self.n_components = n_components
        self.n_features = n_features
        self.device = device
        self.covariance_type = covariance_type
        
        self.gmm = skGaussianMixture(n_components=n_components, covariance_type=covariance_type, n_init=10)
        self._init_params()
        

    def _init_params(self):
        self.mu = nn.Parameter(torch.randn(self.n_components, self.n_features, device=self.device), requires_grad=False)
        if self.covariance_type == "diag":
            self.var = nn.Parameter(torch.ones(self.n_components, self.n_features, device=self.device), requires_grad=False)
            self.var_chol = nn.Parameter(torch.ones(self.n_components, self.n_features, device=self.device), requires_grad=False)
        elif self.covariance_type == "full":
            self.var = nn.Parameter(torch.eye(self.n_features, device=self.device).repeat(self.n_components, 1, 1), requires_grad=False)
            self.var_chol = nn.Parameter(torch.eye(self.n_features, device=self.device).repeat(self.n_components, 1, 1), requires_grad=False)
        else:
            raise Exception("[GaussianMixture] __init__ got invalid covariance_type: {}".format(self.covariance_type))
        self.pi = nn.Parameter(torch.ones(self.n_components, device=self.device) / self.n_components, requires_grad=False)
        
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.gmm.means_ = self.mu.detach().cpu().numpy()
        self.gmm.covariances_ = self.var.detach().cpu().numpy()
        self.gmm.weights_ = self.pi.detach().cpu().numpy()
        if self.covariance_type == "diag":
            self.gmm.precisions_cholesky_ = torch.linalg.inv(self.var).detach().cpu().numpy()
        elif self.covariance_type == "full":
            self.gmm.precisions_cholesky_ = torch.linalg.cholesky(torch.linalg.inv(self.var)).detach().cpu().numpy()
        else:
            raise Exception("[GaussianMixture] __init__ got invalid covariance_type: {}".format(self.covariance_type))

        
    def set_variances(self, var):
        self.var[:] = var
        if self.covariance_type == "diag":
            self.var_chol[:] = var.sqrt()
        elif self.covariance_type == "full":
            self.var_chol[:] = torch.linalg.cholesky(var)
        else:
            raise Exception("[GaussianMixture] __init__ got invalid covariance_type: {}".format(self.covariance_type))
        self.gmm.covariances_ = var.detach().cpu().numpy()
        self.gmm.precisions_cholesky_ = torch.linalg.cholesky(torch.linalg.inv(var)).detach().cpu().numpy()


    def aic(self, x):
        x = x.cpu().numpy()
        aic = self.gmm.aic(x)
        return aic


    def bic(self, x):
        x = x.cpu().numpy()
        bic = self.gmm.bic(x)
        return bic


    def fit(self, x):
        x = x.cpu().numpy()
        self.gmm.fit(x)
        self.mu[:] = torch.tensor(self.gmm.means_, device=self.device, dtype=torch.float, requires_grad=False)
        self.var[:] = torch.tensor(self.gmm.covariances_, device=self.device, dtype=torch.float, requires_grad=False)
        self.pi[:] = torch.tensor(self.gmm.weights_, device=self.device, dtype=torch.float, requires_grad=False)
        if self.covariance_type == "diag":
            self.var_chol[:] = self.var.sqrt()
        elif self.covariance_type == "full":
            self.var_chol[:] = torch.linalg.cholesky(self.var)
        else:
            raise Exception("[GaussianMixture] __init__ got invalid covariance_type: {}".format(self.covariance_type))


    def predict(self, x):
        x = x.cpu().numpy()
        y = self.gmm.predict(x)
        return torch.tensor(y, device=self.device, dtype=torch.long, requires_grad=False)


    def predict_proba(self, x):
        x = x.cpu().numpy()
        resp = self.gmm.predict_proba(x)
        return torch.tensor(resp, device=self.device, dtype=torch.float, requires_grad=False)


    def sample(self, n):
        x, y = self.gmm.sample(n)
        return torch.tensor(x, device=self.device, dtype=torch.float, requires_grad=False), torch.tensor(y, device=self.device, dtype=torch.long, requires_grad=False)


    def sample_class(self, n, k):
        mu_k = self.mu[k, :]
        var_chol_k = self.var_chol[k, :]
        if self.covariance_type == "diag":
            return torch.randn(n, self.n_features, device=self.device, dtype=torch.float, requires_grad=False) * var_chol_k + mu_k
        elif self.covariance_type == "full":
            return (var_chol_k @ torch.randn(n, self.n_features, 1, device=self.device, dtype=torch.float, requires_grad=False)).squeeze(-1) + mu_k
        else:
            raise Exception("[GaussianMixture] __init__ got invalid covariance_type: {}".format(self.covariance_type))


    def score(self, x):
        x = x.cpu().numpy()
        score = self.gmm.score(x)
        return torch.tensor(score, device=self.device, dtype=torch.float, requires_grad=False)


    def score_samples(self, x):
        x = x.cpu().numpy()
        score_samples = self.gmm.score_samples(x)
        return torch.tensor(score_samples, device=self.device, dtype=torch.float, requires_grad=False)
    
    
    def get_block_parameters(self, latent_dim):
        mu = []
        var = []
        num_slices = int(self.n_features / latent_dim)
        for i in range(num_slices):
            mu.append(self.mu[:, i * latent_dim:(i + 1) * latent_dim])
            if self.covariance_type == "full":
                var.append(self.var[:, i * latent_dim:(i + 1) * latent_dim, i * latent_dim:(i + 1) * latent_dim])
            else:
                var.append(self.var[:, i * latent_dim:(i + 1) * latent_dim])
        return mu, var