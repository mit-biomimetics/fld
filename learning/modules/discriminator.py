import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd


class Discriminator(nn.Module):
    def __init__(self,
                 observation_dim,
                 observation_horizon,
                 device,
                 reward_coef=0.1,
                 reward_lerp=0.3,
                 shape=[1024, 512],
                 style_reward_function="quad_mapping",
                 **kwargs,
                 ):
        if kwargs:
            print("Discriminator.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super(Discriminator, self).__init__()
        self.observation_dim = observation_dim
        self.observation_horizon = observation_horizon
        self.input_dim = observation_dim * observation_horizon
        self.device = device
        self.reward_coef = reward_coef
        self.reward_lerp = reward_lerp
        self.style_reward_function = style_reward_function
        self.shape = shape

        discriminator_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.shape:
            discriminator_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            discriminator_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        discriminator_layers.append(nn.Linear(self.shape[-1], 1))
        self.architecture = nn.Sequential(*discriminator_layers).to(self.device)
        self.architecture.train()

    def forward(self, x):
        return self.architecture(x)

    def compute_grad_pen(self, expert_state_buf, lambda_=10):
        expert_data = expert_state_buf.flatten(1, 2)
        expert_data.requires_grad = True

        disc = self.architecture(expert_data)
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_wasabi_reward(self, state_buf, task_reward, dt, state_normalizer=None, style_reward_normalizer=None):
        with torch.no_grad():
            self.eval()
            if state_normalizer is not None:
                for i in range(self.observation_horizon):
                    state_buf[:, i] = state_normalizer.normalize(state_buf[:, i].clone())
            d = self.architecture(state_buf.flatten(1, 2))
            if self.style_reward_function == "quad_mapping":
                style_reward = self.reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            elif self.style_reward_function == "log_mapping":
                style_reward = -torch.log(torch.maximum(1 - 1 / (1 + torch.exp(-d)), torch.tensor(0.0001, device=self.device)))
            elif self.style_reward_function == "wasserstein_mapping":
                if style_reward_normalizer is not None:
                    style_reward = style_reward_normalizer.normalize(d.clone())
                    style_reward_normalizer.update(d)
                else:
                    style_reward = d
                style_reward = self.reward_coef * style_reward
            else:
                raise ValueError("Unexpected style reward mapping specified")
            style_reward = style_reward * dt
            reward = (1.0 - self.reward_lerp) * style_reward + self.reward_lerp * task_reward.unsqueeze(-1)
            self.train()
        return reward.squeeze(), style_reward.squeeze()
