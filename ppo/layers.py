import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal, Categorical
import torch.nn.functional as F
import numpy as np
from utils.util_cls import GAT


class ActionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(ActionLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.device = device
        self.mu_scale = 1


class ContinuousActionLayer(ActionLayer):
    def __init__(self, input_dim, output_dim, device):
        super(ContinuousActionLayer, self).__init__(input_dim, output_dim, device)
        self.layer_mu = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh()
        )
        self.log_sigma = nn.Parameter(torch.zeros(1, output_dim))  # 对角矩阵

    def forward(self, x):
        mu = self.layer_mu(x) * self.mu_scale
        return mu, self.log_sigma

    def sample_action(self, x):
        batch_size = x.shape[0]
        mu = self.layer_mu(x) * self.mu_scale
        sigma = torch.exp(self.log_sigma)
        action = mu + torch.randn(batch_size, self.output_dim).to(self.device) * sigma
        log_p = self.log_p([mu, self.log_sigma], action)
        return action, log_p

    def inference_action(self, x):
        return self.layer_mu(x) * self.mu_scale

    def log_p(self, p, action):
        mu, log_sigma = p
        sigma = torch.exp(log_sigma)
        log_p = - 0.5 * action.shape[1] * np.log(2 * np.pi) \
                - torch.sum(log_sigma, dim=1) \
                - 0.5 * torch.sum(((action - mu) / sigma) ** 2, dim=1)
        return log_p.unsqueeze(1)

    def entropy(self, p):
        mu, log_sigma = p
        return 0.5 * self.output_dim * (np.log(2 * np.pi) + 1) + torch.sum(log_sigma, dim=1)


class DiscreteActionLayer(ActionLayer):
    def __init__(self, input_dim, output_dim, device):
        super(DiscreteActionLayer, self).__init__(input_dim, output_dim, device)
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return F.softmax(self.layer(x), dim=-1)

    def sample_action(self, x):
        probs = self(x)
        dist = Categorical(probs)
        action = dist.sample()
        log_p = dist.log_prob(action)
        return action, log_p

    def inference_action(self, x):
        probs = F.softmax(self(x), dim=-1)
        return torch.argmax(probs, dim=1)

    def log_p(self, p, action):
        if action.type != torch.long:
            action = action.long()
        probs = p[0]
        probs_action = torch.stack([prob[a[0]] for prob, a in zip(probs, action)]).unsqueeze(1)
        log_p = torch.log(probs_action + 1e-6)
        return log_p

    def entropy(self, p):
        probs = p[0]
        return - torch.sum(torch.log(probs + 1e-6) * probs, dim=1)


class FeatureLayer(nn.Module):
    def __init__(self, input_dim, output_dim=64, model_type='FC'):
        super(FeatureLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_type = model_type
        if self.model_type == 'FC':
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
                nn.ReLU(),
            )
        elif self.model_type == 'GAT':
            self.layers = GAT(
                nfeat=input_dim,
                nhid=8,
                nout=2,
                alpha=0.2,
                nheads=8
            )

    def forward(self, inputs):
        if self.model_type == 'FC':
            y = self.layers(inputs)
            return y
        elif self.model_type == 'GAT':
            x, adj = inputs
            y = self.layers(x, adj)
            return y
