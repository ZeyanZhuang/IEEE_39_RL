import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ppo.layers import ContinuousActionLayer, DiscreteActionLayer, FeatureLayer

class ActorCriticNetwork(nn.Module):

    def __init__(self, configs, **kwargs):
        super(ActorCriticNetwork, self).__init__()
        self.configs = configs
        self.feature_net_type = self.configs.alg_config.model_type
        self.alg_config = configs.alg_config
        self.env_config = configs.env_config
        self.device = self.configs.alg_config.device
        self.build_networks()
        self.create_optimizer()
        self.to(self.device)
        self.model_step = 0
        if self.feature_net_type == 'GAT':
            self.adj = torch.from_numpy(kwargs.get('adj')).to(self.device)

    def build_networks(self):
        self.feature_net = self.create_feature_net()
        self.policy_net = self.create_policy_net()
        self.value_net = self.create_value_net()
        self.action_head = self.create_action_head(64)

    def create_feature_net(self):
        n_input = 152
        self.feature_dim = n_output = 64
        if self.feature_net_type == 'GAT':
            n_input = 6
            self.feature_dim = n_output = 78
        elif self.feature_net_type == 'FC':
            n_input = self.env_config.state_dim
        feature_net = FeatureLayer(
            n_input,
            n_output,
            model_type=self.feature_net_type
        )
        return feature_net

    def create_policy_net(self):
        input_dim = self.feature_dim
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
        )

    def create_value_net(self):
        input_dim = self.feature_dim
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def create_action_head(self, input_dim=32):
        action_dim = self.env_config.action_dim
        device = self.alg_config.device
        if self.env_config.action_type == 'discrete':
            return DiscreteActionLayer(input_dim, action_dim, device)
        elif self.env_config.action_type == 'continuous':
            return ContinuousActionLayer(input_dim, action_dim, device)

    def create_optimizer(self):
        self.opt = torch.optim.Adam(
            params=self.parameters(),
            lr=self.alg_config.lr
        )

    def forward(self, inputs):
        inputs = self.feature_net(inputs)
        x = self.policy_net(inputs)
        values = self.value_net(inputs)
        if self.env_config.action_type == 'discrete':
            probs = self.action_head(x)
            return [probs], values
        elif self.env_config.action_type == 'continuous':
            mu, cov = self.action_head(x)
            return [mu, cov], values

    def prepare_train_datas(self, train_datas):
        self.state_batch = torch.as_tensor(train_datas.get('states')).float().to(self.device)
        self.action_batch = torch.as_tensor(train_datas.get('actions')).float().to(self.device)
        self.old_log_p = torch.as_tensor(train_datas.get('log_p')).float().to(self.device)
        self.adv = torch.as_tensor(train_datas.get('adv')).float().to(self.device)
        self.y_r = torch.as_tensor(train_datas.get('y_r')).float().to(self.device)

    def calculate_loss(self):
        inputs = self.state_batch
        if self.feature_net_type == 'GAT':
            inputs = (self.state_batch, self.adj)
        param, self.y = self(inputs)
        new_log_p = self.action_head.log_p(param, self.action_batch)
        self.entropy = torch.mean(self.action_head.entropy(param))
        r_theta = torch.exp(new_log_p - self.old_log_p)
        item_1 = r_theta * self.adv
        item_2 = torch.clamp(r_theta, 1 - self.alg_config.clip_eps, 1 + self.alg_config.clip_eps) * self.adv
        loss_policy = - torch.mean(torch.min(item_1, item_2)) * self.alg_config.p_coeff
        loss_entropy = - self.entropy * self.alg_config.e_coeff
        loss_value = F.mse_loss(self.y, self.y_r) * self.alg_config.v_coeff
        self.loss_all = loss_value + loss_policy + loss_entropy
        return self.loss_all

    def get_summary(self):
        y = torch.mean(self.y).item()
        entropy = self.entropy.item()
        loss_all = self.loss_all.item()
        return {'model_step': self.model_step, 'predict_v': y, 'entropy': entropy, 'loss_all': loss_all}

    def inference_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            if self.feature_net_type == 'GAT':
                inputs = (state, self.adj)
            elif self.feature_net_type == 'FC':
                inputs = state
            x = self.feature_net(inputs)
            x = self.policy_net(x)
            action = self.action_head.inference_action(x)
        action = action.cpu().numpy()[0]
        return action

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            if self.feature_net_type == 'GAT':
                inputs = (state, self.adj)
            elif self.feature_net_type == 'FC':
                inputs = state
            feature = self.feature_net(inputs)
            v = self.value_net(feature)
            x = self.policy_net(feature)
            action, log_p = self.action_head.sample_action(x)
        v = v.cpu().numpy().item()
        log_p = log_p.cpu().numpy()[0]
        action = action.cpu().numpy()[0]
        return action, log_p, v


