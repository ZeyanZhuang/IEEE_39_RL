import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from ppo.networks import ActorCriticNetwork
from ppo.buffer import PPOBuffer


class PPOAgent:
    def __init__(self, config, **kwargs):
        self.config = config
        if self.config.alg_config.model_type == 'GAT':
            self.adj = kwargs.get('adj', None)
            self.ac_network = ActorCriticNetwork(self.config, adj=self.adj)
        elif self.config.alg_config.model_type == 'FC':
            self.ac_network = ActorCriticNetwork(self.config)
        self.buffer = PPOBuffer()
        self.gamma = self.config.env_config.gamma

    def rollout_processed_train_datas(self):
        data = dict()
        data['states'] = np.array(self.buffer.state_buffer, dtype=np.float32)
        data['actions'] = np.array(self.buffer.action_buffer, dtype=np.float32)
        data['rewards'] = np.array(self.buffer.reward_buffer, dtype=np.float32)
        data['log_p'] = np.array(self.buffer.log_p_buffer, dtype=np.float32)
        for k, v in data.items():
            if len(data[k].shape) < 2:
                data[k] = data[k][:, None]

        buffer_len = len(self.buffer.reward_buffer)
        y_r = np.zeros([buffer_len, 1], dtype=np.float32)
        G = 0
        for i in reversed(range(buffer_len)):
            if self.buffer.done_buffer[i] == 1:
                G = 0
            G = data['rewards'][i] + self.gamma * G
            y_r[i, 0] = G
        data['y_r'] = y_r
        V = np.array(self.buffer.value_buffer, dtype=np.float32)[:, None]
        data['adv'] = y_r - V
        data['length'] = len(V)
        self.buffer.clear()
        return data

    def sample_action(self, state):
        return self.ac_network.sample_action(state)

    def inference_action(self, state):
        return self.ac_network.inference_action(state)

    def save_trains(self, **kwargs):
        self.buffer.save_trains(**kwargs)

    def get_buffer_size(self):
        return self.buffer.size

    def train(self):
        loss = self.ac_network.calculate_loss()
        self.ac_network.opt.zero_grad()
        loss.backward()
        self.ac_network.opt.step()
        self.ac_network.model_step += 1
        return loss.item()

    def get_summary(self):
        return self.ac_network.get_summary()

    def save_ckpt(self):
        torch.save(self.ac_network.state_dict(), './model/' + str(self.ac_network.model_step))

    def load_ckpt(self, model_step):
        state_dict = torch.load('./model/' + str(model_step))
        self.ac_network.load_state_dict(state_dict)

