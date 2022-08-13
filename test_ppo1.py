import gym
from ppo.configs import Config
from ppo.agent import PPOAgent
import numpy as np

env = gym.make('Pendulum-v1')
# env = env.unwrapped

config = Config
config.env_config.action_dim = 1
config.env_config.state_dim = 3


class PPO:
    def __init__(self, config):
        self.config = config
        self.env = env
        self.agent = PPOAgent(self.config)
        self.state = self.env.reset()

    def rollout_data(self):
        done = False
        self.state = self.env.reset()
        while not done:
            action, log_p, value = self.agent.sample_action(self.state)
            next_state, reward, done, _ = self.env.step(action)
            reward /= 10
            trans_dict = {'state': self.state, 'action': action, 'reward': reward, 'done': done, 'value': value, 'log_p': log_p}
            self.agent.buffer.save_trains(**trans_dict)
            self.state = next_state

    def train(self):
        self.rollout_data()
        data = self.agent.rollout_processed_train_datas()
        self.agent.ac_network.prepare_train_datas(data)
        for i in range(10):
            loss = self.agent.ac_network.calculate_loss()
            self.agent.ac_network.opt.zero_grad()
            loss.backward()
            self.agent.ac_network.opt.step()

    def test(self):
        total_reward = 0
        env = gym.make('Pendulum-v1')
        state = env.reset()
        done = False
        while not done:
            action = self.agent.ac_network.inference_action(state)
            # random_action = np.random.choice(self.config.env_config.action_dim)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            # env.render()
        env.close()
        return total_reward, self.agent.ac_network.action_head.log_sigma.detach().item()


ppo = PPO(config)
for i in range(15000):
    ppo.train()
    if i % 20 == 0:
        r, s = ppo.test()
        print("episode: {}, total reward: {}, log sigma: {}".format(i, r, s))
