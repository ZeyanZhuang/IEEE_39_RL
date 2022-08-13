import gym
from ppo.configs import Config
from ppo.agent import PPOAgent
import numpy as np

env = gym.make('CartPole-v0')

config = Config
config.env_config.action_dim = env.action_space.n
config.env_config.state_dim = env.observation_space.shape[0]
config.env_config.action_type = 'discrete'

class PPO:
    def __init__(self, config):
        self.config = config
        self.env = env
        self.agent = PPOAgent(self.config)
        self.state = self.env.reset()

    def rollout_data(self):
        done = False
        for i in range(32):
            action, log_p, value = self.agent.sample_action(self.state)
            next_state, reward, done, _ = self.env.step(action)
            reward /= 10
            trans_dict = {'state': self.state, 'action': action, 'reward': reward, 'done': done, 'value': value, 'log_p': log_p}
            self.agent.buffer.save_trains(**trans_dict)
            self.state = next_state
            if done:
                self.state = self.env.reset()

    def train(self):
        self.rollout_data()
        data = self.agent.rollout_processed_train_datas()
        self.agent.ac_network.prepare_train_datas(data)
        for i in range(5):
            loss = self.agent.ac_network.calculate_loss()
            self.agent.ac_network.opt.zero_grad()
            loss.backward()
            self.agent.ac_network.opt.step()

    def test(self):
        total_reward = 0
        env = gym.make('CartPole-v0')
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
        return total_reward


ppo = PPO(config)
for i in range(15000):
    ppo.train()
    if i % 20 == 0:
        r = ppo.test()
        print("episode: {}, total reward: {}".format(i, r))
