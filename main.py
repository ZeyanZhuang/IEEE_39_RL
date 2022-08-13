from env import SimulatorPsse
import psspy
from ppo.agent import PPOAgent
from ppo.configs import Config
from torch.utils.tensorboard import SummaryWriter
import numpy as np

psspy.result_table_output(0)
psspy.report_output(6)
psspy.progress_output(6)
psspy.prompt_output(6)
psspy.close_report()


class IEEE39PPOSolution:
    def __init__(self,
                 config=Config()):
        self.env = SimulatorPsse(config)
        self.config = config
        if self.config.alg_config.model_type == 'FC':
            self.agent = PPOAgent(config)
        elif self.config.alg_config.model_type == 'GAT':
            self.agent = PPOAgent(config, adj=self.env.adj)
        self.state_dim = self.config.env_config.state_dim
        self.update_times_per_train = 10
        self.writer = SummaryWriter()
        self.episode_round = 0
        self.save_model_freq = 10000
        self.episode_length = config.env_config.episode_length


    def run_episode_collect_trajectories(self, index):
        ''' 采集一条轨迹 '''
        state = self.env.reset(_index=index)
        for t in range(self.episode_length):
            action, log_p, value = self.agent.sample_action(state)
            next_state, done, reward = self.env.step(action)
            reward *= self.config.env_config.reward_scale
            tr_dict = {
                'state': state, 'action': action, 'reward': reward, 'done': done, 'value': value, 'log_p': log_p
            }
            self.agent.save_trains(**tr_dict)
            state = next_state
            if done:
                break
        # print('steps: ' + str(t))
        self.episode_round += 1

    def evaluate(self, index):
        state = self.env.reset(_index=index)
        total_reward = 0
        for t in range(self.episode_length):
            action = self.agent.inference_action(state)
            next_state, done, reward = self.env.step(action)
            total_reward += reward * self.config.env_config.reward_scale
            state = next_state
            if done:
                break
        bus_voltage = state[:34]
        shunt_state = self.env.system_shunt_C.copy()
        shunt_state.update(self.env.system_shunt_X.copy())
        wind_Q = self.env.get_wind_Q()
        evaluate_summary = {'total_reward': total_reward,
                            'final_net_loss': self.env.net_loss_P,
                            'adjust_steps': t + 1,
                            'done_reward': self.env.done_reward}

        self.write_evaluate_summary(evaluate_summary)
        return evaluate_summary, bus_voltage, shunt_state, wind_Q

    def train(self, index):
        while self.agent.get_buffer_size() < self.episode_length:
            self.run_episode_collect_trajectories(index)
        data = self.agent.rollout_processed_train_datas()
        self.agent.ac_network.prepare_train_datas(data)
        for i in range(self.update_times_per_train):
            self.agent.train()
        if self.agent.ac_network.model_step % 100 == 0:
            self.write_train_summary()
        if self.agent.ac_network.model_step % self.save_model_freq == 0:
            self.agent.save_ckpt()

    def load_ac_model_ckpt(self, model_step):
        self.agent.load_ckpt(model_step)

    def write_train_summary(self):
        summary = self.agent.get_summary()
        model_step = summary['model_step']
        del summary['model_step']
        for k, v in summary.items():
            self.writer.add_scalar('train/' + k, float(v), model_step)

    def write_evaluate_summary(self, summary):
        for k, v in summary.items():
            self.writer.add_scalar('evaluate/' + k, float(v), self.episode_round)


if __name__ == '__main__':
    import tqdm
    solution = IEEE39PPOSolution()
    # cnt = 0
    # for i in tqdm.tqdm(range(5000)):
    #     solution.env.reset(i)
    #     if solution.env.flag == 0:
    #         cnt += 1
    #     if i % 500 == 0:
    #         print('[{}/{}]'.format(cnt, i))
    # print(cnt)

    for i in range(1000000):
        index = np.random.randint(0, 5001)
        index = 20
        solution.train(index)
        if i % 10 == 1:
            summary, bus, shunt_state, wind_Q = solution.evaluate(index)
            print('train episode: {}, total reward: {}, done reward: {}, final net loss: {}, adjust_steps: {}'.format(solution.episode_round,
                                                                                                    summary['total_reward'],
                                                                                                    summary['done_reward'],
                                                                                                    summary['final_net_loss'],
                                                                                                    summary['adjust_steps']))
            print('bus voltage: ', bus)
            print('shunt state: ', shunt_state)
            print('wind Q: ', wind_Q)

