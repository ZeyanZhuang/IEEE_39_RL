import numpy as np


class PPOBuffer:
    def __init__(self):
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.value_buffer = []
        self.log_p_buffer = []
        self.size = 0

    def save_trains(self, state, action, reward, done, value, log_p):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.value_buffer.append(value)
        self.log_p_buffer.append(log_p)
        self.size += 1

    def clear(self):
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.value_buffer = []
        self.log_p_buffer = []
        self.size = 0
