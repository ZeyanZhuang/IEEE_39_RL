import numpy as np
import time
import sys
import openpyxl
from pandas import read_csv
import os
import math
import pandas as pd
import importlib
import csv
import xlrd
import re
import math

# PSSE_path1 = r'D:\Program Files\PTI\PSSE35\35.0\PSSPY27'  # path of the psspy.pyc
PSSE_path_os = r'D:\Program Files\PTI\PSSE35\35.2\PSSBIN'  # path of the psse.exe
PSSE_path2 = r'D:\Program Files\PTI\PSSE35\35.2\PSSPY37'  # path of the psspy.pyc
# sys.path.append(PSSE_path1)
sys.path.append(PSSE_path_os)
sys.path.append(PSSE_path2)
# os.environ['PATH'] += ';' + PSSE_path1
os.environ['PATH'] += ';' + PSSE_path_os
os.environ['PATH'] += ';' + PSSE_path2
# import psse35
import psspy
from psspy import _i   # 默认整数
from psspy import _f   #浮点数
from psspy import _s   # 字符串

_i = psspy.getdefaultint()
_f = psspy.getdefaultreal()
_s = psspy.getdefaultchar()

psspy.result_table_output(0)
# psspy.report_output(6)
# psspy.progress_output(6)
# psspy.prompt_output(6)
psspy.close_report()
psspy.psseinit()
opensav = r"""E:\PPOnew\files\IEEE 39 bus.sav"""  # find the .sav file
psspy.case(opensav)


def f(x, x_max, x_min):
    return np.abs((2 * x - x_max - x_min) / (x_max - x_min))


class SimulatorPsse():
    def __init__(self, configs):
        self.configs = configs
        self.state_dim = 137  # 34 个 bus 的电压 + 29 个 bus 有功无功 + 5 台火机有功无功 + 5 台风机有功无功 + 24 条支路电流 + 系统有功网损
        self.action_dim = 25
        self.system_state = np.zeros(self.state_dim, dtype=np.float32)
        self.need_graph_data = False
        if self.configs.alg_config.model_type == 'GAT':
            self.need_graph_data = True
            self.graph_feature_dim = 6 # [电压， 有功， 无功， 开关状态， 电流， 网损]
            self.system_graph_state = np.zeros([39, self.graph_feature_dim], dtype=np.float32)
            self.adj = pd.read_excel('datas/constrain/网络的拓扑关系.xlsx').values[:, 1:]
        self.done = False
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.reward = 0
        self.fire_machine_bus_index = [30, 31, 33, 34, 36]  # 火机编号
        self.wind_machine_bus_index = [32, 35, 37, 38, 39]  # 风机编号
        self.shunt_C_bus_index = [3, 4, 5, 6, 7, 8, 12]  # 电容所在母线对号
        self.shunt_X_bus_index = [1, 2, 9, 25, 26, 27, 28, 29]  # 电抗所在母线
        self.shunt_bus_index = self.shunt_C_bus_index + self.shunt_X_bus_index  # 总的电容电抗
        # self.init_C_bus_index = [3, 4, 5, 6, 7, 8, 12] # 需要初始化的电容
        self.init_C_bus_index = {3: 11, 4: 14, 5: 2, 6: 2, 7: 5, 8: 4, 12: 3}  # 需要初始化的电容
        # self.init_X_bus_index = [1, 28, 29] # 需要初始化的电抗
        self.init_X_bus_index = {1: 6, 28: 5, 29: 3}  # 需要初始化的电抗

        self.system_shunt_C = {index: 0 for index in self.shunt_C_bus_index}
        self.system_shunt_X = {index: 0 for index in self.shunt_X_bus_index}

        for k, v in self.init_C_bus_index.items():
            self.system_shunt_C[k] = v

        for k, v in self.init_X_bus_index.items():
            self.system_shunt_X[k] = v

        self.load_bus_index = [i for i in range(1, 30)]  # 负载母线号
        self.load_all_bus_index = self.load_bus_index + self.wind_machine_bus_index  # 负载 + 风机
        self.net_loss_P = 0
        self.net_loss_Q = 0
        self.read_sheet()  # 读取表格
        self.num_G = 15  # 档位
        self.n_branchs = 24  # 分支个数
        self.wind_machine_bound_Q = [-130, -130, -278.98, -166, -200]
        self.flag = 0  # 潮流计算是否收敛，1 收敛
        self.branch_index = []
        self.init_branch()

        self.n_steps = 0

        self.wind_machine_cons_Q = [[-130, 0],
                                    [-130, 0],
                                    [-278.98, 0],
                                    [-166, 0],
                                    [-200, 0]
                                    ]
        self.wind_machine_cons_P = [[-1725, 0],
                                    [- 1687, 0],
                                    [- 1564, 0],
                                    [- 1865, 0],
                                    [- 1100, 0],
                                    ]
        self.wind_machine_cons_V = [0.99, 1.06]
        self.load_cons_V = [0.95, 1.05]
        self.fire_machine_cons_Q = [[0, 400],
                                    [-100, 400],
                                    [0, 350],
                                    [0, 367],
                                    [0, 340]
                                    ]
        self.fire_machine_cons_P = [[0, 1040],
                                    [0, 680],
                                    [0, 652],
                                    [0, 508],
                                    [0, 580]
                                    ]
        self.fire_machine_cons_V = [0.9, 1.1]
        self.branch_cons = [600
            , 1000
            , 2000
            , 2500
            , 1500
            , 1200
            , 1200
            , 1500
            , 2000
            , 1800
            , 900
            , 2500
            , 2500
            , 2000
            , 600
            , 600
            , 1800
            , 2500
            , 1500
            , 600
            , 2500
            , 1500
            , 1200
            , 600]
        self.lamda_I, self.lamda_U, self.lamda_P, self.lamda_Q = 0.25, 0.25, 0.25, 0.25

        self.done_reward = 0
        self.max_episode_length = configs.env_config.episode_length

    def read_sheet(self):
        self.df_gen = pd.read_excel('./files/generators.xlsx')
        self.df_load = pd.read_excel('./files/loads.xlsx')
        self.df_shunt = pd.read_excel('./files/shunt.xlsx')
        self.df_branch = pd.read_excel('./files/brantch_ijbus_24.xlsx')
        self.df_fire_machine_P = pd.read_excel('E:/PPOnew/datas/train/火力发电机的有功数据-数据集.xlsx')
        self.df_load_P_Q = pd.read_excel('E:/PPOnew/datas/train/负荷的有功和无功数据-数据集.xlsx')
        self.df_net_loss_P = pd.read_excel('E:/PPOnew/datas/train/预设有功网损数据-数据集.xlsx')
        self.wind_machine_P_Q = pd.read_excel('E:/PPOnew/datas/train/风机的有功和无功数据-数据集.xlsx')

    def init_branch(self):
        from_ = self.df_branch.iloc[:, 0].values
        to_ = self.df_branch.iloc[:, 2].values
        self.branch_index = [[i, j] for i, j in zip(from_, to_)]

    def init_psse_env_parameters(self, index):
        '''初始化 IEEE 系统要用的数据， 根据 index 这条算例'''
        self.fire_machine_bus_P_ = self.df_fire_machine_P.values[index, 1:]
        # self.fire_machine_bus_P_ = self.df_gen.values[:, 13]
        self.fire_machine_bus_Q_ = self.df_gen.values[:, 16]
        self.fire_machine_bus_V_ = self.df_gen.values[:, 10]
        self.load_bus_P_ = self.df_load_P_Q.values[index, 1:30]
        self.load_bus_Q_ = self.df_load_P_Q.values[index, 30:]
        # self.load_bus_P_ = self.df_load.values[:19, 15]
        # self.load_bus_Q_ = self.df_load.values[:19, 16]
        self.need_init_load_bus_index = self.df_load.values[:19, 0]
        self.wind_machine_bus_P_ = self.wind_machine_P_Q.values[index, 1:6]
        self.wind_machine_bus_Q_ = self.wind_machine_P_Q.values[index, 6:]

        # self.wind_machine_bus_P_ = self.df_load.values[19:, 15]
        # self.wind_machine_bus_Q_ = self.df_load.values[19:, 16]
        self.ref_net_loss_P = self.df_net_loss_P.values[index, 1]

    # def reset_shunt_bus(self):
    #     # 初始化电容档位
    #     for i in range(7):
    #         for j in range(self.num_G):
    #             psspy.shunt_data(self.shunt_C_bus_index[i],
    #                              str(j),
    #                              0,
    #                              [_f, _f])
    #
    #     # 初始化电抗档位
    #     for i in range(8):
    #         for j in range(self.num_G):
    #             psspy.shunt_data(self.shunt_X_bus_index[i],
    #                              str(j),
    #                              0,
    #                              [_f, _f])
    #
    #     for index, K in self.init_C_bus_index.items():
    #         for j in range(K):
    #             psspy.shunt_data(index, str(j), 1, [_f, _f])
    #
    #     for index, K in self.init_X_bus_index.items():
    #         for j in range(K):
    #             psspy.shunt_data(index, str(j), 1, [_f, _f])

    def reset(self, _index):
        self.init_psse_env_parameters(_index)
        opensav = r"""E:\PPOnew\files\IEEE 39 bus.sav"""  # find the .sav file
        psspy.case(opensav)
        # 初始化火机有功

        for i in range(5):
            psspy.machine_data_3(self.fire_machine_bus_index[i],
                                 '1',
                                 [1],
                                 [self.fire_machine_bus_P_[i], _f])
            # ierr, P = psspy.macdat(self.fire_machine_bus_index[i], '1', 'P')
        # 初始化火机电压
        for i in range(5):
            psspy.bus_data_4(self.fire_machine_bus_index[i],
                             0,
                             [_i, _i, _i, _i],
                             [_f, self.fire_machine_bus_V_[i], _f, _f, _f, _f, _f],
                             _s)
            # ierr, v = psspy.busdat(self.fire_machine_bus_index[i], 'PU')
        # 初始化负载数据
        for i in range(29):
            psspy.load_data_4(self.load_bus_index[i],
                              '1',
                              [1],
                              [self.load_bus_P_[i], self.load_bus_Q_[i]])
            # ierr, P_LOAD = psspy.loddt2(self.load_bus_index[i], '1', 'MVA', 'NOM')
            # ierr, P = psspy.loddt2(self.wind_machine_bus_index[i], '1', 'P')
        # 初始化风机负荷
        for i in range(5):
            psspy.load_data_4(self.wind_machine_bus_index[i],
                              '1',
                              [1],
                              [self.wind_machine_bus_P_[i], self.wind_machine_bus_Q_[i]])
            # ierr, P_wind = psspy.loddt2(self.wind_machine_bus_index[i], '1', 'MVA')
        # # 初始化电容档位
        for i in range(7):
            for j in range(self.num_G):
                psspy.shunt_data(self.shunt_C_bus_index[i],
                                 str(j),
                                 0,
                                 [0, 10])

        # 初始化电抗档位
        for i in range(8):
            for j in range(self.num_G):
                psspy.shunt_data(self.shunt_X_bus_index[i],
                                 str(j),
                                 0,
                                 [0, -10])


        for index, K in self.init_C_bus_index.items():
            for j in range(K):
                psspy.shunt_data(index, str(j), 1, [_f, _f])

        for index, K in self.init_X_bus_index.items():
            for j in range(K):
                psspy.shunt_data(index, str(j), 1, [_f, _f])

        self.flag = self.power_flow()
        # print('init flag: ' + str(self.flag))
        ierr, net_loss_PQ = psspy.systot('LOSS')  # 获得网损
        self.net_loss_P = net_loss_PQ.real
        self.net_loss_Q = net_loss_PQ.imag
        self.n_steps = 0
        self.reward = 0
        self.done_reward = 0
        self.system_shunt_C = {index: 0 for index in self.shunt_C_bus_index}
        self.system_shunt_X = {index: 0 for index in self.shunt_X_bus_index}

        for k, v in self.init_C_bus_index.items():
            self.system_shunt_C[k] = v

        for k, v in self.init_X_bus_index.items():
            self.system_shunt_X[k] = v

        self.calculate_system_state()
        if self.need_graph_data:
            return self.system_graph_state
        else:
            return self.system_state

    def get_wind_Q(self):
        wind_Q_list = []
        for i in range(5):  # 风机无功
            ierr, PQ = psspy.loddt2(self.wind_machine_bus_index[i], '1', 'MVA', 'ACT')
            Q = PQ.imag
            wind_Q_list.append(Q)
        return wind_Q_list

    def power_flow(self):
        ''' 进行潮流计算返回收敛标志和网损的 PQ 值 '''
        psspy.fnsl([0, 0, 0, 0, 0, 0, -1, 0])
        ival = psspy.solved()  # 潮流计算成功判断
        if ival == 0:
            flag = 1  # 收敛
        else:
            flag = 0
        return flag

    def get_shunt_gear(self, x):
        if x > 1:
            x = 1
        elif x < -1:
            x = -1
        return int(np.floor((x + 1) * 2))

    def calculate_system_state(self):
        if self.need_graph_data:
            # [电压， 有功， 无功， 电流， 开关状态， 网损]
            # [0  ,   1,    2,     3,       4,    5]
            self.system_graph_state = np.zeros([39, self.graph_feature_dim], dtype=np.float32)
        state = []
        # bus1 - 29 电压
        for index in self.load_bus_index:
            ierr, v = psspy.busdat(index, 'PU')
            state.append(v)
            if self.need_graph_data:
                self.system_graph_state[index - 1, 0] = v
        # 5 风机电压
        for index in self.wind_machine_bus_index:
            ierr, v = psspy.busdat(index, 'PU')
            state.append(v)
            if self.need_graph_data:
                self.system_graph_state[index - 1, 0] = v
        # bus1-29 有功无功
        for i, (P, Q) in enumerate(zip(self.load_bus_P_, self.load_bus_Q_)):
            P_, Q_ = P / 700, Q / 150
            state += [P_, Q_]
            if self.need_graph_data:
                self.system_graph_state[i, 1] = P_
                self.system_graph_state[i, 2] = Q_
        # 5 火机有功无功
        for index in self.fire_machine_bus_index:
            ierr, P = psspy.macdat(index, '1', 'P')
            P_ = P / 1000
            state.append(P_)
            ierr, Q = psspy.macdat(index, '1', 'Q')
            Q_ = Q / 1000
            state.append(Q_)
            if self.need_graph_data:
                self.system_graph_state[index - 1, 1] = P_
                self.system_graph_state[index - 1, 2] = Q_
        # 5 风机有功无功
        for index in self.wind_machine_bus_index:
            ierr, PQ = psspy.loddt2(index, '1', 'MVA', 'ACT')
            P_, Q_ = PQ.real / 1800, PQ.imag / 300
            state += [P_, Q_]
            if self.need_graph_data:
                self.system_graph_state[index - 1, 1] = P_
                self.system_graph_state[index - 1, 2] = Q_
        # 24 条支路的功率
        for i in range(24):
            bus_i, bus_j = self.branch_index[i]
            ierr, rval = psspy.brnmsc(bus_i, bus_j, '1', 'MVA')
            rval_ = rval / 2500
            state.append(rval_)
            if self.need_graph_data:
                self.system_graph_state[bus_i - 1, 3] = rval_
                self.system_graph_state[bus_j - 1, 3] = rval_
        # 15 开关状态
        for index, v in self.system_shunt_C.items():
            state.append(v / self.num_G)
            if self.need_graph_data:
                self.system_graph_state[index - 1, 4] = v / self.num_G
        for index, v in self.system_shunt_X.items():
            state.append(v / self.num_G)
            if self.need_graph_data:
                self.system_graph_state[index - 1, 4] = v / self.num_G
        # 1 有功网损
        net_loss_P_ = self.net_loss_P / 200
        state.append(net_loss_P_)
        if self.need_graph_data:
            for i in range(39):
                self.system_graph_state[i, 5] = net_loss_P_
        self.system_state = np.array(state, dtype=np.float32)

    def get_bounded_wind_machine_Q(self, Q, i):
        if Q < self.wind_machine_bound_Q[i]:
            Q = self.wind_machine_bound_Q[i]
        if Q > 0:
            Q = 0
        return Q

    def reward_func(self):
        if self.flag == 0:
            # 不收敛
            self.done_reward = -20
            return self.done_reward
        if self.n_steps >= self.max_episode_length:
            self.done = True
            self.done_reward = -10
            return self.done_reward
        reward_V = self.voltage_punish()
        if reward_V < 0:
            # 电压越限
            return reward_V
        # if 1.05 * self.ref_net_loss_P > self.net_loss_P:
        #     r_f_t = self.net_param_punish()
        #     r_r_t = ((self.ref_net_loss_P - self.net_loss_P) / 100 + 0.2) * 100
        #     self.done = True
        #     self.done_reward = r_f_t + r_r_t
        #     return self.done_reward
        # reward = ((self.ref_net_loss_P - self.net_loss_P) / 100 + 0.2) * 5
        # return reward
        r_f_t = self.net_param_punish()
        # r_f_t = 0
        r_r_t = ((self.ref_net_loss_P - self.net_loss_P) / 100 + 0.2) * 100
        self.done = True
        self.done_reward = r_f_t + r_r_t
        if self.ref_net_loss_P > self.net_loss_P:
            self.done_reward += 20
        return self.done_reward

    def voltage_punish(self):
        bus_voltage = self.system_state[:29]
        bus_voltage_high = np.sum(bus_voltage > 1.05)
        bus_voltage_low = np.sum(bus_voltage < 0.95)
        wind_machine_voltage = self.system_state[29:34]
        wind_machine_voltage_high = np.sum(wind_machine_voltage > 1.06)
        wind_machine_voltage_low = np.sum(wind_machine_voltage < 0.99)

        return (bus_voltage_high + bus_voltage_low + wind_machine_voltage_high + wind_machine_voltage_low) * - 0.05

    def net_param_punish(self):
        I = self.system_state[112:136]
        R_I = 0
        for i in range(24):
            R_I += (I[i] * 2500 - self.branch_cons[i]) / self.branch_cons[i]
        R_U = 0
        U = self.system_state[:34]
        for i in range(29):
            R_U += f(U[i], self.load_cons_V[1], self.load_cons_V[0])
        for i in range(29, 34):
            R_U += f(U[i], self.wind_machine_cons_V[1], self.wind_machine_cons_V[0])

        R_P, R_Q = 0, 0
        wind = self.system_state[102:112]
        # for i in range(92, 102, 2):
        #     R_P += f(wind[i], self.wind_machine_cons_P[i][1], self.wind_machine_cons_P[i][0])
        #     R_Q += f(wind[i + 1], self.wind_machine_cons_Q[i][1], self.wind_machine_cons_Q[i][0])
        for i in range(5):
            R_P += f(wind[2 * i] * 1800, self.wind_machine_cons_P[i][1], self.wind_machine_cons_P[i][0])
            R_Q += f(wind[2 * i + 1] * 300, self.wind_machine_cons_Q[i][1], self.wind_machine_cons_Q[i][0])

        fire = self.system_state[92:102]
        for i in range(5):
            R_P += f(fire[2 * i] * 1000, self.fire_machine_cons_P[i][1], self.fire_machine_cons_P[i][0])
            R_Q += f(fire[2 * i + 1] * 400, self.fire_machine_cons_Q[i][1], self.fire_machine_cons_Q[i][0])

        return -(self.lamda_I * R_I + self.lamda_U * R_U + self.lamda_P * R_P + self.lamda_Q * R_Q)

    def get_dK(self, x):
        a = np.abs(x)
        frac = 1 if x > 0 else -1
        dK = 0
        if a > 0.2:
            dK = 1
        return frac * dK

    def step(self, action):
        # action 的均值取值都是 [-1, 1]
        for i in range(5):
            ierr, v = psspy.busdat(self.fire_machine_bus_index[i], 'PU')
            V = action[i] * 0.025 + v
            if V > 1.05:
                V = 1.05
            if V < 0.95:
                V = 0.95
            # psspy.plant_data_4(self.fire_machine_bus_index[i], 0, [_i, _i], [V, _f])
            ierr = psspy.plant_data_4(self.fire_machine_bus_index[i], 0, [_i, _i], [V, _f])
        for i in range(5):  # 风机无功
            dQ = action[i + 5] * 10
            ierr, PQ = psspy.loddt2(self.wind_machine_bus_index[i], '1', 'MVA', 'ACT')
            Q = PQ.imag
            Q += dQ
            Q = self.get_bounded_wind_machine_Q(Q, i)
            psspy.load_data_4(self.wind_machine_bus_index[i], '1', [1], [_f, Q])

        for i in range(7):
            dK = self.get_dK(action[i + 10])
            index = self.shunt_C_bus_index[i]
            self.system_shunt_C[index] += dK
            if self.system_shunt_C[index] > self.num_G:
                self.system_shunt_C[index] = self.num_G
            elif self.system_shunt_C[index] < 0:
                self.system_shunt_C[index] = 0
            for j in range(self.num_G):
                psspy.shunt_data(index, str(j), 0, [_f, _f])
            for j in range(self.system_shunt_C[index]):
                psspy.shunt_data(index, str(j), 1, [_f, _f])

        for i in range(8):
            dK = self.get_dK(action[i + 17])
            index = self.shunt_X_bus_index[i]
            self.system_shunt_X[index] += dK
            if self.system_shunt_X[index] > self.num_G:
                self.system_shunt_X[index] = self.num_G
            elif self.system_shunt_X[index] < 0:
                self.system_shunt_X[index] = 0
            for j in range(self.num_G):
                psspy.shunt_data(index, str(j), 0, [_f, _f])
            for j in range(self.system_shunt_X[index]):
                psspy.shunt_data(index, str(j), 1, [_f, _f])

        # for i in range(7):          # 电容
        #     for j in range(self.num_G):
        #         psspy.shunt_data(self.shunt_C_bus_index[i], str(j), 0, [_f, _f])
        #     G = self.get_shunt_gear(action[i + 10])
        #     for g in range(G):
        #         psspy.shunt_data(self.shunt_C_bus_index[i], str(g), 1, [_f, _f])
        #
        # for i in range(8):         # 电抗
        #     for j in range(self.num_G):
        #         psspy.shunt_data(self.shunt_X_bus_index[i], str(j), 0, [_f, _f])
        #     G = self.get_shunt_gear(action[i + 17])
        #     for g in range(G):
        #         psspy.shunt_data(self.shunt_X_bus_index[i], str(g), 1, [_f, _f])

        self.flag = self.power_flow()
        ierr, net_loss_PQ = psspy.systot('LOSS')  # 获得网损
        self.net_loss_P = net_loss_PQ.real
        self.net_loss_Q = net_loss_PQ.imag

        self.calculate_system_state()
        self.done = (self.flag == 0)
        self.n_steps += 1
        self.reward = self.reward_func()

        return_state = self.system_graph_state.copy() if self.need_graph_data else self.system_state
        return return_state, self.done, self.reward
