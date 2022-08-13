import numpy as np


def action_map(a, env):
    # a = a[0]
    t = env.K
    a = np.piecewise(a, [a > env.A_Threadhold, a < -env.A_Threadhold],
                     [1, -1])  # 将a元素中大于A_Threadhold的用1替换掉，小于-A_Threadhold的用-1替换掉，其余的默认以0填充
    temp = t + a[0:4]
    temp = np.piecewise(temp, [temp > 3, temp < 0], [1, 1])
    a[0:4][np.where(temp == 1)] = 0

    env.K = t + a[0:4]

    return a
