class EnvConfig:
    state_dim = 152
    action_dim = 25
    action_type = 'continuous'
    gamma = 0.98
    reward_scale = 1
    episode_length = 128


class AlgConfig:
    model_type = 'GAT' # FC & GAT
    lr = 0.00025
    clip_eps = 0.2
    v_coeff = 0.25
    p_coeff = 1
    e_coeff = 0.001
    device = 'cpu'


class Config:
    env_config = EnvConfig()
    alg_config = AlgConfig()
