

class args():
    def __init__(self, s_d, a_d,  a_bound, e_beta, OPT_A, OPT_C, GAME, max_ep_step, max_global_ep, gamma, up_global_iter):
        self.state_dim = s_d
        self.action_dim = a_d
        self.action_bound = a_bound
        self.entropy_beta = e_beta
        self.OPT_A = OPT_A
        self.OPT_C = OPT_C
        self.GAME = GAME
        self.MAX_EP_STEP = max_ep_step
        self.MAX_GLOBAL_EP = max_global_ep
        self.GAMMA = gamma
        self.UPDATE_GLOBAL_ITER = up_global_iter