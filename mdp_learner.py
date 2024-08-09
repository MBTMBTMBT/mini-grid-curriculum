from customize_minigrid.wrappers import FullyObsSB3MLPWrapper


# decimal_number = int(np.dot(final_obs, np.power(2, np.arange(final_obs.size)[::-1])))

class OneHotEncodingMDPLearner:
    def __init__(self, onehot_env: FullyObsSB3MLPWrapper):
        self.env = onehot_env
    pass
