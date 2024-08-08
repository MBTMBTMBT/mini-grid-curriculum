from customize_minigrid.wrappers import FullyObsSB3MLPWrapper


class OneHotEncodingMDPLearner:
    def __init__(self, onehot_env: FullyObsSB3MLPWrapper):
        self.env = onehot_env
    pass
