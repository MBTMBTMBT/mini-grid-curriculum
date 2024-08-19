import torch
from torch.utils.data import Dataset, DataLoader

from mdp_learner import OneHotEncodingMDPLearner


class OneHotDataset(Dataset):
    def __init__(self, learner: OneHotEncodingMDPLearner):
        pass
