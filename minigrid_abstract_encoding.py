from random import randint, choice
from typing import Dict, Set

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from binary_state_representation.binary2binaryautoencoder import Binary2BinaryFeatureNet
from mdp_learner import string_to_numpy_binary_array, OneHotEncodingMDPLearner


class OneHotDataset(Dataset):
    def __init__(self, state_action_state_to_reward_dict: Dict[str, int], done_state_action_state_set: Set[str]):
        self.state_action_state_to_reward_dict = state_action_state_to_reward_dict
        self.done_state_action_state_set = done_state_action_state_set
        self.key_list = list(self.state_action_state_to_reward_dict.keys())

    def __len__(self):
        return len(self.state_action_state_to_reward_dict)

    def __getitem__(self, idx):
        if idx >= 0:
            state0, action, state1, = self.key_list[idx].split("-")
            reward = self.state_action_state_to_reward_dict[self.key_list[idx]]
            is_terminated = int(self.key_list[idx] in self.done_state_action_state_set)
        else:
            key = choice(list(self.done_state_action_state_set))
            state0, action, state1, = key.split("-")
            reward = self.state_action_state_to_reward_dict[key]
            is_terminated = True

        state0 = torch.tensor(string_to_numpy_binary_array(state0), dtype=torch.float32)
        action = torch.tensor(int(action), dtype=torch.int64)
        state1 = torch.tensor(string_to_numpy_binary_array(state1), dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        is_terminated = torch.tensor(is_terminated, dtype=torch.float32)

        return state0, action, state1, reward, is_terminated


def train_epoch(
        dataset: OneHotDataset,
        batch_size: int,
        model: Binary2BinaryFeatureNet,
        writer: SummaryWriter,
        step_counter: int
):
    num_terminals = len(dataset.done_state_action_state_set)
    if batch_size >= num_terminals * 10:
        keep_terminals = num_terminals
    else:
        keep_terminals = 1
    if num_terminals == 0:
        keep_terminals = 0
    dataloader_batch_size = batch_size - keep_terminals
    dataloader = DataLoader(dataset, batch_size=dataloader_batch_size, shuffle=True)

    for batch in dataloader:
        obs_vec0, actions, obs_vec1, rewards, is_terminated = batch
        if keep_terminals > 0:
            for _ in range(keep_terminals):
                obs_vec0_, actions_, obs_vec1_, rewards_, is_terminated_ = dataset.__getitem__(-1)
                obs_vec0_ = torch.unsqueeze(obs_vec0_, 0)
                actions_ = torch.unsqueeze(actions_, 0)
                obs_vec1_ = torch.unsqueeze(obs_vec1_, 0)
                rewards_ = torch.unsqueeze(rewards_, 0)
                is_terminated_ = torch.unsqueeze(is_terminated_, 0)
                obs_vec0 = torch.cat((obs_vec0, obs_vec0_), 0)
                actions = torch.cat((actions, actions_), 0)
                obs_vec1 = torch.cat((obs_vec1, obs_vec1_), 0)
                rewards = torch.cat((rewards, rewards_), 0)
                is_terminated = torch.cat((is_terminated, is_terminated_), 0)

        num_keep_dim = randint(1, model.n_latent_dims)
        losses = model.run_batch(obs_vec0, actions, obs_vec1, rewards, is_terminated, num_keep_dim, train=True)
        loss, rec_loss, inv_loss, ratio_loss, reward_loss, terminate_loss, neighbour_loss = losses
        if step_counter <= 0:
            step_counter = 1
        names = ['loss', 'rec_loss', 'inv_loss', 'ratio_loss', 'reward_loss', 'terminate_loss', 'neighbour_loss']
        for name, val in zip(names, losses):
            writer.add_scalar(name, val, step_counter)
        step_counter += 1
        return loss, step_counter


if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import math

    import torch
    from customize_minigrid.custom_env import CustomEnv
    from customize_minigrid.wrappers import FullyObsSB3MLPWrapper

    train_list_envs = [
        FullyObsSB3MLPWrapper(
            CustomEnv(
                txt_file_path=r"./maps/short_corridor.txt",
                display_size=11,
                display_mode="random",
                random_rotate=True,
                random_flip=True,
                custom_mission="reach the goal",
                max_steps=500,
            )
        ),
        FullyObsSB3MLPWrapper(
            CustomEnv(
                txt_file_path=r"./maps/long_corridor.txt",
                display_size=11,
                display_mode="random",
                random_rotate=True,
                random_flip=True,
                custom_mission="reach the goal",
                max_steps=500,
            )
        ),
        FullyObsSB3MLPWrapper(
            CustomEnv(
                txt_file_path=r"./maps/extra_long_corridor.txt",
                display_size=11,
                display_mode="random",
                random_rotate=True,
                random_flip=True,
                custom_mission="reach the goal",
                max_steps=500,
            )
        ),
    ]

    test_list_envs = [
        FullyObsSB3MLPWrapper(
            CustomEnv(
                txt_file_path=r"./maps/short_corridor.txt",
                display_size=11,
                display_mode="middle",
                random_rotate=False,
                random_flip=False,
                custom_mission="reach the goal",
                max_steps=500,
            )
        ),
        FullyObsSB3MLPWrapper(
            CustomEnv(
                txt_file_path=r"./maps/long_corridor.txt",
                display_size=11,
                display_mode="middle",
                random_rotate=False,
                random_flip=False,
                custom_mission="reach the goal",
                max_steps=500,
            )
        ),
        FullyObsSB3MLPWrapper(
            CustomEnv(
                txt_file_path=r"./maps/extra_long_corridor.txt",
                display_size=11,
                display_mode="middle",
                random_rotate=False,
                random_flip=False,
                custom_mission="reach the goal",
                max_steps=500,
            )
        ),
    ]

    # model configs
    NUM_ACTIONS = int(train_list_envs[0].env.action_space.n)
    OBS_SPACE = int(train_list_envs[0].total_features)
    LATENT_DIMS = 16

    # train hyperparams
    WEIGHTS = {'inv': 0.3, 'dis': 0.3, 'neighbour': 0.3, 'dec': 0.0001, 'rwd': 0.05, 'terminate': 0.05}
    BATCH_SIZE = 64
    LR = 1e-4

    # train configs
    EPOCHS = int(1e5)
    RESAMPLE_FREQ = int(1e3)
    RESET_TIMES = 25
    SAVE_FREQ = 20

    session_name = "experiments/learn_feature_bin"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Binary2BinaryFeatureNet(NUM_ACTIONS, OBS_SPACE, n_latent_dims=LATENT_DIMS, lr=LR, weights=WEIGHTS, device=device,).to(device)

    from binary_state_representation.binary2binaryautoencoder import find_latest_checkpoint
    import os

    if not os.path.isdir(session_name):
        os.makedirs(session_name)
    # Check for the latest saved model
    latest_checkpoint = find_latest_checkpoint(session_name)

    # load parameters if it has any
    if latest_checkpoint:
        print(f"Loading model from {latest_checkpoint}")
        epoch_counter, step_counter, performance = model.load(latest_checkpoint)
        epoch_counter += 1
        step_counter += 1
    else:
        epoch_counter = 0
        step_counter = 0
        performance = float('inf')

    # init tensorboard writer
    tb_writer = SummaryWriter(log_dir=session_name)

    state_action_state_to_reward_dict = {}
    done_state_action_state_set = set()
    dataset = OneHotDataset(state_action_state_to_reward_dict, done_state_action_state_set)

    progress_bar = tqdm(range(epoch_counter, EPOCHS), desc=f'Training Epoch {epoch_counter}')
    for i, batch in enumerate(progress_bar):
        if epoch_counter % RESAMPLE_FREQ == 0 or len(state_action_state_to_reward_dict) == 0:
            state_action_state_to_reward_dict = {}
            done_state_action_state_set = set()

            for k in range(RESET_TIMES):
                for env in train_list_envs:
                    learner = OneHotEncodingMDPLearner(env)
                    learner.learn()
                    state_action_state_to_reward_dict.update(learner.state_action_state_to_reward_dict)
                    done_state_action_state_set.update(learner.done_state_action_state_set)

                progress_bar.set_description(
                    f'Train Epoch {epoch_counter}: Sampling turn [{k+1}/{RESET_TIMES}], Num samples [{len(state_action_state_to_reward_dict)}]')

            tqdm.write(f"Number of transition pairs: {len(state_action_state_to_reward_dict)}")
            tqdm.write(f"Number of terminals: {len(done_state_action_state_set)}")

            dataset = OneHotDataset(state_action_state_to_reward_dict, done_state_action_state_set)

        loss_val, step_counter = train_epoch(dataset, BATCH_SIZE, model, tb_writer, step_counter)

        if epoch_counter % SAVE_FREQ == 0:
            model.save(f"{session_name}/model_epoch_{epoch_counter}.pth", epoch_counter, step_counter, performance)
        epoch_counter += 1
        progress_bar.set_description(
            f'Train Epoch {epoch_counter}: Loss: {loss_val:.2f}')
