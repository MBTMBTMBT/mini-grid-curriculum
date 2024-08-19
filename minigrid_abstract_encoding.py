from random import randint, choice
from typing import Dict, Set

import torch
from torch.utils.data import Dataset, DataLoader

from binary_state_representation.binary2binaryautoencoder import Binary2BinaryFeatureNet
from mdp_learner import string_to_numpy_binary_array


class OneHotDataset(Dataset):
    def __init__(self, state_action_state_to_reward_dict: Dict[str, int], done_state_action_states: Set[str]):
        self.state_action_state_to_reward_dict = state_action_state_to_reward_dict
        self.done_state_action_states = done_state_action_states
        self.key_list = list(self.state_action_state_to_reward_dict.keys())

    def __len__(self):
        return len(self.state_action_state_to_reward_dict)

    def __getitem__(self, idx):
        if idx >= 0:
            state0, action, state1, = self.key_list[idx].split("-")
            reward = self.state_action_state_to_reward_dict[self.key_list[idx]]
            is_terminated = int(self.key_list[idx] in self.done_state_action_states)
        else:
            key = choice(list(self.done_state_action_states))
            state0, action, state1, = key.split("-")
            reward = self.state_action_state_to_reward_dict[key]
            is_terminated = True

        state0 = torch.tensor(string_to_numpy_binary_array(state0), dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        state1 = torch.tensor(string_to_numpy_binary_array(state1), dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        is_terminated = torch.tensor(is_terminated, dtype=torch.float32)

        return state0, action, state1, reward, is_terminated


def train_epoch(dataset: OneHotDataset, batch_size: int, model: Binary2BinaryFeatureNet):
    num_terminals = len(dataset.done_state_action_states)
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
        # loss, rec_loss, inv_loss, ratio_loss, reward_loss, terminate_loss, neighbour_loss = losses
        return losses


if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
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
                txt_file_path=r"./maps/extra_corridor.txt",
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
                txt_file_path=r"./maps/extra_corridor.txt",
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
    NUM_ACTIONS = 4
    LATENT_DIMS = 2
    RECONSTRUCT_SIZE = (96, 96)
    RECONSTRUCT_SCALE = 2

    # sampler configs
    SAMPLE_SIZE = 16384
    SAMPLE_REPLAY_TIME = 4
    MAX_SAMPLE_STEP = 4096

    # train hyperparams
    WEIGHTS = {
        'inv': 1.0,
        'dis': 1.0,
        'neighbour': 1.0,
        'dec': 1.0,
        'rwd': 0.0,
        'demo': 5.0,
    }
    BATCH_SIZE = 64
    LR = 1e-4

    # train configs
    EPOCHS = 80
    SAVE_FREQ = 1
    TEST_FREQ = 1

    session_name = "learn_feature_maze13_dec_neighbour_demo"
    feature_model_name = 'feature_model_step'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureNet(NUM_ACTIONS, n_latent_dims=LATENT_DIMS, lr=LR, img_size=RECONSTRUCT_SIZE, initial_scale_factor=RECONSTRUCT_SCALE, device=device, weights=WEIGHTS).to(device)

    from utils import find_latest_checkpoint, plot_decoded_images, plot_representations
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

    from env_sampler import TransitionBuffer, RandomSampler
    from utils import make_env

    progress_bar = tqdm(range(epoch_counter, EPOCHS), desc=f'Training Epoch {epoch_counter}')
    for i, batch in enumerate(progress_bar):
        sampler = RandomSampler(not WEIGHTS['demo'] == 0.0)
        envs = [make_env(config) for config in CONFIGS]
        while len(sampler.transition_pairs) < SAMPLE_SIZE:
            for env in envs:
                sampler.sample(env, MAX_SAMPLE_STEP)

        transition_buffer = TransitionBuffer(sampler.transition_pairs)
        dataloader = DataLoader(transition_buffer, batch_size=BATCH_SIZE, shuffle=True)
        loss_val = 0.0
        for _ in range(SAMPLE_REPLAY_TIME):
            for batch in dataloader:
                d = None
                if WEIGHTS['demo'] == 0.0:
                    x0, a, x1, r = batch  # Assuming batch format is [x0, a, x1, r] when 'demo' is not used
                else:
                    x0, a, x1, r, d = batch  # Assuming batch format is [x0, a, x1, r, d] when 'demo' is used

                # Move tensors to the appropriate device
                x0 = x0.to(device)
                a = a.to(device)
                x1 = x1.to(device)
                r = r.to(device)
                if d is not None:
                    d = d.to(device)

                # Train batch and calculate losses
                loss_vals = model.train_batch(x0, x1, a, r, d)

                # Log values to TensorBoard
                names = ['loss', 'inv_loss', 'neighbour_loss', 'ratio_loss', 'pixel_loss', 'reward_loss', 'demo_loss']
                for name, val in zip(names, loss_vals):
                    tb_writer.add_scalar(name, val, step_counter)

                # Increment the step counter
                step_counter += 1

        if epoch_counter % TEST_FREQ == 0:
            _plot_dir = os.path.join(session_name, 'plots')
            for config, env in zip(CONFIGS, envs):
                env_path = config['env_file']
                env_name = env_path.split('/')[-1].split('.')[0]
                if not os.path.isdir(_plot_dir):
                    os.makedirs(_plot_dir)
                save_path = os.path.join(_plot_dir, f"{env_name}-{step_counter}.png")

                if model.decoder is not None:
                    plot_decoded_images(env, model.phi,
                                        model.decoder, save_path, device)
                if LATENT_DIMS == 2 or LATENT_DIMS == 3:
                    plot_representations(env, model.phi, LATENT_DIMS, save_path, device)

        if epoch_counter % SAVE_FREQ == 0:
            model.save(f"{session_name}/model_epoch_{epoch_counter}.pth", epoch_counter, step_counter, performance)
        epoch_counter += 1
        progress_bar.set_description(
            f'Train Epoch {epoch_counter}: Loss: {loss_val:.2f}')
