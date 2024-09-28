import os

import gymnasium as gym
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from customPPO import CustomPPO, CustomActorCriticPolicy, TransformerEncoderExtractor
from customize_minigrid.custom_env import CustomEnv
from customize_minigrid.wrappers import FullyObsSB3MLPWrapper
from train import TaskConfig


class GymDataset(Dataset):
    def __init__(self, env_make_func, epoch_size, num_envs=1):
        self.env_make_func = env_make_func
        self.epoch_size = epoch_size
        self.num_envs = num_envs
        self.data = [{}]

        # Create the vectorized environment with monitoring
        self.env = VecMonitor(SubprocVecEnv([self.env_make_func for _ in range(num_envs)]))

        # Initial sampling to populate the dataset
        self.resample()

    def resample(self):
        """Resample the data by interacting with the environment and collecting a new epoch of data."""
        self.data.clear()  # Clear existing data
        obs, _ = self.env.reset()  # Reset the environment to get the initial observations

        # Collect data for one full epoch with a progress bar
        for _ in tqdm(range(self.epoch_size // self.num_envs), desc="Sampling Data", unit="step"):
            # Sample actions for each parallel environment
            actions = [self.env.action_space.sample() for _ in range(self.num_envs)]
            next_obs, rewards, dones, infos = self.env.step(actions)

            # Adjust the `dones` array to handle truncated episodes
            corrected_dones = dones.copy()
            for idx, done in enumerate(dones):
                # If the episode was truncated, mark `done` as False to avoid incorrect terminal state handling
                if done and infos[idx].get("TimeLimit.truncated", False):
                    corrected_dones[idx] = False

            # Store the data for each parallel environment
            for env_idx in range(self.num_envs):
                self.data.append({
                    'obs': torch.tensor(obs[env_idx], dtype=torch.float32),
                    'action': torch.tensor(actions[env_idx], dtype=torch.int64),
                    'next_obs': torch.tensor(next_obs[env_idx], dtype=torch.float32),
                    'reward': torch.tensor(rewards[env_idx], dtype=torch.float32),
                    'done': torch.tensor(corrected_dones[env_idx], dtype=torch.bool)
                })

            obs = next_obs  # Update `obs` to `next_obs` for the next step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['obs'], item['action'], item['next_obs'], item['reward'], item['done']


def make_env():
    return FullyObsSB3MLPWrapper(
        CustomEnv(
            txt_file_path=None,
            rand_gen_shape=(4, 4),
            display_size=4,
            display_mode='rand',
            random_rotate=True,
            random_flip=True,
            custom_mission="Explore and interact with objects.",
            max_steps=128,
        )
    )


if __name__ == '__main__':
    session_dir = "feature_extractor_test"
    policy_kwargs = dict(
        features_extractor_class=TransformerEncoderExtractor,  # Use the custom encoder extractor
        features_extractor_kwargs=dict(
            net_arch=[32],  # Custom layer sizes
            num_transformer_layers=1,
            n_heads=8,
            activation_fn=nn.LeakyReLU  # Activation function
        ),
        net_arch=dict(pi=[32, 128, 128], vf=[32, 128, 128]),  # Policy and value network architecture
        activation_fn=nn.LeakyReLU,
    )
    os.makedirs(session_dir, exist_ok=True)
    log_dir = os.path.join(session_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    # model_save_dir = os.path.join(session_dir, "saved_models")
    # os.makedirs(model_save_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir)
    model = CustomPPO(CustomActorCriticPolicy, env=make_env(), policy_kwargs=policy_kwargs, verbose=1, log_dir=log_dir,
                      batch_size=4)
    feature_model = model.policy.features_extractor

    # Create the dataset using the provided `make_env` function
    dataset = GymDataset(make_env, 1e6, num_envs=8)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop with the initial data, with progress bar
    for obs, action, next_obs, reward, done in tqdm(dataloader, desc="Training", unit="batch"):
        # Perform training here
        pass
