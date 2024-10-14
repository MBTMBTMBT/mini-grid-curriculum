import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv
from torch.utils.data import Dataset
from tqdm import tqdm


class GymDataset(Dataset):
    def __init__(self, env: VecEnv, epoch_size: int):
        """
        Args:
            env: A pre-created vectorized environment (VecEnv).
            epoch_size: The number of steps for one epoch of data collection.
        """
        self.env = env  # Use the provided VecEnv
        self.epoch_size = epoch_size
        self.num_envs = self.env.num_envs  # Retrieve the number of environments from the provided VecEnv
        self.data = []

        # Initial sampling to populate the dataset
        self.resample()

    def resample(self):
        """Resample the data by interacting with the environment and collecting new data for one epoch."""
        self.data.clear()  # Clear existing data
        obs = self.env.reset()  # Reset the environment to get the initial observations

        # Collect data for the entire epoch with a progress bar
        for _ in tqdm(range(self.epoch_size // self.num_envs), desc="Sampling Data", unit="step"):
            # Sample actions for each parallel environment
            actions = [self.env.action_space.sample() for _ in range(self.num_envs)]
            next_obs, rewards, dones, infos = self.env.step(actions)

            # Copy `next_obs` to avoid modifying the original
            final_next_obs = np.copy(next_obs)

            # If an environment is done, replace values in `final_next_obs`
            done_indices = np.where(dones)[0]  # Optimisation: only handle environments where `dones` is True
            for env_idx in done_indices:
                final_next_obs[env_idx] = infos[env_idx]["terminal_observation"]

            # Store the data for each parallel environment
            for env_idx in range(self.num_envs):
                self.data.append({
                    'obs': torch.tensor(obs[env_idx], dtype=torch.float32),
                    'action': torch.tensor(actions[env_idx], dtype=torch.int64),
                    'next_obs': torch.tensor(final_next_obs[env_idx], dtype=torch.float32),
                    'reward': torch.tensor(rewards[env_idx], dtype=torch.float32),
                    'done': torch.tensor(dones[env_idx], dtype=torch.bool)
                })

            obs = next_obs  # Update `obs` to `next_obs` for the next step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['obs'], item['action'], item['next_obs'], item['reward'], item['done']
