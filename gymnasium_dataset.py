import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from torch.utils.data import Dataset
from tqdm import tqdm

from customize_minigrid.wrappers import FullyObsImageWrapper
from task_config import TaskConfig, make_env


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

        # Collect data for the entire epoch with a progress bar showing the number of actual samples
        total_samples = self.epoch_size
        with tqdm(total=total_samples, desc="Sampling Data", unit="sample") as pbar:
            while len(self.data) < total_samples:
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
                    if len(self.data) < total_samples:  # Ensure we don't overshoot the target samples
                        self.data.append({
                            'obs': torch.tensor(obs[env_idx], dtype=torch.float32),
                            'action': torch.tensor(actions[env_idx], dtype=torch.int64),
                            'next_obs': torch.tensor(final_next_obs[env_idx], dtype=torch.float32),
                            'reward': torch.tensor(rewards[env_idx], dtype=torch.float32),
                            'done': torch.tensor(dones[env_idx], dtype=torch.bool)
                        })

                # Update the observation for the next step
                obs = next_obs

                # Update the progress bar with the number of samples collected in this step
                pbar.update(self.num_envs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['obs'], item['action'], item['next_obs'], item['reward'], item['done']


if __name__ == '__main__':
    configs = []
    for i in range(1, 7):
        config = TaskConfig()
        config.name = f"7-{i}"
        config.rand_gen_shape = None
        config.txt_file_path = f"./maps/7-{i}.txt"
        config.custom_mission = "reach the goal"
        config.minimum_display_size = 7
        config.display_mode = "random"
        config.random_rotate = True
        config.random_flip = True
        config.max_steps = 1024
        config.start_pos = (5, 5)
        config.train_total_steps = 2.5e7
        config.difficulty_level = 0
        config.add_random_door_key = False
        configs.append(config)

    max_minimum_display_size = 0
    for config in configs:
        if config.minimum_display_size > max_minimum_display_size:
            max_minimum_display_size = config.minimum_display_size

    venv = SubprocVecEnv([
        lambda: make_env(each_task_config, FullyObsImageWrapper, max_minimum_display_size)
        for each_task_config in configs
    ])

    dataset = GymDataset(venv, 16384)
