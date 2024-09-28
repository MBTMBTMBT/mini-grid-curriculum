import os

import gymnasium as gym
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from torch import nn, optim
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
        obs = self.env.reset()  # Reset the environment to get the initial observations

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
            display_mode='random',
            random_rotate=True,
            random_flip=True,
            custom_mission="Explore and interact with objects.",
            max_steps=128,
        )
    )


if __name__ == '__main__':
    session_dir = "experiments/feature_extractor_test"
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
    feature_model = feature_model.cuda()

    feature_model.weights = {'total': 1.0, 'inv': 1.0, 'dis': 0.0, 'neighbour': 0.1, 'dec': 0.0, 'rwd': 0.1, 'terminate': 1.0}

    # Create the dataset using the provided `make_env` function
    dataset = GymDataset(make_env, int(50e5), num_envs=12)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = optim.Adam(feature_model.parameters(), lr=1e-4)

    counter = 0

    # Training loop with the initial data, with progress bar
    for obs, actions, next_obs, rewards, dones in tqdm(dataloader, desc="Training", unit="batch"):
        # Perform training here
        differences = obs - next_obs
        norms = torch.norm(differences, p=2, dim=1)
        same_states = norms < 0.5

        obs = obs.cuda()
        actions = actions.cuda()
        next_obs = next_obs.cuda()
        rewards = rewards.cuda()
        dones = dones.cuda()

        z0 = feature_model(obs)
        z1 = feature_model(next_obs)

        # filter out the cases that the agent did not move
        z0_filtered = z0[~same_states]
        z1_filtered = z1[~same_states]

        # get fake z1
        idx = torch.randperm(len(obs))
        fake_z1 = z1.view(len(z1), -1)[idx].view(z1.size())
        fake_z1_filtered = fake_z1[~same_states]

        actions_filtered = actions[~same_states]

        loss, loss_vals = feature_model.compute_loss(
            obs, next_obs, z0, z1, z0_filtered, z1_filtered,
            fake_z1_filtered, actions, actions_filtered, rewards,
            dones,
        )

        names = ['feature_loss', 'rec_loss', 'inv_loss', 'ratio_loss', 'reward_loss', 'terminate_loss',
                 'neighbour_loss']
        for name, val in zip(names, loss_vals):
            log_writer.add_scalar(name, val, counter)

        counter += 1

        loss.backward()
        optimizer.step()
