import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from customPPO import CustomPPO, CustomActorCriticPolicy, CNNEncoderExtractor
from customize_minigrid.custom_env import CustomEnv
from customize_minigrid.wrappers import FullyObsImageWrapper
from gymnasium_dataset import GymDataset


def make_env():
    return FullyObsImageWrapper(
        CustomEnv(
            txt_file_path=None,
            rand_gen_shape=(4, 4),
            display_size=4,
            display_mode='random',
            random_rotate=True,
            random_flip=True,
            custom_mission="Explore and interact with objects.",
            max_steps=1024,
        )
    )


if __name__ == '__main__':
    session_dir = "experiments/feature_extractor_test"
    policy_kwargs = dict(
        features_extractor_class=CNNEncoderExtractor,  # Use the custom encoder extractor
        features_extractor_kwargs=dict(
            net_arch=[32],  # Custom layer sizes
            cnn_net_arch=[
                (64, 3, 2, 1),
                (128, 3, 2, 1),
                (256, 3, 2, 1),
                (512, 3, 2, 1),
                (1024, 3, 2, 1),
            ],
            activation_fn=nn.LeakyReLU,  # Activation function
            encoder_only=True,
        ),
        net_arch=dict(pi=[32, 32], vf=[32, 32]),  # Policy and value network architecture
        activation_fn=nn.LeakyReLU,
    )
    os.makedirs(session_dir, exist_ok=True)
    log_dir = os.path.join(session_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    # model_save_dir = os.path.join(session_dir, "saved_models")
    # os.makedirs(model_save_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir)

    model = CustomPPO(CustomActorCriticPolicy, env=make_env(), policy_kwargs=policy_kwargs, verbose=1, log_dir=log_dir)
    feature_model = model.policy.features_extractor
    # feature_model.weights = {'total': 1.0, 'inv': 0.0, 'dis': 1.0, 'neighbour': 0.0, 'dec': 0.0, 'rwd': 0.0, 'terminate': 1.0}
    # feature_model.set_up()
    feature_model = feature_model.cuda()

    # Create the dataset using the provided `make_env` function
    dataset = GymDataset(make_env, int(1e4), num_envs=8)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = optim.Adam(feature_model.parameters(), lr=5e-4)

    counter = 0

    for k in range(100):
        # Training loop with the initial data, with progress bar
        for obs, actions, next_obs, rewards, dones in tqdm(dataloader, desc="Training", unit="batch"):
            # Perform training here

            # # visualize the first example in the batch
            # batch_idx = 0
            # # Extract data for visualization
            # obs_image = obs[batch_idx].numpy().transpose(1, 2, 0)  # Convert to (height, width, channels)
            # next_obs_image = next_obs[batch_idx].numpy().transpose(1, 2, 0)  # Convert to (height, width, channels)
            # action = actions[batch_idx].item()
            # reward = rewards[batch_idx].item()
            # done = dones[batch_idx].item()
            #
            # if done:
            #     # Plotting
            #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            #
            #     # Display the current observation
            #     axes[0].imshow(obs_image)
            #     axes[0].set_title("Current Observation")
            #     axes[0].axis('off')
            #
            #     # Display the next observation
            #     axes[1].imshow(next_obs_image)
            #     axes[1].set_title("Next Observation")
            #     axes[1].axis('off')
            #
            #     # Display the action, reward, and done status
            #     action_names = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
            #     info_text = f"Action: {action_names[action]}\nReward: {reward}\nDone: {done}"
            #     axes[2].text(0.5, 0.5, info_text, fontsize=14, ha='center', va='center')
            #     axes[2].set_title("Action/Reward/Done")
            #     axes[2].axis('off')
            #
            #     plt.tight_layout()
            #     plt.show()

            obs = obs.cuda()
            actions = actions.cuda()
            next_obs = next_obs.cuda()
            rewards = rewards.cuda()
            dones = dones.cuda()

            # if not dones.any():
            #     continue

            # Compute the differences
            differences = obs - next_obs
            # Flatten the differences along all non-batch dimensions
            flattened_differences = differences.flatten(start_dim=1)
            # Compute the L2 norm along the flattened dimension
            norms = torch.norm(flattened_differences, p=2, dim=1)
            # Compare with the threshold to obtain a boolean tensor
            same_states = norms < 0.5

            if len(~same_states) < 1:
                continue

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

            # try:
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

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            # except Exception as e:
            #     print(e)


