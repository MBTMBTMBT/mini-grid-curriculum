from typing import Callable, Optional, List, Dict, Union, Type

import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from torch.distributions import Categorical, Normal
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MLPEncoderExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor that applies a neural network (MLP) as an encoder.
    It also supports freezing/unfreezing the network.

    :param observation_space: (gym.Space) The observation space of the environment
    :param net_arch: (list of int) The architecture of the MLP encoder (list of layer sizes)
    :param activation_fn: (nn.Module) The activation function to use (ReLU, Tanh, etc.)
    """

    def __init__(self, observation_space: gym.spaces.Box, net_arch=None, activation_fn=nn.ReLU):
        # Initialize the base feature extractor with the flattened observation size
        super().__init__(observation_space, th.prod(th.tensor(observation_space.shape)).item())

        # Build the encoder network layers
        if net_arch is None:
            net_arch = [64, 64]
        input_dim = self.features_dim  # Flattened observation dimension
        layers = []
        for layer_size in net_arch:
            layers.append(nn.Linear(input_dim, layer_size))  # Fully connected layer
            layers.append(activation_fn())  # Activation function
            input_dim = layer_size

        # Create the sequential model
        self.encoder = nn.Sequential(*layers)

        # reset the output dimension
        self._features_dim = net_arch[-1]

        # Initially, the network is not frozen
        self.frozen = False

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # First flatten the input (same as FlattenExtractor)
        observations = observations.view(observations.size(0), -1)
        # Then pass through the encoder network
        encoded = self.encoder(observations)
        return encoded

    def freeze(self):
        """
        Freeze the parameters of the encoder, preventing them from being updated during training.
        """
        self.frozen = True
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze the parameters of the encoder, allowing them to be updated during training.
        """
        self.frozen = False
        for param in self.encoder.parameters():
            param.requires_grad = True


class TransformerEncoderExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor that applies a Transformer encoder followed by an MLP network.
    The Transformer layers keep the input dimension, and the MLP structure is determined by net_arch.

    :param observation_space: (gym.Space) The observation space of the environment
    :param net_arch: (list of int) The architecture of the MLP layers (list of layer sizes)
    :param num_transformer_layers: (int) The number of Transformer layers to apply
    :param n_heads: (int) Number of attention heads in the Transformer
    :param activation_fn: (nn.Module) The activation function to use (ReLU, Tanh, etc.)
    """

    def __init__(self, observation_space: gym.spaces.Box, net_arch=None, num_transformer_layers=2, n_heads=8,
                 activation_fn=nn.ReLU):
        # Initialize the base feature extractor with the flattened observation size
        super().__init__(observation_space, th.prod(th.tensor(observation_space.shape)).item())

        # Default MLP architecture if none is provided
        if net_arch is None:
            net_arch = [128, 64]

        # Set the input sequence length and model dimension (d_model) to match the input
        self.seq_length = self.features_dim
        d_model = self.seq_length

        # Ensure d_model is divisible by n_heads by padding if necessary
        if d_model % n_heads != 0:
            # Find the nearest multiple of n_heads greater than or equal to d_model
            padded_d_model = ((d_model // n_heads) + 1) * n_heads
            self.pad_size = padded_d_model - d_model
            d_model = padded_d_model
        else:
            self.pad_size = 0

        # Build the Transformer encoder layers with batch_first=True
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_transformer_layers):
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, activation=activation_fn(),
                                                       batch_first=True)
            self.transformer_layers.append(nn.TransformerEncoder(encoder_layer, num_layers=1))

        # Define the MLP layers according to net_arch
        mlp_layers = []
        input_dim = d_model  # Start with the output dimension of the Transformer layers
        for layer_size in net_arch:
            mlp_layers.append(nn.Linear(input_dim, layer_size))
            mlp_layers.append(activation_fn())
            input_dim = layer_size

        # Create the MLP network as a sequential model
        self.mlp = nn.Sequential(*mlp_layers)

        # Set the final feature dimension as the last layer size of net_arch
        self._features_dim = net_arch[-1]

        # Initially, the network is not frozen
        self.frozen = False

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Flatten the input observations
        observations = observations.view(observations.size(0), -1)

        # Apply padding if necessary
        if self.pad_size > 0:
            padding = th.zeros((observations.size(0), self.pad_size), device=observations.device)
            observations = th.cat((observations, padding), dim=1)

        # Reshape to [batch_size, sequence_length, d_model] for Transformer processing
        embedded = observations.unsqueeze(1)  # Change to (batch_size, 1, d_model)

        # Pass through each Transformer encoder layer
        for transformer_layer in self.transformer_layers:
            embedded = transformer_layer(embedded)

        # Flatten the output from the Transformer
        encoded = embedded.squeeze(1)  # Remove the extra sequence dimension

        # Pass through the MLP network
        output = self.mlp(encoded)

        return output

    def freeze(self):
        """
        Freeze the parameters of the encoder, preventing them from being updated during training.
        """
        self.frozen = True
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze the parameters of the encoder, allowing them to be updated during training.
        """
        self.frozen = False
        for param in self.parameters():
            param.requires_grad = True


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    A custom ActorCriticPolicy that allows freezing/unfreezing the mlp_extractor.
    """

    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)

    def freeze_mlp_extractor(self):
        """
        Freeze the parameters of the mlp_extractor, preventing them from being updated during training.
        """
        for param in self.mlp_extractor.parameters():
            param.requires_grad = False
        print("MLP Extractor is now frozen.")

    def unfreeze_mlp_extractor(self):
        """
        Unfreeze the parameters of the mlp_extractor, allowing them to be updated during training.
        """
        for param in self.mlp_extractor.parameters():
            param.requires_grad = True
        print("MLP Extractor is now unfrozen.")


# Evaluation function
def evaluate_model(model, env, n_eval_episodes=5):
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=False)
    return mean_reward


if __name__ == '__main__':
    # # Create environment
    # env = make_vec_env("CartPole-v1", n_envs=1)
    #
    # # Define policy kwargs
    # policy_kwargs = dict(
    #     features_extractor_class=MLPEncoderExtractor,  # Use the custom encoder extractor
    #     features_extractor_kwargs=dict(
    #         net_arch=[128, 64],  # Custom layer sizes
    #         activation_fn=nn.ReLU  # Activation function
    #     ),
    #     net_arch=[dict(pi=[64, 64], vf=[64, 64])],  # Policy and value network architecture
    #     activation_fn=nn.ReLU,
    # )
    #
    # # Store rewards
    # rewards_all_frozen = []
    # rewards_first_half_frozen = []
    # rewards_second_half_frozen = []
    # rewards_unfrozen = []
    #
    # # Number of epochs for each training (each epoch is 1000 timesteps)
    # epochs = 5
    # timesteps_per_epoch = 1000
    #
    # # Training case 1: All layers frozen (mlp_extractor and feature_extractor)
    # print("\nTraining model with all layers frozen...")
    # model_all_frozen = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
    # model_all_frozen.policy.freeze_mlp_extractor()
    # model_all_frozen.policy.features_extractor.freeze()
    # for _ in range(epochs):
    #     model_all_frozen.learn(total_timesteps=timesteps_per_epoch, reset_num_timesteps=False)
    #     rewards_all_frozen.append(evaluate_model(model_all_frozen, env))
    #
    # # Training case 2: First half of layers frozen
    # print("\nTraining model with first half of layers frozen...")
    # model_first_half_frozen = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
    #
    # # Freeze the first half of the mlp_extractor
    # total_mlp_layers = len(list(model_first_half_frozen.policy.mlp_extractor.children()))
    # for idx, layer in enumerate(model_first_half_frozen.policy.mlp_extractor.children()):
    #     if idx < total_mlp_layers // 2:
    #         for param in layer.parameters():
    #             param.requires_grad = False
    #
    # # Freeze the first half of the feature extractor
    # total_encoder_layers = len(list(model_first_half_frozen.policy.features_extractor.encoder.children()))
    # for idx, layer in enumerate(model_first_half_frozen.policy.features_extractor.encoder.children()):
    #     if idx < total_encoder_layers // 2:
    #         for param in layer.parameters():
    #             param.requires_grad = False
    #
    # for _ in range(epochs):
    #     model_first_half_frozen.learn(total_timesteps=timesteps_per_epoch, reset_num_timesteps=False)
    #     rewards_first_half_frozen.append(evaluate_model(model_first_half_frozen, env))
    #
    # # Training case 3: Second half of layers frozen
    # print("\nTraining model with second half of layers frozen...")
    # model_second_half_frozen = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
    #
    # # Freeze the second half of the mlp_extractor
    # for idx, layer in enumerate(model_second_half_frozen.policy.mlp_extractor.children()):
    #     if idx >= total_mlp_layers // 2:
    #         for param in layer.parameters():
    #             param.requires_grad = False
    #
    # # Freeze the second half of the feature extractor
    # for idx, layer in enumerate(model_second_half_frozen.policy.features_extractor.encoder.children()):
    #     if idx >= total_encoder_layers // 2:
    #         for param in layer.parameters():
    #             param.requires_grad = False
    #
    # for _ in range(epochs):
    #     model_second_half_frozen.learn(total_timesteps=timesteps_per_epoch, reset_num_timesteps=False)
    #     rewards_second_half_frozen.append(evaluate_model(model_second_half_frozen, env))
    #
    # # Training case 4: All layers unfrozen
    # print("\nTraining model with all layers unfrozen...")
    # model_unfrozen = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
    # for _ in range(epochs):
    #     model_unfrozen.learn(total_timesteps=timesteps_per_epoch, reset_num_timesteps=False)
    #     rewards_unfrozen.append(evaluate_model(model_unfrozen, env))
    #
    # # Plotting the results
    # epochs_range = np.arange(1, epochs + 1)
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs_range, rewards_all_frozen, label='All Frozen', marker='o')
    # plt.plot(epochs_range, rewards_first_half_frozen, label='First Half Frozen', marker='o')
    # plt.plot(epochs_range, rewards_second_half_frozen, label='Second Half Frozen', marker='o')
    # plt.plot(epochs_range, rewards_unfrozen, label='Unfrozen', marker='o')
    # plt.xlabel('Epochs (1000 timesteps per epoch)')
    # plt.ylabel('Mean Reward')
    # plt.title('Training Trend: Frozen vs Unfrozen Networks (Different Models)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Create a dummy observation space (e.g., a flattened 16x16 image or other data)
    observation_shape = (256,)  # Example flattened observation of length 256
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32)

    # Define your network architecture
    net_arch = [128, 64]  # Example MLP architecture after the Transformer layers

    # Instantiate the custom feature extractor
    feature_extractor = TransformerEncoderExtractor(
        observation_space=observation_space,
        net_arch=net_arch,
        num_transformer_layers=2,  # Example: using 2 Transformer layers
        n_heads=8,  # Example: using 8 attention heads
        activation_fn=nn.ReLU
    )

    # Create a dummy batch of observations with batch_size = 4
    batch_size = 4
    dummy_observations = th.rand((batch_size, *observation_shape), dtype=th.float32)

    # Pass the dummy observations through the feature extractor
    output = feature_extractor(dummy_observations)

    # Print the output shape
    print(f"Input shape: {dummy_observations.shape}")
    print(f"Output shape: {output.shape}")

    # Check that the output shape matches the expected feature dimension
    expected_output_dim = net_arch[-1]
    assert output.shape == (batch_size, expected_output_dim), \
        f"Expected output shape {(batch_size, expected_output_dim)}, but got {output.shape}"

    print("Test passed: The TransformerEncoderExtractor works correctly with the given input.")

