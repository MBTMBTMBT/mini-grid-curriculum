from typing import Callable, Optional, List, Dict, Union, Type

import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from torch.distributions import Categorical, Normal
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomEncoderExtractor(BaseFeaturesExtractor):
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

        # Initially, the network is not frozen
        self.frozen = False

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # First flatten the input (same as FlattenExtractor)
        observations = observations.view(observations.size(0), -1)
        # Then pass through the encoder network
        return self.encoder(observations)

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


if __name__ == '__main__':
    # Create environment
    env = make_vec_env('CartPole-v1', n_envs=1)

    # Define custom architecture parameters
    policy_kwargs = dict(
        shared_layers=[128, 128],  # Shared feature extractor layers
        policy_layers=[128, 64],   # Policy network layers
        value_layers=[128, 64],    # Value network layers
        activation_fn=nn.Tanh      # Custom activation function (Tanh)
    )

    # Create PPO model with the custom policy network
    model = PPO(CustomGeneralPolicy, env, policy_kwargs=policy_kwargs, verbose=1)

    # Freeze the policy network, unfreeze the value network
    model.policy.freeze_layers(part="policy", freeze=True)
    model.policy.freeze_layers(part="value", freeze=False)

    # Begin training
    timesteps = 10000
    for i in range(timesteps // 1000):
        if i % 2 == 0:
            # Every 1000 steps, freeze the shared feature extractor
            model.policy.freeze_layers(part="shared", freeze=True)
        else:
            # Unfreeze the shared feature extractor
            model.policy.freeze_layers(part="shared", freeze=False)

        # Train the model for 1000 steps
        model.learn(total_timesteps=1000)
