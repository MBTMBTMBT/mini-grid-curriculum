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
    # Create environment
    env = make_vec_env("CartPole-v1", n_envs=1)

    # Define policy kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomEncoderExtractor,  # Use the custom encoder extractor
        features_extractor_kwargs=dict(
            net_arch=[128, 64],  # Custom layer sizes
            activation_fn=nn.ReLU  # Activation function
        ),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],  # Policy and value network architecture
        activation_fn=nn.ReLU,
    )

    # Store rewards
    rewards_all_frozen = []
    rewards_first_half_frozen = []
    rewards_second_half_frozen = []
    rewards_unfrozen = []

    # Number of epochs for each training (each epoch is 1000 timesteps)
    epochs = 5
    timesteps_per_epoch = 1000

    # Training case 1: All layers frozen (mlp_extractor and feature_extractor)
    print("\nTraining model with all layers frozen...")
    model_all_frozen = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
    model_all_frozen.policy.freeze_mlp_extractor()
    model_all_frozen.policy.features_extractor.freeze()
    for _ in range(epochs):
        model_all_frozen.learn(total_timesteps=timesteps_per_epoch, reset_num_timesteps=False)
        rewards_all_frozen.append(evaluate_model(model_all_frozen, env))

    # Training case 2: First half of layers frozen
    print("\nTraining model with first half of layers frozen...")
    model_first_half_frozen = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)

    # Freeze the first half of the mlp_extractor
    total_mlp_layers = len(list(model_first_half_frozen.policy.mlp_extractor.children()))
    for idx, layer in enumerate(model_first_half_frozen.policy.mlp_extractor.children()):
        if idx < total_mlp_layers // 2:
            for param in layer.parameters():
                param.requires_grad = False

    # Freeze the first half of the feature extractor
    total_encoder_layers = len(list(model_first_half_frozen.policy.features_extractor.encoder.children()))
    for idx, layer in enumerate(model_first_half_frozen.policy.features_extractor.encoder.children()):
        if idx < total_encoder_layers // 2:
            for param in layer.parameters():
                param.requires_grad = False

    for _ in range(epochs):
        model_first_half_frozen.learn(total_timesteps=timesteps_per_epoch, reset_num_timesteps=False)
        rewards_first_half_frozen.append(evaluate_model(model_first_half_frozen, env))

    # Training case 3: Second half of layers frozen
    print("\nTraining model with second half of layers frozen...")
    model_second_half_frozen = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)

    # Freeze the second half of the mlp_extractor
    for idx, layer in enumerate(model_second_half_frozen.policy.mlp_extractor.children()):
        if idx >= total_mlp_layers // 2:
            for param in layer.parameters():
                param.requires_grad = False

    # Freeze the second half of the feature extractor
    for idx, layer in enumerate(model_second_half_frozen.policy.features_extractor.encoder.children()):
        if idx >= total_encoder_layers // 2:
            for param in layer.parameters():
                param.requires_grad = False

    for _ in range(epochs):
        model_second_half_frozen.learn(total_timesteps=timesteps_per_epoch, reset_num_timesteps=False)
        rewards_second_half_frozen.append(evaluate_model(model_second_half_frozen, env))

    # Training case 4: All layers unfrozen
    print("\nTraining model with all layers unfrozen...")
    model_unfrozen = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
    for _ in range(epochs):
        model_unfrozen.learn(total_timesteps=timesteps_per_epoch, reset_num_timesteps=False)
        rewards_unfrozen.append(evaluate_model(model_unfrozen, env))

    # Plotting the results
    epochs_range = np.arange(1, epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, rewards_all_frozen, label='All Frozen', marker='o')
    plt.plot(epochs_range, rewards_first_half_frozen, label='First Half Frozen', marker='o')
    plt.plot(epochs_range, rewards_second_half_frozen, label='Second Half Frozen', marker='o')
    plt.plot(epochs_range, rewards_unfrozen, label='Unfrozen', marker='o')
    plt.xlabel('Epochs (1000 timesteps per epoch)')
    plt.ylabel('Mean Reward')
    plt.title('Training Trend: Frozen vs Unfrozen Networks (Different Models)')
    plt.legend()
    plt.grid(True)
    plt.show()