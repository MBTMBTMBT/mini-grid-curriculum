from typing import Callable, Optional, List, Dict, Union, Type, Generator

import numpy as np
import torch
import torch as th
import torch.nn as nn
import gymnasium as gym
from gymnasium.vector.utils import spaces
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor, explained_variance
from stable_baselines3.common.vec_env import VecNormalize, VecEnv
from torch.distributions import Categorical, Normal
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import RolloutBuffer

from torch.nn import functional as F


class CustomRolloutBufferSamples(RolloutBufferSamples):
    next_observations: th.Tensor


class CustomRolloutBuffer(RolloutBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device='cpu', gamma=0.99, gae_lambda=0.95,
                 n_envs=1):
        super().__init__(buffer_size, observation_space, action_space, device, gamma=gamma, gae_lambda=gae_lambda,
                         n_envs=n_envs)

        # Initialize buffer for next observations
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)

    def reset(self) -> None:
        super().reset()
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)

    def add(self, *args, next_obs=None, **kwargs):
        """
        Add a new transition to the buffer
        :param next_obs: The next observation at time t+1
        """
        # Call the parent add method without next_obs
        super().add(*args, **kwargs)

        # Store the next observation if provided
        if next_obs is not None:
            self.next_observations[self.pos - 1] = th.tensor(next_obs, device=self.device)

    def get(self, batch_size: Optional[int] = None) -> Generator[CustomRolloutBufferSamples, None, None]:
        assert self.full, "Buffer is not full"
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "next_observations",  # Include next_observations
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.next_observations[batch_inds],  # Add next_observations to data
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class CustomPPO(PPO):
    def __init__(self, *args, **kwargs):
        super(CustomPPO, self).__init__(*args, **kwargs)

        self.rollout_buffer = CustomRolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device
        )

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: CustomRolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``CustomRolloutBuffer``.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        # Sample new weights for state-dependent exploration (if applicable)
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # Step in the environment using the clipped actions
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Callback update
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with the value function
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # Add data to the CustomRolloutBuffer, including next_obs
            rollout_buffer.add(
                self._last_obs,  # actual current observation
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                next_obs=new_obs  # new_obs as next_obs given to buffer
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        # Compute the returns and advantages
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer with support for `next_obs`.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Extract next observations from rollout data
                next_obs = rollout_data.next_observations  # Extract next_obs for future use

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batch size == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss to favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


class BaseEncoderExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, activation_fn=nn.ReLU, slope=1.0, binary_output=False):
        # Initialize the base feature extractor with the flattened observation size
        super().__init__(observation_space, th.prod(th.tensor(observation_space.shape)).item())

        # Additional attributes
        self.slope = slope
        self.binary_output = binary_output
        self.frozen = False

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # First flatten the input (same as FlattenExtractor)
        observations = observations.view(observations.size(0), -1)
        encoded = self.process(observations)

        # Sigmoid activation
        encoded = torch.sigmoid(self.slope * encoded)

        # Conditional output formatting based on binary_output flag
        if self.binary_output:
            encoded_binary = (encoded > 0.5).float().detach() + encoded - encoded.detach()
            return encoded_binary
        else:
            return encoded

    def process(self, observations: th.Tensor) -> th.Tensor:
        # Placeholder method to be implemented by subclasses
        raise NotImplementedError

    def freeze(self):
        """
        Freeze the parameters, preventing them from being updated during training.
        """
        self.frozen = True
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze the parameters, allowing them to be updated during training.
        """
        self.frozen = False
        for param in self.parameters():
            param.requires_grad = True


class MLPEncoderExtractor(BaseEncoderExtractor):
    def __init__(self, observation_space: gym.spaces.Box, net_arch=None, activation_fn=nn.ReLU):
        super().__init__(observation_space, activation_fn=activation_fn)

        # Build the encoder network layers
        if net_arch is None:
            net_arch = [64, 64]
        input_dim = self.features_dim  # Flattened observation dimension
        layers = []
        for layer_size in net_arch[:-1]:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(activation_fn())
            input_dim = layer_size

        # Add last layer without activation
        layers.append(nn.Linear(input_dim, net_arch[-1]))

        # Create the MLP model
        self.mlp = nn.Sequential(*layers)

        # Set the features_dim and _features_dim to the last layer size to maintain the binding relationship
        self._features_dim = net_arch[-1]

    def process(self, observations: th.Tensor) -> th.Tensor:
        # Use the MLP network from the parent class
        return self.mlp(observations)


class TransformerEncoderExtractor(BaseEncoderExtractor):
    def __init__(self, observation_space: gym.spaces.Box, net_arch=None, num_transformer_layers=2, n_heads=8,
                 activation_fn=nn.ReLU):
        super().__init__(observation_space, activation_fn=activation_fn)

        # Set the input sequence length and model dimension (d_model) to match the input
        self.seq_length = self.features_dim
        d_model = self.seq_length

        # Ensure d_model is divisible by n_heads by padding if necessary
        if d_model % n_heads != 0:
            padded_d_model = ((d_model // n_heads) + 1) * n_heads
            self.pad_size = padded_d_model - d_model
            d_model = padded_d_model
        else:
            self.pad_size = 0

        # Build the Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, activation=activation_fn(),
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Define the MLP layers according to net_arch
        if net_arch is None:
            net_arch = [128, 64]
        mlp_layers = []
        input_dim = d_model  # Start with the output dimension of the Transformer layers
        for layer_size in net_arch[:-1]:
            mlp_layers.append(nn.Linear(input_dim, layer_size))
            mlp_layers.append(activation_fn())
            input_dim = layer_size

        # Add last layer without activation
        mlp_layers.append(nn.Linear(input_dim, net_arch[-1]))

        # Create the MLP network as a sequential model
        self.mlp = nn.Sequential(*mlp_layers)

        # Set the features_dim and _features_dim to the last layer size to maintain the binding relationship
        self._features_dim = net_arch[-1]

    def process(self, observations: th.Tensor) -> th.Tensor:
        # Apply padding if necessary
        if self.pad_size > 0:
            padding = th.zeros((observations.size(0), self.pad_size), device=observations.device)
            observations = th.cat((observations, padding), dim=1)

        # Reshape to [batch_size, sequence_length, d_model] for Transformer processing
        embedded = observations.unsqueeze(1)  # Change to (batch_size, 1, d_model)

        # Pass through the Transformer encoder
        encoded = self.transformer_encoder(embedded)

        # Flatten the output from the Transformer
        encoded = encoded.squeeze(1)  # Remove the extra sequence dimension

        return self.mlp(encoded)



# class MLPEncoderExtractor(BaseFeaturesExtractor):
#     """
#     A custom feature extractor that applies a neural network (MLP) as an encoder.
#     It also supports freezing/unfreezing the network.
#
#     :param observation_space: (gym.Space) The observation space of the environment
#     :param net_arch: (list of int) The architecture of the MLP encoder (list of layer sizes)
#     :param activation_fn: (nn.Module) The activation function to use (ReLU, Tanh, etc.)
#     """
#
#     def __init__(self, observation_space: gym.spaces.Box, net_arch=None, activation_fn=nn.ReLU):
#         # Initialize the base feature extractor with the flattened observation size
#         super().__init__(observation_space, th.prod(th.tensor(observation_space.shape)).item())
#
#         # Build the encoder network layers
#         if net_arch is None:
#             net_arch = [64, 64]
#         input_dim = self.features_dim  # Flattened observation dimension
#         layers = []
#         for layer_size in net_arch[:-1]:
#             layers.append(nn.Linear(input_dim, layer_size))
#             layers.append(activation_fn())
#             input_dim = layer_size
#
#         # add last layer without activation
#         layers.append(nn.Linear(input_dim, net_arch[-1]))
#
#         # Create the sequential model
#         self.encoder = nn.Sequential(*layers)
#
#         # reset the output dimension
#         self._features_dim = net_arch[-1]
#
#         # Initially, the network is not frozen
#         self.frozen = False
#
#         # slop of last activation layer
#         self.slope = 1.0
#
#         self.binary_output = False
#
#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         # First flatten the input (same as FlattenExtractor)
#         observations = observations.view(observations.size(0), -1)
#         # Then pass through the encoder network
#         encoded = self.encoder(observations)
#
#         # sigmoid activation
#         encoded = torch.sigmoid(self.slope * encoded)
#
#         # Conditional output formatting based on binary_output flag
#         if self.binary_output:
#             # Output binary version using STE-like method for backprop compatibility
#             encoded_binary = (encoded > 0.5).float().detach() + encoded - encoded.detach()
#             return encoded_binary
#         else:
#             # Output continuous values
#             return encoded
#
#     def freeze(self):
#         """
#         Freeze the parameters of the encoder, preventing them from being updated during training.
#         """
#         self.frozen = True
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#
#     def unfreeze(self):
#         """
#         Unfreeze the parameters of the encoder, allowing them to be updated during training.
#         """
#         self.frozen = False
#         for param in self.encoder.parameters():
#             param.requires_grad = True
#
#
# class TransformerEncoderExtractor(BaseFeaturesExtractor):
#     """
#     A custom feature extractor that applies a Transformer encoder followed by an MLP network.
#     The Transformer layers keep the input dimension, and the MLP structure is determined by net_arch.
#
#     :param observation_space: (gym.Space) The observation space of the environment
#     :param net_arch: (list of int) The architecture of the MLP layers (list of layer sizes)
#     :param num_transformer_layers: (int) The number of Transformer layers to apply
#     :param n_heads: (int) Number of attention heads in the Transformer
#     :param activation_fn: (nn.Module) The activation function to use (ReLU, Tanh, etc.)
#     """
#
#     def __init__(self, observation_space: gym.spaces.Box, net_arch=None, num_transformer_layers=2, n_heads=8,
#                  activation_fn=nn.ReLU):
#         # Initialize the base feature extractor with the flattened observation size
#         super().__init__(observation_space, th.prod(th.tensor(observation_space.shape)).item())
#
#         # Default MLP architecture if none is provided
#         if net_arch is None:
#             net_arch = [128, 64]
#
#         # Set the input sequence length and model dimension (d_model) to match the input
#         self.seq_length = self.features_dim
#         d_model = self.seq_length
#
#         # Ensure d_model is divisible by n_heads by padding if necessary
#         if d_model % n_heads != 0:
#             # Find the nearest multiple of n_heads greater than or equal to d_model
#             padded_d_model = ((d_model // n_heads) + 1) * n_heads
#             self.pad_size = padded_d_model - d_model
#             d_model = padded_d_model
#         else:
#             self.pad_size = 0
#
#         # Build the Transformer encoder layers with batch_first=True
#         self.transformer_layers = nn.ModuleList()
#         for _ in range(num_transformer_layers):
#             encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, activation=activation_fn(),
#                                                        batch_first=True)
#             self.transformer_layers.append(nn.TransformerEncoder(encoder_layer, num_layers=1))
#
#         # Define the MLP layers according to net_arch
#         mlp_layers = []
#         input_dim = d_model  # Start with the output dimension of the Transformer layers
#         for layer_size in net_arch[:-1]:
#             mlp_layers.append(nn.Linear(input_dim, layer_size))
#             mlp_layers.append(activation_fn())
#             input_dim = layer_size
#
#         # add last layer without activation
#         mlp_layers.append(nn.Linear(input_dim, net_arch[-1]))
#
#         # Create the MLP network as a sequential model
#         self.mlp = nn.Sequential(*mlp_layers)
#
#         # Set the final feature dimension as the last layer size of net_arch
#         self._features_dim = net_arch[-1]
#
#         # Initially, the network is not frozen
#         self.frozen = False
#
#         # slop of last activation layer
#         self.slope = 1.0
#
#         self.binary_output = False
#
#     def forward(self, observations: th.Tensor, binary_output=True) -> th.Tensor:
#         # Flatten the input observations
#         observations = observations.view(observations.size(0), -1)
#
#         # Apply padding if necessary
#         if self.pad_size > 0:
#             padding = th.zeros((observations.size(0), self.pad_size), device=observations.device)
#             observations = th.cat((observations, padding), dim=1)
#
#         # Reshape to [batch_size, sequence_length, d_model] for Transformer processing
#         embedded = observations.unsqueeze(1)  # Change to (batch_size, 1, d_model)
#
#         # Pass through each Transformer encoder layer
#         for transformer_layer in self.transformer_layers:
#             embedded = transformer_layer(embedded)
#
#         # Flatten the output from the Transformer
#         encoded = embedded.squeeze(1)  # Remove the extra sequence dimension
#
#         # sigmoid activation
#         encoded = torch.sigmoid(self.slope * encoded)
#
#         # Pass through the MLP network
#         encoded = self.mlp(encoded)
#
#         # Conditional output formatting based on binary_output flag
#         if self.binary_output:
#             # Output binary version using STE-like method for backprop compatibility
#             encoded_binary = (encoded > 0.5).float().detach() + encoded - encoded.detach()
#             return encoded_binary
#         else:
#             # Output continuous values
#             return encoded
#
#     def freeze(self):
#         """
#         Freeze the parameters of the encoder, preventing them from being updated during training.
#         """
#         self.frozen = True
#         for param in self.parameters():
#             param.requires_grad = False
#
#     def unfreeze(self):
#         """
#         Unfreeze the parameters of the encoder, allowing them to be updated during training.
#         """
#         self.frozen = False
#         for param in self.parameters():
#             param.requires_grad = True


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

