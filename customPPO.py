from collections import defaultdict
from typing import Callable, Optional, List, Dict, Union, Type, Generator, Tuple, Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import Space
from gymnasium.vector.utils import spaces
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.type_aliases import RolloutBufferSamples, PyTorchObs
from stable_baselines3.common.utils import obs_as_tensor, explained_variance
from stable_baselines3.common.vec_env import VecNormalize, VecEnv
from torch import Tensor
from torch.distributions import Categorical, Normal
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import RolloutBuffer
from torch.utils.tensorboard import SummaryWriter

from binary_state_representation.binary2binaryautoencoder import InvNet, ContrastiveNet, Binary2BinaryDecoder, \
    RewardPredictor, TerminationPredictor

from torch.nn import functional as F


class CustomRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class CustomRolloutBuffer(RolloutBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device='cpu', gamma=0.99, gae_lambda=0.95, n_envs=1):
        super().__init__(buffer_size, observation_space, action_space, device, gamma=gamma, gae_lambda=gae_lambda, n_envs=n_envs)

        # Initialize buffer for next observations and dones
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=bool)  # Initialize buffer for dones

    def reset(self) -> None:
        super().reset()
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=bool)  # Reset the dones buffer

    def add(self, *args, next_obs=None, dones=None, **kwargs):
        """
        Add a new transition to the buffer
        :param next_obs: The next observation at time t+1
        :param dones: Done signals indicating if the episode has ended
        """
        # Call the parent add method without next_obs and dones
        super().add(*args, **kwargs)

        # Store the next observation if provided
        if next_obs is not None:
            self.next_observations[self.pos - 1] = torch.tensor(next_obs)

        # Store the dones if provided
        if dones is not None:
            self.dones[self.pos - 1] = dones

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
                "rewards",  # Include rewards in the tensor names
                "dones",  # Include dones in the tensor names
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
    ) -> CustomRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.next_observations[batch_inds],  # Add next_observations to data
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.rewards[batch_inds].flatten(),  # Retrieve rewards from the parent buffer
            self.dones[batch_inds].flatten(),    # Include dones in the output
        )
        return CustomRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class CustomPPO(PPO):
    def __init__(self, *args, log_dir: str, **kwargs):
        super(CustomPPO, self).__init__(*args, **kwargs)

        # self.policy.features_extractor.encoder_only = False
        # self.policy.features_extractor.action_space = self.action_space

        self.rollout_buffer = CustomRolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.log_dir = log_dir
        self.train_counter = 0

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

            with torch.no_grad():
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

            # Correct the `dones` array to handle truncated episodes
            corrected_dones = dones.copy()  # Create a copy of dones to store the corrected dones
            for idx, done in enumerate(dones):
                if done and infos[idx].get("TimeLimit.truncated", False):
                    # If it's a truncated episode, set done to False to avoid incorrect terminal state handling
                    corrected_dones[idx] = False

                    # Handle bootstrapping with value function for truncated episodes
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # Add data to the CustomRolloutBuffer, using `corrected_dones` instead of `dones`
            rollout_buffer.add(
                self._last_obs,  # actual current observation
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                next_obs=new_obs,  # new_obs as next_obs given to buffer
                dones=corrected_dones,  # Use the corrected `corrected_dones` here
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones  # Keep the original `dones` to properly reset the episode start

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        # Compute the returns and advantages
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=corrected_dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer with support for `next_obs`.
        """
        log_writer = SummaryWriter(self.log_dir)

        try:
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

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    values, log_prob, entropy, features \
                        = self.policy.evaluate_actions(rollout_data.observations, actions, return_features=True)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batch size == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # Ratio between old and new policy, should be one at the first iteration
                    ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                    # Clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + torch.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss to favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -torch.mean(-log_prob)
                    else:
                        entropy_loss = -torch.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # #######################################
                    # loss = torch.tensor(0.0).to(loss.device)
                    # #######################################

                    # if do train the encoder with constrains:
                    if not self.policy.features_extractor.encoder_only:
                        # Extract next observations from rollout data
                        obs = rollout_data.observations
                        next_obs = rollout_data.next_observations  # Extract next_obs for future use

                        differences = obs - next_obs
                        # Compute norms along all non-batch dimensions
                        norms = torch.norm(differences, p=2, dim=list(range(1, obs.dim())))
                        same_states = norms < 0.5

                        z0 = features
                        z1 = self.policy.features_extractor(next_obs)

                        # filter out the cases that the agent did not move
                        z0_filtered = z0[~same_states]
                        z1_filtered = z1[~same_states]

                        # get fake z1
                        idx = torch.randperm(len(obs))
                        fake_z1 = z1.view(len(z1), -1)[idx].view(z1.size())
                        fake_z1_filtered = fake_z1[~same_states]

                        actions_filtered = rollout_data.actions[~same_states]

                        _loss, loss_vals = self.policy.features_extractor.compute_loss(
                            obs, next_obs, z0, z1, z0_filtered, z1_filtered,
                            fake_z1_filtered, rollout_data.actions, actions_filtered, rollout_data.rewards,
                            rollout_data.dones,
                        )

                        names = ['feature_loss', 'rec_loss', 'inv_loss', 'ratio_loss', 'reward_loss', 'terminate_loss',
                                 'neighbour_loss']
                        for name, val in zip(names, loss_vals):
                            log_writer.add_scalar(name, val, self.train_counter)

                    else:
                        _loss = torch.tensor(0.0).to(loss.device)

                    constant_loss_val = 0.0
                    for each_constant_loss in self.policy.features_extractor.constant_losses:
                        _loss += each_constant_loss()
                        constant_loss_val += each_constant_loss().clone().detach().cpu().item()
                    log_writer.add_scalar('constant loss', constant_loss_val, self.train_counter)
                    loss += _loss

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    with torch.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    self.train_counter += 1

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")

                        # Optimization step
                        self.policy.optimizer.zero_grad()
                        _loss.backward()
                        # Clip grad norm
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.policy.optimizer.step()
                        log_writer.close()
                        break

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                    log_writer.add_scalar("loss", loss.detach().cpu().item(), self.train_counter - 1)

                self._n_updates += 1
                if not continue_training:
                    log_writer.close()
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
                self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)

        except Exception as e:
            e.printStackTrace()

        log_writer.close()


class BaseEncoderExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, net_arch: List[int], slope=1.0, binary_output=False, encoder_only=True, weights=None, action_space: gym.spaces.Discrete = None):
        # Initialize the base feature extractor with the flattened observation size
        super().__init__(observation_space, torch.prod(torch.tensor(observation_space.shape)).item())

        # Additional attributes
        self.slope = slope
        self.binary_output = binary_output
        self.frozen = False
        self.encoder_only = encoder_only

        self.inv_model = None
        self.discriminator = None
        self.decoder = None
        self.reward_predictor = None
        self.termination_predictor = None

        self.weights = weights
        self.action_space = action_space

        self.net_arch = net_arch

        self.obs_dim = self.features_dim

        if self.weights is None:
            self.weights = {'total': 1.0, 'inv': 1.0, 'dis': 1.0, 'neighbour': 0.1, 'dec': 0.0, 'rwd': 0.1, 'terminate': 1.0}

        if not encoder_only:
            self.weights = defaultdict(lambda: 0.0)
            self.set_up()

        self.constant_losses = []

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        encoded = self.process(observations)

        # Sigmoid activation
        encoded = torch.sigmoid(self.slope * encoded)

        # Conditional output formatting based on binary_output flag
        if self.binary_output:
            encoded_binary = (encoded > 0.5).float().detach() + encoded - encoded.detach()
            return encoded_binary
        else:
            return encoded

    def process(self, observations: torch.Tensor) -> torch.Tensor:
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

    def set_up(self):
        assert self.action_space is not None, "Must specify action space."
        n_actions = self.action_space.n
        n_latent_dims = self.net_arch[-1]
        if self.weights['inv'] > 0.0:
            self.inv_model = InvNet(
                n_actions=n_actions,
                n_latent_dims=n_latent_dims,
                n_hidden_layers=3,
                n_units_per_layer=128,
            )

        if self.weights['dis'] > 0.0:
            self.discriminator = ContrastiveNet(
                n_latent_dims=n_latent_dims,
                n_hidden_layers=3,
                n_units_per_layer=128,
            )

        if self.weights['dec'] > 0.0:
            self.decoder = Binary2BinaryDecoder(
                n_latent_dims=n_latent_dims,
                output_dim=self.obs_dim,
                n_hidden_layers=3,
                n_units_per_layer=128,
            )

        if self.weights['rwd'] > 0.0:
            self.reward_predictor = RewardPredictor(
                n_actions=n_actions,
                n_latent_dims=n_latent_dims,
                n_hidden_layers=3,
                n_units_per_layer=128,
            )

        if self.weights['terminate'] > 0.0:
            self.termination_predictor = TerminationPredictor(
                n_latent_dims=n_latent_dims,
                n_hidden_layers=3,
                n_units_per_layer=128,
            )

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.bce_loss_ = torch.nn.BCELoss(reduction="none")
        self.mse_loss = torch.nn.MSELoss()

    def compute_loss(
            self,
            obs: torch.Tensor,
            next_obs: torch.Tensor,
            z0: torch.Tensor,
            z1: torch.Tensor,
            z0_filtered: torch.Tensor,
            z1_filtered: torch.Tensor,
            fake_z1_filtered: torch.Tensor,
            actions: torch.Tensor,
            actions_filtered: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple]:
        device = next(self.parameters()).device
        if self.encoder_only:
            return torch.tensor(0.0, requires_grad=True).to(device)

        # obs = obs.to(device)
        # next_obs = next_obs.to(device)
        # z0 = z0.to(device)
        # z1 = z1.to(device)
        # z0_filtered = z0_filtered.to(device)
        # z1_filtered = z1_filtered.to(device)
        # fake_z1_filtered = fake_z1_filtered.to(device)
        # actions = actions.to(device)
        # actions_filtered = actions_filtered.to(device)
        # rewards = rewards.to(device)
        # dones = dones.to(device)

        self.cross_entropy = self.cross_entropy.to(device)
        self.bce_loss = self.bce_loss.to(device)
        self.bce_loss_ = self.bce_loss_.to(device)
        self.mse_loss = self.mse_loss.to(device)

        rec_loss = torch.tensor(0.0, requires_grad=True).to(device)
        if self.decoder:
            self.decoder = self.decoder.to(device)
            # compute reconstruct loss
            decoded_z0 = self.decoder(z0)
            decoded_z1 = self.decoder(z1)
            rec_loss = self.bce_loss(
                torch.cat((decoded_z0, decoded_z1), dim=0),
                torch.cat((obs, next_obs), dim=0),
            )

        inv_loss = torch.tensor(0.0, requires_grad=True).to(device)
        if self.inv_model:
            self.inv_model = self.inv_model.to(device)
            if z0_filtered.size(0) > 0:
                pred_actions = self.inv_model(z0_filtered, z1_filtered)
                inv_loss = self.cross_entropy(pred_actions, actions_filtered.squeeze(dim=-1).type(torch.int64))

        ratio_loss = torch.tensor(0.0, requires_grad=True).to(device)
        if self.discriminator:
            self.discriminator = self.discriminator.to(device)
            labels = torch.cat((
                torch.ones(len(z1_filtered), device=z1_filtered.device),
                torch.zeros(len(fake_z1_filtered), device=fake_z1_filtered.device),
            ), dim=0)
            pred_fakes = torch.cat((
                self.discriminator(z0_filtered, z1_filtered),
                self.discriminator(z0_filtered, fake_z1_filtered),
            ), dim=0).squeeze()
            ratio_loss = self.bce_loss(pred_fakes, labels)

        reward_loss = torch.tensor(0.0, requires_grad=True).to(device)
        if self.reward_predictor:
            self.reward_predictor = self.reward_predictor.to(device)
            # compute reward loss
            pred_rwds = self.reward_predictor(z0, actions.type(torch.int64).squeeze(), z1).squeeze()
            reward_loss = self.mse_loss(pred_rwds, rewards)
            a = rewards
            b = pred_rwds

        terminate_loss = torch.tensor(0.0, requires_grad=True).to(device)
        if self.termination_predictor:
            self.termination_predictor = self.termination_predictor.to(device)
            # compute terminate loss
            pred_terminated = self.termination_predictor(z1).squeeze()
            dones_val = dones.type(torch.float32).squeeze()
            terminate_loss = F.binary_cross_entropy(pred_terminated, dones_val)
            # print(pred_terminated)
            # print(dones_val)
            # print(terminate_loss)
            # c = pred_terminated
            # d = dones
            pass

        # compute neighbour loss
        # neighbour_loss = torch.tensor(0.0).to(device)
        distances = torch.abs(z0 - z1) * 0.1
        weights = torch.linspace(1.0, 0.0, steps=z0.size(1)).to(z0.device)
        weights = weights.unsqueeze(0)
        weighted_distances = distances * weights
        weighted_distance = torch.sum(weighted_distances, dim=1)
        neighbour_loss = torch.mean(torch.pow(weighted_distance, 2))

        # compute total loss
        loss = torch.tensor(0.0, requires_grad=True).to(device)
        loss += rec_loss * self.weights['dec']
        loss += inv_loss * self.weights['inv']
        loss += ratio_loss * self.weights['dis']
        loss += reward_loss * self.weights['rwd']
        loss += terminate_loss * self.weights['terminate']
        loss += neighbour_loss * self.weights['neighbour']
        loss *= self.weights['total']

        return terminate_loss, (
            loss.clone().detach().cpu().item(),
            rec_loss.clone().detach().cpu().item(),
            inv_loss.clone().detach().cpu().item(),
            ratio_loss.clone().detach().cpu().item(),
            reward_loss.clone().detach().cpu().item(),
            terminate_loss.clone().detach().cpu().item(),
            neighbour_loss.clone().detach().cpu().item(),
        )


class MLPEncoderExtractor(BaseEncoderExtractor):
    def __init__(self, observation_space: gym.spaces.Box, net_arch=None, activation_fn=nn.ReLU, encoder_only=True,
                 action_space: gym.spaces.Discrete = None, weights=None):
        # Build the encoder network layers
        if net_arch is None:
            net_arch = [64, 64]

        super().__init__(observation_space, net_arch=net_arch, encoder_only=encoder_only, action_space=action_space, weights=weights)

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

    def process(self, observations: torch.Tensor) -> torch.Tensor:
        # First flatten the input (same as FlattenExtractor)
        observations = observations.view(observations.size(0), -1)
        # Use the MLP network from the parent class
        return self.mlp(observations)


class TransformerEncoderExtractor(BaseEncoderExtractor):
    def __init__(self, observation_space: gym.spaces.Box, net_arch=None, num_transformer_layers=2, n_heads=8,
                 activation_fn=nn.ReLU, encoder_only=True, action_space: gym.spaces.Discrete = None, weights=None):
        # Define the MLP layers according to net_arch
        if net_arch is None:
            net_arch = [128, 64]

        super().__init__(observation_space, net_arch=net_arch, encoder_only=encoder_only, action_space=action_space, weights=weights)

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

    def process(self, observations: torch.Tensor) -> torch.Tensor:
        # First flatten the input (same as FlattenExtractor)
        observations = observations.view(observations.size(0), -1)
        # Apply padding if necessary
        if self.pad_size > 0:
            padding = torch.zeros((observations.size(0), self.pad_size), device=observations.device)
            observations = torch.cat((observations, padding), dim=1)

        # Reshape to [batch_size, sequence_length, d_model] for Transformer processing
        embedded = observations.unsqueeze(1)  # Change to (batch_size, 1, d_model)

        # Pass through the Transformer encoder
        encoded = self.transformer_encoder(embedded)

        # Flatten the output from the Transformer
        encoded = encoded.squeeze(1)  # Remove the extra sequence dimension

        return self.mlp(encoded)


class CNNEncoderExtractor(BaseEncoderExtractor):
    def __init__(self, observation_space: gym.spaces.Box, net_arch=None, cnn_net_arch=None, activation_fn=nn.ReLU,
                 encoder_only=True, action_space: gym.spaces.Discrete=None, weights=None):
        if net_arch is None:
            net_arch = [64, 64]

        if cnn_net_arch is None:
            # Default CNN architecture: [(out_channels, kernel_size, stride, padding)]
            cnn_net_arch = [
                (32, 3, 2, 1),  # out_channels=32, kernel_size=3, stride=2, padding=1
                (64, 3, 2, 1),  # out_channels=64, kernel_size=3, stride=2, padding=1
                (128, 3, 2, 1),  # out_channels=128, kernel_size=3, stride=2, padding=1
            ]

        super().__init__(observation_space, net_arch=net_arch, encoder_only=encoder_only, action_space=action_space,
                         weights=weights)

        # Create convolutional layers based on cnn_net_arch
        cnn_layers = []
        in_channels = observation_space.shape[0]  # The input channel corresponds to observation space's first dimension
        for out_channels, kernel_size, stride, padding in cnn_net_arch:
            cnn_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding))
            cnn_layers.append(activation_fn())
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate the output shape after convolutions for the observation space size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)
            conv_output = self.cnn(dummy_input)
            pooled_output = F.adaptive_avg_pool2d(conv_output,
                                                  (1, 1))  # Apply average pooling to get a fixed-size output
            conv_output_flattened_dim = pooled_output.numel()  # Use numel() to get the total number of elements

        # Fully connected layers defined by net_arch
        input_dim = conv_output_flattened_dim
        fc_layers = []
        for layer_size in net_arch[:-1]:
            fc_layers.append(nn.Linear(input_dim, layer_size))
            fc_layers.append(activation_fn())
            input_dim = layer_size

        # Add the last layer without activation
        fc_layers.append(nn.Linear(input_dim, net_arch[-1]))

        # Create the fully connected model
        self.fc = nn.Sequential(*fc_layers)

        # Set the features_dim and _features_dim to the last layer size to maintain the binding relationship
        self._features_dim = net_arch[-1]

    def process(self, observations: torch.Tensor) -> torch.Tensor:
        # Reshape observations to match the expected input for CNN: (batch_size, channels, height, width)
        batch_size = observations.size(0)
        observations = observations.view(batch_size, *self.observation_space.shape)

        # Apply the CNN layers
        conv_features = self.cnn(observations)

        # Apply average pooling to make the feature map adaptable to any input size
        pooled_features = F.adaptive_avg_pool2d(conv_features, (1, 1))

        # Flatten the pooled features
        flattened_features = pooled_features.view(batch_size, -1)

        # Pass through the fully connected layers
        return self.fc(flattened_features)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        # Reshape z to match embedding size
        z_flattened = z.view(-1, z.size(-1))
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)
                     - 2 * torch.matmul(z_flattened, self.embeddings.weight.t())
                     + torch.sum(self.embeddings.weight ** 2, dim=1))
        min_indices = torch.argmin(distances, dim=1)
        z_q = self.embeddings(min_indices).view(z.shape)
        return z_q, min_indices


# class MLPVectorQuantizerEncoderExtractor(BaseEncoderExtractor):
#     def __init__(self, observation_space: gym.spaces.Box, net_arch=None, embedding_dim=64, num_embeddings=512,
#                  activation_fn=nn.ReLU, encoder_only=True, action_space: gym.spaces.Discrete = None, weights=None):
#         if net_arch is None:
#             net_arch = [128, 64]
#
#         # UNSOLVED ISSUE WITH LAST LAYER OF NET ARCH
#
#         super(MLPVectorQuantizerEncoderExtractor, self).__init__(observation_space, net_arch=net_arch,
#                                                                  encoder_only=encoder_only, action_space=action_space,
#                                                                  weights=weights)
#
#         input_dim = observation_space.shape[0]
#         encoder_layers = []
#         for layer_size in net_arch:
#             encoder_layers.append(nn.Linear(input_dim, layer_size))
#             encoder_layers.append(activation_fn())
#             input_dim = layer_size
#
#         self.encoder = nn.Sequential(*encoder_layers)
#
#         # Vector Quantizer
#         self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)
#
#         post_quant_layers = []
#         post_quant_layers.append(nn.Linear(embedding_dim, net_arch[-1]))  # Map to final output layer
#         self.post_quant_mlp = nn.Sequential(*post_quant_layers)
#
#         # Set the features_dim and _features_dim to the final output layer size
#         self._features_dim = net_arch[-1]
#         self.commit_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
#         self.constant_losses.append(self.get_commit_loss)
#
#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         observations = observations.view(observations.size(0), -1)
#         encoded = self.encoder(observations)
#         z_q, min_indices = self.vector_quantizer(encoded)
#         self.commit_loss = F.mse_loss(z_q.detach(), encoded) + F.mse_loss(z_q, encoded.detach())
#         return self.post_quant_mlp(z_q)
#
#     def get_commit_loss(self):
#         return self.commit_loss


class CNNVectorQuantizerEncoderExtractor(BaseEncoderExtractor):
    def __init__(self, observation_space: gym.spaces.Box, net_arch=None, cnn_net_arch=None, embedding_dim=64,
                 num_embeddings=512, activation_fn=nn.ReLU, encoder_only=True, action_space: gym.spaces.Discrete=None,
                 weights=None):
        if net_arch is None:
            net_arch = [embedding_dim]  # Default fully connected architecture after quantization

        if cnn_net_arch is None:
            # Default CNN architecture: [(out_channels, kernel_size, stride, padding)]
            cnn_net_arch = [
                (32, 3, 2, 1),  # out_channels=32, kernel_size=3, stride=2, padding=1
                (64, 3, 2, 1),  # out_channels=64, kernel_size=3, stride=2, padding=1
                (128, 3, 2, 1),  # out_channels=128, kernel_size=3, stride=2, padding=1
            ]

        # net_arch used only for knowing the input dimension for super so it should be embedding_dim
        super().__init__(observation_space, net_arch=[embedding_dim], encoder_only=encoder_only, action_space=action_space,
                         weights=weights)

        # Create CNN layers
        cnn_layers = []
        in_channels = observation_space.shape[0]  # The input channel corresponds to observation space's first dimension
        for out_channels, kernel_size, stride, padding in cnn_net_arch:
            cnn_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding))
            cnn_layers.append(activation_fn())
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate the output shape after convolutions for the observation space size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)
            conv_output = self.cnn(dummy_input)
            pooled_output = F.adaptive_avg_pool2d(conv_output,
                                                  (1, 1))  # Apply average pooling to get a fixed-size output
            conv_output_flattened_dim = pooled_output.numel()  # Use numel() to get the total number of elements

        # MLP layers (from net_arch) applied after CNN and before Vector Quantizer
        mlp_layers = []
        input_dim = conv_output_flattened_dim
        for layer_size in net_arch:
            mlp_layers.append(nn.Linear(input_dim, layer_size))
            mlp_layers.append(activation_fn())
            input_dim = layer_size

        self.mlp = nn.Sequential(*mlp_layers)

        # Linear layer to map MLP output to the embedding_dim
        self.fc_to_embedding_dim = nn.Linear(input_dim, embedding_dim)

        # Vector Quantizer
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        # Set the features_dim and _features_dim to the final output layer size
        self._features_dim = embedding_dim

        # Commit loss for the vector quantization process
        self.commit_loss = torch.tensor(0.0, requires_grad=True)
        self.constant_losses.append(self.get_commit_loss)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Reshape observations to match the expected input for CNN: (batch_size, channels, height, width)
        batch_size = observations.size(0)
        observations = observations.view(batch_size, *self.observation_space.shape)

        # Apply CNN layers
        conv_features = self.cnn(observations)

        # Apply average pooling
        pooled_features = F.adaptive_avg_pool2d(conv_features, (1, 1))

        # Flatten the pooled features
        flattened_features = pooled_features.view(batch_size, -1)

        # Pass through MLP layers (from net_arch)
        mlp_output = self.mlp(flattened_features)

        # Map MLP output to embedding_dim
        embedded_features = self.fc_to_embedding_dim(mlp_output)

        # constrained with tanh activation
        embedded_features = F.tanh(embedded_features)

        # Vector quantization
        z_q, min_indices = self.vector_quantizer(embedded_features)

        # Calculate commit loss
        self.commit_loss = F.mse_loss(z_q.detach(), embedded_features) + F.mse_loss(z_q, embedded_features.detach())

        # Return the quantized result
        return z_q

    def get_commit_loss(self):
        return self.commit_loss


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

    def evaluate_actions(self, obs: PyTorchObs, actions: torch.Tensor, return_features=False) -> (
            Tuple[Any, Tensor, Optional[Tensor]] or
            Tuple[Any, Tensor, Optional[Tensor], Union[Tensor, Tuple[Tensor, Tensor]]]
    ):
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        if return_features:
            return values, log_prob, entropy, features
        return values, log_prob, entropy


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
    dummy_observations = torch.rand((batch_size, *observation_shape), dtype=torch.float32)

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

