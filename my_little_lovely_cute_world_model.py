from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import FrameStack, AtariPreprocessing, LazyFrames
from gymnasium.vector import AsyncVectorEnv, VectorEnv
import torch.nn.functional as F
import torchvision.transforms as T


# Function to calculate the input size based on the cnn_net_arch and target latent size
def calculate_input_size(latent_size, cnn_net_arch):
    size = latent_size
    for (out_channels, kernel_size, stride, padding) in reversed(cnn_net_arch):
        size = (size - 1) * stride - 2 * padding + kernel_size
    return size


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_shape, cnn_net_arch):
        super(Encoder, self).__init__()
        channels, height, width = input_shape
        latent_channels, latent_height, latent_width = latent_shape

        # Calculate the input image size
        target_input_size = calculate_input_size(min(latent_height, latent_width), cnn_net_arch)
        self.resize = T.Resize((target_input_size, target_input_size), antialias=True)

        conv_layers = []
        in_channels = channels
        for out_channels, kernel_size, stride, padding in cnn_net_arch:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels

        # Add the final convolution layer to ensure output channels match latent_channels
        conv_layers.append(nn.Conv2d(in_channels, latent_channels, kernel_size=3, stride=1, padding=1))
        self.encoder = nn.Sequential(*conv_layers)

    def forward(self, x):
        # Ensure the input has a batch dimension (batch_size, channels, height, width)
        if len(x.shape) == 3:  # If the input is missing the batch dimension
            x = x.unsqueeze(0)  # Add the batch dimension
        x = self.resize(x)
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_shape, output_shape, cnn_net_arch):
        super(Decoder, self).__init__()
        latent_channels, latent_height, latent_width = latent_shape
        channels, height, width = output_shape

        deconv_layers = []
        in_channels = latent_channels  # Decoder input channels should match the encoder output channels

        # Use transposed convolution to reconstruct the image
        for out_channels, kernel_size, stride, padding in reversed(cnn_net_arch):
            deconv_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
            deconv_layers.append(nn.ReLU())
            in_channels = out_channels

        # Add the final transposed convolution to ensure output channels match the original image channels
        deconv_layers.append(nn.ConvTranspose2d(in_channels, channels, kernel_size=3, stride=1, padding=1))
        deconv_layers.append(nn.Sigmoid())  # Output pixel values in the range [0, 1]

        self.decoder = nn.Sequential(*deconv_layers)

    def forward(self, latent_state):
        # Ensure the input has a batch dimension (batch_size, channels, height, width)
        if len(latent_state.shape) == 3:  # If the input is missing the batch dimension
            latent_state = latent_state.unsqueeze(0)  # Add the batch dimension
        return self.decoder(latent_state)


class Discriminator(nn.Module):
    def __init__(self, input_shape, conv_arch):
        """
        :param input_shape: Shape of the input image (channels, height, width)
        :param conv_arch: Architecture of the convolutional layers for the discriminator
        """
        super(Discriminator, self).__init__()
        channels, height, width = input_shape

        # Convolutional layers architecture for the discriminator
        conv_layers = []
        in_channels = channels
        for out_channels, kernel_size, stride, padding in conv_arch:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            conv_layers.append(nn.LeakyReLU(0.2))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Global average pooling to reduce the feature map to (batch_size, out_channels, 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Calculate flattened size dynamically using a dummy input
        flattened_size = self._get_flattened_size(input_shape, conv_arch)

        # Fully connected layer that outputs a scalar, representing the probability of the image being "real"
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 1),  # Input is the channel count after global average pooling
            nn.Sigmoid()  # Output range is [0, 1]
        )

    def _get_flattened_size(self, input_shape, conv_arch):
        """
        Dynamically calculate the flattened size based on the given convolutional architecture and input shape
        """
        # Create a dummy input tensor
        sample_tensor = torch.zeros(1, *input_shape)  # Assuming batch_size = 1
        sample_tensor = self.conv_layers(sample_tensor)
        sample_tensor = self.global_avg_pool(sample_tensor)
        return sample_tensor.numel()  # Return the flattened size

    def forward(self, x):
        x = self.conv_layers(x)  # Process through convolutional layers
        x = self.global_avg_pool(x)  # Global average pooling (batch_size, out_channels, 1, 1)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, out_channels)
        return self.fc(x)


class TransitionModelVAE(nn.Module):
    def __init__(self, latent_shape, action_dim, conv_arch):
        """
        :param latent_shape: Shape of the latent space (channels, height, width)
        :param action_dim: Dimensionality of the action space
        :param conv_arch: Convolutional network architecture, e.g., [(64, 4, 2, 1), (128, 4, 2, 1)]
        """
        super(TransitionModelVAE, self).__init__()
        latent_channels, latent_height, latent_width = latent_shape

        # Action embedding: Map the action to a vector with the same size as the latent state
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, latent_channels * latent_height * latent_width),
            nn.ReLU()
        )

        # Convolutional layers for merging the action and latent state
        conv_layers = []
        in_channels = latent_channels * 2  # Channel count is doubled after concatenation
        for out_channels, kernel_size, stride, padding in conv_arch:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv_encoder = nn.Sequential(*conv_layers)

        # Dynamically calculate the flattened size after the convolutional layers
        self.flattened_size = self._get_flattened_size(latent_shape, conv_arch)

        # Fully connected layers to generate mean and logvar
        self.fc_mean = nn.Linear(self.flattened_size, latent_channels * latent_height * latent_width)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_channels * latent_height * latent_width)

        # Added: Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Predict reward, output a scalar
        )

    def _get_flattened_size(self, latent_shape, conv_arch):
        """
        Dynamically calculate the flattened size based on the given convolutional architecture and input shape
        """
        # Assume batch_size = 1, create a dummy input
        sample_tensor = torch.zeros(1, latent_shape[0] * 2, latent_shape[1], latent_shape[2])  # Channels doubled after concatenation
        sample_tensor = self.conv_encoder(sample_tensor)
        return sample_tensor.numel()

    def forward(self, latent_state, action):
        """
        :param latent_state: (batch_size, latent_channels, latent_height, latent_width)
        :param action: (batch_size, action_dim)
        :return: z_next: Next latent state, mean: Mean of the latent distribution, logvar: Log variance, reward_pred: Predicted reward
        """
        batch_size, latent_channels, latent_height, latent_width = latent_state.shape

        # Embed the action and reshape to match the latent state
        action_embed = self.action_embed(action)
        action_embed = action_embed.view(batch_size, latent_channels, latent_height, latent_width)

        # Concatenate action embedding and latent state, rather than adding them
        x = torch.cat([latent_state, action_embed], dim=1)  # Channels doubled after concatenation

        # Process through the convolutional encoder
        x = self.conv_encoder(x)

        # Flatten for the fully connected layers
        x_flat = x.view(batch_size, -1)

        # Generate mean and logvar
        mean = self.fc_mean(x_flat)
        logvar = self.fc_logvar(x_flat)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_next = mean + eps * std

        # Predict reward
        reward_pred = self.reward_predictor(x_flat)

        # Reshape z_next to (batch_size, latent_channels, latent_height, latent_width)
        z_next_reshaped = z_next.view(batch_size, latent_channels, latent_height, latent_width)

        return z_next_reshaped, mean, logvar, reward_pred


class WorldModel:
    def __init__(
            self,
            latent_shape: Tuple[int, int, int],
            num_homomorphism_channels: int,
            cnn_net_arch: List[Tuple[int, int, int, int]],
            transition_model_conv_arch: List[Tuple[int, int, int, int]],
            disc_conv_arch: List[Tuple[int, int, int, int]],
            lr: float = 1e-4,
            discriminator_lr: float = 1e-4,
    ):
        self.latent_shape = latent_shape
        self.homomorphism_latent_space = (num_homomorphism_channels, latent_shape[1], latent_shape[2])
        self.encoder = Encoder(env.single_observation_space.shape, latent_shape, cnn_net_arch)
        self.decoder = Decoder(latent_shape, env.single_observation_space.shape, cnn_net_arch)
        self.transition_model = TransitionModelVAE(latent_shape, env.single_action_space.n, transition_model_conv_arch)
        self.discriminator = Discriminator(env.single_observation_space.shape, disc_conv_arch)

        # Optimizer for all components except the discriminator
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.transition_model.parameters()),
            lr=lr,
        )

        # Separate optimizer for the discriminator
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_lr
        )

        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
