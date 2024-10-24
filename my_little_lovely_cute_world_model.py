import io
import os
from typing import Tuple, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from jedi.inference.gradual.typing import Callable
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.models import vgg16
from piq import ssim

from customize_minigrid.wrappers import FullyObsImageWrapper
from gymnasium_dataset import GymDataset
from task_config import TaskConfig, make_env


# Simple dictionary to map action numbers to string labels
action_dict = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done"
}


# Enhanced loss with separate handling for L1 and L2 (MSE) losses with optional thresholding
class FlexibleThresholdedLoss(nn.Module):
    def __init__(self, use_mse_threshold=False, use_mae_threshold=True, mse_threshold=None, mae_threshold=None,
                 reduction='mean', l1_weight=1.0, l2_weight=1.0, threshold_weight=1.0, non_threshold_weight=1.0,
                 mse_clip_ratio=None, mae_clip_ratio=1e2):
        """
        use_mse_threshold: Whether to apply a threshold to L2 (MSE)-based loss.
        use_mae_threshold: Whether to apply a threshold to L1-based loss.
        mse_threshold: Static L2 (MSE)-based threshold if provided; otherwise, will use adaptive MSE threshold.
        mae_threshold: Static L1-based threshold if provided; otherwise, will use adaptive MAE threshold.
        reduction: Specifies the reduction to apply to the output. ('mean' or 'sum').
        l1_weight: The weight for L1 loss in both thresholded and non-thresholded parts.
        l2_weight: The weight for L2 (MSE) loss in both thresholded and non-thresholded parts.
        threshold_weight: Weight for the thresholded loss part.
        non_threshold_weight: Weight for the non-thresholded loss part.
        mse_clip_ratio: Ratio to apply the upper limit clipping for MSE thresholded loss.
        mae_clip_ratio: Ratio to apply the upper limit clipping for MAE thresholded loss.
        """
        super(FlexibleThresholdedLoss, self).__init__()
        self.use_mse_threshold = use_mse_threshold
        self.use_mae_threshold = use_mae_threshold
        self.mse_threshold = mse_threshold
        self.mae_threshold = mae_threshold
        self.reduction = reduction
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.threshold_weight = threshold_weight
        self.non_threshold_weight = non_threshold_weight
        self.mse_clip_ratio = mse_clip_ratio
        self.mae_clip_ratio = mae_clip_ratio

    def forward(self, input_img, target_img):
        # Calculate pixel-wise absolute difference (for L1) and squared difference (for L2/MSE)
        pixel_diff = torch.abs(input_img - target_img)  # For L1
        pixel_diff_squared = (input_img - target_img) ** 2  # For L2 (MSE)

        # General mean for normalization
        general_mean = pixel_diff.mean()

        # Part 1: L2 (MSE)-based threshold handling with optional normalization
        mse_loss = pixel_diff_squared.mean()  # MSE (L2) loss
        if self.use_mse_threshold:
            if self.mse_threshold is None:
                self.mse_threshold = mse_loss  # Set adaptive threshold based on MSE

            # Filter values based on threshold
            mse_thresholded_diff = pixel_diff_squared[pixel_diff_squared >= self.mse_threshold]

            # Calculate the thresholded loss and normalize if necessary
            if mse_thresholded_diff.numel() > 0:
                mse_thresholded_loss = mse_thresholded_diff.mean()
                if self.mse_clip_ratio is not None:
                    clip_value = self.mse_clip_ratio * general_mean
                    # Normalize the loss if it exceeds the clip_value
                    if mse_thresholded_loss > clip_value:
                        mse_thresholded_loss = clip_value * mse_thresholded_loss / mse_thresholded_loss.detach()
            else:
                mse_thresholded_loss = torch.tensor(0.0, device=pixel_diff.device)
        else:
            # No thresholding, use all squared differences for L2 (MSE)
            mse_thresholded_loss = pixel_diff_squared.mean()

        # Part 2: L1-based threshold handling with optional normalization
        mae_loss = pixel_diff.mean()  # L1 (absolute difference) loss
        if self.use_mae_threshold:
            if self.mae_threshold is None:
                self.mae_threshold = mae_loss  # Set adaptive threshold based on MAE

            # Filter values based on threshold
            mae_thresholded_diff = pixel_diff[pixel_diff >= self.mae_threshold]

            # Calculate the thresholded loss and normalize if necessary
            if mae_thresholded_diff.numel() > 0:
                mae_thresholded_loss = mae_thresholded_diff.mean()
                if self.mae_clip_ratio is not None:
                    clip_value = self.mae_clip_ratio * general_mean
                    # Normalize the loss if it exceeds the clip_value
                    if mae_thresholded_loss > clip_value:
                        mae_thresholded_loss = clip_value * mae_thresholded_loss / mae_thresholded_loss.detach()
            else:
                mae_thresholded_loss = torch.tensor(0.0, device=pixel_diff.device)
        else:
            # No thresholding, use all absolute differences for L1
            mae_thresholded_loss = pixel_diff.mean()

        # Part 3: Non-thresholded loss (all differences are considered for both L1 and L2)
        non_thresholded_l1_loss = pixel_diff.mean()  # L1 part without threshold
        non_thresholded_l2_loss = pixel_diff_squared.mean()  # L2 (MSE) part without threshold

        # Combine thresholded L1 and L2 losses
        combined_thresholded_loss = self.l1_weight * mae_thresholded_loss + self.l2_weight * mse_thresholded_loss

        # Combine non-thresholded L1 and L2 losses
        combined_non_thresholded_loss = self.l1_weight * non_thresholded_l1_loss + self.l2_weight * non_thresholded_l2_loss

        # Apply reduction (mean or sum) to each part
        if self.reduction == 'mean':
            combined_thresholded_loss = combined_thresholded_loss.mean()
            combined_non_thresholded_loss = combined_non_thresholded_loss.mean()
        elif self.reduction == 'sum':
            combined_thresholded_loss = combined_thresholded_loss.sum()
            combined_non_thresholded_loss = combined_non_thresholded_loss.sum()

        # Combine thresholded and non-thresholded losses with respective weights
        total_loss = self.threshold_weight * combined_thresholded_loss + self.non_threshold_weight * combined_non_thresholded_loss

        return total_loss


# Function to calculate the input size based on the cnn_net_arch and target latent size
def calculate_input_size(latent_size, cnn_net_arch):
    size = latent_size
    for (out_channels, kernel_size, stride, padding) in reversed(cnn_net_arch):
        size = (size - 1) * stride - 2 * padding + kernel_size
    return size


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out += residual  # Add the input (skip connection)
        return self.activation(out)


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_shape, cnn_net_arch):
        super(Encoder, self).__init__()
        channels, height, width = input_shape
        latent_channels, latent_height, latent_width = latent_shape

        # Calculate the input image size
        target_input_size = calculate_input_size(min(latent_height, latent_width), cnn_net_arch)
        self.resize = T.Resize((target_input_size, target_input_size), antialias=True)
        print(f"The input observations will be resized into {target_input_size} * {target_input_size} for encoding.")

        conv_layers = []
        in_channels = channels
        for out_channels, kernel_size, stride, padding in cnn_net_arch:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            conv_layers.append(nn.LeakyReLU())

            # Add a residual block after each Conv2D layer
            conv_layers.append(ResidualBlock(out_channels))

            in_channels = out_channels

        # Add the final convolution layer to ensure output channels match latent_channels
        conv_layers.append(nn.Conv2d(in_channels, latent_channels, kernel_size=3, stride=1, padding=1))
        self.encoder = nn.Sequential(*conv_layers)

    def forward(self, x):
        # Ensure the input has a batch dimension (batch_size, channels, height, width)
        if len(x.shape) == 3:  # If the input is missing the batch dimension
            x = x.unsqueeze(0)  # Add the batch dimension
        x = self.resize(x)
        return F.sigmoid(self.encoder(x))


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
            deconv_layers.append(nn.LeakyReLU())

            # Add a residual block for additional feature extraction
            deconv_layers.append(ResidualBlock(out_channels))

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


class TransitionModel(nn.Module):
    def __init__(self, latent_shape, action_dim, conv_arch):
        """
        Deterministic Transition Model based on CNN architecture.
        :param latent_shape: Shape of the latent space (channels, height, width)
        :param action_dim: Dimensionality of the action space (int)
        :param conv_arch: Convolutional network architecture for the encoder and decoder, e.g., [(64, 4, 2, 1), (128, 4, 2, 1)]
        """
        super(TransitionModel, self).__init__()
        latent_channels, latent_height, latent_width = latent_shape

        self.action_dim = action_dim

        # Initial convolution to map input to the expected number of channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(latent_channels + action_dim, conv_arch[0][0], kernel_size=3, stride=1, padding=1),  # e.g. 19 to 64 channels
            nn.LeakyReLU(0.2),
            nn.LayerNorm([conv_arch[0][0], latent_height, latent_width])  # Add LayerNorm after the initial convolution
        )

        # Convolutional layers with Residual Blocks
        conv_layers = []
        in_channels = conv_arch[0][0]  # Start with output from initial_conv (e.g. 64)
        for out_channels, kernel_size, stride, padding in conv_arch:
            conv_layers.append(ResidualBlock(in_channels))  # Residual block keeps the channels the same
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))  # Now update the channels
            conv_layers.append(nn.LeakyReLU(0.2))
            conv_layers.append(nn.LayerNorm([out_channels, latent_height, latent_width]))  # Add LayerNorm after conv layers
            in_channels = out_channels  # Update in_channels for the next layer

        self.conv_encoder = nn.Sequential(*conv_layers)

        # Dynamically calculate the output shape after conv layers
        self.output_shape = self._get_output_shape(latent_shape)

        # Deconvolutional layers
        deconv_layers = []
        in_channels = self.output_shape[0]  # Start with the encoder output channels (e.g. 128)
        for out_channels, kernel_size, stride, padding in self._create_deconv_arch(conv_arch, latent_channels):
            deconv_layers.append(ResidualBlock(in_channels))  # Residual block keeps channels unchanged
            deconv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))  # Reduce channels
            deconv_layers.append(nn.LeakyReLU(0.2))
            deconv_layers.append(nn.LayerNorm([out_channels, latent_height, latent_width]))  # Add LayerNorm after deconv layers
            in_channels = out_channels  # Update channels for next layer

        self.deconv_decoder = nn.Sequential(*deconv_layers)

        # Extra conv layer for reward and done prediction
        self.reward_done_conv = nn.Conv2d(self.output_shape[0], self.output_shape[0], kernel_size=3, padding=1)

        # Shared part of reward and done predictor
        self.shared_reward_done = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.output_shape[0], 256),
            nn.LeakyReLU(0.2)
        )

        # Reward predictor
        self.reward_predictor = nn.Sequential(
            self.shared_reward_done,
            nn.Linear(256, 1)
        )

        # Done predictor
        self.done_predictor = nn.Sequential(
            self.shared_reward_done,
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def _get_output_shape(self, latent_shape):
        sample_tensor = torch.zeros(1, latent_shape[0] + self.action_dim, latent_shape[1], latent_shape[2]).to(next(self.parameters()).device)
        sample_tensor = self.initial_conv(sample_tensor)
        sample_tensor = self.conv_encoder(sample_tensor)
        return sample_tensor.shape[1:]

    def _create_deconv_arch(self, conv_arch, latent_channels):
        deconv_arch = []
        for i, (out_channels, kernel_size, stride, padding) in enumerate(reversed(conv_arch)):
            if i == len(conv_arch) - 1:
                out_channels = latent_channels
            deconv_arch.append((out_channels, kernel_size, stride, padding))
        return deconv_arch

    def forward(self, latent_state, action):
        batch_size, latent_channels, latent_height, latent_width = latent_state.shape

        # Reshape action to match latent state dimensions
        action_reshaped = action.view(batch_size, self.action_dim, 1, 1).expand(batch_size, self.action_dim, latent_height, latent_width)
        x = torch.cat([latent_state, action_reshaped], dim=1)

        # Process through encoder
        x = self.initial_conv(x)
        x = self.conv_encoder(x)

        # Process through decoder
        z_next_decoded = self.deconv_decoder(x)
        z_next_decoded = F.sigmoid(z_next_decoded)  # Sigmoid activation for output scaling

        # Reward and done prediction
        x = self.reward_done_conv(x)
        reward_pred = self.reward_predictor(x)
        done_pred = self.done_predictor(x)

        return z_next_decoded, reward_pred, done_pred


class TransitionModelVAE(TransitionModel):
    def __init__(self, latent_shape, action_dim, conv_arch):
        """
        VAE-based Transition Model, inheriting from the deterministic Transition Model.
        """
        super(TransitionModelVAE, self).__init__(latent_shape, action_dim, conv_arch)

        # Overwrite the conv_mean_logvar layer to generate both mean and logvar
        self.conv_mean_logvar = nn.Conv2d(self.output_shape[0], self.output_shape[0] * 2, kernel_size=3, padding=1)

    def forward(self, latent_state, action):
        batch_size, latent_channels, latent_height, latent_width = latent_state.shape

        # Reshape action to match latent state dimensions
        action_reshaped = action.view(batch_size, self.action_dim, 1, 1).expand(batch_size, self.action_dim, latent_height, latent_width)
        x = torch.cat([latent_state, action_reshaped], dim=1)

        # Process through encoder
        x = self.initial_conv(x)
        x = self.conv_encoder(x)

        # Generate mean and logvar
        mean_logvar = self.conv_mean_logvar(x)
        mean, logvar = torch.split(mean_logvar, self.output_shape[0], dim=1)

        # Reparameterization trick (mean + eps * std)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_next = mean + eps * std

        # Process through decoder
        z_next_decoded = self.deconv_decoder(z_next)
        z_next_decoded = F.sigmoid(z_next_decoded)  # Sigmoid activation for output scaling

        # Reward and done prediction
        x = self.reward_done_conv(x)
        reward_pred = self.reward_predictor(x)
        done_pred = self.done_predictor(x)

        return z_next_decoded, mean, logvar, reward_pred, done_pred


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        """
        Vector Quantizer for VQ-VAE
        :param num_embeddings: Number of vectors in the codebook
        :param embedding_dim: Dimensionality of the embeddings
        :param commitment_cost: Weight for the commitment loss
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Embedding table
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, inputs):
        # Flatten inputs except the last dimension
        flat_inputs = inputs.view(-1, self.embedding_dim)

        # Compute L2 distances between input vectors and embeddings
        distances = torch.sum(flat_inputs ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding ** 2, dim=1) - \
                    2 * torch.matmul(flat_inputs, self.embedding.t())

        # Get the closest embedding index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Quantize by replacing with closest embedding
        quantized = F.embedding(encoding_indices, self.embedding).view_as(inputs)

        # Compute losses
        embedding_loss = F.mse_loss(quantized.detach(), inputs)
        commitment_loss = self.commitment_cost * F.mse_loss(inputs.detach(), quantized)
        quantized = inputs + (quantized - inputs).detach()

        return quantized, encoding_indices, embedding_loss + commitment_loss


class TransitionModelVQVAE(TransitionModel):
    def __init__(self, latent_shape, action_dim, conv_arch, num_embeddings=512):
        super(TransitionModelVQVAE, self).__init__(latent_shape, action_dim, conv_arch)
        embedding_dim = latent_shape[0]  # embedding_dim based on the output channels of decoder

        # Vector quantizer
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)

    def forward(self, latent_state, action):
        batch_size, latent_channels, latent_height, latent_width = latent_state.shape

        # Reshape action to match latent state dimensions
        action_reshaped = action.view(batch_size, self.action_dim, 1, 1).expand(batch_size, self.action_dim,
                                                                                latent_height, latent_width)
        x = torch.cat([latent_state, action_reshaped], dim=1)

        # Process through encoder
        x = self.initial_conv(x)
        x = self.conv_encoder(x)

        # Process through decoder
        z_next_decoded = self.deconv_decoder(x)

        # Apply vector quantization on the output of decoder
        z_quantized, _, vq_loss = self.vector_quantizer(z_next_decoded)

        # Apply Sigmoid to limit the output range between [0, 1]
        z_quantized = torch.sigmoid(z_quantized)

        # Reward and done predictions
        reward_pred = self.reward_predictor(x)
        done_pred = self.done_predictor(x)

        return z_quantized, reward_pred, done_pred, vq_loss


class TransitionModelVQVAEVAE(TransitionModelVAE):
    def __init__(self, latent_shape, action_dim, conv_arch, num_embeddings=512):
        super(TransitionModelVQVAEVAE, self).__init__(latent_shape, action_dim, conv_arch)
        embedding_dim = latent_shape[0]  # embedding_dim based on the output channels of decoder

        # Vector quantizer
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)

    def forward(self, latent_state, action):
        batch_size, latent_channels, latent_height, latent_width = latent_state.shape

        # Reshape action to match latent state dimensions
        action_reshaped = action.view(batch_size, self.action_dim, 1, 1).expand(batch_size, self.action_dim,
                                                                                latent_height, latent_width)
        x = torch.cat([latent_state, action_reshaped], dim=1)

        # Process through encoder
        x = self.initial_conv(x)
        x = self.conv_encoder(x)

        # Generate mean and logvar for VAE
        mean_logvar = self.conv_mean_logvar(x)
        mean, logvar = torch.split(mean_logvar, self.output_shape[0], dim=1)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_next = mean + eps * std

        # Process through decoder
        z_next_decoded = self.deconv_decoder(z_next)

        # Apply vector quantization on the output of decoder
        z_quantized, _, vq_loss = self.vector_quantizer(z_next_decoded)

        # Apply Sigmoid to limit the output range between [0, 1]
        z_quantized = torch.sigmoid(z_quantized)

        # Reward and done predictions
        reward_pred = self.reward_predictor(x)
        done_pred = self.done_predictor(x)

        return z_quantized, mean, logvar, reward_pred, done_pred, vq_loss


class SimpleTransitionModel(nn.Module):
    def __init__(self, latent_shape, action_dim, conv_arch):
        """
        A simplified transition model that replaces VAE with a deterministic CNN-based approach.
        :param latent_shape: Shape of the latent space (channels, height, width)
        :param action_dim: Dimensionality of the action space (int)
        :param conv_arch: Convolutional network architecture for the encoder, e.g., [(64, 4, 2, 1), (128, 4, 2, 1)]
        """
        super(SimpleTransitionModel, self).__init__()
        latent_channels, latent_height, latent_width = latent_shape

        self.action_dim = action_dim

        # Initial convolution to map input (latent + action) to expected channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(latent_channels + action_dim, conv_arch[0][0], kernel_size=3, stride=1, padding=1),  # e.g. 19 to 64 channels
            nn.LeakyReLU()
        )

        # Encoder: Convolutional layers (simplified)
        conv_layers = []
        in_channels = conv_arch[0][0]  # Start with output from initial_conv (e.g. 64)
        for out_channels, kernel_size, stride, padding in conv_arch:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))  # Standard Conv layer
            conv_layers.append(nn.LeakyReLU())  # ReLU activation after each Conv
            in_channels = out_channels  # Update the number of channels

        self.conv_encoder = nn.Sequential(*conv_layers)

        # Decoder: Simplified version to reverse the process
        deconv_layers = []
        in_channels = conv_arch[-1][0]  # Start with the encoder's final output channels
        for out_channels, kernel_size, stride, padding in reversed(conv_arch):
            deconv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))  # Conv for decoding
            deconv_layers.append(nn.LeakyReLU())  # ReLU activation
            in_channels = out_channels  # Update the number of channels

        self.deconv_decoder = nn.Sequential(*deconv_layers)

        # Final convolution to ensure the output matches the latent shape
        self.final_conv = nn.Conv2d(in_channels, latent_channels, kernel_size=3, stride=1, padding=1)

        # Reward and Done prediction layers (as in the original model)
        self.reward_done_conv = nn.Conv2d(conv_arch[-1][0], conv_arch[-1][0], kernel_size=3, padding=1)

        # Shared part for reward and done prediction
        self.shared_reward_done = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(conv_arch[-1][0], 256),
            nn.LeakyReLU()
        )

        # Reward predictor
        self.reward_predictor = nn.Sequential(
            self.shared_reward_done,
            nn.Linear(256, 1)
        )

        # Done predictor
        self.done_predictor = nn.Sequential(
            self.shared_reward_done,
            nn.Linear(256, 1),
            nn.Sigmoid()  # Sigmoid for binary classification of "done" state
        )

    def forward(self, latent_state, action):
        """
        Forward pass through the deterministic transition model.
        :param latent_state: (batch_size, latent_channels, latent_height, latent_width)
        :param action: (batch_size, action_dim), one-hot encoded action
        :return: z_next: Next latent state (determined by CNN), reward_pred: Predicted reward, done_pred: Predicted done state
        """
        batch_size, latent_channels, latent_height, latent_width = latent_state.shape

        # Reshape action to match latent state dimensions
        action_reshaped = action.view(batch_size, self.action_dim, 1, 1).expand(batch_size, self.action_dim, latent_height, latent_width)

        # Concatenate action and latent state
        x = torch.cat([latent_state, action_reshaped], dim=1)

        # Process through encoder
        x = self.initial_conv(x)
        x = self.conv_encoder(x)

        # Process through decoder
        z_next_decoded = self.deconv_decoder(x)
        z_next_decoded = self.final_conv(z_next_decoded)  # Final conv to match latent shape
        z_next_decoded = F.sigmoid(z_next_decoded)

        # Reward and Done prediction
        x = self.reward_done_conv(x)
        reward_pred = self.reward_predictor(x)
        done_pred = self.done_predictor(x)

        return z_next_decoded, reward_pred, done_pred


class WorldModel(nn.Module):
    def __init__(
            self,
            latent_shape: Tuple[int, int, int],
            num_homomorphism_channels: int,
            obs_shape: Tuple[int, int, int],
            num_actions: int,
            cnn_net_arch: List[Tuple[int, int, int, int]],
            transition_model_conv_arch: List[Tuple[int, int, int, int]],
            lr: float = 1e-4,
    ):
        super(WorldModel, self).__init__()
        self.latent_shape = latent_shape
        self.num_homomorphism_channels = num_homomorphism_channels
        self.homomorphism_latent_space = (num_homomorphism_channels, latent_shape[1], latent_shape[2])
        self.encoder = Encoder(obs_shape, latent_shape, cnn_net_arch)
        self.decoder = Decoder(latent_shape, obs_shape, cnn_net_arch)
        self.transition_model = TransitionModel(self.homomorphism_latent_space, num_actions, transition_model_conv_arch)
        # self.transition_model = SimpleTransitionModel(self.homomorphism_latent_space, num_actions, transition_model_conv_arch)

        self.num_actions = num_actions

        # Optimizer for all components
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.transition_model.parameters()),
            lr=lr,
        )

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.mbt_loss = FlexibleThresholdedLoss()

    def forward(self, state, action):
        device = next(self.parameters()).device
        state = state.to(device)
        action = action.to(device)

        # Encode the current and next state
        latent_state = self.encoder(state)

        # Make homomorphism state
        homo_latent_state = latent_state[:, 0:self.num_homomorphism_channels, :, :]

        # Predict the next latent state and reward with the transition model
        action = F.one_hot(action, self.num_actions).type(torch.float)
        # predicted_next_homo_latent_state, mean, logvar, predicted_reward, predicted_done \
        #     = self.transition_model(homo_latent_state, action)
        predicted_next_homo_latent_state, predicted_reward, predicted_done \
            = self.transition_model(homo_latent_state, action)

        # Make homomorphism next state
        predicted_next_state = torch.cat(
            [predicted_next_homo_latent_state, latent_state[:, self.num_homomorphism_channels:, :, :]], dim=1)

        # Reconstruct the predicted next state
        predicted_reconstructed_state = self.decoder(latent_state)
        predicted_reconstructed_next_state = self.decoder(predicted_next_state)

        # **Resize the next state** to match the size of the reconstructed state
        resized_predicted_next_state = F.interpolate(predicted_reconstructed_next_state, size=state.shape[2:],
                                                     mode='bilinear',
                                                     align_corners=False)
        resized_next_state = F.interpolate(predicted_reconstructed_state, size=state.shape[2:],
                                           mode='bilinear',
                                           align_corners=False)

        return resized_next_state, resized_predicted_next_state, predicted_reward, predicted_done

    def save_model(self, epoch, loss, save_dir='models', is_best=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }

        latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        print(f"Saved latest model checkpoint at epoch {epoch} with loss {loss:.4f}")

        if is_best:
            best_path = os.path.join(save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model checkpoint at epoch {epoch} with loss {loss:.4f}")

    def load_model(self, save_dir='models', best=False):
        checkpoint_path = os.path.join(save_dir, 'best_checkpoint.pth' if best else 'latest_checkpoint.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Loaded {'best' if best else 'latest'} model checkpoint from epoch {epoch} with loss {loss:.4f}")
            return epoch, loss
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    def train_minibatch(self, state, action, reward, next_state, done):
        device = next(self.parameters()).device
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)

        # Encode the current and next state
        latent_state = self.encoder(state)
        latent_next_state = self.encoder(next_state).detach()

        # Make homomorphism states
        homo_latent_state = latent_state[:, 0:self.num_homomorphism_channels, :, :]
        homo_latent_next_state = latent_next_state[:, 0:self.num_homomorphism_channels, :, :]

        # Predict the next latent state and reward with the transition model
        action = F.one_hot(action, self.num_actions).type(torch.float)
        # predicted_next_homo_latent_state, mean, logvar, predicted_reward, predicted_done \
        #     = self.transition_model(homo_latent_state, action)
        predicted_next_homo_latent_state, predicted_reward, predicted_done \
            = self.transition_model(homo_latent_state, action)

        # Make homomorphism next state
        predicted_latent_next_state = torch.cat(
            [predicted_next_homo_latent_state,
             latent_state[:, self.num_homomorphism_channels:, :, :]],
            dim=1,
        )
        latent_next_state = torch.cat(
            [homo_latent_next_state,
             latent_state[:, self.num_homomorphism_channels:, :, :]],
            dim=1,
        )  # still use the obs layers from the first observation

        # Reconstruct the state and predicted next state
        # reconstructed_state = self.decoder(latent_state)
        reconstructed_next_state = self.decoder(latent_next_state)
        reconstructed_predicted_next_state = self.decoder(predicted_latent_next_state)

        # **Resize the states** to match the size of the reconstructed state
        # resized_state = F.interpolate(state, size=reconstructed_predicted_next_state.shape[2:], mode='bilinear',
        #                                    align_corners=False)
        resized_next_state = F.interpolate(next_state, size=reconstructed_predicted_next_state.shape[2:], mode='bilinear',
                                           align_corners=False)

        # Compute reconstruction loss (MSE) between the reconstructed and resized next state
        reconstruction_loss = self.mbt_loss(reconstructed_predicted_next_state, resized_next_state)
        encoder_reconstruction_loss = self.mbt_loss(reconstructed_next_state, resized_next_state)
        reconstruction_mae_loss = self.mae_loss(reconstructed_predicted_next_state, resized_next_state)


        # --------------------
        # VAE Loss (Reconstruction + KL Divergence)
        # --------------------
        # latent_transition_loss_mse = self.mse_loss(homo_latent_next_state, predicted_next_homo_latent_state)
        # latent_transition_loss_mae = self.mae_loss(homo_latent_next_state, predicted_next_homo_latent_state)
        # latent_transition_loss = latent_transition_loss_mse  # + latent_transition_loss_mae

        # kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())  # * 0.1
        # vae_loss = latent_transition_loss + 0.1 * kl_loss

        # --------------------
        # Reward Prediction Loss
        # --------------------
        reward_loss = self.mbt_loss(predicted_reward.squeeze(), reward)

        # --------------------
        # Done Prediction Loss
        # --------------------
        done_loss = F.binary_cross_entropy(predicted_done.squeeze(), done.float())  # Convert done to float

        # --------------------
        # Total Loss
        # --------------------
        # total_loss = vae_loss + reward_loss + generator_loss + done_loss
        total_loss = reconstruction_loss + reward_loss + done_loss + encoder_reconstruction_loss  # + kl_loss * 0.1
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Return a dictionary with all the loss values
        loss_dict = {
            "reconstruction_loss": reconstruction_loss.detach().cpu().item(),
            "reconstruction_mae_loss": reconstruction_mae_loss.detach().cpu().item(),
            "encoder_reconstruction_loss": encoder_reconstruction_loss.detach().cpu().item(),
            # "kl_loss": kl_loss.detach().cpu().item(),
            "reward_loss": reward_loss.detach().cpu().item(),
            "done_loss": done_loss.detach().cpu().item(),
            "total_loss": total_loss.detach().cpu().item(),
        }

        return loss_dict

    def train_epoch(self, dataloader: DataLoader, log_writer: SummaryWriter, start_num_batches=0,):
        total_samples = len(dataloader) * dataloader.batch_size
        loss_sum = 0.0

        with tqdm(total=total_samples, desc="Training", unit="sample") as pbar:
            for i, batch in enumerate(dataloader):
                obs, actions, next_obs, rewards, dones = batch
                loss_dict = self.train_minibatch(obs, actions, rewards, next_obs, dones)
                total_loss = loss_dict['total_loss']
                running_loss = loss_dict['reconstruction_mae_loss']
                loss_sum += running_loss
                avg_loss = loss_sum / (i + 1)
                pbar.update(len(obs))
                pbar.set_postfix({'total_loss': total_loss, 'reconstruction_mae_loss': running_loss, 'avg_reconstruction_mae_loss': avg_loss})
                for key in loss_dict.keys():
                    log_writer.add_scalar(f'{key}', loss_dict[key], i + start_num_batches)
            else:
                with torch.no_grad():
                    # Logging single combined image for each sample
                    with torch.no_grad():
                        for idx, (ob, action, next_ob, reward, done) in enumerate(
                                zip(obs, actions, next_obs, rewards, dones)):
                            if idx >= 10:
                                break

                            rec_ob, pred_next_ob, pred_reward, pred_done = self.forward(ob.unsqueeze(dim=0),
                                                                                        action.unsqueeze(dim=0))

                            # Convert tensors from GPU to CPU and adjust dimensions for display (HWC format)
                            ob_cpu = ob.cpu().detach().permute(1, 2, 0)  # Current observation
                            rec_ob_cpu = rec_ob.squeeze(0).cpu().detach().permute(1, 2, 0)  # Reconstructed observation
                            pred_next_ob_cpu = pred_next_ob.squeeze(0).cpu().detach().permute(1, 2, 0)  # Predicted next observation
                            next_ob_cpu = next_ob.cpu().detach().permute(1, 2, 0)  # Actual next observation

                            # Get the string label for the action from the dictionary
                            action_str = action_dict.get(action.item(), "Unknown Action")

                            # Create a figure to combine all visualizations and scalar information
                            fig, axs = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns for layout

                            # Plot images: Observation, Reconstructed Observation, Predicted Next Observation, Actual Next Observation
                            axs[0, 0].imshow(ob_cpu)
                            axs[0, 0].set_title('Observation')

                            axs[0, 1].imshow(rec_ob_cpu)  # Draw the reconstructed observation
                            axs[0, 1].set_title('Reconstructed Observation')

                            axs[0, 2].imshow(pred_next_ob_cpu)
                            axs[0, 2].set_title('Predicted Next Observation')

                            axs[0, 3].imshow(next_ob_cpu)  # Draw the actual next observation
                            axs[0, 3].set_title('Actual Next Observation')

                            # Plot scalar values: Action, Predicted Reward/Done, Actual Reward/Done
                            axs[1, 0].text(0.5, 0.5, f'Action: {action_str}', horizontalalignment='center',
                                           verticalalignment='center')
                            axs[1, 0].set_title('Action')
                            axs[1, 0].axis('off')

                            axs[1, 1].text(0.5, 0.5,
                                           f'Predicted Reward: {pred_reward.item():.2f}\nPredicted Done: {pred_done.item():.2f}',
                                           horizontalalignment='center', verticalalignment='center')
                            axs[1, 1].set_title('Predicted Reward & Done')
                            axs[1, 1].axis('off')

                            axs[1, 2].text(0.5, 0.5,
                                           f'Actual Reward: {reward.item():.2f}\nActual Done: {done.item():.2f}',
                                           horizontalalignment='center', verticalalignment='center')
                            axs[1, 2].set_title('Actual Reward & Done')
                            axs[1, 2].axis('off')

                            # Remove the extra axis in the second row, fourth column
                            axs[1, 3].axis('off')

                            # Convert the figure to a PIL Image and then to a Tensor
                            buf = io.BytesIO()
                            plt.tight_layout()
                            plt.savefig(buf, format='png')
                            buf.seek(0)
                            image = Image.open(buf)
                            image_tensor = T.ToTensor()(image)

                            # Log the combined image to TensorBoard
                            log_writer.add_image(f'{i + start_num_batches}-{idx}_combined', image_tensor,
                                                 i + start_num_batches)

                            # Close the figure to free memory
                            plt.close(fig)

        return avg_loss, start_num_batches


class EnsembleTransitionModel(nn.Module):
    def __init__(self, latent_shape, action_dim, conv_arch, num_models=5, epsilon=0.1):
        """
        Ensemble of SimpleTransitionModel to measure uncertainty for exploration.
        :param latent_shape: Shape of the latent space (channels, height, width)
        :param action_dim: Dimensionality of the action space (int)
        :param conv_arch: Convolutional architecture for the transition model
        :param num_models: Number of models in the ensemble
        :param epsilon: Epsilon for epsilon-greedy exploration (for random action selection)
        """
        super(EnsembleTransitionModel, self).__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([SimpleTransitionModel(latent_shape, action_dim, conv_arch) for _ in range(num_models)])
        self.epsilon = epsilon

    def compute_minibatch_loss(self, latent_state, action, next_latent_state, reward, done, loss_fn):
        """
        Train all models in the ensemble on the provided data.
        :param latent_state: Current latent state (batch_size, latent_channels, latent_height, latent_width)
        :param action: Actions taken (batch_size, action_dim)
        :param next_latent_state: True next latent state (batch_size, latent_channels, latent_height, latent_width)
        :param reward: Reward received (batch_size, 1)
        :param done: Done flag indicating episode termination (batch_size, 1)
        :param loss_fn: Loss function to be used for training
        """
        device = next(self.models[0].parameters()).device
        latent_state = latent_state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_latent_state = next_latent_state.to(device)
        done = done.to(device)

        total_loss = torch.tensor(0.0).to(device)
        for model in self.models:
            # Forward pass
            z_next_pred, reward_pred, done_pred = model(latent_state, action)

            # Compute losses: next state prediction loss, reward prediction loss, done prediction loss
            state_loss = loss_fn(z_next_pred, next_latent_state)
            reward_loss = loss_fn(reward_pred, reward)
            done_loss = loss_fn(done_pred, done)

            # Total loss
            total_loss += state_loss + reward_loss + done_loss

        return total_loss / self.num_models

    def compute_uncertainty(self, latent_states, actions):
        """
        Compute uncertainty ranking for a batch of state-action pairs.
        :param latent_states: Current latent states (batch_size, latent_channels, latent_height, latent_width)
        :param actions: Actions taken (batch_size, action_dim)
        :return: Combined uncertainty for next latent state, reward, and done predictions (normalized).
        """
        device = next(self.models[0].parameters()).device
        latent_states = latent_states.to(device)
        actions = actions.to(device)

        state_predictions = []
        reward_predictions = []
        done_predictions = []

        # Gather predictions from all models
        with torch.no_grad():
            for model in self.models:
                z_next_pred, reward_pred, done_pred = model(latent_states, actions)
                state_predictions.append(z_next_pred)
                reward_predictions.append(reward_pred)
                done_predictions.append(done_pred)

        # Stack predictions to compute variance (uncertainty)
        state_predictions = torch.stack(state_predictions,
                                        dim=0)  # Shape: (num_models, batch_size, latent_channels, height, width)
        reward_predictions = torch.stack(reward_predictions, dim=0)  # Shape: (num_models, batch_size, 1)
        done_predictions = torch.stack(done_predictions, dim=0)  # Shape: (num_models, batch_size, 1)

        # Compute variances for next state, reward, and done
        state_uncertainty = state_predictions.var(dim=0).mean(
            dim=[1, 2, 3])  # Batch variance over all latent state dimensions
        reward_uncertainty = reward_predictions.var(dim=0).mean(dim=1)  # Batch variance for reward predictions
        done_uncertainty = done_predictions.var(dim=0).mean(dim=1)  # Batch variance for done predictions

        # Normalize the uncertainties (min-max normalization) for the batch
        uncertainties = torch.stack([state_uncertainty, reward_uncertainty, done_uncertainty], dim=1)
        min_val, _ = uncertainties.min(dim=1, keepdim=True)
        max_val, _ = uncertainties.max(dim=1, keepdim=True)
        normalized_uncertainties = (uncertainties - min_val) / (
                    max_val - min_val + 1e-5)  # Add small value to avoid division by zero

        # Combine the normalized uncertainties
        combined_uncertainty = normalized_uncertainties.sum(dim=1)

        return combined_uncertainty

    def select_action(self, latent_states, action_space, temperature=1.0):
        """
        Select actions for a batch of latent states based on uncertainty-driven exploration using softmax with mapped temperature.
        :param latent_states: Current latent states (batch_size, latent_channels, latent_height, latent_width)
        :param action_space: Available actions (list of one-hot encoded actions)
        :param temperature: The temperature value (0-inf) controlling randomness in action selection
        :return: Selected actions (one-hot encoded for each sample in the batch)
        """
        batch_size = latent_states.size(0)

        # Epsilon-greedy: with probability epsilon, choose random actions for each sample
        if np.random.rand() < self.epsilon:
            random_indices = np.random.choice(len(action_space), batch_size)
            return [action_space[idx] for idx in random_indices]

        # Compute uncertainties for each action in the action space for all latent states
        uncertainties = []
        for action in action_space:
            action_tensor = torch.FloatTensor(action).unsqueeze(0).expand(batch_size,
                                                                          -1)  # Broadcast action across the batch
            uncertainty = self.compute_uncertainty(latent_states, action_tensor)
            uncertainties.append(uncertainty)

        uncertainties = torch.stack(uncertainties, dim=1)  # Shape: (batch_size, num_actions)

        # Map temperature in the range [0, inf] for softmax scaling
        actual_temperature = temperature if temperature > 0 else 1e-5  # Ensure temperature is not zero

        # Apply temperature scaling for softmax
        scaled_uncertainties = uncertainties / actual_temperature
        probabilities = torch.softmax(scaled_uncertainties, dim=1).cpu().numpy()  # Shape: (batch_size, num_actions)

        # Select actions based on the calculated softmax probabilities for each sample in the batch
        selected_action_indices = [np.random.choice(len(action_space), p=prob) for prob in probabilities]

        # Return the selected one-hot encoded actions
        return [action_space[idx] for idx in selected_action_indices]

    def select_action_integers(self, latent_states, num_actions, temperature=1.0):
        """
        Call select_action and return the actions as integer indices for a batch of latent states.
        :param latent_states: Current latent states (batch_size, latent_channels, latent_height, latent_width)
        :param num_actions: Number of possible actions (integer)
        :param temperature: The temperature value (0-inf) controlling randomness in action selection
        :return: List of selected action indices (integer for each sample in the batch)
        """
        # Generate one-hot encoded actions based on num_actions
        action_space = [np.eye(num_actions)[i] for i in range(num_actions)]  # Creates one-hot encoded actions

        # Call select_action to get the selected actions in one-hot encoding
        selected_actions = self.select_action(latent_states, action_space, temperature)

        # Convert each selected one-hot action back to its corresponding integer index
        selected_action_indices = [np.argmax(action) for action in selected_actions]

        return selected_action_indices


class WorldModelAgentDataset(GymDataset):
    def __init__(self, data_size: int, repeat: int = 1, movement_augmentation: int = 0):
        super().__init__(None, data_size, repeat, movement_augmentation)

    def resample(self, env: VecEnv = None, action_selection_func=None, temperature=1.0):
        """Resample the data by interacting with the environment and collecting new data for one epoch."""
        assert env is not None, "I hate to see warnings from the parent, but nah, you need to give an env to sample!"
        num_envs = env.num_envs

        self.data.clear()  # Clear existing data
        obs = env.reset()  # Reset the environment to get the initial observations

        # Collect data for the entire epoch with a progress bar showing the number of actual samples
        total_samples = self.data_size
        augmented = 0
        next_obs, infos = env.reset()
        with tqdm(total=total_samples, desc="Sampling Data", unit="sample") as pbar:
            while len(self.data) < total_samples:
                # Sample actions for each parallel environment
                if action_selection_func is None:
                    actions = np.array([env.action_space.sample() for _ in range(num_envs)])
                else:
                    actions = np.array(action_selection_func(next_obs, env.action_space.n, temperature))
                next_obs, rewards, dones, infos = env.step(actions)

                # Copy `next_obs` to avoid modifying the original
                final_next_obs = np.copy(next_obs)

                # If an environment is done, replace values in `final_next_obs`
                done_indices = np.where(dones)[0]  # Optimisation: only handle environments where `dones` is True
                for env_idx in done_indices:
                    final_next_obs[env_idx] = infos[env_idx]["terminal_observation"]

                # Store the data for each parallel environment
                for env_idx in range(num_envs):
                    if len(self.data) < total_samples:  # Ensure we don't overshoot the target samples
                        if self.movement_augmentation > 0:
                            repeat = 0 if np.allclose(
                                obs[env_idx], final_next_obs[env_idx], rtol=1e-5, atol=1e-8,
                            ) else self.movement_augmentation
                        else:
                            repeat = 0
                        for _ in range(1 + repeat):
                            self.data.append({
                                'obs': torch.tensor(obs[env_idx], dtype=torch.float32),
                                'action': torch.tensor(actions[env_idx], dtype=torch.int64),
                                'next_obs': torch.tensor(final_next_obs[env_idx], dtype=torch.float32),
                                'reward': torch.tensor(rewards[env_idx], dtype=torch.float32),
                                'done': torch.tensor(dones[env_idx], dtype=torch.bool)
                            })
                        augmented += repeat

                        # Update the progress bar with the number of samples collected in this step
                        pbar.update(1 + repeat)

                # Update the observation for the next step
                obs = next_obs

        print(f"{total_samples} samples collected, including {augmented} augmented.")


class WorldModelAgent(WorldModel):
    def __init__(
            self,
            latent_shape: Tuple[int, int, int],
            num_homomorphism_channels: int,
            obs_shape: Tuple[int, int, int],
            num_actions: int,
            cnn_net_arch: List[Tuple[int, int, int, int]],
            transition_model_conv_arch: List[Tuple[int, int, int, int]],
            lr: float = 1e-4,
            samples_per_epoch: int = 4096,
            dataset_repeat_times: int = 10,
            ensemble_num_models: int = 4,
            ensemble_net_arch: List[Tuple[int, int, int, int]] = None,
            ensemble_lr: float = 1e-4,
            ensemble_epsilon: float = 0.1,
            batch_size: int = 32,
    ):
        super(WorldModelAgent, self).__init__(
            latent_shape,
            num_homomorphism_channels,
            obs_shape,
            num_actions,
            cnn_net_arch,
            transition_model_conv_arch,
            lr,
        )
        self.dataset = WorldModelAgentDataset(samples_per_epoch, dataset_repeat_times)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.ensemble_model = EnsembleTransitionModel(
            latent_shape,
            num_actions,
            ensemble_net_arch,
            num_models=ensemble_num_models,
            epsilon=ensemble_epsilon,
        )
        self.ensemble_optimizer = optim.Adam(self.ensemble_model.parameters(), lr=ensemble_lr)
        self.sample_counter = 0
        self.batch_counter = 0

    def save_model(self, epoch, loss, save_dir='models', is_best=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'trained_samples': self.sample_counter,
            'ensemble_optimizer': self.ensemble_optimizer.state_dict(),
        }

        latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        print(f"Saved latest model checkpoint at epoch {epoch} with loss {loss:.4f}")

        if is_best:
            best_path = os.path.join(save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model checkpoint at epoch {epoch} with loss {loss:.4f}")

    def load_model(self, save_dir='models', best=False):
        checkpoint_path = os.path.join(save_dir, 'best_checkpoint.pth' if best else 'latest_checkpoint.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.ensemble_optimizer = checkpoint['ensemble_optimizer']
            self.trained_samples = checkpoint['trained_samples']
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Loaded {'best' if best else 'latest'} model checkpoint from epoch {epoch} with loss {loss:.4f}")
            return epoch, loss
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    def _select_action_integers(self, state: np.ndarray, num_actions: int, temperature=1.0):
        device = next(self.parameters()).device
        state = torch.tensor(state)
        state = state.to(device)
        with torch.no_grad():
            # Encode the current and next state
            latent_state = self.encoder(state)
            # Make homomorphism state
            homo_latent_state = latent_state[:, 0:self.num_homomorphism_channels, :, :]
        return self.ensemble_model.select_action_integers(homo_latent_state, num_actions, temperature)

    def train_epoch(self, dataloader: DataLoader, log_writer: SummaryWriter, start_num_batches=0,):
        device = next(self.parameters()).device

        _avg_loss, _start_num_batches = super(WorldModelAgent, self).train_epoch(
            dataloader, log_writer, start_num_batches
        )

        with tqdm(total=len(self.dataset), desc="Training Ensemble Models", unit="sample") as pbar:
            loss_sum = 0.0
            for i, batch in enumerate(dataloader):
                obs, actions, next_obs, rewards, dones = batch
                state = state.to(device)
                action = action.to(device)
                reward = reward.to(device)
                next_state = next_state.to(device)
                done = done.to(device)
                latent_state, next_latent_state = self.encoder(state), self.encoder(next_state)
                ensemble_loss = self.ensemble_model.compute_minibatch_loss(
                    latent_state, actions, rewards, next_latent_state, dones, self.mse_loss
                )
                self.ensemble_optimizer.zero_grad()
                ensemble_loss.backward()
                self.ensemble_optimizer.step()
                ensemble_loss = ensemble_loss.detach().cpu().item()
                loss_sum += ensemble_loss
                avg_loss = loss_sum / (i + 1)
                pbar.update(len(obs))
                pbar.set_postfix({'ensemble_loss': ensemble_loss, 'avg_ensemble_loss_loss': avg_loss})
                log_writer.add_scalar(f'ensemble_loss', ensemble_loss, i + start_num_batches)
        return _avg_loss, _start_num_batches

    def train_session(self, env: VecEnv, session_dir: str, total_collected_samples: int):
        log_dir = os.path.join(session_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)

        min_loss = float('inf')
        start_epoch = 0

        try:
            _, min_loss = self.load_model(save_dir=os.path.join(session_dir, 'models'), best=True)
            start_epoch, _ = self.load_model(save_dir=os.path.join(session_dir, 'models'))
            start_epoch += 1
        except FileNotFoundError:
            pass

        total_num_epochs = total_collected_samples // self.dataset_size
        for epoch in range(start_epoch, total_num_epochs):
            # sample dataset
            self.dataset.resample(env, self._select_action_integers, temperature=1.0)

            # train on epoch
            loss, _ = self.train_epoch(self.dataloader, log_writer, start_num_batches=self.sample_counter)

            if loss < min_loss:
                min_loss = loss
                self.save_model(epoch, loss, is_best=True, save_dir=os.path.join(session_dir, 'models'))
            self.save_model(epoch, loss, save_dir=os.path.join(session_dir, 'models'))


def train_world_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    session_dir = r"./experiments/full-world_model-door_key"
    dataset_samples = int(1.5e4)
    dataset_repeat_each_epoch = 5
    num_epochs = 100
    batch_size = 32
    lr = 1e-4
    num_parallel = 6

    latent_shape = (8, 24, 24)  # channel, height, width
    num_homomorphism_channels = 5

    movement_augmentation = 6

    encoder_decoder_net_arch = [
        (16, 3, 2, 1),
        (32, 3, 2, 1),
        (64, 3, 2, 1),
    ]

    transition_model_conv_arch = [
        (64, 3, 1, 1),
        (64, 3, 1, 1),
    ]

    configs = []
    for _ in range(num_parallel):
        # for i in range(1, 7):
        config = TaskConfig()
        config.name = f"door_key"
        config.rand_gen_shape = None
        config.txt_file_path = f"./maps/base_env.txt"
        config.custom_mission = "reach the goal"
        config.minimum_display_size = 10
        config.display_mode = "random"
        config.random_rotate = True
        config.random_flip = True
        config.max_steps = 1024
        # config.start_pos = (5, 5)
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

    dataset = GymDataset(venv, data_size=dataset_samples, repeat=dataset_repeat_each_epoch,
                         movement_augmentation=movement_augmentation)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(len(dataloader))

    log_dir = os.path.join(session_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=log_dir)

    world_model = WorldModel(
        latent_shape=latent_shape,
        num_homomorphism_channels=num_homomorphism_channels,
        obs_shape=venv.observation_space.shape,
        num_actions=venv.action_space.n,
        cnn_net_arch=encoder_decoder_net_arch,
        transition_model_conv_arch=transition_model_conv_arch,
        lr=lr,
    ).to(device)

    min_loss = float('inf')
    start_epoch = 0
    try:
        _, min_loss = world_model.load_model(save_dir=os.path.join(session_dir, 'models'), best=True)
        start_epoch, _ = world_model.load_model(save_dir=os.path.join(session_dir, 'models'))
        start_epoch += 1
    except FileNotFoundError:
        pass

    for epoch in range(start_epoch, num_epochs):
        print(f"Start epoch {epoch + 1} / {num_epochs}:")
        print("Resampling dataset...")
        dataset.resample()
        print("Starting training...")
        loss, _ = world_model.train_epoch(
            dataloader,
            log_writer,
            start_num_batches=epoch * len(dataloader),
        )
        if loss < min_loss:
            min_loss = loss
            world_model.save_model(epoch, loss, is_best=True, save_dir=os.path.join(session_dir, 'models'))
        world_model.save_model(epoch, loss, save_dir=os.path.join(session_dir, 'models'))

if __name__ == '__main__':
    pass
