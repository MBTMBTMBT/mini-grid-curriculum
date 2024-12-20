import io
import os
import random
from typing import Tuple, List, Optional, Any

import PIL
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from vector_quantize_pytorch import ResidualVQ

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


def z_shape_activation(x, lower_bound=0.25, upper_bound=0.75, steepness=1.0):
    """
    Z-shape activation function with three regions:
    - Low region: output close to 0
    - Middle region: output close to y = x
    - High region: output close to 1
    Args:
        x (torch.Tensor): Input tensor.
        lower_bound (float): Start of the linear middle region.
        upper_bound (float): End of the linear middle region.
        steepness (float): Controls the smoothness of transitions.
    Returns:
        torch.Tensor: Output tensor in the range [0, 1] with a Z-shape.
    """
    # Transition functions to smooth the edges of each region
    lower_transition = torch.sigmoid((x - lower_bound) * steepness)
    upper_transition = torch.sigmoid((upper_bound - x) * steepness)

    # Middle region approximates y = x
    linear_region = x

    # Combine regions for Z shape
    y = (1 - lower_transition) * 0 + lower_transition * upper_transition * linear_region + (1 - upper_transition) * 1
    y /= torch.max(y)

    return y


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
        self.target_input_size = target_input_size

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
        return F.tanh(self.encoder(x))


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
        z_next_decoded = F.tanh(z_next_decoded)  # activation for output scaling

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
        z_next_decoded = F.tanh(z_next_decoded)  # activation for output scaling

        # Reward and done prediction
        x = self.reward_done_conv(x)
        reward_pred = self.reward_predictor(x)
        done_pred = self.done_predictor(x)

        return z_next_decoded, mean, logvar, reward_pred, done_pred


class TransitionModelVQVAE(TransitionModel):
    def __init__(
            self,
            latent_shape,
            action_dim,
            conv_arch,
            num_embeddings=128,
            num_quantizers=1,
            commitment_cost=0.25,
            range_target_mean=0.0,
            range_target_std=1.0,
            range_loss_weight=0.1,
            uniformity_weight=0.1,
            uniformity_batch_size=64,
    ):
        super(TransitionModelVQVAE, self).__init__(latent_shape, action_dim, conv_arch)
        # embedding_dim = latent_shape[0]  # embedding_dim based on the output channels of decoder
        self.latent_shape = latent_shape
        self.latent_dim = latent_shape[0] * latent_shape[1] * latent_shape[2]

        # Vector quantizer
        self.vector_quantizer = ResidualVQ(
            dim=self.latent_dim,
            codebook_size=num_embeddings,
            codebook_dim=self.latent_dim,
            num_quantizers=num_quantizers,
            shared_codebook=False,
            quantize_dropout=False,
            accept_image_fmap=False,
            # implicit_neural_codebook=True,
            # mlp_kwargs={'depth': 3, 'l2norm_output': True},
            commitment_weight=commitment_cost,
            kmeans_init=True,
            rotation_trick=True,
            # orthogonal_reg_weight=10,
            # orthogonal_reg_max_codes=64,
            # orthogonal_reg_active_codes_only=False,
        )

    def forward(self, latent_state, action):
        batch_size, latent_channels, latent_height, latent_width = latent_state.shape

        # Quantization at the beginning (without grad)
        # with torch.no_grad():
        #     latent_state_flattened = latent_state.view(batch_size, -1)  # form [batch, channels * height * width]
        #     latent_state_quantized, _, _ = self.vector_quantizer(latent_state_flattened)
        #     latent_state_quantized = latent_state_quantized.view(batch_size, latent_channels, latent_height,
        #                                                          latent_width)

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
        z_next_decoded_flattened = z_next_decoded.view(batch_size, -1)
        z_quantized, _, vq_loss = self.vector_quantizer(z_next_decoded_flattened)
        z_quantized = z_quantized.view(batch_size, latent_channels, latent_height, latent_width)

        # # Apply Sigmoid to limit the output range between [0, 1]
        # z_quantized = torch.sigmoid(z_quantized)

        # Reward and done predictions
        reward_pred = self.reward_predictor(x)
        done_pred = self.done_predictor(x)

        vq_loss = vq_loss.sum()

        return z_quantized, reward_pred, done_pred, vq_loss

    def idx_forward(self, idx, action):
        with torch.no_grad():
            latent_state = self.vector_quantizer.get_output_from_indices(idx)
            batch_size = latent_state.shape[0]
            latent_channels, latent_height, latent_width = self.latent_shape
            latent_state = latent_state.view(batch_size, latent_channels, latent_height, latent_width)

            # Reshape action to match latent state dimensions
            action_reshaped = action.view(batch_size, self.action_dim, 1, 1).expand(batch_size, self.action_dim,
                                                                                    latent_height, latent_width)
            x = torch.cat([latent_state.view((batch_size, *self.latent_shape)), action_reshaped], dim=1)

            # Process through encoder
            x = self.initial_conv(x)
            x = self.conv_encoder(x)

            # Process through decoder
            z_next_decoded = self.deconv_decoder(x)

            # Apply vector quantization on the output of decoder
            z_next_decoded_flattened = z_next_decoded.view(batch_size, -1)
            z_quantized, pred_idx, vq_loss = self.vector_quantizer(z_next_decoded_flattened)

            # # Apply Sigmoid to limit the output range between [0, 1]
            # z_quantized = torch.sigmoid(z_quantized)

            # Reward and done predictions
            reward_pred = self.reward_predictor(x)
            done_pred = self.done_predictor(x)

            vq_loss = vq_loss.sum()

        return pred_idx, reward_pred, done_pred, vq_loss

    def get_state_action_graph(self, batch_size=16):
        ideces = [i for i in range(self.vector_quantizer.codebook_size)]
        # Initialise an empty list to store the batches
        batched_ideces = []
        # Iterate through the list with step of batch_size
        for i in range(0, len(ideces), batch_size):
            # Slice the list to create a batch
            batch = ideces[i:i + batch_size]
            # Convert batch to a torch tensor and reshape to (-1, 1) for shape (x, 1)
            batch_tensor = torch.tensor(batch, dtype=torch.int).reshape(-1, 1).to(next(self.parameters()).device)
            batched_ideces.append(batch_tensor)

        graph_dict = {}
        for i in ideces:
            graph_dict[i] = {}
        for idxs in batched_ideces:
            for act in range(self.action_dim):
                acts = (torch.ones_like(idxs) * act).squeeze().long()
                acts_ = F.one_hot(acts, self.action_dim).type(torch.float)
                pred_idx, reward_pred, done_pred, _ = self.idx_forward(idxs, acts_)
                for i, a, ni, r, d in zip(idxs, acts, pred_idx, reward_pred, done_pred):
                    graph_dict[i.cpu().item()][a.cpu().item()] = (ni.cpu().item(), r.cpu().item(), d.cpu().item() > 0.5)

        return graph_dict


def plot_graph(graph_dict, writer: SummaryWriter, draw_self_loops=False, highlight_done=True, show_edge_labels=False, tag="graph"):
    """
    Plot the graph and save it to TensorBoard.

    Parameters:
        graph_dict (dict): Definition of nodes and edges, format is {start_node: {action: (next_node, reward, done)}}.
        writer (SummaryWriter): TensorBoard writer to save the plot.
        draw_self_loops (bool): Whether to plot self-loops (edges where start_node == next_node).
        highlight_done (bool): Whether to highlight edges with done status as True.
        show_edge_labels (bool): Whether to display labels on edges.
        tag (str): Name used as tag in TensorBoard, default is 'graph'.
    """

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges based on the graph dictionary
    for start_node, actions in graph_dict.items():
        for action, (next_node, reward, done) in actions.items():
            # Skip self-loops if draw_self_loops is False
            if not draw_self_loops and start_node == next_node:
                continue

            # Create edge label if show_edge_labels is True
            edge_label = f"action: {action}, reward: {reward: .2f}, done: {done}" if show_edge_labels else None
            # Set edge colour based on highlight_done parameter
            edge_colour = 'red' if highlight_done and done else 'grey'

            # Add the edge to the graph with additional colour attribute
            G.add_edge(start_node, next_node, label=edge_label, colour=edge_colour)

    # Define node layout
    pos = nx.spring_layout(G)  # Layout for nodes

    # Draw nodes and edges with specified attributes
    fig, ax = plt.subplots(figsize=(15, 15))  # Use subplots for more control
    edge_colours = [G[u][v]['colour'] for u, v in G.edges()]  # Use defined edge colours
    nx.draw(G, pos, with_labels=True, node_size=700, font_size=10, font_weight='bold', node_color='skyblue',
            edge_color=edge_colours, ax=ax)

    # Optionally add edge labels
    if show_edge_labels:
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5, ax=ax)

    # Render the plot to an in-memory file
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")  # Use bbox_inches="tight" to ensure no clipping
    buf.seek(0)
    img = PIL.Image.open(buf)

    # Convert the image to a NumPy array and then to a tensor
    img_array = np.array(img)  # Convert PIL image to NumPy array
    img_tensor = torch.tensor(img_array).permute(2, 0, 1) / 255.0  # Scale and permute to match [C, H, W]

    # Add image to TensorBoard with the specified tag
    writer.add_image(tag, img_tensor, global_step=0)  # Use tag as the name in TensorBoard

    # Clean up resources
    buf.close()
    plt.close(fig)  # Only close specific figure


class _TransitionModelVQVAEVAE(TransitionModelVAE):
    def __init__(self, latent_shape, action_dim, conv_arch, num_embeddings=512):
        super(_TransitionModelVQVAEVAE, self).__init__(latent_shape, action_dim, conv_arch)
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
            nn.Conv2d(latent_channels + action_dim, conv_arch[0][0], kernel_size=3, stride=1, padding=1),
            # e.g., 19 to 64 channels
            nn.LeakyReLU()
        )

        # Encoder: Convolutional layers (simplified)
        conv_layers = []
        in_channels = conv_arch[0][0]  # Start with output from initial_conv
        for out_channels, kernel_size, stride, padding in conv_arch:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))  # Conv layer
            conv_layers.append(nn.LeakyReLU())  # ReLU activation
            in_channels = out_channels  # Update in_channels

        self.conv_encoder = nn.Sequential(*conv_layers)

        # Decoder: Reversed architecture for the deconvolutional layers
        deconv_layers = []
        in_channels = conv_arch[-1][0]  # Start with the encoder's final output channels
        for out_channels, kernel_size, stride, padding in reversed(conv_arch):
            # Adjust padding for ConvTranspose2d to ensure matching output size
            deconv_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                                    output_padding=(stride - 1)))
            deconv_layers.append(nn.LeakyReLU())
            in_channels = out_channels

        self.deconv_decoder = nn.Sequential(*deconv_layers)

        # Final convolution to match the latent space dimensions
        self.final_conv = nn.Conv2d(in_channels, latent_channels, kernel_size=3, stride=1, padding=1)

        # Reward and Done prediction layers
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
        action_reshaped = action.view(batch_size, self.action_dim, 1, 1).expand(batch_size, self.action_dim,
                                                                                latent_height, latent_width)

        # Concatenate action and latent state
        x = torch.cat([latent_state, action_reshaped], dim=1)

        # Process through encoder
        x = self.initial_conv(x)
        x = self.conv_encoder(x)

        # Process through decoder
        z_next_decoded = self.deconv_decoder(x)
        z_next_decoded = self.final_conv(z_next_decoded)  # Final conv to match latent shape
        z_next_decoded = torch.sigmoid(z_next_decoded)  # Sigmoid for output range [0, 1]

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
            num_embeddings=128,
            commitment_cost=2.0,
            range_target_mean=0.0,
            range_target_std=1.0,
            range_loss_weight=0.1,
            uniformity_weight=0.1,
            uniformity_batch_size=64,
    ):
        super(WorldModel, self).__init__()
        self.latent_shape = latent_shape
        self.num_homomorphism_channels = num_homomorphism_channels
        self.homomorphism_latent_space = (num_homomorphism_channels, latent_shape[1], latent_shape[2])
        self.encoder = Encoder(obs_shape, latent_shape, cnn_net_arch)
        self.decoder = Decoder(latent_shape, obs_shape, cnn_net_arch)
        self.transition_model = TransitionModelVQVAE(
            self.homomorphism_latent_space,
            num_actions,
            transition_model_conv_arch,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            range_target_mean=range_target_mean,
            range_target_std=range_target_std,
            range_loss_weight=range_loss_weight,
            uniformity_weight=uniformity_weight,
            uniformity_batch_size=uniformity_batch_size,
        )
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
        predicted_next_homo_latent_state, predicted_reward, predicted_done, _ \
            = self.transition_model(homo_latent_state, action)

        # Make homomorphism next state
        predicted_next_state = torch.cat(
            [predicted_next_homo_latent_state, latent_state[:, self.num_homomorphism_channels:, :, :]], dim=1)

        # Reconstruct the predicted next state
        predicted_reconstructed_state = self.decoder(latent_state)
        predicted_reconstructed_next_state = self.decoder(predicted_next_state)

        # predicted_reconstructed_state = z_shape_activation(predicted_reconstructed_state)
        # predicted_reconstructed_next_state = z_shape_activation(predicted_reconstructed_next_state)

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
        predicted_next_homo_latent_state, predicted_reward, predicted_done, vq_loss \
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
        total_loss = reconstruction_loss + reward_loss + done_loss + encoder_reconstruction_loss + vq_loss  # + kl_loss * 0.1
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Return a dictionary with all the loss values
        loss_dict = {
            "reconstruction_loss": reconstruction_loss.detach().cpu().item(),
            "reconstruction_mae_loss": reconstruction_mae_loss.detach().cpu().item(),
            "encoder_reconstruction_loss": encoder_reconstruction_loss.detach().cpu().item(),
            "vq_loss": vq_loss.detach().cpu().item(),
            # "kl_loss": kl_loss.detach().cpu().item(),
            "reward_loss": reward_loss.detach().cpu().item(),
            "done_loss": done_loss.detach().cpu().item(),
            "total_loss": total_loss.detach().cpu().item(),
        }

        # for k in vq_loss_dict.keys():
        #     loss_dict[k] = vq_loss_dict[k]

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
        self.num_actions = action_dim

        self.total_state_uncertainty = 0.0
        self.total_reward_uncertainty = 0.0
        self.total_done_uncertainty = 0.0
        self.uncertainty_computation_count = 0
        self.count_actions = {}
        for i in range(self.num_actions):
            self.count_actions[i] = 0

    def reset_uncertainty(self):
        self.total_state_uncertainty = 0.0
        self.total_reward_uncertainty = 0.0
        self.total_done_uncertainty = 0.0
        self.uncertainty_computation_count = 0
        self.count_actions = {}
        for i in range(self.num_actions):
            self.count_actions[i] = 0

    def compute_minibatch_loss(self, latent_state, action, next_latent_state, reward, done, loss_fn):
        """
        Train a randomly selected subset of models in the ensemble on the provided data.
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
        done = done.float().to(device)

        # One-hot encode the action
        action = F.one_hot(action, self.num_actions).type(torch.float)

        # Convert self.models to a list for random sampling
        models_list = list(self.models)
        num_models_to_use = random.randint(1, len(models_list))  # Randomly choose 1 to total models
        selected_models = random.sample(models_list, num_models_to_use)

        total_loss = torch.tensor(0.0, device=device)
        for model in selected_models:
            # Forward pass through the selected model
            z_next_pred, reward_pred, done_pred = model(latent_state, action)

            # Compute losses: next state prediction loss, reward prediction loss, done prediction loss
            state_loss = loss_fn(z_next_pred, next_latent_state)
            reward_loss = loss_fn(reward_pred.squeeze(), reward)
            done_loss = F.binary_cross_entropy(done_pred.squeeze(), done)

            # Accumulate loss
            total_loss += state_loss + reward_loss + done_loss

        # Average the loss over the number of selected models
        return total_loss / num_models_to_use

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
        state_uncertainty = state_predictions.std(dim=0).mean(
            dim=[1, 2, 3])  # Batch variance over all latent state dimensions
        reward_uncertainty = reward_predictions.std(dim=0).mean(dim=1)  # Batch variance for reward predictions
        done_uncertainty = done_predictions.std(dim=0).mean(dim=1)  # Batch variance for done predictions

        self.total_state_uncertainty += state_uncertainty.sum().cpu().item()
        self.total_reward_uncertainty += reward_uncertainty.sum().cpu().item()
        self.total_done_uncertainty += done_uncertainty.sum().cpu().item()
        self.uncertainty_computation_count += 1

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

        for action in selected_action_indices:
            self.count_actions[action] += 1

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
        next_obs = env.reset()
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
            movement_augmentation: int = 0,
            ensemble_num_models: int = 4,
            ensemble_net_arch: List[Tuple[int, int, int, int]] = None,
            ensemble_lr: float = 1e-4,
            dataset_repeat_times_ensemble: int = 5,
            ensemble_epsilon: float = 0.1,
            batch_size: int = 32,
            num_embeddings=128,
            commitment_cost=0.25,
            range_target_mean=0.0,
            range_target_std=1.0,
            range_loss_weight=0.1,
            uniformity_weight=0.1,
            uniformity_batch_size=64,
    ):
        if ensemble_net_arch is None:
            ensemble_net_arch = [
                (64, 3, 2, 1),
                (64, 3, 2, 1),
            ]
        super(WorldModelAgent, self).__init__(
            latent_shape,
            num_homomorphism_channels,
            obs_shape,
            num_actions,
            cnn_net_arch,
            transition_model_conv_arch,
            lr,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            range_target_mean=range_target_mean,
            range_target_std=range_target_std,
            range_loss_weight=range_loss_weight,
            uniformity_weight=uniformity_weight,
            uniformity_batch_size=uniformity_batch_size,
        )
        self.dataset = WorldModelAgentDataset(samples_per_epoch, dataset_repeat_times, movement_augmentation=movement_augmentation)
        self.dataset_ensemble = WorldModelAgentDataset(samples_per_epoch, dataset_repeat_times_ensemble, movement_augmentation=movement_augmentation)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.dataloader_ensemble = DataLoader(self.dataset_ensemble, batch_size=batch_size, shuffle=True)
        self.ensemble_model = EnsembleTransitionModel(
            #(num_homomorphism_channels, latent_shape[1], latent_shape[2]),
            (3, self.encoder.target_input_size, self.encoder.target_input_size),
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
            self.ensemble_optimizer.load_state_dict(checkpoint['ensemble_optimizer'])
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
        # with torch.no_grad():
            # Encode the current and next state
            # latent_state = self.encoder(state)
            # Make homomorphism state
            # homo_latent_state = latent_state[:, 0:self.num_homomorphism_channels, :, :]
        return self.ensemble_model.select_action_integers(state, num_actions, temperature)

    def train_epoch(self, dataloader: DataLoader, dataloader_ensemble: DataLoader, log_writer: SummaryWriter, start_num_batches=0,):
        device = next(self.parameters()).device

        _avg_loss, _start_num_batches = super(WorldModelAgent, self).train_epoch(
            dataloader, log_writer, start_num_batches
        )

        with tqdm(total=len(self.dataset_ensemble), desc="Training Ensemble Models", unit="sample") as pbar:
            loss_sum = 0.0
            for i, batch in enumerate(dataloader_ensemble):
                obs, actions, next_obs, rewards, dones = batch
                state = obs.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_state = next_obs.to(device)
                dones = dones.to(device)
                # latent_state, next_latent_state = self.encoder(state), self.encoder(next_state)
                # Make homomorphism state
                # homo_latent_state = latent_state[:, 0:self.num_homomorphism_channels, :, :]
                # next_homo_latent_state = next_latent_state[:, 0:self.num_homomorphism_channels, :, :]
                ensemble_loss = self.ensemble_model.compute_minibatch_loss(
                    state, actions, next_state, rewards, dones, self.mbt_loss
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

    def train_session(self, env: VecEnv, session_dir: str, total_collected_samples: int, temperature: float):
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

        total_num_epochs = total_collected_samples // self.dataset.data_size
        for epoch in range(start_epoch, total_num_epochs):
            print(f"Start epoch {epoch + 1} / {total_num_epochs}:")

            print(
                f"Resampling dataset samples: {self.sample_counter}+{self.dataset.data_size} / {total_collected_samples}..."
            )
            # sample dataset
            self.dataset.resample(env, self._select_action_integers, temperature=temperature)
            print("Action selected:", self.ensemble_model.count_actions)
            self.dataset_ensemble.data = self.dataset.data
            self.sample_counter += self.dataset.data_size
            avg_state_uncertainty = self.ensemble_model.total_state_uncertainty / self.ensemble_model.uncertainty_computation_count
            avg_reward_uncertainty = self.ensemble_model.total_reward_uncertainty / self.ensemble_model.uncertainty_computation_count
            avg_done_uncertainty = self.ensemble_model.total_done_uncertainty / self.ensemble_model.uncertainty_computation_count
            self.ensemble_model.reset_uncertainty()
            log_writer.add_scalar(f'state uncertainty', avg_state_uncertainty, epoch)
            log_writer.add_scalar(f'reward uncertainty',avg_reward_uncertainty, epoch)
            log_writer.add_scalar(f'done uncertainty', avg_done_uncertainty,epoch)
            print(f"Average state uncertainty: {avg_state_uncertainty:.6f}, reward uncertainty: {avg_reward_uncertainty:.6f}, done uncertainty: {avg_done_uncertainty:.6f}")

            print("Starting training...")
            # train on epoch
            loss, _ = self.train_epoch(self.dataloader, self.dataloader_ensemble, log_writer, start_num_batches=epoch * len(self.dataloader))

            graph_dict = self.transition_model.get_state_action_graph()
            print("State-Action graph:")
            print(graph_dict)
            plot_graph(
                graph_dict,
                writer=log_writer,
                draw_self_loops=False,
                highlight_done=True,
                show_edge_labels=True,
                tag=f'{epoch}-graph',
            )

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
        config.max_steps = 4096
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


def train_world_model_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    session_dir = r"./experiments/discrete-world_model_agent-empty"
    dataset_samples = 1024
    dataset_repeat_each_epoch = 10
    dataset_repeat_times_ensemble = 1
    total_samples = 4096 * 100
    ensemble_num_models = 7
    batch_size = 32
    lr = 1e-4
    num_parallel = 6
    ensemble_epsilon = 0.9
    num_embeddings = 14
    commitment_cost = 2.0
    range_target_mean = 0.0
    range_target_std = 1.0
    range_loss_weight = 0.1
    uniformity_weight = 0.1
    uniformity_batch_size = 32
    temperature = 0.5

    latent_shape = (8, 24, 24)  # channel, height, width
    num_homomorphism_channels = 8

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

    ensemble_transition_model_conv_arch = [
        (16, 3, 2, 1),
        (32, 3, 2, 1),
        (64, 3, 2, 1),
    ]

    configs = []
    for _ in range(num_parallel):
        # for i in range(1, 7):
        config = TaskConfig()
        config.name = f"door_key"
        config.rand_gen_shape = None
        config.txt_file_path = f"./maps/empty_small.txt"
        config.custom_mission = "reach the goal"
        config.minimum_display_size = 4
        config.display_mode = "random"
        config.random_rotate = False
        config.random_flip = False
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

    world_model = WorldModelAgent(
        latent_shape=latent_shape,
        num_homomorphism_channels=num_homomorphism_channels,
        obs_shape=venv.observation_space.shape,
        num_actions=venv.action_space.n,
        cnn_net_arch=encoder_decoder_net_arch,
        transition_model_conv_arch=transition_model_conv_arch,
        lr=lr,
        samples_per_epoch = dataset_samples,
        dataset_repeat_times = dataset_repeat_each_epoch,
        movement_augmentation = movement_augmentation,
        ensemble_num_models= ensemble_num_models,
        ensemble_net_arch = ensemble_transition_model_conv_arch,
        ensemble_lr=lr,
        dataset_repeat_times_ensemble=dataset_repeat_times_ensemble,
        ensemble_epsilon=ensemble_epsilon,
        batch_size=batch_size,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        range_target_mean=range_target_mean,
        range_target_std=range_target_std,
        range_loss_weight=range_loss_weight,
        uniformity_weight=uniformity_weight,
        uniformity_batch_size=uniformity_batch_size,
    ).to(device)

    world_model.train_session(venv, session_dir, total_samples, temperature)


if __name__ == '__main__':
    train_world_model_agent()
