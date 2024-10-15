import os
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
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from customize_minigrid.wrappers import FullyObsImageWrapper
from gymnasium_dataset import GymDataset
from task_config import TaskConfig, make_env


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


class WorldModel(nn.Module):
    def __init__(
            self,
            latent_shape: Tuple[int, int, int],
            num_homomorphism_channels: int,
            obs_shape: Tuple[int, int, int],
            num_actions: int,
            cnn_net_arch: List[Tuple[int, int, int, int]],
            transition_model_conv_arch: List[Tuple[int, int, int, int]],
            disc_conv_arch: List[Tuple[int, int, int, int]],
            lr: float = 1e-4,
            discriminator_lr: float = 1e-4,
    ):
        super(WorldModel, self).__init__()
        self.latent_shape = latent_shape
        self.num_homomorphism_channels = num_homomorphism_channels
        self.homomorphism_latent_space = (num_homomorphism_channels, latent_shape[1], latent_shape[2])
        self.encoder = Encoder(obs_shape, latent_shape, cnn_net_arch)
        self.decoder = Decoder(latent_shape, obs_shape, cnn_net_arch)
        self.transition_model = TransitionModelVAE(self.homomorphism_latent_space, num_actions, transition_model_conv_arch)
        self.discriminator = Discriminator(obs_shape, disc_conv_arch)

        self.num_actions = num_actions

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

    def save_model(self, epoch, loss, save_dir='models', is_best=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
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
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
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
        latent_next_state = self.encoder(next_state)

        # Make homomorphism state
        homo_latent_state = latent_state[:, 0:self.num_homomorphism_channels, :, :]

        # Predict the next latent state and reward with the transition model
        action = F.one_hot(action, self.num_actions).type(torch.float)
        predicted_next_homo_latent_state, mean, logvar, predicted_reward = self.transition_model(homo_latent_state, action)

        # Make homomorphism next state
        predicted_next_state = torch.cat([predicted_next_homo_latent_state, latent_next_state[:, self.num_homomorphism_channels:, :, :]], dim=1)

        # Reconstruct the predicted next state
        reconstructed_state = self.decoder(predicted_next_state)

        # **Resize the next state** to match the size of the reconstructed state
        resized_next_state = F.interpolate(next_state, size=reconstructed_state.shape[2:], mode='bilinear',
                                           align_corners=False)

        # --------------------
        # Discriminator Training
        # --------------------
        real_labels = torch.ones(state.size(0), 1).to(device)
        fake_labels = torch.zeros(state.size(0), 1).to(device)

        # Train discriminator on real and fake images
        real_outputs = self.discriminator(resized_next_state)
        fake_outputs = self.discriminator(reconstructed_state.detach())

        d_real_loss = self.adversarial_loss(real_outputs, real_labels)
        d_fake_loss = self.adversarial_loss(fake_outputs, fake_labels)
        discriminator_loss = (d_real_loss + d_fake_loss) / 2

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # --------------------
        # Generator (Decoder) Loss
        # --------------------
        # Try to fool the discriminator with the generated image
        g_fake_outputs = self.discriminator(reconstructed_state)
        generator_loss = self.adversarial_loss(g_fake_outputs, real_labels)

        # --------------------
        # VAE Loss (Reconstruction + KL Divergence)
        # --------------------
        recon_loss = self.mse_loss(reconstructed_state, resized_next_state)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        vae_loss = recon_loss + kl_loss

        # --------------------
        # Reward Prediction Loss
        # --------------------
        reward_loss = self.mse_loss(predicted_reward.squeeze(), reward)

        # --------------------
        # Total Loss (except Discriminator)
        # --------------------
        total_loss = vae_loss + reward_loss + generator_loss

        # Optimize the components except for the discriminator
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Return a dictionary with all the loss values
        loss_dict = {
            "discriminator_loss": discriminator_loss.detach().cpu().item(),
            "generator_loss": generator_loss.detach().cpu().item(),
            "vae_loss": vae_loss.detach().cpu().item(),
            "reconstruction_loss": recon_loss.detach().cpu().item(),
            "kl_loss": kl_loss.detach().cpu().item(),
            "reward_loss": reward_loss.detach().cpu().item(),
            "total_loss": total_loss.detach().cpu().item(),
        }

        return loss_dict

    def train_epoch(self, dataloader: DataLoader, log_writer: SummaryWriter, start_num_samples=0):
        total_samples = len(dataloader) * dataloader.batch_size
        total_loss = 0.0
        with tqdm(total=total_samples, desc="Training", unit="sample") as pbar:
            for i, batch in enumerate(dataloader):
                obs, actions, next_obs, rewards, dones = batch
                loss_dict = self.train_minibatch(obs, actions, rewards, next_obs, dones)
                running_loss = loss_dict['total_loss']
                total_loss += running_loss
                avg_loss = total_loss / (i + 1)
                pbar.update(len(obs))
                pbar.set_postfix({'loss': running_loss, 'avg_loss': avg_loss})
                for key in loss_dict.keys():
                    log_writer.add_scalar(f'{key}', loss_dict[key], i + start_num_samples)

        return avg_loss, start_num_samples


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    session_dir = r"./experiments/world_model-door_key-7"
    dataset_samples = int(1e2)
    dataset_repeat_each_epoch = 10
    num_epochs = 20
    batch_size = 32
    lr = 1e-4
    discriminator_lr=5e-5

    latent_shape = (16, 16, 16)  # channel, height, width
    encoder_decoder_net_arch = [
        (64, 3, 2, 1),
        (128, 3, 2, 1),
        (256, 3, 2, 1),
    ]

    disc_conv_arch = [
        (64, 3, 2, 1),
        (128, 3, 2, 1),
        (256, 3, 2, 1),
    ]

    transition_model_conv_arch = [
        (64, 4, 2, 1),
        (128, 4, 2, 1),
        (256, 4, 2, 1),
    ]

    configs = []
    for i in range(1, 7):
        config = TaskConfig()
        config.name = f"7-{i}"
        config.rand_gen_shape = None
        config.txt_file_path = f"./maps/7-{i}.txt"
        config.custom_mission = "reach the goal"
        config.minimum_display_size = 7
        config.display_mode = "random"
        config.random_rotate = True
        config.random_flip = True
        config.max_steps = 1024
        config.start_pos = (5, 5)
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

    dataset = GymDataset(venv, data_size=dataset_samples, repeat=dataset_repeat_each_epoch)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(len(dataloader))

    log_dir = os.path.join(session_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=log_dir)

    world_model = WorldModel(
        latent_shape=latent_shape,
        num_homomorphism_channels=8,
        obs_shape=venv.observation_space.shape,
        num_actions=venv.action_space.n,
        cnn_net_arch=encoder_decoder_net_arch,
        transition_model_conv_arch=transition_model_conv_arch,
        disc_conv_arch=disc_conv_arch,
        lr=lr,
        discriminator_lr=discriminator_lr,
    )

    for epoch in range(num_epochs):
        print(f"Start epoch {epoch + 1} / {num_epochs}:")
        print("Resampling dataset...")
        dataset.resample()
        print("Starting training...")
        min_loss = float('inf')
        loss, _ = world_model.train_epoch(
            dataloader,
            log_writer,
            start_num_samples=epoch * dataset_samples * dataset_repeat_each_epoch,
        )
        if loss < min_loss:
            min_loss = loss
            world_model.save_model(epoch, loss, is_best=True)
        world_model.save_model(epoch, loss)
