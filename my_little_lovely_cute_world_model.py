import io
import os
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from PIL import Image
from gymnasium.wrappers import FrameStack, AtariPreprocessing, LazyFrames
from gymnasium.vector import AsyncVectorEnv, VectorEnv
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2.functional import resize
from tqdm import tqdm

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


class TransitionModelVAE(nn.Module):
    def __init__(self, latent_shape, action_dim, conv_arch):
        super(TransitionModelVAE, self).__init__()
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

        # Single convolution layer to generate both mean and logvar
        self.conv_mean_logvar = nn.Conv2d(self.output_shape[0], self.output_shape[0] * 2, kernel_size=3, padding=1)

        # Adaptive pooling if needed
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

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

        # Generate mean and logvar
        mean_logvar = self.conv_mean_logvar(x)
        mean, logvar = torch.split(mean_logvar, self.output_shape[0], dim=1)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_next = mean + eps * std

        # Process through decoder
        z_next_decoded = self.deconv_decoder(z_next)
        z_next_decoded = F.sigmoid(z_next_decoded)

        # Reward and done prediction
        x = self.reward_done_conv(x)
        reward_pred = self.reward_predictor(x)
        done_pred = self.done_predictor(x)

        return z_next_decoded, mean, logvar, reward_pred, done_pred


class WorldModel(nn.Module):
    def __init__(
            self,
            latent_shape: Tuple[int, int, int],
            # num_homomorphism_channels: int,
            obs_shape: Tuple[int, int, int],
            num_actions: int,
            cnn_net_arch: List[Tuple[int, int, int, int]],
            transition_model_conv_arch: List[Tuple[int, int, int, int]],
            ae_lr: float = 1e-4,
            trvae_lr: float = 1e-4,
    ):
        super(WorldModel, self).__init__()
        self.latent_shape = latent_shape
        # self.num_homomorphism_channels = num_homomorphism_channels
        # self.homomorphism_latent_space = (num_homomorphism_channels, latent_shape[1], latent_shape[2])
        self.encoder = Encoder(obs_shape, latent_shape, cnn_net_arch)
        self.decoder = Decoder(latent_shape, obs_shape, cnn_net_arch)
        self.transition_model = TransitionModelVAE(latent_shape, num_actions, transition_model_conv_arch)

        self.num_actions = num_actions

        # Optimizer for all components except the discriminator
        self.ae_optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()),
            lr=ae_lr,
        )

        # Separate optimizer for the discriminator
        self.trvae_optimizer = optim.Adam(
            list(self.transition_model.parameters()),
            lr=trvae_lr
        )

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, state, action):
        device = next(self.parameters()).device
        state = state.to(device)
        action = action.to(device)

        # Encode the current and next state
        latent_state = self.encoder(state)

        # Make homomorphism state
        # homo_latent_state = latent_state[:, 0:self.num_homomorphism_channels, :, :]

        # Predict the next latent state and reward with the transition model
        action = F.one_hot(action, self.num_actions).type(torch.float)
        # predicted_next_homo_latent_state, mean, logvar, predicted_reward, predicted_done \
        #     = self.transition_model(homo_latent_state, action)
        predicted_next_latent_state, mean, logvar, predicted_reward, predicted_done\
            = self.transition_model(latent_state, action)

        # Make homomorphism next state
        # predicted_next_state = torch.cat(
        #     [predicted_next_homo_latent_state, latent_state[:, self.num_homomorphism_channels:, :, :]], dim=1)

        # Reconstruct the predicted next state
        # predicted_reconstructed_state = self.decoder(predicted_next_state)
        predicted_reconstructed_state = self.decoder(predicted_next_latent_state)

        # **Resize the next state** to match the size of the reconstructed state
        resized_predicted_next_state = F.interpolate(predicted_reconstructed_state, size=state.shape[2:], mode='bilinear',
                                           align_corners=False)

        return resized_predicted_next_state, predicted_reward, predicted_done

    def save_model(self, epoch, gen_loss, trans_loss, save_dir='models', is_best=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'ae_optimizer_state_dict': self.ae_optimizer.state_dict(),
            'trvae_optimizer_state_dict': self.trvae_optimizer.state_dict(),
            'gen_loss': gen_loss,
            'trans_loss': trans_loss,
        }

        latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        print(f"Saved latest model checkpoint at epoch {epoch} with loss {gen_loss:.4f}, {trans_loss:.4f}")

        if is_best:
            best_path = os.path.join(save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model checkpoint at epoch {epoch} with loss {gen_loss:.4f}, {trans_loss:.4f}")

    def load_model(self, save_dir='models', best=False):
        checkpoint_path = os.path.join(save_dir, 'best_checkpoint.pth' if best else 'latest_checkpoint.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.ae_optimizer.load_state_dict(checkpoint['ae_optimizer_state_dict'])
            self.trvae_optimizer.load_state_dict(checkpoint['trvae_optimizer_state_dict'])
            epoch = checkpoint['epoch']
            gen_loss = checkpoint['gen_loss']
            trans_loss = checkpoint['trans_loss']
            print(f"Loaded {'best' if best else 'latest'} model checkpoint from epoch {epoch} with loss {gen_loss:.4f}, {trans_loss:.4f}")
            return epoch, gen_loss, trans_loss
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    def train_autoencoder_minibatch(self, state, next_state):
        device = next(self.parameters()).device
        state = state.to(device)
        next_state = next_state.to(device)

        # Encode the current and next state
        latent_state = self.encoder(state)
        latent_next_state = self.encoder(next_state)

        # # make the 'predicted' next latent state by merging the two latent states
        # predicted_latent_next_state = torch.cat(
        #     [latent_next_state[:, 0:self.num_homomorphism_channels, :, :],
        #      latent_state[:, self.num_homomorphism_channels:, :, :]],
        #     dim=1,
        # )
        #
        # # make fake reconstructed current state by merging random noised channels
        # fake_latent_state = torch.cat(
        #     [torch.randn_like(latent_next_state[:, 0:self.num_homomorphism_channels, :, :]),
        #      latent_state[:, self.num_homomorphism_channels:, :, :]],
        #     dim=1,
        # )

        # get reconstructed states
        reconstructed_state = self.decoder(latent_state)
        reconstructed_next_state = self.decoder(latent_next_state)
        # reconstructed_predicted_next_state = self.decoder(predicted_latent_next_state)
        # reconstructed_fake_state = self.decoder(fake_latent_state)
        #
        # get expected 'reconstructed' states
        resized_state = F.interpolate(state, size=reconstructed_state.shape[2:],
                                           mode='bilinear', align_corners=False)
        resized_next_state = F.interpolate(next_state, size=reconstructed_state.shape[2:],
                                           mode='bilinear', align_corners=False)

        reconstruction_loss = (
                # self.mse_loss(reconstructed_state, resized_state) +
                # self.mae_loss(reconstructed_state, resized_state) +
                # self.mse_loss(reconstructed_next_state, resized_next_state) +
                # self.mae_loss(reconstructed_next_state, resized_next_state) +
                self.mse_loss(reconstructed_state, resized_state) +
                self.mae_loss(reconstructed_state, resized_state) +
                self.mse_loss(reconstructed_next_state, resized_next_state) +
                self.mae_loss(reconstructed_next_state, resized_next_state)
        )

        # get hidden observation channels and the loss
        # observation_channel_loss = self.mse_loss(
        #     latent_state[:, self.num_homomorphism_channels:, :, :],  # .detach(),
        #     latent_next_state[:, self.num_homomorphism_channels:, :, :],
        # )

        ae_loss = reconstruction_loss # + observation_channel_loss

        self.ae_optimizer.zero_grad()
        ae_loss.backward()
        self.ae_optimizer.step()

        return {
            # "reconstruction_loss": reconstruction_loss.detach().cpu().item(),
            # "observation_channel_loss": observation_channel_loss.detach().cpu().item(),
            "ae_loss": ae_loss.detach().cpu().item(),
        }

    def train_transition_model_minibatch(self, state, action, reward, next_state, done):
        device = next(self.parameters()).device
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)

        # Encode the current and next state
        with torch.no_grad():
            latent_state = self.encoder(state)
            latent_next_state = self.encoder(next_state)

        # Make homomorphism states
        # homo_latent_state = latent_state[:, 0:self.num_homomorphism_channels, :, :]
        # homo_latent_next_state = latent_next_state[:, 0:self.num_homomorphism_channels, :, :]

        # Predict the next latent state and reward with the transition model
        action = F.one_hot(action, self.num_actions).type(torch.float)
        # predicted_next_homo_latent_state, mean, logvar, predicted_reward, predicted_done \
        #     = self.transition_model(homo_latent_state, action)
        predicted_latent_next_state, mean, logvar, predicted_reward, predicted_done \
            = self.transition_model(latent_state, action)

        latent_transition_loss = self.mse_loss(latent_next_state, predicted_latent_next_state)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        vae_loss = latent_transition_loss + 0.1 * kl_loss
        reward_loss = self.mse_loss(predicted_reward.squeeze(), reward)
        done_loss = F.binary_cross_entropy(predicted_done.squeeze(), done.float())  # Convert done to float

        trvae_loss = vae_loss + reward_loss + done_loss

        # Optimize the components except for the discriminator
        self.trvae_optimizer.zero_grad()
        trvae_loss.backward()
        self.trvae_optimizer.step()

        # Return a dictionary with all the loss values
        loss_dict = {
            "latent_transition_loss": latent_transition_loss.detach().cpu().item(),
            "kl_loss": kl_loss.detach().cpu().item(),
            "vae_loss": vae_loss.detach().cpu().item(),
            "reward_loss": reward_loss.detach().cpu().item(),
            "done_loss": done_loss.detach().cpu().item(),
            "trvae_loss": trvae_loss.detach().cpu().item(),
        }

        return loss_dict

    def train_epoch(
            self,
            dataloader: DataLoader,
            log_writer: SummaryWriter,
            start_num_batches=0,
            train_ae=False,
            train_trvae=False,
    ):
        total_samples = len(dataloader) * dataloader.batch_size
        loss_sum = 0.0

        assert not (train_ae and train_trvae), "Cannot train autoencoder and the transition model at the same time!"
        assert train_ae or train_trvae, "No model is selected to be trained!"

        obs, actions, next_obs, rewards, dones = None, None, None, None, None,

        if train_ae:
            with tqdm(total=total_samples, desc="Training Auto Encoder", unit="sample") as pbar:
                for i, batch in enumerate(dataloader):
                    obs, actions, next_obs, rewards, dones = batch
                    loss_dict = self.train_autoencoder_minibatch(obs, next_obs)
                    running_loss = loss_dict['ae_loss']
                    loss_sum += running_loss
                    avg_loss = loss_sum / (i + 1)
                    pbar.update(len(obs))
                    pbar.set_postfix(
                        {
                            'ae_loss': loss_dict['ae_loss'],
                            # 'reconstruction_loss': loss_dict['reconstruction_loss'],
                            # 'observation_channel_loss': loss_dict['observation_channel_loss'],
                            'avg_reconstruction_loss': avg_loss,
                        }
                    )
                    for key in loss_dict.keys():
                        log_writer.add_scalar(f'{key}', loss_dict[key], i + start_num_batches)

        elif train_trvae:
            with tqdm(total=total_samples, desc="Training Transition Model", unit="sample") as pbar:
                for i, batch in enumerate(dataloader):
                    obs, actions, next_obs, rewards, dones = batch
                    loss_dict = self.train_transition_model_minibatch(obs, actions, rewards, next_obs, dones,)
                    running_loss = loss_dict['trvae_loss']
                    loss_sum += running_loss
                    avg_loss = loss_sum / (i + 1)
                    pbar.update(len(obs))
                    pbar.set_postfix(
                        {
                            'latent_transition_loss': loss_dict['latent_transition_loss'],
                            'trvae_loss': loss_dict['trvae_loss'],
                            'avg_trvae_loss': avg_loss,
                        }
                    )
                    for key in loss_dict.keys():
                        log_writer.add_scalar(f'{key}', loss_dict[key], i + start_num_batches)

        # Logging single combined image for each sample
        with torch.no_grad():
            for idx, (ob, action, next_ob, reward, done) in enumerate(
                    zip(obs, actions, next_obs, rewards, dones)):
                if idx >= 10:
                    break

                pred_next_ob, pred_reward, pred_done = self.forward(ob.unsqueeze(dim=0),
                                                                    action.unsqueeze(dim=0))

                # Convert tensors from GPU to CPU
                ob_cpu = ob.cpu().detach().permute(1, 2, 0)  # Convert to HWC format for image display
                next_ob_cpu = next_ob.cpu().detach().permute(1, 2, 0)
                pred_next_ob_cpu = pred_next_ob.squeeze(0).cpu().detach().permute(1, 2, 0)

                # Get the string label for the action from the simple dictionary
                action_str = action_dict.get(action.item(), "Unknown Action")

                # Create a figure to combine all visualizations and scalar information
                fig, axs = plt.subplots(2, 3, figsize=(12, 8))

                # Plot images: Observation, Predicted Next Observation, Actual Next Observation
                axs[0, 0].imshow(ob_cpu)
                axs[0, 0].set_title('Observation')
                axs[0, 1].imshow(pred_next_ob_cpu)
                axs[0, 1].set_title('Predicted Next Observation')
                axs[0, 2].imshow(next_ob_cpu)
                axs[0, 2].set_title('Actual Next Observation')

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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    session_dir = r"./experiments/world_model-door_key"
    dataset_samples = int(1e4)
    dataset_repeat_each_epoch = 10
    train_ae_epochs = 30
    train_trvae_epochs = 60
    batch_size = 32
    ae_lr = 2.5e-4
    trvae_lr = 1e-4
    num_parallel = 4

    latent_shape = (32, 32, 32)  # channel, height, width
    # num_homomorphism_channels = 16
    movement_augmentation = 3

    encoder_decoder_net_arch = [
        (32, 3, 2, 1),
        # (32, 3, 1, 1),
        (64, 3, 1, 1),
        # (64, 3, 1, 1),
        (128, 3, 1, 1),
        # (128, 3, 1, 1),
    ]

    transition_model_conv_arch = [
        (32, 3, 1, 1),
        (64, 3, 1, 1),
        (128, 3, 1, 1),
    ]

    configs = []
    for _ in range(num_parallel):
        # for i in range(1, 7):
            config = TaskConfig()
            config.name = f"door_key"
            config.rand_gen_shape = None
            config.txt_file_path = f"./maps/door_key.txt"
            config.custom_mission = "reach the goal"
            config.minimum_display_size = 7
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

    dataset = GymDataset(venv, data_size=dataset_samples, repeat=dataset_repeat_each_epoch, movement_augmentation=movement_augmentation)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    log_dir = os.path.join(session_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=log_dir)

    world_model = WorldModel(
        latent_shape=latent_shape,
        # num_homomorphism_channels=num_homomorphism_channels,
        obs_shape=venv.observation_space.shape,
        num_actions=venv.action_space.n,
        cnn_net_arch=encoder_decoder_net_arch,
        transition_model_conv_arch=transition_model_conv_arch,
        ae_lr=ae_lr,
        trvae_lr=trvae_lr,
    ).to(device)

    min_ae_loss = float('inf')
    min_trvae_loss = float('inf')
    start_epoch_ae = 0
    start_epoch_trvae = 0
    try:
        _, min_ae_loss, min_trvae_loss = world_model.load_model(save_dir=os.path.join(session_dir, 'models'), best=True)
        start_epoch, _, _ = world_model.load_model(save_dir=os.path.join(session_dir, 'models'))
        if min_trvae_loss < float('inf'):
            start_epoch_trvae = start_epoch
            start_epoch_trvae += 1
        else:
            start_epoch_ae = start_epoch
            start_epoch_ae += 1
    except FileNotFoundError:
        pass

    for epoch in range(start_epoch_ae, train_ae_epochs):
        print(f"Start epoch {epoch + 1} / {train_ae_epochs} for AutoEncoder training...:")
        print("Resampling dataset...")
        dataset.resample()
        print("Starting training...")
        loss, _ = world_model.train_epoch(
            dataloader,
            log_writer,
            start_num_batches=epoch * len(dataloader),
            train_ae=True,
            train_trvae=False,
        )
        if loss < min_ae_loss:
            min_ae_loss = loss
            world_model.save_model(epoch, min_ae_loss, min_trvae_loss, is_best=True, save_dir=os.path.join(session_dir, 'models'))
        world_model.save_model(epoch, min_ae_loss, min_trvae_loss, save_dir=os.path.join(session_dir, 'models'))

    for epoch in range(start_epoch_trvae, train_trvae_epochs):
        print(f"Start epoch {epoch + 1} / {train_trvae_epochs} for Transition Model training...:")
        print("Resampling dataset...")
        dataset.resample()
        print("Starting training...")
        loss, _ = world_model.train_epoch(
            dataloader,
            log_writer,
            start_num_batches=epoch * len(dataloader) + train_ae_epochs * len(dataloader),
            train_ae=False,
            train_trvae=True,
        )
        if loss < min_trvae_loss:
            min_trvae_loss = loss
            world_model.save_model(epoch, min_ae_loss, min_trvae_loss, is_best=True,
                                   save_dir=os.path.join(session_dir, 'models'))
        world_model.save_model(epoch, min_ae_loss, min_trvae_loss, save_dir=os.path.join(session_dir, 'models'))
