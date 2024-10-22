import io
import os
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg16
from piq import ssim
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


# Define a unified loss function class that combines Perceptual, SSIM, and MAE/MSE losses
class UnifiedLoss(nn.Module):
    def __init__(self, perc_weight=0.6666, ssim_weight=0.3334, pixel_loss_weight=1.0):
        super(UnifiedLoss, self).__init__()
        # Load pre-trained VGG16 layers for perceptual loss
        self.vgg_layers = vgg16(pretrained=True).features[:16]
        for param in self.vgg_layers.parameters():
            param.requires_grad = False  # Freeze VGG parameters

        # Weights for each loss component
        self.perc_weight = perc_weight
        self.ssim_weight = ssim_weight
        self.pixel_loss_weight = pixel_loss_weight

    def forward(self, gen_img, target_img):
        # Perceptual Loss: Extract features and compute L1 loss between them
        gen_features = self.vgg_layers(gen_img)
        target_features = self.vgg_layers(target_img)
        perceptual_loss = nn.functional.l1_loss(gen_features, target_features)

        # SSIM Loss: 1 - SSIM(generated, target) as loss
        ssim_loss = 1 - ssim(gen_img, target_img, data_range=1.0)

        # Pixel-wise loss
        pixel_loss = (nn.functional.mse_loss(gen_img, target_img)
                      + nn.functional.l1_loss(gen_img, target_img))

        # Combine all losses with respective weights
        total_loss = (self.perc_weight * perceptual_loss +
                      self.ssim_weight * ssim_loss +
                      self.pixel_loss_weight * pixel_loss)

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


class SimpleTransitionModel(nn.Module):
    def __init__(self, latent_shape, action_dim, conv_arch):
        """
        A simplified transition model that replaces VAE with a deterministic CNN-based approach.
        :param latent_shape: Shape of the latent space (channels, height, width)
        :param action_dim: Dimensionality of the action space (one-hot encoded)
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

        # Reward and Done prediction
        x = self.reward_done_conv(x)
        reward_pred = self.reward_predictor(x)
        done_pred = self.done_predictor(x)

        return z_next_decoded, reward_pred, done_pred


class ForwardableWorldModel(nn.Module):
    def __init__(
            self,
            latent_shape: Tuple[int, int, int],
            obs_shape: Tuple[int, int, int],
            num_actions: int,
            cnn_net_arch: List[Tuple[int, int, int, int]],
            transition_model_conv_arch: List[Tuple[int, int, int, int]],
            lr=1e-4,
    ):
        super(ForwardableWorldModel, self).__init__()
        self.latent_shape = latent_shape
        self.encoder = Encoder(obs_shape, latent_shape, cnn_net_arch)
        self.decoder = Decoder(latent_shape, obs_shape, cnn_net_arch)
        self.transition_model = SimpleTransitionModel(latent_shape, num_actions, transition_model_conv_arch)
        self.num_actions = num_actions
        self.optimizer = torch.optim.Adam(
            params = self.parameters(),
            lr = lr,
        )

    def forward(self, state, action):
        device = next(self.parameters()).device
        state = state.to(device)
        action = action.to(device)

        # Encode the current and next state
        latent_state = self.encoder(state)

        # Predict the next latent state and reward with the transition model
        action = F.one_hot(action, self.num_actions).type(torch.float)
        predicted_latent_next_state, predicted_reward, predicted_done \
            = self.transition_model(latent_state, action)

        # Reconstruct the predicted next state
        predicted_reconstructed_state = self.decoder(latent_state)
        predicted_reconstructed_next_state = self.decoder(predicted_latent_next_state)

        return predicted_reconstructed_state, predicted_reconstructed_next_state, predicted_latent_next_state, predicted_reward, predicted_done

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
                # self.mse_loss(reconstructed_state, resized_state) +
                # self.mae_loss(reconstructed_state, resized_state) +
                # self.mse_loss(reconstructed_next_state, resized_next_state) +
                # self.mae_loss(reconstructed_next_state, resized_next_state)
                self.uni_loss(reconstructed_state, resized_state) +
                self.uni_loss(reconstructed_next_state, resized_next_state)
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
        # predicted_latent_next_state, mean, logvar, predicted_reward, predicted_done \
        #     = self.transition_model(latent_state, action)
        predicted_latent_next_state, predicted_reward, predicted_done \
            = self.transition_model(latent_state, action)

        latent_transition_loss = self.mse_loss(latent_next_state, predicted_latent_next_state)
        # kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        vae_loss = latent_transition_loss  # + 0.1 * kl_loss
        reward_loss = self.mse_loss(predicted_reward.squeeze(), reward)
        done_loss = F.binary_cross_entropy(predicted_done.squeeze(), done.float())  # Convert done to float

        predicted_next_state = self.decoder(predicted_latent_next_state)
        resized_next_state = F.interpolate(next_state, size=predicted_next_state.shape[2:],
                                           mode='bilinear', align_corners=False)
        reconstruction_disc_loss = self.uni_loss(predicted_next_state, resized_next_state)

        trvae_loss = vae_loss + reward_loss + done_loss + reconstruction_disc_loss

        # Optimize the components except for the discriminator
        self.trvae_optimizer.zero_grad()
        trvae_loss.backward()
        self.trvae_optimizer.step()

        # Return a dictionary with all the loss values
        loss_dict = {
            # "latent_transition_loss": latent_transition_loss.detach().cpu().item(),
            "reconstruction_disc_loss": reconstruction_disc_loss.detach().cpu().item(),
            # "kl_loss": kl_loss.detach().cpu().item(),
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
                            'vae_loss': loss_dict['vae_loss'],
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

                axs[1, 2].text(0.5, 0.5, f'Actual Reward: {reward.item():.2f}\nActual Done: {done.item():.2f}',
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
                log_writer.add_image(f'{i + start_num_batches}-{idx}_combined', image_tensor, i + start_num_batches)

                # Close the figure to free memory
                plt.close(fig)

        return avg_loss, start_num_batches


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    session_dir = r"./experiments/world_model-door_key"
    dataset_samples = int(1e4)
    dataset_repeat_each_epoch = 10
    train_ae_epochs = 15
    train_trvae_epochs = 50
    batch_size = 64
    ae_lr = 1e-4
    trvae_lr = 1e-4
    num_parallel = 4

    latent_shape = (8, 32, 32)  # channel, height, width
    # num_homomorphism_channels = 16
    movement_augmentation = 3

    encoder_decoder_net_arch = [
        (32, 3, 2, 1),
        # (32, 3, 1, 1),
        (64, 3, 2, 1),
        # (64, 3, 1, 1),
        (128, 3, 1, 1),
        # (128, 3, 1, 1),
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
            config.txt_file_path = f"./maps/door_key.txt"
            config.custom_mission = "reach the goal"
            config.minimum_display_size = 7
            config.display_mode = "middle"
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
            start_epoch_ae = train_ae_epochs
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
