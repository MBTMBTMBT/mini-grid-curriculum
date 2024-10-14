import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import FrameStack, AtariPreprocessing, LazyFrames
from gymnasium.vector import AsyncVectorEnv
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

        # 计算输入图像尺寸
        target_input_size = calculate_input_size(min(latent_height, latent_width), cnn_net_arch)
        self.resize = T.Resize((target_input_size, target_input_size), antialias=True)

        conv_layers = []
        in_channels = channels
        for out_channels, kernel_size, stride, padding in cnn_net_arch:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels

        # 添加最后一层卷积层，确保输出通道数等于latent_channels
        conv_layers.append(nn.Conv2d(in_channels, latent_channels, kernel_size=3, stride=1, padding=1))
        self.encoder = nn.Sequential(*conv_layers)

    def forward(self, x):
        # 保证输入有batch维度 (batch_size, channels, height, width)
        if len(x.shape) == 3:  # 如果输入缺少批次维度
            x = x.unsqueeze(0)  # 添加批次维度
        x = self.resize(x)
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_shape, output_shape, cnn_net_arch):
        super(Decoder, self).__init__()
        latent_channels, latent_height, latent_width = latent_shape
        channels, height, width = output_shape

        deconv_layers = []
        in_channels = latent_channels  # 解码器的输入通道应与Encoder的输出通道一致

        # 使用反卷积网络还原图像
        for out_channels, kernel_size, stride, padding in reversed(cnn_net_arch):
            deconv_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
            deconv_layers.append(nn.ReLU())
            in_channels = out_channels

        # 添加最后一层反卷积层，确保输出通道数等于原始图像的通道数
        deconv_layers.append(nn.ConvTranspose2d(in_channels, channels, kernel_size=3, stride=1, padding=1))
        deconv_layers.append(nn.Sigmoid())  # 输出像素值范围[0, 1]

        self.decoder = nn.Sequential(*deconv_layers)

    def forward(self, latent_state):
        # 保证输入有batch维度 (batch_size, channels, height, width)
        if len(latent_state.shape) == 3:  # 如果输入缺少批次维度
            latent_state = latent_state.unsqueeze(0)  # 添加批次维度
        return self.decoder(latent_state)


class TransitionModelVAE(nn.Module):
    def __init__(self, latent_shape, action_dim, conv_arch):
        """
        :param latent_shape: 潜在空间的形状 (channels, height, width)
        :param action_dim: 动作的维度
        :param conv_arch: 卷积网络的架构，例如 [(64, 4, 2, 1), (128, 4, 2, 1)]
        """
        super(TransitionModelVAE, self).__init__()
        latent_channels, latent_height, latent_width = latent_shape

        # 动作嵌入：将动作映射为与潜在状态同样大小的向量
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, latent_channels * latent_height * latent_width),
            nn.ReLU()
        )

        # 动作和潜在状态融合的卷积层
        conv_layers = []
        in_channels = latent_channels * 2  # 由于拼接，通道数是原来的2倍
        for out_channels, kernel_size, stride, padding in conv_arch:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv_encoder = nn.Sequential(*conv_layers)

        # 动态计算展平后的大小
        self.flattened_size = self._get_flattened_size(latent_shape, conv_arch)

        # Flatten后生成均值和logvar
        self.fc_mean = nn.Linear(self.flattened_size, latent_channels * latent_height * latent_width)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_channels * latent_height * latent_width)

        # 新增：奖励预测器
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 预测奖励，输出标量
        )

    def _get_flattened_size(self, latent_shape, conv_arch):
        """
        通过给定的卷积架构和输入形状，动态计算展平后的大小
        """
        # 假设 batch_size = 1，创建一个假定输入
        sample_tensor = torch.zeros(1, latent_shape[0] * 2, latent_shape[1], latent_shape[2])  # 拼接后，通道数翻倍
        sample_tensor = self.conv_encoder(sample_tensor)
        return sample_tensor.numel()

    def forward(self, latent_state, action):
        """
        :param latent_state: (batch_size, latent_channels, latent_height, latent_width)
        :param action: (batch_size, action_dim)
        :return: z_next: 下一个潜在状态, mean: 均值, logvar: 方差的对数, reward_pred: 预测奖励
        """
        batch_size, latent_channels, latent_height, latent_width = latent_state.shape

        # 嵌入动作，并将其形状调整为与潜在状态匹配
        action_embed = self.action_embed(action)
        action_embed = action_embed.view(batch_size, latent_channels, latent_height, latent_width)

        # 将动作嵌入与潜在状态拼接，而不是相加
        x = torch.cat([latent_state, action_embed], dim=1)  # 拼接后通道数翻倍

        # 通过卷积编码器处理
        x = self.conv_encoder(x)

        # Flatten 以进行全连接层处理
        x_flat = x.view(batch_size, -1)

        # 生成均值和logvar
        mean = self.fc_mean(x_flat)
        logvar = self.fc_logvar(x_flat)

        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_next = mean + eps * std

        # 奖励预测
        reward_pred = self.reward_predictor(x_flat)

        # 将 z_next 恢复为 (batch_size, latent_channels, latent_height, latent_width)
        z_next_reshaped = z_next.view(batch_size, latent_channels, latent_height, latent_width)

        return z_next_reshaped, mean, logvar, reward_pred
