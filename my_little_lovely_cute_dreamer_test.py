import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import FrameStack, AtariPreprocessing, LazyFrames
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
    def __init__(self, latent_shape, action_dim):
        super(TransitionModelVAE, self).__init__()
        latent_channels, latent_height, latent_width = latent_shape
        latent_dim = latent_channels * latent_height * latent_width

        self.fc_mean = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, latent_state, action):
        if len(latent_state.shape) == 4:  # (batch_size, channels, height, width)
            latent_state = latent_state.view(latent_state.size(0), -1)  # 展平成向量
        x = torch.cat([latent_state, action], dim=-1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_next = mean + eps * std
        return z_next.view_as(latent_state), mean, logvar


class Actor(nn.Module):
    def __init__(self, latent_shape, action_dim, conv_arch):
        super(Actor, self).__init__()
        latent_channels, latent_height, latent_width = latent_shape

        # 动态添加卷积层
        conv_layers = []
        in_channels = latent_channels
        for out_channels, kernel_size, stride, padding in conv_arch:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层，输出离散动作的概率分布
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, latent_state):
        # 经过卷积层处理
        x = self.conv_layers(latent_state)

        # 对卷积输出进行全局平均池化
        x = self.global_avg_pool(x)

        # 去掉多余的维度 (batch_size, in_channels)
        x = torch.flatten(x, 1)

        # 全连接层输出动作分布
        return self.fc(x)


class Critic(nn.Module):
    def __init__(self, latent_shape, conv_arch):
        super(Critic, self).__init__()
        latent_channels, latent_height, latent_width = latent_shape

        # 动态添加卷积层
        conv_layers = []
        in_channels = latent_channels
        for out_channels, kernel_size, stride, padding in conv_arch:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层，输出状态价值
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 输出一个标量作为状态价值
        )

    def forward(self, latent_state):
        # 经过卷积层处理
        x = self.conv_layers(latent_state)

        # 对卷积输出进行全局平均池化
        x = self.global_avg_pool(x)

        # 去掉多余的维度 (batch_size, in_channels)
        x = torch.flatten(x, 1)

        # 全连接层输出状态价值
        return self.fc(x)


# DreamerAgent: 包含Encoder, Decoder, TransitionModel, Actor, Critic
class DreamerAgent:
    def __init__(self, env, latent_shape, cnn_net_arch, actor_conv_arch, critic_conv_arch, gamma=0.99):
        self.env = env
        self.encoder = Encoder(env.observation_space.shape, latent_shape, cnn_net_arch).to(device)
        self.decoder = Decoder(latent_shape, env.observation_space.shape, cnn_net_arch).to(device)
        self.transition_model = TransitionModelVAE(latent_shape, env.action_space.n).to(device)
        self.actor = Actor(latent_shape, env.action_space.n, actor_conv_arch).to(device)
        self.critic = Critic(latent_shape, critic_conv_arch).to(device)
        self.gamma = gamma
        self.optimizer = optim.Adam(list(self.encoder.parameters()) +
                                    list(self.decoder.parameters()) +
                                    list(self.transition_model.parameters()) +
                                    list(self.actor.parameters()) +
                                    list(self.critic.parameters()), lr=1e-4)

    def reset(self):
        return self.env.reset()

    def select_action(self, state):
        with torch.no_grad():
            latent_state = self.encoder(state)
            action_probs = self.actor(latent_state)
            # 对每个批次样本选择动作
            action = torch.multinomial(action_probs, 1).squeeze(1)  # (batch_size,)

            # 将动作转换为one-hot编码
            action_one_hot = F.one_hot(action, num_classes=self.env.action_space.n).float()  # (batch_size, action_dim)
            return action_one_hot

    def train(self, state, action, reward, next_state, done):
        state = self.encoder(state)
        next_state = self.encoder(next_state)

        # 使用Transition Model预测下一个潜在状态
        predicted_state, mean, logvar = self.transition_model(state, torch.tensor(action).unsqueeze(0).to(device))

        # 通过Decoder重建图像观测
        reconstructed_state = self.decoder(predicted_state)

        # 计算VAE损失：重构损失和KL散度
        recon_loss = nn.MSELoss()(reconstructed_state, next_state)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        vae_loss = recon_loss + kl_loss

        # Critic损失：TD误差
        critic_value = self.critic(state)
        next_value = self.critic(next_state) * (1 - done)
        target_value = reward + self.gamma * next_value
        value_loss = nn.MSELoss()(critic_value, target_value.detach())

        # 总损失
        total_loss = vae_loss + value_loss

        # 优化模型
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def run_episode(self):
        state, _ = self.reset()  # 在gymnasium中reset返回(state, info)

        # 检查并转换LazyFrames为 PyTorch Tensor
        if isinstance(state, LazyFrames):
            state = np.array(state)  # 转换为 NumPy 数组
            state = torch.tensor(state, dtype=torch.float32).to(device)  # 转换为 PyTorch Tensor，并移动到设备上

        done = False
        total_reward = 0
        while not done:
            # 选择动作
            action = self.select_action(state)

            # 环境交互
            next_state, reward, done, _, _ = self.env.step(action)

            # 检查并转换LazyFrames为 PyTorch Tensor
            if isinstance(next_state, LazyFrames):
                next_state = np.array(next_state)  # 转换为 NumPy 数组
                next_state = torch.tensor(next_state, dtype=torch.float32).to(device)  # 转换为 PyTorch Tensor，并移动到设备上

            # 训练模型
            self.train(state, action, reward, next_state, done)

            # 状态更新
            state = next_state
            total_reward += reward

        return total_reward


# 使用Atari环境作为示例（Pong-v4）
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建 Atari Pong 环境，并添加预处理和 FrameStack
    env = gym.make('PongNoFrameskip-v4')

    # Atari 预处理：灰度化、下采样、动作重复等
    env = AtariPreprocessing(env, scale_obs=True)  # 将观测归一化为[0, 1]

    # 堆叠4帧，构成时序观测
    env = FrameStack(env, num_stack=4)

    # 定义潜在表示形状和卷积网络架构
    latent_shape = (16, 8, 8)  # 潜在空间为16通道的8x8图像
    cnn_net_arch = [
        (64, 3, 2, 1),
        (128, 3, 2, 1),
        (256, 3, 2, 1),
    ]

    # 定义Actor和Critic的卷积架构
    actor_conv_arch = [
        (128, 3, 2, 1),
        (256, 3, 2, 1),
    ]

    critic_conv_arch = [
        (128, 3, 2, 1),
        (256, 3, 2, 1),
    ]

    agent = DreamerAgent(env, latent_shape, cnn_net_arch, actor_conv_arch, critic_conv_arch)

    for episode in range(1000):
        reward = agent.run_episode()
        print(f"Episode {episode}, Reward: {reward}")
