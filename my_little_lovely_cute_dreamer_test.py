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
        :return: z_next: 下一个潜在状态, mean: 均值, logvar: 方差的对数
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

        # 将 z_next 恢复为 (batch_size, latent_channels, latent_height, latent_width)
        z_next_reshaped = z_next.view(batch_size, latent_channels, latent_height, latent_width)

        return z_next_reshaped, mean, logvar


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
    def __init__(self, env, latent_shape, cnn_net_arch, transition_model_conv_arch, actor_conv_arch, critic_conv_arch, gamma=0.99):
        self.env = env
        self.encoder = Encoder(env.single_observation_space.shape, latent_shape, cnn_net_arch).to(device)
        self.decoder = Decoder(latent_shape, env.single_observation_space.shape, cnn_net_arch).to(device)
        self.transition_model = TransitionModelVAE(latent_shape, env.single_action_space.n, transition_model_conv_arch).to(device)
        self.actor = Actor(latent_shape, env.single_action_space.n, actor_conv_arch).to(device)
        self.critic = Critic(latent_shape, critic_conv_arch).to(device)
        self.gamma = gamma
        self.optimizer = optim.Adam(list(self.encoder.parameters()) +
                                    list(self.decoder.parameters()) +
                                    list(self.transition_model.parameters()) +
                                    list(self.actor.parameters()) +
                                    list(self.critic.parameters()), lr=1e-4)

    def reset(self):
        obs, _ = self.env.reset()  # 在 gymnasium 中 reset 返回 (obs, info)
        return obs

    def select_action(self, state):
        with torch.no_grad():
            latent_state = self.encoder(state)
            action_probs = self.actor(latent_state)
            action = torch.multinomial(action_probs, 1).squeeze(1)  # (batch_size,)

            # 将动作转换为one-hot编码
            action_one_hot = F.one_hot(action, num_classes=self.env.single_action_space.n).float()  # (batch_size, action_dim)
            return action_one_hot, action

    def train(self, state, action, reward, next_state, done):
        # 编码当前状态和下一个真实状态
        latent_state = self.encoder(state)
        latent_next_state = self.encoder(next_state)  # 保持next_state为编码后的真实下一个状态

        # 使用Transition Model预测下一个潜在状态
        predicted_next_state, mean, logvar = self.transition_model(latent_state, action)

        # 通过Decoder重建预测的图像观测
        reconstructed_state = self.decoder(predicted_next_state)

        # 将next_state调整到与重构图像相同的大小
        resized_next_state = F.interpolate(next_state, size=reconstructed_state.shape[2:], mode='bilinear',
                                           align_corners=False)

        # 计算VAE损失：重构损失和KL散度
        recon_loss = nn.MSELoss()(reconstructed_state, resized_next_state)  # 使用调整后的next_state作为目标
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        vae_loss = recon_loss + kl_loss

        # Critic损失：TD误差
        critic_value = self.critic(latent_state)
        next_value = self.critic(latent_next_state) * (1 - done)
        target_value = reward + self.gamma * next_value
        value_loss = nn.MSELoss()(critic_value, target_value.detach())

        # **Actor loss**: The Actor should maximize the expected value predicted by the Critic
        action_probs = self.actor(latent_state)
        log_action_probs = torch.log(action_probs + 1e-6)  # avoid log(0)

        # The Actor wants to maximize Critic's value, so we minimize negative value
        actor_loss = -(log_action_probs * critic_value.detach()).mean()

        # 总损失 = VAE损失 + Critic损失 + Actor损失
        total_loss = vae_loss + value_loss + actor_loss

        # 优化模型
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def run_episode(self):
        states = self.reset()

        done = np.zeros(self.env.num_envs, dtype=bool)  # 每个环境的完成状态
        total_rewards = np.zeros(self.env.num_envs, dtype=np.float32)  # 每个环境的累积奖励

        while not done.all():  # 所有环境都完成时结束
            # 检查并转换 LazyFrames 为 PyTorch Tensor
            if isinstance(states, LazyFrames):
                states = np.array(states)
            states = torch.tensor(states, dtype=torch.float32).to(device)

            # 选择动作
            action_one_hot, action_int = self.select_action(states)

            # 环境交互
            next_states, rewards, terminated, truncated, infos = self.env.step(action_int.cpu().numpy())
            print(rewards)

            # 终止条件更新：当 terminated 或 truncated 为 True 时认为环境完成
            dones = np.logical_or(terminated, truncated)

            # 训练模型
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

            self.train(states, action_one_hot, rewards, next_states, dones)

            # 状态更新
            states = next_states
            total_rewards += rewards.cpu().numpy()  # 更新累积奖励

        return total_rewards


# Vectorized environment creator
def create_env(env_name):
    def make_env():
        env = gym.make(env_name)
        env = AtariPreprocessing(env, scale_obs=True)  # 归一化为[0, 1]
        env = FrameStack(env, num_stack=4)  # 堆叠4帧
        return env
    return make_env


# 主函数：创建向量化环境和训练
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义向量化环境：并行运行多个Pong环境
    num_envs = 1  # 并行环境的数量
    env_name = 'PongNoFrameskip-v4'
    envs = AsyncVectorEnv([create_env(env_name) for _ in range(num_envs)])

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

    transition_model_conv_arch = [
        (64, 4, 2, 1),
        (128, 4, 2, 1),
    ]

    agent = DreamerAgent(envs, latent_shape, cnn_net_arch, transition_model_conv_arch, actor_conv_arch, critic_conv_arch)

    for episode in range(1000):
        rewards = agent.run_episode()
        print(f"Episode {episode}, Rewards: {rewards}")
