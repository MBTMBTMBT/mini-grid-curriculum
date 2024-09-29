import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


class SimpleCNN(nn.Module):
    def __init__(self, observation_space: gym.spaces.Space, net_arch=None, cnn_net_arch=None, activation_fn=nn.ReLU, output_activation_fn=nn.Sigmoid):
        super(SimpleCNN, self).__init__()

        if net_arch is None:
            net_arch = [64, 64]

        if cnn_net_arch is None:
            # Default CNN architecture: [(out_channels, kernel_size, stride, padding)]
            cnn_net_arch = [
                (32, 3, 2, 1),  # out_channels=32, kernel_size=3, stride=2, padding=1
                (64, 3, 2, 1),  # out_channels=64, kernel_size=3, stride=2, padding=1
                (128, 3, 2, 1),  # out_channels=128, kernel_size=3, stride=2, padding=1
            ]

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
            pooled_output = F.adaptive_avg_pool2d(conv_output, (1, 1))  # Apply average pooling to get a fixed-size output
            conv_output_flattened_dim = pooled_output.numel()  # Use numel() to get the total number of elements

        # Fully connected layers defined by net_arch
        input_dim = conv_output_flattened_dim
        fc_layers = []
        for i, layer_size in enumerate(net_arch):
            fc_layers.append(nn.Linear(input_dim, layer_size))
            if i < len(net_arch) - 1:
                fc_layers.append(activation_fn())  # Apply activation function to all layers except the last
            input_dim = layer_size

        # Create the fully connected model and add the final output activation
        self.fc = nn.Sequential(*fc_layers)
        self.output_activation = output_activation_fn()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Ensure the observations match expected input shape (batch_size, channels, height, width)
        batch_size = observations.size(0)
        observations = observations.view(batch_size, *observations.shape[1:])

        # Apply the CNN layers
        conv_features = self.cnn(observations)

        # Apply average pooling to make the feature map adaptable to any input size
        pooled_features = F.adaptive_avg_pool2d(conv_features, (1, 1))

        # Flatten the pooled features
        flattened_features = pooled_features.view(batch_size, -1)

        # Pass through the fully connected layers
        fc_output = self.fc(flattened_features)

        # Apply the output activation function (default is Sigmoid)
        return self.output_activation(fc_output)
