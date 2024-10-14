import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3 import A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

# Custom CNN feature extractor that inherits from BaseFeaturesExtractor
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        # Call the parent constructor
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # Define a simple CNN architecture
        # Assuming the input is a 3-channel RGB image
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),  # 3 channels for RGB image
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the shape of the output after passing through the CNN
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        # Define a final fully connected layer that outputs the desired number of features
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Pass the input through the CNN layers
        cnn_output = self.cnn(observations)
        # Pass the result through the fully connected layer
        return self.linear(cnn_output)

# Use an environment with image input and discrete action space
env_id = "Breakout-v4"  # Example of an Atari game with image input and discrete action space
num_envs = 4  # Number of environments for parallel training
seed = 42

# Create a vectorized environment without frame stacking
env = DummyVecEnv([lambda: gym.make(env_id) for _ in range(num_envs)])

# Create the model using the custom CNN feature extractor
policy_kwargs = dict(
    features_extractor_class=CustomCNN,  # Set our custom CNN class
    features_extractor_kwargs=dict(features_dim=512),  # Define the output feature dimension
)

# Instantiate the A2C model
model = A2C(
    "CnnPolicy",  # We are using a CNN-based policy
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
)

# Train the model
model.learn(total_timesteps=1_000_000, progress_bar=True)

# Save the trained model
model.save("a2c_breakout_custom_cnn")

# Optionally: Load the model and continue training or evaluation
# loaded_model = A2C.load("a2c_breakout_custom_cnn")
# loaded_model.set_env(env)
# loaded_model.learn(total_timesteps=500_000)
