import torch as th
import torch.nn as nn
from gymnasium.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from torch.distributions import Categorical, Normal


# Custom general policy network with freeze functionality
class CustomGeneralPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # Accept custom parameters for network architecture
        self.shared_layers = kwargs.pop('shared_layers', [128, 64])  # Shared layers for feature extractor
        self.policy_layers = kwargs.pop('policy_layers', [128, 64])   # Policy network layers
        self.value_layers = kwargs.pop('value_layers', [128, 64])     # Value network layers
        self.activation_fn = kwargs.pop('activation_fn', nn.ReLU)     # Activation function

        # Step 1: Call the parent constructor to initialize observation_space and action_space
        super(CustomGeneralPolicy, self).__init__(*args, **kwargs)

        # Initialize frozen states for each network part
        self.shared_frozen = False
        self.policy_frozen = False
        self.value_frozen = False

    def build_mlp(self, input_dim, layers, activation_fn):
        # Helper function to build fully connected layers
        modules = []
        for layer_size in layers:
            modules.append(nn.Linear(input_dim, layer_size))
            modules.append(activation_fn())
            input_dim = layer_size
        return nn.Sequential(*modules)

    def _build_mlp_extractor(self):
        """
        Override this method to avoid using the default mlp_extractor.
        We will define our own feature extractor and policy/value networks.
        """
        # Build custom feature extractor
        self.shared_feature_extractor = self.build_mlp(self.observation_space.shape[0], self.shared_layers, self.activation_fn)

        # Build custom policy and value networks
        self.policy_net = self.build_mlp(self.shared_layers[-1], self.policy_layers, self.activation_fn)
        self.value_net = self.build_mlp(self.shared_layers[-1], self.value_layers, self.activation_fn)
        self.value_net.add_module("output", nn.Linear(self.value_layers[-1], 1))

        # Set mlp_extractor to None as we are overriding it
        self.mlp_extractor = None

    def _build(self, lr_schedule: callable):
        """
        Override the _build method to manually control network building.
        """
        # Call the custom _build_mlp_extractor to set up the networks
        self._build_mlp_extractor()

        # Set up action distribution based on policy latent size
        latent_dim_pi = self.policy_net[-1].out_features  # Get last layer output size
        latent_dim_vf = self.value_net[-2].out_features  # For value network

        if isinstance(self.action_dist, Normal):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init)
        elif isinstance(self.action_dist, Categorical):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        # Set optimizer
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs, deterministic=False):
        # Forward pass using custom feature extractor and policy/value networks
        features = self.shared_feature_extractor(obs)

        # Compute action logits and state value
        action_logits = self.policy_net(features)
        value = self.value_net(features)

        # Get action distribution
        if isinstance(self.action_space, Discrete):
            action_dist = Categorical(logits=action_logits)
        else:
            action_std = th.ones_like(action_logits)
            action_dist = Normal(action_logits, action_std)

        actions = action_dist.sample() if not deterministic else action_logits
        log_prob = action_dist.log_prob(actions)

        return actions, value, log_prob

    def evaluate_actions(self, obs, actions):
        # Evaluate actions using custom networks
        features = self.shared_feature_extractor(obs)
        action_logits = self.policy_net(features)
        value = self.value_net(features)

        # Get action distribution
        if isinstance(self.action_space, Discrete):
            action_dist = Categorical(logits=action_logits)
        else:
            action_std = th.ones_like(action_logits)
            action_dist = Normal(action_logits, action_std)

        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        return value, log_prob, entropy

    def predict_values(self, obs):
        # Predict values using the custom value network
        features = self.shared_feature_extractor(obs)
        return self.value_net(features)

    def get_distribution(self, obs):
        # Get the action distribution using the custom policy network
        features = self.shared_feature_extractor(obs)
        action_logits = self.policy_net(features)

        if isinstance(self.action_space, Discrete):
            return Categorical(logits=action_logits)
        else:
            action_std = th.ones_like(action_logits)
            return Normal(action_logits, action_std)

    def freeze_layers(self, part, freeze=True):
        """
        Lock or unlock the parameters of specific parts of the network.
        :param part: Can be "shared", "policy", or "value" to specify which part to freeze/unfreeze.
        :param freeze: Set to True to freeze the parameters (no gradient updates), False to unfreeze.
        """
        if part == "shared":
            self.shared_frozen = freeze
            for param in self.shared_feature_extractor.parameters():
                param.requires_grad = not freeze  # Set whether gradients are required
        elif part == "policy":
            self.policy_frozen = freeze
            for param in self.policy_net.parameters():
                param.requires_grad = not freeze
        elif part == "value":
            self.value_frozen = freeze
            for param in self.value_net.parameters():
                param.requires_grad = not freeze


if __name__ == '__main__':
    # Create environment
    env = make_vec_env('CartPole-v1', n_envs=1)

    # Define custom architecture parameters
    policy_kwargs = dict(
        shared_layers=[128, 128],  # Shared feature extractor layers
        policy_layers=[128, 64],   # Policy network layers
        value_layers=[128, 64],    # Value network layers
        activation_fn=nn.Tanh      # Custom activation function (Tanh)
    )

    # Create PPO model with the custom policy network
    model = PPO(CustomGeneralPolicy, env, policy_kwargs=policy_kwargs, verbose=1)

    # Freeze the policy network, unfreeze the value network
    model.policy.freeze_layers(part="policy", freeze=True)
    model.policy.freeze_layers(part="value", freeze=False)

    # Begin training
    timesteps = 10000
    for i in range(timesteps // 1000):
        if i % 2 == 0:
            # Every 1000 steps, freeze the shared feature extractor
            model.policy.freeze_layers(part="shared", freeze=True)
        else:
            # Unfreeze the shared feature extractor
            model.policy.freeze_layers(part="shared", freeze=False)

        # Train the model for 1000 steps
        model.learn(total_timesteps=1000)
