from typing import Set, Hashable, List
import torch
from stable_baselines3.common.base_class import BaseAlgorithm

from binary_state_representation.binary2binaryautoencoder import Binary2BinaryEncoder
from mdp_graph.mdp_graph import PolicyGraph
from mdp_learner import string_to_numpy_binary_array


def sample_model_with_onehot_encoding(
        model: BaseAlgorithm,
        states: Set[Hashable],
        actions: List[Hashable],
        policy_graph: PolicyGraph,
        sample_as_prior: bool = False,
        encoder: Binary2BinaryEncoder or None = None,
        keep_dims: int = -1,
        #  device: torch.device = torch.device('cpu'),
):
    """
    No need to use env cause states are observations, if ever use image obs in the future env will be needed.
    """
    device = model.device
    with torch.no_grad():
        for state in states:
            obs_tensor = string_to_numpy_binary_array(state)
            obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32, device=device).unsqueeze(0)
            if encoder:
                encoder.to(device)
                encoder.eval()
                obs_tensor = encoder(obs_tensor)[:, 0:keep_dims]
            _, action_logits, _ = model.policy.evaluate_actions(obs_tensor, torch.tensor([actions], device=device))
            action_probs = torch.nn.functional.softmax(action_logits, dim=-1).cpu().squeeze().numpy().tolist()
            for action, action_prob in zip(actions, action_probs):
                if sample_as_prior:
                    policy_graph.set_prior_prob(state, action, action_prob)
                else:
                    policy_graph.set_state_prob(state, action, action_prob)


if __name__ == '__main__':
    from stable_baselines3 import PPO
    from customize_minigrid.wrappers import FullyObsSB3MLPWrapper
    from customize_minigrid.custom_env import CustomEnv
    from mdp_graph.mdp_graph import OptimalPolicyGraph
    from mdp_learner import OneHotEncodingMDPLearner

    # Initialize the environment and wrapper
    env = CustomEnv(
        txt_file_path='maps/door_key.txt',
        display_size=13,
        display_mode="random",
        random_rotate=True,
        random_flip=True,
        custom_mission="Find the key and open the door.",
        render_mode=None,
    )
    env = FullyObsSB3MLPWrapper(env, to_print=False)
    learner = OneHotEncodingMDPLearner(env)
    learner.learn()
    optimal_graph = OptimalPolicyGraph()
    optimal_graph.load_graph(learner.mdp_graph)
    optimal_graph.visualize(highlight_states=[learner.start_state, *learner.done_states], use_grid_layout=False,
                            display_state_name=False)
    optimal_graph.optimal_value_iteration(0.999, threshold=1e-5)
    optimal_graph.compute_optimal_policy(0.999, threshold=1e-5)

    model = PPO("MlpPolicy", env)
    sample_model_with_onehot_encoding(
        model, learner.state_set, learner.possible_actions, optimal_graph, sample_as_prior=True
    )

    optimal_graph.control_info_iteration(1.0, threshold=1e-5)
    optimal_graph.value_iteration(1.0, threshold=1e-5)
    optimal_policy = optimal_graph.free_energy_iteration(beta=1.0, gamma=1.0)
    optimal_graph.visualize_policy_and_values(title="Policy and Values", value_type="value",
                                              highlight_states=[learner.start_state, *learner.done_states],
                                              use_grid_layout=False, display_state_name=False)
    optimal_graph.visualize_policy_and_values(title="Policy and Control Info", value_type="control_info",
                                              highlight_states=[learner.start_state, *learner.done_states],
                                              use_grid_layout=False, display_state_name=False)
    optimal_graph.visualize_policy_and_values(title="Policy and Free Energy", value_type="free_energy",
                                              highlight_states=[learner.start_state, *learner.done_states],
                                              use_grid_layout=False, display_state_name=False)
