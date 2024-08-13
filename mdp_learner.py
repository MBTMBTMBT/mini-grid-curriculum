import numpy as np

from customize_minigrid.wrappers import FullyObsSB3MLPWrapper
from mdp_graph.mdp_graph import MDPGraph, PolicyGraph, OptimalPolicyGraph


def numpy_binary_array_to_string(binary_array):
    binary_array = np.array(binary_array, dtype=np.uint8)
    binary_string = ''.join(binary_array.astype(str))
    return binary_string


def string_to_numpy_binary_array(binary_string):
    binary_array = np.array(list(binary_string), dtype=int)
    return binary_array


class OneHotEncodingMDPLearner:
    def __init__(self, onehot_env: FullyObsSB3MLPWrapper):
        self.env = onehot_env
        self.state_set = set()
        self.state_action_set = set()
        self.start_state = None
        self.done_states = set()
        self.possible_actions = list(range(self.env.action_space.n))
        self.mdp_graph: MDPGraph = MDPGraph()

    def learn(self):
        obs, _ = self.env.reset()
        current_state_code = numpy_binary_array_to_string(obs)
        self.state_set.add(current_state_code)
        self.start_state = current_state_code
        state_action_count = 0
        while True:
            new_state_set = set()
            new_state_action_set = set()
            for current_state_code in self.state_set:
                current_state_obs = string_to_numpy_binary_array(current_state_code)
                if current_state_code in self.done_states:
                    continue
                for action in self.possible_actions:
                    current_state_action_code = str(current_state_code) + str(action)
                    if current_state_action_code not in self.state_action_set:
                        self.env.set_env_with_code(current_state_obs)
                        next_obs, reward, done, truncated, info = self.env.step(action)
                        next_state_code = numpy_binary_array_to_string(next_obs)
                        if done:
                            self.done_states.add(next_state_code)
                        if current_state_action_code not in self.state_action_set:
                            new_state_action_set.add(current_state_action_code)
                        if next_state_code not in self.state_set:
                            new_state_set.add(next_state_code)
                        state_action_count += 1
                        print(f"Added [state-action pair num: {state_action_count}]: {hash(current_state_action_code)} -- {action} -> {hash(next_state_code)} Reward: {reward}")
                        self.mdp_graph.add_transition(current_state_code, action, next_state_code, 1.0)
                        self.mdp_graph.add_reward(current_state_code, action, next_state_code, float(reward))
            for new_state_code in new_state_set:
                self.state_set.add(new_state_code)
            for new_state_action_code in new_state_action_set:
                self.state_action_set.add(new_state_action_code)
            if len(new_state_set) == 0 and len(new_state_action_set) == 0:
                break

        self.env.force_reset()


if __name__ == '__main__':
    from customize_minigrid.custom_env import CustomEnv
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
    optimal_graph.visualize(highlight_states=[learner.start_state, *learner.done_states], use_grid_layout=False, display_state_name=False)
    optimal_graph.uniform_prior_policy()
    optimal_graph.optimal_value_iteration(0.999, threshold=1e-5)
    optimal_graph.compute_optimal_policy(0.999, threshold=1e-5)
    optimal_graph.control_info_iteration(1.0, threshold=1e-5)
    optimal_graph.value_iteration(1.0, threshold=1e-5)
    optimal_policy = optimal_graph.get_free_energy(beta=1e1)
    optimal_graph.visualize_policy_and_values(title="Policy and Values", value_type="value",
                                              highlight_states=[learner.start_state, *learner.done_states],
                                              use_grid_layout=False, display_state_name=False)
    optimal_graph.visualize_policy_and_values(title="Policy and Control Info", value_type="control_info",
                                              highlight_states=[learner.start_state, *learner.done_states],
                                              use_grid_layout=False, display_state_name=False)
    optimal_graph.visualize_policy_and_values(title="Policy and Free Energy", value_type="free_energy",
                                              highlight_states=[learner.start_state, *learner.done_states],
                                              use_grid_layout=False, display_state_name=False)
