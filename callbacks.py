import os
from typing import Union, List
import statistics

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from stable_baselines3.common.type_aliases import PolicyPredictor
from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

from mdp_graph.mdp_graph import PolicyGraph
from mdp_learner import OneHotEncodingMDPLearner
from model_sampler import sample_model_with_onehot_encoding


def eval_envs(
        envs: List[VecEnv],
        env_names: List[str],
        model: PolicyPredictor,
        num_eval_episodes: int,
        deterministic: bool,
        log_writer: SummaryWriter,
        num_timesteps: int,
        verbose: int = 1,
) -> float:
    print("Time step: ", num_timesteps)
    rewards = []
    for env, env_name in zip(envs, env_names):
        episode_rewards, _ = evaluate_policy(
            model,
            env,
            n_eval_episodes=num_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=True,
        )

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        # Log results
        if verbose >= 1:
            print(f"Evaluation of {env_name}: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Log results to TensorBoard
        log_writer.add_scalar(f'{env_name}/mean_reward', mean_reward, num_timesteps)
        log_writer.add_scalar(f'{env_name}/std_reward', std_reward, num_timesteps)

        rewards.append(mean_reward)

    return np.mean(rewards)


class EvalCallback(EventCallback):
    def __init__(
            self,
            eval_envs: List[VecEnv],
            eval_env_names: List[str],
            log_writer: SummaryWriter,
            eval_freq: int,
            n_eval_episodes: int,
            deterministic: bool = True,
            verbose: int = 1,
            start_timestep: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_envs = eval_envs
        self.eval_env_names = eval_env_names
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.log_writer = log_writer
        self.evaluations: List[float] = []
        self.start_timestep = start_timestep

    def _on_step(self) -> bool:
        # Evaluate the model at specified frequency
        if self.n_calls % self.eval_freq == 0:
            self.eval()
        return True

    def _on_training_end(self) -> None:
        self.eval()

    def eval(self):
        print("Evaluating model over targets...")
        mean_reward = eval_envs(
            envs=self.eval_envs,
            env_names=self.eval_env_names,
            model=self.model,
            num_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            log_writer=self.log_writer,
            num_timesteps=self.num_timesteps + self.start_timestep,
            verbose=self.verbose,
        )


class EvalSaveCallback(EventCallback):
    def __init__(
            self,
            eval_envs: List[VecEnv],
            eval_env_names: List[str],
            model_save_dir: str,
            model_save_name: str,
            log_writer: SummaryWriter,
            eval_freq: int,
            n_eval_episodes: int,
            deterministic: bool = True,
            verbose: int = 1,
            start_timestep: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_envs = eval_envs
        self.eval_env_names = eval_env_names
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.model_save_dir = model_save_dir
        self.model_save_name = model_save_name
        self.log_writer = log_writer
        self.evaluations: List[float] = []
        self.start_timestep = start_timestep

    def _on_step(self) -> bool:
        # Evaluate the model at specified frequency
        if self.n_calls % self.eval_freq == 0:
            self.eval()
        return True

    def _on_training_end(self) -> None:
        self.eval()

    def eval(self):
        print("Evaluating model...")
        mean_reward = eval_envs(
            envs=self.eval_envs,
            env_names=self.eval_env_names,
            model=self.model,
            num_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            log_writer=self.log_writer,
            num_timesteps=self.num_timesteps + self.start_timestep,
            verbose=self.verbose,
        )

        latest_path = os.path.join(self.model_save_dir, f"{self.model_save_name}_latest.zip")
        self.model.save(latest_path)
        if self.verbose >= 1:
            print(f"Saved latest model to {latest_path}")

        best_path = os.path.join(self.model_save_dir, f"{self.model_save_name}_best.zip")
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(best_path)
            if self.verbose >= 1:
                print(f"New best model with mean reward {mean_reward:.2f} saved to {best_path}")


class InfoEvalSaveCallback(EvalSaveCallback):
    def __init__(
            self,
            eval_envs: List[VecEnv],
            eval_env_names: List[str],
            model: BaseAlgorithm,
            model_save_dir: str,
            model_save_name: str,
            log_writer: SummaryWriter,
            eval_freq: int,
            compute_info_freq: int,
            n_eval_episodes: int,
            deterministic: bool = True,
            verbose: int = 1,
            start_timestep: int = 0,
            iter_gamma: float = 0.999,
            iter_threshold: float = 1e-5,
            max_iter: int = 1e5,
    ):
        super(InfoEvalSaveCallback, self).__init__(
            eval_envs=eval_envs,
            eval_env_names=eval_env_names,
            model_save_dir=model_save_dir,
            model_save_name=model_save_name,
            log_writer=log_writer,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            verbose=verbose,
            start_timestep=start_timestep,
        )

        self.model = model
        self.compute_info_freq = compute_info_freq
        self.iter_gamma = iter_gamma
        self.iter_threshold = iter_threshold
        self.max_iter = max_iter

        # get all the prior distributions.
        self.mdp_learners = {}
        self.policy_graphs_agent_prior = {}
        self.policy_graphs_uniform_prior = {}

        for env, env_name in zip(eval_envs, eval_env_names):
            print("Observing environment {}...".format(env_name))
            mdp_learner = OneHotEncodingMDPLearner(env.venv.envs[0])
            mdp_learner.learn()
            self.mdp_learners[env_name] = mdp_learner
            # self.prior_model = prior_model
            policy_graph = PolicyGraph()
            policy_graph.load_graph(mdp_learner.mdp_graph)
            sample_model_with_onehot_encoding(
                self.model,
                mdp_learner.state_set,
                mdp_learner.possible_actions,
                policy_graph,
                sample_as_prior=True,
            )
            self.policy_graphs_agent_prior[env_name] = policy_graph

            policy_graph = PolicyGraph()
            policy_graph.load_graph(mdp_learner.mdp_graph)
            policy_graph.uniform_prior_policy()
            self.policy_graphs_uniform_prior[env_name] = policy_graph

    def _on_step(self) -> bool:
        # Evaluate the model at specified frequency
        if self.n_calls % self.eval_freq == 0:
            self.eval()

        if self.n_calls % self.compute_info_freq == 0:
            self.compute_info_gains()

        return True

    def _on_training_end(self) -> None:
        self.eval()
        self.compute_info_gains()

    def compute_info_gains(self):
        print("Computing Information Gain...")
        for env, env_name in zip(self.eval_envs, self.eval_env_names):
            sample_model_with_onehot_encoding(
                self.model,
                self.mdp_learners[env_name].state_set,
                self.mdp_learners[env_name].possible_actions,
                self.policy_graphs_agent_prior[env_name],
                sample_as_prior=False,
            )
            sample_model_with_onehot_encoding(
                self.model,
                self.mdp_learners[env_name].state_set,
                self.mdp_learners[env_name].possible_actions,
                self.policy_graphs_uniform_prior[env_name],
                sample_as_prior=False,
            )

            self.policy_graphs_agent_prior[env_name].value_iteration(
                gamma=self.iter_gamma,
                threshold=self.iter_threshold,
                max_iterations=self.max_iter,
            )
            self.policy_graphs_agent_prior[env_name].control_info_iteration(
                gamma=self.iter_gamma,
                threshold=self.iter_threshold,
                max_iterations=self.max_iter,
            )
            self.policy_graphs_uniform_prior[env_name].control_info_iteration(
                gamma=self.iter_gamma,
                threshold=self.iter_threshold,
                max_iterations=self.max_iter,
            )

            min_value = min(self.policy_graphs_agent_prior[env_name].policy_value)
            mean_value = statistics.mean(self.policy_graphs_agent_prior[env_name].policy_value)
            start_position_value \
                = self.policy_graphs_agent_prior[env_name].policy_value[self.mdp_learners[env_name].start_state]

            min_control_info_gain = min(self.policy_graphs_agent_prior[env_name].control_info)
            mean_control_info_gain = statistics.mean(self.policy_graphs_agent_prior[env_name].control_info)
            start_position_control_info_gain \
                = self.policy_graphs_agent_prior[env_name].control_info[self.mdp_learners[env_name].start_state]

            min_control_info = min(self.policy_graphs_uniform_prior[env_name].control_info)
            mean_control_info = statistics.mean(self.policy_graphs_uniform_prior[env_name].control_info)
            start_position_control_info \
                = self.policy_graphs_uniform_prior[env_name].control_info[self.mdp_learners[env_name].start_state]

            if self.verbose >= 1:
                print(f"Value of {env_name}: min: {min_value:.2f}, mean:{mean_value:.2f}, start position: {start_position_value:.2f}")
                print(f"Info Gain of {env_name}: min: {min_control_info_gain:.2f}, mean:{mean_control_info_gain:.2f}, start position: {start_position_control_info_gain:.2f}")
                print(f"Control Info of {env_name}: min: {min_control_info:.2f}, mean:{mean_control_info:.2f}, start position: {start_position_control_info:.2f}")

            self.log_writer.add_scalar(
                f'{env_name}/value_min', min_value, self.num_timesteps + self.start_timestep
            )
            self.log_writer.add_scalar(
                f'{env_name}/value_mean', mean_value, self.num_timesteps + self.start_timestep
            )
            self.log_writer.add_scalar(
                f'{env_name}/value_start_pos', start_position_value, self.num_timesteps + self.start_timestep
            )
            self.log_writer.add_scalar(
                f'{env_name}/info_gain_min', min_control_info_gain, self.num_timesteps + self.start_timestep
            )
            self.log_writer.add_scalar(
                f'{env_name}/info_gain_mean', mean_control_info_gain, self.num_timesteps + self.start_timestep
            )
            self.log_writer.add_scalar(
                f'{env_name}/info_gain_start_pos', start_position_control_info_gain, self.num_timesteps + self.start_timestep
            )
            self.log_writer.add_scalar(
                f'{env_name}/control_info_min', min_control_info, self.num_timesteps + self.start_timestep
            )
            self.log_writer.add_scalar(
                f'{env_name}/control_info_mean', mean_control_info, self.num_timesteps + self.start_timestep
            )
            self.log_writer.add_scalar(
                f'{env_name}/control_info_start_pos', start_position_control_info, self.num_timesteps + self.start_timestep
            )
