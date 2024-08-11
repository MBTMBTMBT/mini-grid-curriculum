import os
import numpy as np
from typing import Dict, List, Union

from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import PolicyPredictor
from torch.utils.tensorboard import SummaryWriter
from customize_minigrid.custom_env import CustomEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv
from stable_baselines3.common.callbacks import EventCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

from customize_minigrid.wrappers import FullyObsSB3MLPWrapper


class TaskConfig:
    def __init__(self):
        self.name = ""

        self.txt_file_path: str = ""
        self.custom_mission: str = ""

        self.minimum_display_size: int = 0

        self.train_display_mode: str = "random"
        self.train_random_rotate: bool = True
        self.train_random_flip: bool = True
        self.train_max_steps: int = 0
        self.train_total_steps: int = 0

        self.eval_display_mode: str = "middle"
        self.eval_random_rotate: bool = False
        self.eval_random_flip: bool = False
        self.eval_max_steps: int = 0

        self.difficulty_level: int = 0


class TargetConfig:
    def __init__(self):
        self.name = ""

        self.txt_file_path: str = ""
        self.custom_mission: str = ""

        self.minimum_display_size: int = 0

        self.display_mode: str = "middle"
        self.random_rotate: bool = False
        self.random_flip: bool = False
        self.max_steps: int = 0


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


class CurriculumRunner:
    def __init__(
            self,
            task_configs: List[TaskConfig],
            target_configs: List[TargetConfig],
    ):
        self.task_dict: Dict[int, List[TaskConfig]] = dict()
        self.target_configs = target_configs
        self.max_minimum_display_size: int = 0

        # add tasks, find the minimum display size for all
        for each_task_config in task_configs:
            if each_task_config.minimum_display_size > self.max_minimum_display_size:
                self.max_minimum_display_size = each_task_config.minimum_display_size
            self.task_dict.setdefault(each_task_config.difficulty_level, []).append(each_task_config)
        for each_target_config in target_configs:
            if each_target_config.minimum_display_size > self.max_minimum_display_size:
                self.max_minimum_display_size = each_target_config.minimum_display_size

    def train(
            self,
            session_dir: str,
            eval_freq: int,
            num_eval_episodes: int,
            eval_deterministic: bool,
            force_sequential: bool = False,
            start_time_step: int = 0,
            load_best: bool = False
    ):
        """
        Trainer function for mini-grid curriculum
        Args:
            session_dir: directory to save the trained model and logs
            eval_freq: frequency of evaluation steps
            force_sequential: if True, make all tasks sequentially even with same difficulty level

        Returns:

        """
        # prepare environments
        train_env_list = []
        train_env_step_list = []
        # train_env_name_list = []
        eval_env_list = []
        eval_env_name_list = []
        target_env_list = []
        target_env_name_list = []

        # iterate through task dict to make sequence of environments.
        for difficulty_level in sorted(self.task_dict.keys()):
            tasks = self.task_dict[difficulty_level]
            train_envs = []
            train_env_steps = []
            train_env_names = []
            for each_task_config in tasks:
                train_env = FullyObsSB3MLPWrapper(
                    CustomEnv(
                        txt_file_path=each_task_config.txt_file_path,
                        display_size=self.max_minimum_display_size,
                        display_mode=each_task_config.train_display_mode,
                        random_rotate=each_task_config.train_random_rotate,
                        random_flip=each_task_config.train_random_flip,
                        custom_mission=each_task_config.custom_mission,
                        max_steps=each_task_config.train_max_steps,
                    )
                )
                eval_env = FullyObsSB3MLPWrapper(
                    CustomEnv(
                        txt_file_path=each_task_config.txt_file_path,
                        display_size=self.max_minimum_display_size,
                        display_mode=each_task_config.eval_display_mode,
                        random_rotate=each_task_config.eval_random_rotate,
                        random_flip=each_task_config.eval_random_flip,
                        custom_mission=each_task_config.custom_mission,
                        max_steps=each_task_config.eval_max_steps,
                    )
                )
                train_envs.append(train_env)
                train_env_steps.append(each_task_config.train_total_steps)
                train_env_names.append(each_task_config.name)
                eval_env_list.append(VecMonitor(DummyVecEnv([lambda: eval_env])))
                eval_env_name_list.append(each_task_config.name)
            if force_sequential:
                for env in train_envs:
                    train_env_list.append(VecMonitor(DummyVecEnv([lambda: env])))
                for steps in train_env_steps:
                    train_env_step_list.append(steps)
                # for name in train_env_names:
                #     train_env_name_list.append(name)
            else:
                train_env_list.append(VecMonitor(DummyVecEnv([lambda: env for env in train_envs])))
                train_env_step_list.append(sum(train_env_steps))
                # train_env_name_list.append(train_env_names)
        for each_target_config in self.target_configs:
            target_env = CustomEnv(
                txt_file_path=each_target_config.txt_file_path,
                display_size=self.max_minimum_display_size,
                display_mode=each_target_config.display_mode,
                random_rotate=each_target_config.random_rotate,
                random_flip=each_target_config.random_flip,
                custom_mission=each_target_config.custom_mission,
                max_steps=each_target_config.max_steps,
            )
            target_env_list.append(VecMonitor(DummyVecEnv([lambda: target_env])))
            target_env_name_list.append(each_target_config.name)

        # prepare workspace and log writer
        os.makedirs(session_dir, exist_ok=True)
        log_dir = os.path.join(session_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        model_save_dir = os.path.join(session_dir, "saved_models")
        os.makedirs(model_save_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir)

        # start training
        for i in range(len(train_env_list)):
            print(f"Training stage [{i+1} / {len(train_env_list)}]:")
            print(f"Time step starts from: {start_time_step}...")
            # make call back function for testing.
            target_callback = EvalCallback(
                eval_envs=target_env_list,
                eval_env_names=target_env_name_list,
                log_writer=log_writer,
                eval_freq=eval_freq,
                n_eval_episodes=num_eval_episodes,
                deterministic=eval_deterministic,
                verbose=1,
                start_timestep=start_time_step,
            )
            eval_callback = EvalSaveCallback(
                eval_envs=eval_env_list,
                eval_env_names=eval_env_name_list,
                model_save_dir=model_save_dir,
                model_save_name=f"task_{i+1}",
                log_writer=log_writer,
                eval_freq=eval_freq,
                n_eval_episodes=num_eval_episodes,
                deterministic=eval_deterministic,
                verbose=1,
                start_timestep=start_time_step,
            )

            callback_list = CallbackList(callbacks=[target_callback, eval_callback])

            # get environment for this round
            env = train_env_list[i]

            # Load or create a new model
            best_path = os.path.join(model_save_dir, f"task_{i}_best.zip")
            latest_path = os.path.join(model_save_dir, f"task_{i}_latest.zip")
            if os.path.exists(best_path):
                if load_best:
                    model = PPO.load(best_path, env=env)
                    print(f"Loaded model from {best_path}.")
                else:
                    model = PPO.load(latest_path, env=env)
                    print(f"Loaded model from {latest_path}.")
            else:
                model = PPO("MlpPolicy", env, verbose=1)
                print("Initialized new model.")
            pass

            # train
            model.learn(total_timesteps=train_env_step_list[i], callback=callback_list, progress_bar=True)

            # accumulated timestep
            start_time_step += eval_callback.num_timesteps

        # close the writer
        log_writer.close()


if __name__ == '__main__':
    task_configs = []

    config = TaskConfig()
    config.name = "short_corridor"
    config.txt_file_path = r"./maps/short_corridor.txt"
    config.custom_mission = "reach the goal"
    config.minimum_display_size = 6
    config.train_display_mode = "random"
    config.train_random_rotate = True
    config.train_random_flip = True
    config.train_max_steps = 500
    config.train_total_steps = 5e5
    config.eval_display_mode = "middle"
    config.eval_random_rotate = False
    config.eval_random_flip = False
    config.eval_max_steps = 50
    config.difficulty_level = 0
    task_configs.append(config)

    config = TaskConfig()
    config.name = "long_corridor"
    config.txt_file_path = r"./maps/long_corridor.txt"
    config.custom_mission = "reach the goal"
    config.minimum_display_size = 9
    config.train_display_mode = "random"
    config.train_random_rotate = True
    config.train_random_flip = True
    config.train_max_steps = 500
    config.train_total_steps = 5e5
    config.eval_display_mode = "middle"
    config.eval_random_rotate = False
    config.eval_random_flip = False
    config.eval_max_steps = 50
    config.difficulty_level = 1
    task_configs.append(config)

    config = TaskConfig()
    config.name = "square_space"
    config.txt_file_path = r"./maps/square_space.txt"
    config.custom_mission = "reach the goal"
    config.minimum_display_size = 7
    config.train_display_mode = "random"
    config.train_random_rotate = True
    config.train_random_flip = True
    config.train_max_steps = 1000
    config.train_total_steps = 5e5
    config.eval_display_mode = "middle"
    config.eval_random_rotate = False
    config.eval_random_flip = False
    config.eval_max_steps = 50
    config.difficulty_level = 2
    task_configs.append(config)

    config = TaskConfig()
    config.name = "small_maze"
    config.txt_file_path = r"./maps/small_maze.txt"
    config.custom_mission = "reach the goal"
    config.minimum_display_size = 7
    config.train_display_mode = "random"
    config.train_random_rotate = True
    config.train_random_flip = True
    config.train_max_steps = 1000
    config.train_total_steps = 5e5
    config.eval_display_mode = "middle"
    config.eval_random_rotate = False
    config.eval_random_flip = False
    config.eval_max_steps = 50
    config.difficulty_level = 3
    task_configs.append(config)

    config = TaskConfig()
    config.name = "big_maze"
    config.txt_file_path = r"./maps/big_maze.txt"
    config.custom_mission = "reach the goal"
    config.minimum_display_size = 13
    config.train_display_mode = "random"
    config.train_random_rotate = True
    config.train_random_flip = True
    config.train_max_steps = 2000
    config.train_total_steps = 10e5
    config.eval_display_mode = "middle"
    config.eval_random_rotate = False
    config.eval_random_flip = False
    config.eval_max_steps = 100
    config.difficulty_level = 4
    task_configs.append(config)

    target_configs = []

    runner = CurriculumRunner(task_configs, target_configs)
    runner.train(
        session_dir="./experiments/curriculum_example",
        eval_freq=int(5e3),
        num_eval_episodes=100,
        eval_deterministic=False,
        force_sequential=True,
        start_time_step=0,
    )
