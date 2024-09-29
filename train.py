import os
from typing import Dict, List, Tuple, Optional

import torch
from stable_baselines3 import PPO
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from binary_state_representation.binary2binaryautoencoder import Binary2BinaryEncoder, Binary2BinaryFeatureNet
from callbacks import EvalCallback, EvalSaveCallback, InfoEvalSaveCallback, SigmoidSlopeManagerCallback
from customPPO import MLPEncoderExtractor, CustomActorCriticPolicy, TransformerEncoderExtractor, CustomPPO
from customize_minigrid.custom_env import CustomEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList

from customize_minigrid.wrappers import FullyObsSB3MLPWrapper
from minigrid_abstract_encoding import EncodingWrapper


class TaskConfig:
    def __init__(self):
        self.name = ""

        self.txt_file_path: Optional[str] = ""
        self.rand_gen_shape: Tuple[int, int] = (5, 5)
        self.custom_mission: str = ""

        self.minimum_display_size: int = 0

        self.display_mode: str = "random"
        self.random_rotate: bool = False
        self.random_flip: bool = False
        self.max_steps: int = 0
        self.start_pos: Tuple[int, int] or None = None
        self.start_dir: Optional[int] = None

        self.train_total_steps: int = 0
        self.difficulty_level: int = 0

    def clone(self, config: "TaskConfig") -> "TaskConfig":
        new = TaskConfig()
        new.name = config.name
        new.txt_file_path = config.txt_file_path
        new.rand_gen_shape = config.rand_gen_shape
        new.custom_mission = config.custom_mission
        new.minimum_display_size = config.minimum_display_size
        new.display_mode = config.display_mode
        new.random_rotate = config.random_rotate
        new.random_flip = config.random_flip
        new.max_steps = config.max_steps
        new.start_pos = config.start_pos
        new.start_dir = config.start_dir
        new.train_total_steps = config.train_total_steps
        new.difficulty_level = config.difficulty_level
        return new


class Trainer:
    def __init__(
            self,
            train_configs: List[TaskConfig],
            eval_configs: List[TaskConfig],
            policy_kwargs=None,
    ):
        self.train_configs = train_configs
        self.eval_configs = eval_configs
        self.max_minimum_display_size: int = 0
        self.train_task_dict: Dict[int, List[TaskConfig]] = dict()

        # add tasks, find the minimum display size for all
        for each_task_config in train_configs + eval_configs:
            if each_task_config.minimum_display_size > self.max_minimum_display_size:
                self.max_minimum_display_size = each_task_config.minimum_display_size
            if each_task_config in train_configs:
                self.train_task_dict.setdefault(each_task_config.difficulty_level, []).append(each_task_config)

        if policy_kwargs is None:
            self.policy_kwargs = dict(
                features_extractor_class=MLPEncoderExtractor,  # Use the custom encoder extractor
                features_extractor_kwargs=dict(
                    net_arch=[128, 64],  # Custom layer sizes
                    activation_fn=nn.LeakyReLU  # Activation function
                ),
                net_arch=[dict(pi=[64, 64], vf=[64, 64])],  # Policy and value network architecture
                activation_fn=nn.LeakyReLU,
            )
        else:
            self.policy_kwargs = policy_kwargs

    def train(
            self,
            session_dir: str,
            eval_freq: int,
            compute_info_freq: int,
            num_eval_episodes: int,
            eval_deterministic: bool,
            num_parallel: int,
            start_time_step: int = 0,
            load_path: str or None = None,
            iter_gamma: float = 0.999,
            iter_threshold: float = 1e-5,
            max_iter: int = 1e5,
    ):
        # Helper function to create a new environment instance for training, evaluation, or target
        def make_env(env_config):
            """
            Create a new environment instance based on the configuration and environment type.

            :param env_config: Configuration object for the environment
            :param env_type: A string indicating the type of environment: "train", "eval", or "target"
            :return: A new environment instance
            """
            return FullyObsSB3MLPWrapper(
                CustomEnv(
                    txt_file_path=env_config.txt_file_path,
                    rand_gen_shape=env_config.rand_gen_shape,
                    display_size=self.max_minimum_display_size,
                    display_mode=env_config.display_mode,
                    random_rotate=env_config.random_rotate,
                    random_flip=env_config.random_flip,
                    custom_mission=env_config.custom_mission,
                    max_steps=env_config.max_steps,
                )
            )

        # Prepare environments
        train_env_list = []
        train_env_step_list = []
        eval_env_list = []
        eval_env_name_list = []
        train_env_steps = []
        train_env_names = []

        for each_task_config in self.eval_configs:
            # Store evaluation environments and names
            eval_env_list.append(VecMonitor(DummyVecEnv([lambda: make_env(each_task_config)])))
            eval_env_name_list.append(each_task_config.name)

        for difficulty_level in sorted(self.train_task_dict.keys()):
            train_tasks = self.train_task_dict[difficulty_level]
            vec_train_steps = 0

            # Iterate through task configs to make sequences of training and evaluation environments
            for each_task_config in train_tasks:

                # Store training environments and names
                train_env_steps.append(each_task_config.train_total_steps)
                train_env_names.append(each_task_config.name)

                vec_train_steps += each_task_config.train_total_steps

            # Create VecMonitor for the combined training environments
            vec_train_env = VecMonitor(
                SubprocVecEnv(
                    [lambda: make_env(each_task_config) for each_task_config in train_tasks]))

            train_env_list.append(vec_train_env)
            train_env_step_list.append(vec_train_steps)

        # prepare workspace and log writer
        os.makedirs(session_dir, exist_ok=True)
        log_dir = os.path.join(session_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        model_save_dir = os.path.join(session_dir, "saved_models")
        os.makedirs(model_save_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir)

        for i in range(len(train_env_list)):
            print(f"Training stage [{i + 1} / {len(train_env_list)}]:")
            print(f"Time step starts from: {start_time_step}...")

            # get environment for this round
            env = train_env_list[i]
            steps = train_env_step_list[i] // num_parallel

            if load_path and os.path.exists(load_path):
                print(f"Loading the model from {load_path}...")
                model = PPO.load(load_path, env=env)
                model.log_dir = log_dir
            else:
                model = PPO(CustomActorCriticPolicy, env=env, policy_kwargs=self.policy_kwargs, verbose=1, batch_size=4)
                print("Initialized new model.")
                load_path = os.path.join(model_save_dir, f"saved_model_latest.zip")

            info_eval_callback = InfoEvalSaveCallback(
                eval_envs=eval_env_list,
                eval_env_names=eval_env_name_list,
                model=model,
                model_save_dir=model_save_dir,
                model_save_name=f"saved_model",
                log_writer=log_writer,
                eval_freq=eval_freq // num_parallel,
                compute_info_freq=compute_info_freq // num_parallel,
                n_eval_episodes=num_eval_episodes,
                deterministic=eval_deterministic,
                verbose=1,
                start_timestep=start_time_step,
                iter_gamma=iter_gamma,
                iter_threshold=iter_threshold,
                max_iter=max_iter,
            )

            sigmoid_slope_manager_callback = SigmoidSlopeManagerCallback(
                feature_model=model.policy.features_extractor,
                total_train_steps=steps // num_parallel,
                log_writer=log_writer,
                start_timestep=start_time_step,
            )

            model.policy.features_extractor.unfreeze()
            model.policy.unfreeze_mlp_extractor()

            # callback_list = CallbackList(callbacks=[target_callback, eval_callback])
            callback_list = CallbackList(callbacks=[info_eval_callback, sigmoid_slope_manager_callback])

            # train
            model.learn(total_timesteps=steps, callback=callback_list, progress_bar=True)

            # accumulated timestep
            start_time_step += info_eval_callback.num_timesteps

        # close the writer
        log_writer.close()


if __name__ == '__main__':
    train_configs = []
    eval_configs = []
    num_parallel: int = 12

    ##################################################################
    config = TaskConfig()
    config.name = "6"
    config.rand_gen_shape = (6, 6)
    config.txt_file_path = None
    config.custom_mission = "reach the goal"
    config.minimum_display_size = 6
    config.display_mode = "random"
    config.random_rotate = True
    config.random_flip = True
    config.max_steps = 500
    config.train_total_steps = 5e7
    config.difficulty_level = 0
    for _ in range(num_parallel):
        train_configs.append(config)

    config = TaskConfig()
    config.name = "6-1"
    config.rand_gen_shape = None
    config.txt_file_path = r"./maps/6-1.txt"
    config.custom_mission = "reach the goal"
    config.minimum_display_size = 6
    config.display_mode = "middle"
    config.random_rotate = True
    config.random_flip = True
    config.max_steps = 50
    config.start_pos = (6, 6)
    config.start_dir = 1
    eval_configs.append(config)

    config = TaskConfig()
    config.name = "6-2"
    config.rand_gen_shape = None
    config.txt_file_path = r"./maps/6-2.txt"
    config.custom_mission = "reach the goal"
    config.minimum_display_size = 6
    config.display_mode = "middle"
    config.random_rotate = True
    config.random_flip = True
    config.max_steps = 50
    config.start_pos = (6, 6)
    config.start_dir = 1
    eval_configs.append(config)

    config = TaskConfig()
    config.name = "6-3"
    config.rand_gen_shape = None
    config.txt_file_path = r"./maps/6-3.txt"
    config.custom_mission = "reach the goal"
    config.minimum_display_size = 6
    config.display_mode = "middle"
    config.random_rotate = True
    config.random_flip = True
    config.max_steps = 50
    config.start_pos = (6, 6)
    config.start_dir = 1
    eval_configs.append(config)

    config = TaskConfig()
    config.name = "6-4"
    config.rand_gen_shape = None
    config.txt_file_path = r"./maps/6-4.txt"
    config.custom_mission = "reach the goal"
    config.minimum_display_size = 6
    config.display_mode = "middle"
    config.random_rotate = True
    config.random_flip = True
    config.max_steps = 50
    config.start_pos = (6, 6)
    config.start_dir = 1
    eval_configs.append(config)

    ##################################################################

    # encoder = None  # test non encoding case
    for i in range(3):
        runner = Trainer(
            train_configs,
            eval_configs,
            policy_kwargs=dict(
                features_extractor_class=TransformerEncoderExtractor,  # Use the custom encoder extractor
                features_extractor_kwargs=dict(
                    net_arch=[6],  # Custom layer sizes
                    num_transformer_layers=2,
                    n_heads=8,
                    activation_fn=nn.LeakyReLU,  # Activation function
                    encoder_only=True,
                ),
                net_arch=dict(pi=[6, 128, 128], vf=[6, 128, 128]),  # Policy and value network architecture
                activation_fn=nn.LeakyReLU,
            )
        )
        runner.train(
            session_dir=f"./experiments/mazes/run{i}",
            eval_freq=int(50e4),
            compute_info_freq=int(50e4),
            num_eval_episodes=10,
            eval_deterministic=False,
            start_time_step=0,
            iter_gamma=0.999,
            iter_threshold=1e-5,
            max_iter=int(1e5),
            num_parallel=num_parallel,
        )
