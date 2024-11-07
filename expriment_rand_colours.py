import mlflow

from experiment_helpers import *


import os
from typing import Dict, List, Optional

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from callbacks import EvalSaveCallback
from customPPO import MLPEncoderExtractor, CustomActorCriticPolicy, CustomPPO, \
    CNNEncoderExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList

from customize_minigrid.wrappers import *
from task_config import TaskConfig, make_env


class RandColourTrainerConfig:
    def __init__(
            self,
            session_dir: str = "_experiment",
            num_models: int = 10,
            num_parallel: int = 8,
            init_seed: int = 0,
            eval_freq: int = 10000,
            num_eval_episodes: int = 20,
            eval_deterministic: bool = False,
            policy_kwargs: Optional[dict] = None,
            output_wrapper = FullyObsImageWrapper,
    ):
        self.session_dir = session_dir
        self.train_config = TaskConfig()  # Assuming TaskConfig is defined elsewhere
        self.eval_configs: List[TaskConfig] = []
        self.num_models = num_models
        self.num_parallel = num_parallel
        self.init_seed = init_seed
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.eval_deterministic = eval_deterministic
        self.policy_kwargs = policy_kwargs
        self.output_wrapper = output_wrapper


def train(
        trainer_config: RandColourTrainerConfig,
):
    trainer_config = trainer_config
    model_save_dir = os.path.join(trainer_config.session_dir, "saved_models")
    os.makedirs(model_save_dir, exist_ok=True)
    log_dir = os.path.join(trainer_config.session_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    mlflow.set_tracking_uri(f'file://{os.path.abspath(log_dir)}')
    mlflow.set_experiment(trainer_config.session_dir)

    max_minimum_display_size = 0

    # add tasks, find the minimum display size for all
    for each_task_config in [trainer_config.train_config] + trainer_config.eval_configs:
        if each_task_config.minimum_display_size > max_minimum_display_size:
            max_minimum_display_size = each_task_config.minimum_display_size

    train_configs = []
    for _ in range(trainer_config.num_parallel):
        train_configs.append(trainer_config.train_config)

    vec_train_env = VecMonitor(
        SubprocVecEnv([
            lambda: make_env(
                each_task_config,
                trainer_config.output_wrapper,
                max_minimum_display_size
            ) for each_task_config in train_configs
        ])
    )

    eval_env_list = []
    eval_env_name_list = []
    for each_task_config in trainer_config.eval_configs:
        # Store evaluation environments and names
        eval_env_list.append(VecMonitor(
            DummyVecEnv([
                lambda: make_env(
                    each_task_config,
                    trainer_config.output_wrapper,
                    max_minimum_display_size
                )]
            ))
        )
        eval_env_name_list.append(each_task_config.name)

if __name__ == '__main__':
    train_configs = []
    eval_configs = []
    # num_parallel: int = 8

    ##################################################################
    for i in range(1, 2):
        config = TaskConfig()
        config.name = f"7-{i}"
        config.rand_gen_shape = None
        config.txt_file_path = f"./maps/7-{i}.txt"
        config.custom_mission = "reach the goal"
        config.minimum_display_size = 7
        config.display_mode = "random"
        config.random_rotate = True
        config.random_flip = True
        config.max_steps = 1024
        config.start_pos = (5, 5)
        config.train_total_steps = 2.5e7
        config.difficulty_level = 0
        config.add_random_door_key=False
        train_configs.append(config)
    # for _ in range(num_parallel):
    #     train_configs.append(config)

    for i in range(1, 7):
        config = TaskConfig()
        config.name = f"7-{i}"
        config.rand_gen_shape = None
        config.txt_file_path = f"./maps/7-{i}.txt"
        config.custom_mission = "reach the goal"
        config.minimum_display_size = 7
        config.display_mode = "middle"
        config.random_rotate = True
        config.random_flip = True
        config.max_steps = 50
        config.start_pos = (5, 5)
        config.start_dir = 1
        eval_configs.append(config)

    ##################################################################

    # encoder = None  # test non encoding case
    for i in range(3):
        runner = Trainer(
            train_configs,
            eval_configs,
            policy_kwargs=dict(
                # features_extractor_class=CNNVectorQuantizerEncoderExtractor,  # Use the custom encoder extractor
                # features_extractor_kwargs=dict(
                #     net_arch=[],  # Custom layer sizes
                #     cnn_net_arch=[
                #         (64, 3, 2, 1),
                #         (64, 3, 2, 1),
                #         (64, 3, 2, 1),
                #         (64, 3, 2, 1),
                #         (64, 3, 2, 1),
                #     ],
                #     embedding_dim=16,
                #     num_embeddings=4096,
                #     activation_fn=nn.LeakyReLU,  # Activation function
                #     encoder_only=True,
                # ),
                features_extractor_class=CNNEncoderExtractor,  # Use the custom encoder extractor
                features_extractor_kwargs=dict(
                    net_arch=[32],  # Custom layer sizes
                    cnn_net_arch=[
                        (64, 3, 2, 1),
                        (128, 3, 2, 1),
                        (256, 3, 2, 1),
                        (512, 3, 2, 1),
                        (1024, 3, 2, 1),
                    ],
                    activation_fn=nn.LeakyReLU,  # Activation function
                    encoder_only=True,
                ),
                net_arch=dict(pi=[32, 32], vf=[32, 32]),  # Policy and value network architecture
                activation_fn=nn.LeakyReLU,
            ),
            output_wrapper=FullyObsImageWrapper,
        )
        runner.train(
            session_dir=f"./experiments/mazes-bin-32/run{i}",
            eval_freq=int(50e4),
            compute_info_freq=int(50e4),
            num_eval_episodes=50,
            eval_deterministic=False,
            start_time_step=0,
            iter_gamma=0.999,
            iter_threshold=5e-5,
            max_iter=int(1e5),
            num_parallel=8,
        )

