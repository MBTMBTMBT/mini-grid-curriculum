import os
from typing import Dict, List

import torch
from stable_baselines3 import PPO
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from binary_state_representation.binary2binaryautoencoder import Binary2BinaryEncoder, Binary2BinaryFeatureNet
from callbacks import EvalCallback, EvalSaveCallback, InfoEvalSaveCallback
from customPPO import CustomEncoderExtractor, CustomActorCriticPolicy
from customize_minigrid.custom_env import CustomEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CallbackList

from customize_minigrid.wrappers import FullyObsSB3MLPWrapper
from minigrid_abstract_encoding import EncodingWrapper


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


class CurriculumRunner:
    def __init__(
            self,
            task_configs: List[TaskConfig],
            target_configs: List[TargetConfig],
            feature_encoder_net_arch: List[int],
            mlp_net_arch: List[int] or List[Dict],
    ):
        self.task_configs = task_configs
        self.target_configs = target_configs
        self.max_minimum_display_size: int = 0

        # add tasks, find the minimum display size for all
        for each_task_config in task_configs:
            if each_task_config.minimum_display_size > self.max_minimum_display_size:
                self.max_minimum_display_size = each_task_config.minimum_display_size
        for each_target_config in target_configs:
            if each_target_config.minimum_display_size > self.max_minimum_display_size:
                self.max_minimum_display_size = each_target_config.minimum_display_size

        self.policy_kwargs = dict(
            features_extractor_class=CustomEncoderExtractor,  # Use the custom encoder extractor
            features_extractor_kwargs=dict(
                net_arch=[128, 64],  # Custom layer sizes
                activation_fn=nn.ReLU  # Activation function
            ),
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],  # Policy and value network architecture
            activation_fn=nn.ReLU,
        )

    def train(
            self,
            session_dir: str,
            eval_freq: int,
            compute_info_freq: int,
            num_eval_episodes: int,
            eval_deterministic: bool,
            start_time_step: int = 0,
            load_path: str or None = None,
            iter_gamma: float = 0.999,
            iter_threshold: float = 1e-5,
            max_iter: int = 1e5,
    ):
        # prepare environments
        eval_env_list = []
        eval_env_name_list = []
        target_env_list = []
        target_env_name_list = []

        # # iterate through task dict to make sequence of environments.
        train_envs = []
        train_env_steps = []
        train_env_names = []
        vec_train_steps = 0
        for each_task_config in self.task_configs:
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
            vec_train_steps += each_task_config.train_total_steps
        vec_train_env = VecMonitor(DummyVecEnv([lambda: env for env in train_envs]))

        for each_target_config in self.target_configs:
            target_env = FullyObsSB3MLPWrapper(
                CustomEnv(
                    txt_file_path=each_target_config.txt_file_path,
                    display_size=self.max_minimum_display_size,
                    display_mode=each_target_config.display_mode,
                    random_rotate=each_target_config.random_rotate,
                    random_flip=each_target_config.random_flip,
                    custom_mission=each_target_config.custom_mission,
                    max_steps=each_target_config.max_steps,
                )
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

        if load_path and os.path.exists(load_path):
            model = PPO.load(load_path, env=vec_train_env)
        else:
            model = PPO(CustomActorCriticPolicy, env=vec_train_env, policy_kwargs=self.policy_kwargs, verbose=1)
            print("Initialized new model.")

        info_eval_callback = InfoEvalSaveCallback(
            eval_envs=eval_env_list,
            eval_env_names=eval_env_name_list,
            model=model,
            model_save_dir=model_save_dir,
            model_save_name=f"saved_model",
            log_writer=log_writer,
            eval_freq=eval_freq,
            compute_info_freq=compute_info_freq,
            n_eval_episodes=num_eval_episodes,
            deterministic=eval_deterministic,
            verbose=1,
            start_timestep=start_time_step,
            iter_gamma=iter_gamma,
            iter_threshold=iter_threshold,
            max_iter=max_iter,
            encoder=encoder,
        )

        model.policy.features_extractor.unfreeze()
        model.policy.unfreeze_mlp_extractor()

        # callback_list = CallbackList(callbacks=[target_callback, eval_callback])
        callback_list = CallbackList(callbacks=[info_eval_callback,])

        # train
        model.learn(total_timesteps=vec_train_steps, callback=callback_list, progress_bar=True)

        # start re-training by freezing the second half of the net and replace the first half.
        for eval_env, eval_env_name in zip(eval_env_list, eval_env_name_list):
            _model = PPO(CustomActorCriticPolicy, env=vec_train_env, policy_kwargs=self.policy_kwargs, verbose=1)
            _model.policy.mlp_extractor.load_state_dict(model.policy.mlp_extractor.state_dict())
            _model.policy.freeze_mlp_extractor()
            _info_eval_callback = InfoEvalSaveCallback(
                eval_envs=eval_env_list,
                eval_env_names=eval_env_name_list,
                model=model,
                model_save_dir=model_save_dir,
                model_save_name=f"saved_model",
                log_writer=log_writer,
                eval_freq=eval_freq,
                compute_info_freq=compute_info_freq,
                n_eval_episodes=num_eval_episodes,
                deterministic=eval_deterministic,
                verbose=1,
                start_timestep=start_time_step,
                iter_gamma=iter_gamma,
                iter_threshold=iter_threshold,
                max_iter=max_iter,
                encoder=encoder,
            )

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
    config.train_total_steps = 5e4
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
    config.train_total_steps = 5e4
    config.eval_display_mode = "middle"
    config.eval_random_rotate = False
    config.eval_random_flip = False
    config.eval_max_steps = 50
    config.difficulty_level = 1
    task_configs.append(config)

    config = TaskConfig()
    config.name = "extra_long_corridor"
    config.txt_file_path = r"./maps/extra_long_corridor.txt"
    config.custom_mission = "reach the goal"
    config.minimum_display_size = 11
    config.train_display_mode = "random"
    config.train_random_rotate = True
    config.train_random_flip = True
    config.train_max_steps = 500
    config.train_total_steps = 5e4
    config.eval_display_mode = "middle"
    config.eval_random_rotate = False
    config.eval_random_flip = False
    config.eval_max_steps = 50
    config.difficulty_level = 1
    task_configs.append(config)

    # config = TaskConfig()
    # config.name = "square_space"
    # config.txt_file_path = r"./maps/square_space.txt"
    # config.custom_mission = "reach the goal"
    # config.minimum_display_size = 7
    # config.train_display_mode = "random"
    # config.train_random_rotate = True
    # config.train_random_flip = True
    # config.train_max_steps = 1000
    # config.train_total_steps = 5e5
    # config.eval_display_mode = "middle"
    # config.eval_random_rotate = False
    # config.eval_random_flip = False
    # config.eval_max_steps = 50
    # config.difficulty_level = 2
    # task_configs.append(config)
    #
    # config = TaskConfig()
    # config.name = "small_maze"
    # config.txt_file_path = r"./maps/small_maze.txt"
    # config.custom_mission = "reach the goal"
    # config.minimum_display_size = 7
    # config.train_display_mode = "random"
    # config.train_random_rotate = True
    # config.train_random_flip = True
    # config.train_max_steps = 1000
    # config.train_total_steps = 5e5
    # config.eval_display_mode = "middle"
    # config.eval_random_rotate = False
    # config.eval_random_flip = False
    # config.eval_max_steps = 50
    # config.difficulty_level = 3
    # task_configs.append(config)
    #
    # config = TaskConfig()
    # config.name = "big_maze"
    # config.txt_file_path = r"./maps/big_maze.txt"
    # config.custom_mission = "reach the goal"
    # config.minimum_display_size = 13
    # config.train_display_mode = "random"
    # config.train_random_rotate = True
    # config.train_random_flip = True
    # config.train_max_steps = 2000
    # config.train_total_steps = 10e5
    # config.eval_display_mode = "middle"
    # config.eval_random_rotate = False
    # config.eval_random_flip = False
    # config.eval_max_steps = 100
    # config.difficulty_level = 4
    # task_configs.append(config)

    target_configs = []

    # ========= This section need to be optimized =========
    NUM_ACTIONS = 7
    OBS_SPACE = 2575
    LATENT_DIMS = 16
    LR = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    WEIGHTS = {'inv': 1.0, 'dis': 1.0, 'neighbour': 0.0, 'dec': 0.0, 'rwd': 0.1, 'terminate': 1.0}

    model = Binary2BinaryFeatureNet(NUM_ACTIONS, OBS_SPACE, n_latent_dims=LATENT_DIMS, lr=LR, weights=WEIGHTS,
                                    device=device, )
    model.load(r'experiments/learn_feature_corridor_16/model_epoch_200.pth')
    encoder = model.encoder.to(device)
    # =====================================================

    # encoder = None  # test non encoding case
    runner = CurriculumRunner(task_configs, target_configs)
    runner.train(
        session_dir="./experiments/train_16",
        eval_freq=int(5e3),
        compute_info_freq=int(5e3),
        num_eval_episodes=50,
        eval_deterministic=False,
        force_sequential=True,
        start_time_step=0,
        iter_gamma=0.999,
        iter_threshold=1e-5,
        max_iter=int(1e5),
        encoder=encoder,
        keep_dims=16,
    )

    # ========= This section need to be optimized =========
    NUM_ACTIONS = 7
    OBS_SPACE = 2575
    LATENT_DIMS = 24
    LR = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    WEIGHTS = {'inv': 1.0, 'dis': 1.0, 'neighbour': 0.0, 'dec': 0.0, 'rwd': 0.1, 'terminate': 1.0}

    model = Binary2BinaryFeatureNet(NUM_ACTIONS, OBS_SPACE, n_latent_dims=LATENT_DIMS, lr=LR, weights=WEIGHTS,
                                    device=device, )
    model.load(r'experiments/learn_feature_corridor_24/model_epoch_200.pth')
    encoder = model.encoder.to(device)
    # =====================================================

    # encoder = None  # test non encoding case
    runner = CurriculumRunner(task_configs, target_configs)
    runner.train(
        session_dir="./experiments/train_24",
        eval_freq=int(5e3),
        compute_info_freq=int(5e3),
        num_eval_episodes=50,
        eval_deterministic=False,
        force_sequential=True,
        start_time_step=0,
        iter_gamma=0.999,
        iter_threshold=1e-5,
        max_iter=int(1e5),
        encoder=encoder,
        keep_dims=24,
    )

    # ========= This section need to be optimized =========
    NUM_ACTIONS = 7
    OBS_SPACE = 2575
    LATENT_DIMS = 32
    LR = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    WEIGHTS = {'inv': 1.0, 'dis': 1.0, 'neighbour': 0.0, 'dec': 0.0, 'rwd': 0.1, 'terminate': 1.0}

    model = Binary2BinaryFeatureNet(NUM_ACTIONS, OBS_SPACE, n_latent_dims=LATENT_DIMS, lr=LR, weights=WEIGHTS,
                                    device=device, )
    model.load(r'experiments/learn_feature_corridor_32/model_epoch_200.pth')
    encoder = model.encoder.to(device)
    # =====================================================

    # encoder = None  # test non encoding case
    runner = CurriculumRunner(task_configs, target_configs)
    runner.train(
        session_dir="./experiments/train_32",
        eval_freq=int(5e3),
        compute_info_freq=int(5e3),
        num_eval_episodes=50,
        eval_deterministic=False,
        force_sequential=True,
        start_time_step=0,
        iter_gamma=0.999,
        iter_threshold=1e-5,
        max_iter=int(1e5),
        encoder=encoder,
        keep_dims=32,
    )

    # encoder = None  # test non encoding case
    runner = CurriculumRunner(task_configs, target_configs)
    runner.train(
        session_dir="./experiments/train_no_encode",
        eval_freq=int(5e3),
        compute_info_freq=int(5e3),
        num_eval_episodes=50,
        eval_deterministic=False,
        force_sequential=True,
        start_time_step=0,
        iter_gamma=0.999,
        iter_threshold=1e-5,
        max_iter=int(1e5),
        encoder=None,
        keep_dims=-1,
    )

