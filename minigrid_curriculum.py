import os
from typing import Dict, List

import torch
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

from binary_state_representation.binary2binaryautoencoder import Binary2BinaryEncoder, Binary2BinaryFeatureNet
from callbacks import EvalCallback, EvalSaveCallback, InfoEvalSaveCallback
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
            compute_info_freq: int,
            num_eval_episodes: int,
            eval_deterministic: bool,
            force_sequential: bool = False,
            start_time_step: int = 0,
            load_best: bool = False,
            iter_gamma: float = 0.999,
            iter_threshold: float = 1e-5,
            max_iter: int = 1e5,
            encoder: Binary2BinaryEncoder or None = None,
            keep_dims: int = -1
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
                if encoder is not None:
                    train_env = EncodingWrapper(train_env.env, encoder, device=torch.device("cpu"), keep_dims=keep_dims)
                    eval_env = EncodingWrapper(eval_env.env, encoder, device=torch.device("cpu"), keep_dims=keep_dims)
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
            if encoder is not None:
                target_env = EncodingWrapper(target_env.env, encoder, device=torch.device("cpu"), keep_dims=keep_dims)
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

            info_eval_callback = InfoEvalSaveCallback(
                eval_envs=eval_env_list,
                eval_env_names=eval_env_name_list,
                model=model,
                model_save_dir=model_save_dir,
                model_save_name=f"task_{i + 1}",
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
                keep_dims=keep_dims,
            )

            # callback_list = CallbackList(callbacks=[target_callback, eval_callback])
            callback_list = CallbackList(callbacks=[info_eval_callback,])

            # train
            model.learn(total_timesteps=train_env_step_list[i], callback=callback_list, progress_bar=True)

            # accumulated timestep
            start_time_step += info_eval_callback.num_timesteps

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
    WEIGHTS = {'inv': 0.3, 'dis': 0.3, 'neighbour': 0.3, 'dec': 0.0001, 'rwd': 0.05, 'terminate': 0.05}

    model = Binary2BinaryFeatureNet(NUM_ACTIONS, OBS_SPACE, n_latent_dims=LATENT_DIMS, lr=LR, weights=WEIGHTS,
                                    device=device, )
    model.load(r'experiments/learn_feature_corridor/model_epoch_19000.pth')
    encoder = model.encoder.to(device)
    # =====================================================

    # encoder = None  # test non encoding case
    runner = CurriculumRunner(task_configs, target_configs)
    runner.train(
        session_dir="./experiments/corridor_encode_16",
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
    runner.train(
        session_dir="./experiments/corridor_encode_10",
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
        keep_dims=10,
    )
    runner.train(
        session_dir="./experiments/corridor_not_encoded",
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
