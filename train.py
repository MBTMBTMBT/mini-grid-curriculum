import os
from typing import Dict, List

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from callbacks import EvalSaveCallback
from customPPO import MLPEncoderExtractor, CustomActorCriticPolicy, CustomPPO, \
    CNNEncoderExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList

from customize_minigrid.wrappers import FullyObsSB3MLPWrapper, FullyObsImageWrapper
from task_config import TaskConfig, make_env


class Trainer:
    def __init__(
            self,
            train_configs: List[TaskConfig],
            eval_configs: List[TaskConfig],
            policy_kwargs=None,
            output_wrapper=FullyObsSB3MLPWrapper,
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

        self.output_wrapper = output_wrapper

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
        # Prepare environments
        train_env_list = []
        train_env_step_list = []
        eval_env_list = []
        eval_env_name_list = []
        train_env_steps = []
        train_env_names = []

        for each_task_config in self.eval_configs:
            # Store evaluation environments and names
            eval_env_list.append(VecMonitor(DummyVecEnv([lambda: make_env(each_task_config, self.output_wrapper, self.max_minimum_display_size)])))
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
                    [lambda: make_env(each_task_config, self.output_wrapper, self.max_minimum_display_size) for each_task_config in train_tasks]))

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
                model = CustomPPO.load(load_path, env=env)
                model.log_dir = log_dir
            else:
                model = CustomPPO(
                    CustomActorCriticPolicy,
                    env=env,
                    # n_steps=8192,
                    # batch_size=32,
                    # n_epochs=10,
                    # clip_range=0.4,
                    # clip_range_vf=None,
                    policy_kwargs=self.policy_kwargs,
                    verbose=1,
                    log_dir=log_dir,
                )
                print("Initialized new model.")
                load_path = os.path.join(model_save_dir, f"saved_model_latest.zip")

            # info_eval_callback = InfoEvalSaveCallback(
            #     eval_envs=eval_env_list,
            #     eval_env_names=eval_env_name_list,
            #     model=model,
            #     model_save_dir=model_save_dir,
            #     model_save_name=f"saved_model",
            #     log_writer=log_writer,
            #     eval_freq=eval_freq // num_parallel,
            #     compute_info_freq=compute_info_freq // num_parallel,
            #     n_eval_episodes=num_eval_episodes,
            #     deterministic=eval_deterministic,
            #     verbose=1,
            #     start_timestep=start_time_step,
            #     iter_gamma=iter_gamma,
            #     iter_threshold=iter_threshold,
            #     max_iter=max_iter,
            # )

            eval_callback = EvalSaveCallback(
                eval_envs=eval_env_list,
                eval_env_names=eval_env_name_list,
                model_save_dir=model_save_dir,
                model_save_name=f"saved_model",
                log_writer=log_writer,
                eval_freq=eval_freq // num_parallel,
                n_eval_episodes=num_eval_episodes,
                deterministic=eval_deterministic,
                verbose=1,
                start_timestep=start_time_step,
            )

            # sigmoid_slope_manager_callback = SigmoidSlopeManagerCallback(
            #     feature_model=model.policy.features_extractor,
            #     total_train_steps=steps // num_parallel,
            #     log_writer=log_writer,
            #     start_timestep=start_time_step,
            # )

            model.policy.features_extractor.unfreeze()
            model.policy.unfreeze_mlp_extractor()

            # callback_list = CallbackList(callbacks=[target_callback, eval_callback])
            # callback_list = CallbackList(callbacks=[info_eval_callback, sigmoid_slope_manager_callback])
            # callback_list = CallbackList(callbacks=[eval_callback, sigmoid_slope_manager_callback])
            # callback_list = CallbackList(callbacks=[info_eval_callback,])
            callback_list = CallbackList(callbacks=[eval_callback])

            # train
            model.learn(total_timesteps=steps, callback=callback_list, progress_bar=True)

            # accumulated timestep
            start_time_step += eval_callback.num_timesteps

        # close the writer
        log_writer.close()


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
