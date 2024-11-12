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
            train_config = TaskConfig(),
            eval_configs: List[TaskConfig] = [],
            train_output_wrapper = FullyObsImageWrapper,
            eval_output_wrappers: List = [],
    ):
        self.session_dir = session_dir
        self.train_config = train_config  # Assuming TaskConfig is defined elsewhere
        self.eval_configs = eval_configs
        self.num_models = num_models
        self.num_parallel = num_parallel
        self.init_seed = init_seed
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.eval_deterministic = eval_deterministic
        self.policy_kwargs = policy_kwargs
        self.train_output_wrapper = train_output_wrapper
        self.eval_output_wrappers = eval_output_wrappers


def train(
        trainer_config: RandColourTrainerConfig,
):
    trainer_config = trainer_config
    model_save_dir = os.path.join(trainer_config.session_dir, "saved_models")
    os.makedirs(model_save_dir, exist_ok=True)
    log_dir = os.path.join(trainer_config.session_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir)

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
                trainer_config.train_output_wrapper,
                max_minimum_display_size
            ) for each_task_config in train_configs
        ])
    )

    eval_env_list = []
    eval_env_name_list = []
    for each_task_config, eval_wrapper in zip(trainer_config.eval_configs, trainer_config.eval_output_wrappers):
        # Store evaluation environments and names
        eval_env_list.append(VecMonitor(
            DummyVecEnv([
                lambda: make_env(
                    each_task_config,
                    eval_wrapper,
                    max_minimum_display_size
                )]
            ))
        )
        eval_env_name_list.append(each_task_config.name)

    callbacks = []
    for i in range(trainer_config.num_models):
        print(f"Training model [{i + 1} / {trainer_config.num_models}]:")
        steps = trainer_config.train_config.train_total_steps
        callback = EvalSaveCallback(
            eval_envs=eval_env_list,
            eval_env_names=[f"{name}_{i}" for name in eval_env_name_list],
            model_save_dir=model_save_dir,
            model_save_name=f"saved_model_{i}",
            log_writer=log_writer,
            eval_freq=trainer_config.eval_freq // trainer_config.num_parallel,
            n_eval_episodes=trainer_config.num_eval_episodes,
            deterministic=trainer_config.eval_deterministic,
            verbose=1,
        )
        callbacks.append(callback)

        model = CustomPPO(
            CustomActorCriticPolicy,
            env=vec_train_env,
            policy_kwargs=trainer_config.policy_kwargs,
            verbose=1,
            log_dir=log_dir,
        )

        reinitialize_model(model.policy.features_extractor, seed=trainer_config.init_seed)
        print("Initialized feature extractor with hash:", hash_model(model.policy.features_extractor))
        reinitialize_model(model.policy.mlp_extractor, seed=trainer_config.init_seed)
        print("Initialized mlp extractor with hash:", hash_model(model.policy.mlp_extractor))

        model.policy.features_extractor.unfreeze()
        model.policy.unfreeze_mlp_extractor()

        model.learn(
            total_timesteps=steps,
            callback=CallbackList([callback]),
            progress_bar=True,
        )
        model.save(os.path.join(model_save_dir, f"saved_model_{i}.zip"))
        print("Finished training model.")

    # Dictionary to collect rewards by base environment name and step
    env_step_rewards = {}

    # Iterate over each callback instance
    for callback in callbacks:
        for full_env_name, steps_rewards in callback.rewards_dict.items():
            # Extract the base environment name by removing the index suffix
            base_env_name = "_".join(full_env_name.split("_")[:-1])  # Assumes the suffix is always of the form _i

            if base_env_name not in env_step_rewards:
                env_step_rewards[base_env_name] = {}
            for step, rewards in steps_rewards:
                if step not in env_step_rewards[base_env_name]:
                    env_step_rewards[base_env_name][step] = []
                env_step_rewards[base_env_name][step].extend(rewards)

    # Calculate and log mean and std for each base environment and step
    for base_env_name, steps_rewards in env_step_rewards.items():
        for step, rewards in sorted(steps_rewards.items()):
            mean_reward = np.mean(rewards)
            std_dev_reward = np.std(rewards)

            # Log to TensorBoard with the current step for each base environment
            log_writer.add_scalar(f'{base_env_name}/Mean_Reward', mean_reward, step)
            log_writer.add_scalar(f'{base_env_name}/Std_Dev_Reward', std_dev_reward, step)

            # Print the computed results
            print(
                f"Base Environment: {base_env_name}, Step: {step}, Mean Reward: {mean_reward:.2f}, Std Dev: {std_dev_reward:.2f}")

    # Close TensorBoard writer
    log_writer.close()


if __name__ == '__main__':
    config = TaskConfig()
    config.name = f"small_maze"
    config.rand_gen_shape = None
    config.txt_file_path = f"./maps/small_maze.txt"
    config.custom_mission = "reach the goal"
    config.minimum_display_size = 5
    config.display_mode = "middle"
    config.random_rotate = False
    config.random_flip = False
    config.max_steps = 4096
    config.start_pos = None
    config.start_dir = None
    config.train_total_steps = int(500e3)
    config.difficulty_level = 0
    config.add_random_door_key=False
    train_config = config

    eval_configs = []
    eval_wrappers = []
    config = TaskConfig()
    config.name = f"small_maze"
    config.rand_gen_shape = None
    config.txt_file_path = f"./maps/small_maze.txt"
    config.custom_mission = "reach the goal"
    config.minimum_display_size = 5
    config.display_mode = "middle"
    config.random_rotate = False
    config.random_flip = False
    config.max_steps = 256
    config.start_pos = (1, 1)
    config.start_dir = 1
    eval_configs.append(config)
    eval_wrappers.append(FullyObsImageWrapper)
    config = config.clone()
    config.name = f"small_maze-random_colour"
    eval_configs.append(config)
    eval_wrappers.append(RandomChannelSwapWrapper)

    trainer_config = RandColourTrainerConfig(
        session_dir=f"./experiments/normal_colour-maze",
        num_models=5,
        num_parallel=8,
        init_seed=0,
        eval_freq=int(25e3),
        num_eval_episodes=10,
        eval_deterministic=False,
        policy_kwargs=dict(
            features_extractor_class=CNNEncoderExtractor,  # Use the custom encoder extractor
            features_extractor_kwargs=dict(
                net_arch=[16],  # Custom layer sizes
                cnn_net_arch=[
                    (64, 3, 2, 1),
                    (128, 3, 2, 1),
                    (256, 3, 2, 1),
                ],
                activation_fn=nn.LeakyReLU,  # Activation function
                encoder_only=True,
            ),
            net_arch=dict(pi=[32, 32], vf=[32, 32]),  # Policy and value network architecture
            activation_fn=nn.LeakyReLU,
        ),
        train_config=train_config,
        eval_configs=eval_configs,
        train_output_wrapper=FullyObsImageWrapper,
        eval_output_wrappers=eval_wrappers,
    )

    train(trainer_config)

    trainer_config = RandColourTrainerConfig(
        session_dir=f"./experiments/rand_colour-maze",
        num_models = 5,
        num_parallel = 8,
        init_seed = 0,
        eval_freq = int(50e3),
        num_eval_episodes = 10,
        eval_deterministic = False,
        policy_kwargs=dict(
                features_extractor_class=CNNEncoderExtractor,  # Use the custom encoder extractor
                features_extractor_kwargs=dict(
                    net_arch=[16],  # Custom layer sizes
                    cnn_net_arch=[
                        (64, 3, 2, 1),
                        (128, 3, 2, 1),
                        (256, 3, 2, 1),
                    ],
                    activation_fn=nn.LeakyReLU,  # Activation function
                    encoder_only=True,
                ),
                net_arch=dict(pi=[32, 32], vf=[32, 32]),  # Policy and value network architecture
                activation_fn=nn.LeakyReLU,
            ),
        train_config = train_config,
        eval_configs = eval_configs,
        train_output_wrapper=FullyObsImageWrapper,
        eval_output_wrappers=eval_wrappers,
    )

    train(trainer_config)
