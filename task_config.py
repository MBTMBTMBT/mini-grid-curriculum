from typing import Optional, Tuple, Type, Any

from gymnasium import Wrapper

from customize_minigrid.custom_env import CustomEnv


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
        self.add_random_door_key: bool = False

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


def make_env(env_config, wrapper, max_minimum_display_size: int):
    """
    Create a new environment instance based on the configuration and environment type.

    :param env_config: Configuration object for the environment
    :param env_type: A string indicating the type of environment: "train", "eval", or "target"
    :return: A new environment instance

    Args:
        wrapper: wrapper constructor.
    """
    return wrapper(
        CustomEnv(
            txt_file_path=env_config.txt_file_path,
            rand_gen_shape=env_config.rand_gen_shape,
            display_size=max_minimum_display_size,
            display_mode=env_config.display_mode,
            random_rotate=env_config.random_rotate,
            random_flip=env_config.random_flip,
            custom_mission=env_config.custom_mission,
            max_steps=env_config.max_steps,
            agent_start_pos=env_config.start_pos,
            agent_start_dir=env_config.start_dir,
            add_random_door_key=env_config.add_random_door_key,
        )
    )