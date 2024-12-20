from typing import Optional, Tuple
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

        self.rand_colours = ['R', 'G', 'B', 'P', 'Y', 'E']

    def clone(self,) -> "TaskConfig":
        new = TaskConfig()
        new.name = self.name
        new.txt_file_path = self.txt_file_path
        new.rand_gen_shape = self.rand_gen_shape
        new.custom_mission = self.custom_mission
        new.minimum_display_size = self.minimum_display_size
        new.display_mode = self.display_mode
        new.random_rotate = self.random_rotate
        new.random_flip = self.random_flip
        new.max_steps = self.max_steps
        new.start_pos = self.start_pos
        new.start_dir = self.start_dir
        new.train_total_steps = self.train_total_steps
        new.difficulty_level = self.difficulty_level
        new.rand_colours = self.rand_colours
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
            rand_colours=env_config.rand_colours,
        )
    )