from typing import Optional, Tuple


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
