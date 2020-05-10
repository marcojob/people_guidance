import pathlib
import numpy as np

from time import sleep
from math import pi, sqrt, sin

from people_guidance.modules.module import Module
from people_guidance.utils import project_path



class PositionModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super().__init__(name="position_module",
                         outputs=[],
                         inputs=["drivers_module:accelerations"],
                         log_dir=log_dir)
