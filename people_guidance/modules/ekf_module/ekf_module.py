from pathlib import Path
from time import sleep

from ..module import Module
from ...utils import DEFAULT_DATASET

class EkfModule(Module):
    def __init__(self, log_dir: Path, args=None):
        super(EkfModule, self).__init__(name="ekf_module",
                                            outputs=[],
                                            inputs=["drivers_module:accelerations"], log_dir=log_dir)
        self.args = args

    def start(self):
        while True:
            sleep(1)
            print(self.get("drivers_module:accelerations"))