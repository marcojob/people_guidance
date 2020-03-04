import time
import pathlib

import numpy as np

from ..module import Module


class SpamModule(Module):

    def __init__(self, log_dir: pathlib.Path):
        super(SpamModule, self).__init__(name="spam_module", outputs=[("spam", 10)],
                                         input_topics=["echo_module:echo"], log_dir=log_dir)

    def start(self):
        while True:
            time.sleep(1)
            self.logger.info("Spamming...")
            spam = np.random.random((20, 20, 3))
            self.outputs["spam"].put(spam)
            data = self.get("echo_module:echo")
            self.logger.info(f"Received Echo with shape {data.shape} ")

