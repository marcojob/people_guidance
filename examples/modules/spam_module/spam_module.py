import pathlib
import numpy as np

from time import sleep

from people_guidance.modules.module import Module


class SpamModule(Module):

    def __init__(self, log_dir: pathlib.Path):
        super(SpamModule, self).__init__(name="spam_module", outputs=[("spam", 10)],
                                         input_topics=["echo_module:echo"], log_dir=log_dir)

    def start(self):
        self.logger.info("Starting spam module...")
        while True:
            sleep(1)
            self.logger.warning("Spamming...")

            # Publish data with validity of 2000 ms
            spam = np.random.random((20, 20, 3))
            self.publish("spam", spam, 2000)

            # Get data from echo module and check if data is not empty
            data_dict = self.get("echo_module:echo")
            if data_dict:
                self.logger.info(f"Received Echo with shape {data_dict['data'].shape} ")
