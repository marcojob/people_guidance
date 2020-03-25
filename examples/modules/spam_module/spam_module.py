import pathlib
import numpy as np

from time import sleep

from people_guidance.modules.module import Module


class SpamModule(Module):

    def __init__(self, log_dir: pathlib.Path, args=None):
        super(SpamModule, self).__init__(name="spam_module", outputs=[],
                                         input_topics=[], log_dir=log_dir,
                                         services=[], requests=["echo_module:echo"])

    def start(self):
        self.logger.info("Starting spam module...")
        while True:

            self.make_request("echo_module:echo", {"id": 0, "payload": "hello world"})
            sleep(2) # do some work in your process while you wait for the reponse!
            response = self.await_response("echo_module:echo")
            self.logger.info(f"Service returned response {response}")
