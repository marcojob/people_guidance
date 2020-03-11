import pathlib

from time import sleep

from people_guidance.modules.module import Module


class EchoModule(Module):

    def __init__(self, log_dir: pathlib.Path):
        super(EchoModule, self).__init__(name="echo_module", outputs=[("echo", 10)],
                                         input_topics=["spam_module:spam"], log_dir=log_dir)

    def start(self):
        self.logger.info("Starting echo module...")
        while True:
            sleep(1)
            # Get data from spam module and check if data is not empty
            data_dict = self.get("spam_module:spam")
            if data_dict:
                self.logger.info(f"Received data {data_dict['data'].shape} from spam module.")
                self.publish("echo", data_dict['data'], 2000)
