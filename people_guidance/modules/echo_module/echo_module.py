import pathlib


from ..module import Module


class EchoModule(Module):

    def __init__(self, log_dir: pathlib.Path):
        super(EchoModule, self).__init__(name="echo_module", outputs=[("echo", 10)],
                                         input_topics=["spam_module:spam"], log_dir=log_dir)

    def start(self):
        while True:
            data = self.get("spam_module:spam")
            self.logger.info(f"Received data {data.shape} from spam module.")
            self.publish("echo", data)
