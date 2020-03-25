import pathlib

from time import sleep

from people_guidance.modules.module import Module


class EchoModule(Module):

    def __init__(self, log_dir: pathlib.Path, args=None):
        super(EchoModule, self).__init__(name="echo_module", outputs=[],
                                         input_topics=[], log_dir=log_dir,
                                         services=["echo"], requests=[])

    def start(self):
        self.logger.info("Starting echo module...")
        self.services["echo"].register_handler(self.create_echo)
        while True:
            sleep(1)
            self.handle_requests()

    def create_echo(self, request):
        return {"id": request["id"], "payload": request["payload"]}