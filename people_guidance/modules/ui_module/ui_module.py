import pathlib

from ..module import Module
from .beeper import Beeper


class UIModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super().__init__(name="ui_module",
                         inputs=["reprojection_module:collision_prob",
                                 "reprojection_module:uncertainty"],
                         log_dir=log_dir)

        self.beeper = None

    def start(self):
        self.beeper = Beeper()
        with self.beeper:
            while True:
                proba_payload = self.get("reprojection_module:collision_prob")
                if proba_payload:
                    self.logger.critical(proba_payload)

