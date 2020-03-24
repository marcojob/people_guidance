import pathlib
from typing import Dict, List

import numpy as np

from ..module import Module


class FPSLoggerModule(Module):

    def __init__(self, log_dir: pathlib.Path, args=None):
        super(FPSLoggerModule, self).__init__(name="fps_logger_module", outputs=[],
                                              input_topics=["drivers_module:accelerations"], log_dir=log_dir)

    def start(self):
        payload_idxs: Dict = {input_name: 0 for input_name in self.input_topics}
        start_times: Dict = {input_name: 0 for input_name in self.input_topics}

        smoothed_fps: Dict[str, List] = {input_name: [] for input_name in self.input_topics}

        while True:

            for input_name in self.input_topics:

                if len(smoothed_fps[input_name]) > 10:
                    self.logger.info(f"Smoothed fps for input {input_name}: {np.mean(smoothed_fps[input_name])}")
                    smoothed_fps[input_name] = []

                if payload_idxs[input_name] == 0:
                    start_times[input_name] = self.get_time_ms()

                payload = self.get(input_name)
                if payload:
                    payload_idxs[input_name] += 1

                if payload_idxs[input_name] > 10:
                    fps = (payload_idxs[input_name] / ((float(self.get_time_ms() - start_times[input_name])) / 1000.0))
                    smoothed_fps[input_name].append(fps)
                    self.logger.debug(f"Module {input_name} is running at {fps} fps.")
                    payload_idxs[input_name] = 0
