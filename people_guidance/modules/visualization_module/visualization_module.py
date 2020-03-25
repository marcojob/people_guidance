import pathlib

import cv2
import numpy as np

from ..module import Module


class VisualizationModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super(VisualizationModule, self).__init__(name="visualization_module", outputs=[],
                                                  input_topics=["drivers_module:images", "drivers_module:accelerations"]
                                                  , log_dir=log_dir)

        self.display_fps = 0.0

    def start(self):
        self.logger.info("Starting visualization module...")
        frm_idx = 0
        ms_time = 0
        while True:
            # Get data from spam module and check if data is not empty
            data_dict = self.get("drivers_module:images")
            if data_dict:

                if frm_idx == 1:
                    ms_time = self.get_time_ms()

                if frm_idx > 10:
                    self.display_fps = frm_idx / ((self.get_time_ms() - ms_time)/1000.0)
                    frm_idx = 0

                frm_idx += 1
                self.visualize_image_data(data_dict["data"])

    def visualize_image_data(self, data: bytes) -> None:
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), -1)

        cv2.putText(decoded, f"Display fps: {self.display_fps}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Frame", decoded)
        cv2.waitKey(1)