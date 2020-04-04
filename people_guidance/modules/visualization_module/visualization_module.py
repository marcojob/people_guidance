import pathlib

import cv2
import numpy as np

from ..module import Module


class VisualizationModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super(VisualizationModule, self).__init__(name="visualization_module",
                                                  inputs=["feature_tracking_module:matches_visualization",
                                                          "drivers_module:accelerations"],
                                                  log_dir=log_dir)

        self.display_fps = 0.0

    def start(self):
        self.logger.info("Starting visualization module...")
        frm_idx = 0
        ms_time = 0
        while True:
            # Get data from feature tracking module and check if data is not empty
            data_dict = self.get("feature_tracking_module:matches_visualization")

            if data_dict:
                if frm_idx == 1:
                    ms_time = self.get_time_ms()

                if frm_idx > 10:
                    self.display_fps = frm_idx / ((self.get_time_ms() - ms_time)/1000.0)
                    frm_idx = 0

                frm_idx += 1
                self.visualize_image_data(data_dict["data"])

    def visualize_image_data(self, img: np.ndarray) -> None:
        cv2.putText(img, f"Display fps: {self.display_fps:5.2f}",
                    (img.shape[1]-300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Frame", img)
        cv2.waitKey(1)

    def cleanup(self):
        self.logger.debug("Closing all windwos")
        cv2.destroyAllWindows()