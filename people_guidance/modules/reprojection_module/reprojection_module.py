import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from ..module import Module


class ReprojectionModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super(ReprojectionModule, self).__init__(name="reprojection_module",
                                                 inputs=["feature_tracking_module:feature_point_pairs"],
                                                 log_dir=log_dir)

        self.intrinsic_matrix: Optional[np.array] = np.eye(3, 4)

    def start(self):
        while True:
            payload = self.get("feature_tracking_module:feature_point_pairs")
            if payload:
                self.logger.info("got payload")
                camera_positions, point_pairs = self.extract_payload(payload)
                pm1, pm2 = self.create_projection_matrices(camera_positions)

                reconstructed_points = cv2.triangulatePoints(pm1,
                                                             pm2,
                                                             point_pairs[0, ...],
                                                             point_pairs[1, ...])

                """
                camera2_xyz = np.append(camera_positions[1, :, 3], [0]).reshape(4, 1)
                print(camera2_xyz)
                camera_relative_locations =  np.repeat(camera2_xyz, 10, axis=1)
                print(camera_relative_locations)
                distances_to_camera = np.linalg.norm(reconstructed_points, axis=0)
                print(distances_to_camera)
                """
                print(reconstructed_points)

    def extract_payload(self, payload: Dict):
        return np.random.rand(2, 3, 4), np.random.rand(2, 2, 10)

    def create_projection_matrices(self, camera_positions: np.array) -> Tuple[np.array, np.array]:
        pm1 = self.intrinsic_matrix * np.eye(3, 4)
        pm2 = self.intrinsic_matrix * camera_positions[1, ...]
        return pm1, pm2
