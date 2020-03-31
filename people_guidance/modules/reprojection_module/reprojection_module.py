import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from ..module import Module


class ReprojectionModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super(ReprojectionModule, self).__init__(name="reprojection_module",
                                                 inputs=[],
                                                 log_dir=log_dir)

        self.intrinsic_matrix: Optional[np.array] = np.eye(3, 4)

    def start(self):
        payload = {}  # self.get("feature_tracking_module:feature_point_pairs")
        camera_positions, point_pairs = self.extract_payload(payload)
        pm1, pm2 = self.create_projection_matrices(camera_positions)

        reconstructed_points = cv2.triangulatePoints(pm1,
                                                     pm2,
                                                     point_pairs[0, ...],
                                                     point_pairs[1, ...])

        camera2_xyz = np.append(camera_positions[1, :, 3], [0]).reshape(4, 1)
        camera_relative_locations = reconstructed_points - np.repeat(camera2_xyz, 10, axis=1)
        distances_to_camera = np.linalg.norm(camera_relative_locations, axis=0)
        print(distances_to_camera)

    @staticmethod
    def extract_payload(payload: Dict):
        m = 10  # number of points
        return np.random.rand(2, 3, 4), np.random.rand(2, 2, m)

    def create_projection_matrices(self, camera_positions: np.array) -> Tuple[np.array, np.array]:
        pm1 = self.intrinsic_matrix * camera_positions[0, ...]
        pm2 = self.intrinsic_matrix * camera_positions[1, ...]
        return pm1, pm2
