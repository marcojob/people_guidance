import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt

from ..module import Module


def normalize(v: np.array) -> np.array:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    else:
        return v / norm


class ReprojectionModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super(ReprojectionModule, self).__init__(name="reprojection_module",
                                                 inputs=["position_module:homography"],
                                                 outputs=[("points3d", 10)],
                                                 log_dir=log_dir)

        self.intrinsic_matrix: Optional[np.array] = [[2581.33211, 0, 320], [0, 2576, 240], [0, 0, 1]]

        self.point_buffer: List[np.array] = []
        self.request_counter = 0

        self.vo_pos_buffer: List[Tuple[np.array, np.array, Tuple[float, float]]] = []
        self.imu_pos_buffer: List[Tuple[np.array, Tuple[float, float]]] = []

    def start(self):
        while True:
            homog_payload = self.get("position_module:homography")
            if homog_payload:
                homography = homog_payload["data"]["homography"]
                point_pairs = homog_payload["data"]["point_pairs"]

                pm1, pm2 = self.create_projection_matrices(homography)

                points_homo = cv2.triangulatePoints(pm1, pm2, point_pairs[0, ...], point_pairs[1, ...])
                points3d = cv2.convertPointsFromHomogeneous(points_homo.T)
                self.publish("points3d", data=points3d, validity=100, timestamp=self.get_time_ms())

                user_pos = np.array((0, 0, 0))
                user_trajectory: np.array = normalize(homography[:, 3])
                point_vectors = np.subtract(points3d, user_pos)
                point_vectors = point_vectors.reshape((point_vectors.shape[0], 3))

                #  how far away are the points from the user?
                distances = np.linalg.norm(point_vectors, axis=1, keepdims=False)
                #  how close are the points to the trajectory of the user?
                alignment = np.dot(point_vectors, user_trajectory)
                # weigh the distance and alignment to obtain an estimate of how likely a collision is.
                criticality = (0.3 * distances) + (0.7 * alignment)

                if (criticality > 0.99).mean() != 0:
                    self.logger.critical("Collision Warning!")

                self.logger.info(f"Reconstructed points \n{criticality.shape}")

    def create_projection_matrices(self, homography) -> Tuple[np.array, np.array]:
        pm1 = np.matmul(self.intrinsic_matrix, np.eye(3, 4))
        pm2 = np.matmul(self.intrinsic_matrix, homography)
        return pm1, pm2
