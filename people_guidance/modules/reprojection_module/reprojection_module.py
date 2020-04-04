import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

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
                                                 inputs=["feature_tracking_module:feature_point_pairs"],
                                                 outputs=[("points3d", 10)],
                                                 log_dir=log_dir)

        self.intrinsic_matrix: Optional[np.array] = [[2581.33211, 0, 320], [0, 2576, 240], [0, 0, 1]]

        self.point_buffer: List[np.array] = []

    def start(self):
        while True:
            payload = self.get("feature_tracking_module:feature_point_pairs")
            if payload:
                camera_positions, point_pairs = self.extract_payload(payload)
                pm1, pm2 = self.create_projection_matrices(camera_positions)

                points_homo = cv2.triangulatePoints(pm1, pm2, point_pairs[0, ...], point_pairs[1, ...])
                points3d = cv2.convertPointsFromHomogeneous(points_homo.T)
                self.publish("points3d", data=points3d, validity=100, timestamp=self.get_time_ms())

                user_pos = camera_positions[1, ...][:, 3]
                user_trajectory: np.array = normalize((camera_positions[1, ...] - camera_positions[0, ...])[:, 3])
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

    @staticmethod
    def extract_payload(payload: Dict) -> Tuple[np.array, np.array]:
        return payload["data"]["camera_positions"], payload["data"]["point_pairs"]

    def create_projection_matrices(self, camera_positions: np.array) -> Tuple[np.array, np.array]:
        pm1 = np.matmul(self.intrinsic_matrix, camera_positions[0, ...])
        pm2 = np.matmul(self.intrinsic_matrix, camera_positions[1, ...])
        return pm1, pm2
