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
                                                 inputs=["feature_tracking_module:feature_point_pairs"],
                                                 outputs=[("points3d", 10)],
                                                 log_dir=log_dir)

        self.intrinsic_matrix: Optional[np.array] = [[2581.33211, 0, 320], [0, 2576, 240], [0, 0, 1]]

        self.point_buffer: np.zeros((3000, 3), dtype=float)

    def start(self):
        while True:
            payload = self.get("feature_tracking_module:feature_point_pairs")
            if payload:

                camera_positions, point_pairs = self.extract_payload(payload)
                user_pos = camera_positions[1, ...][:, 3]

                plt.scatter(user_pos[0], user_pos[1], c="r")
                plt.pause(0.05)


                """
                pm1, pm2 = self.create_projection_matrices(camera_positions)
                    
                points_homo = cv2.triangulatePoints(pm1, pm2, point_pairs[0, ...], point_pairs[1, ...])
                points3d = cv2.convertPointsFromHomogeneous(points_homo.T)
                points3d = points3d.reshape((points3d.shape[0], 3))
                points3d = self.remove_outliers(points3d)
                self.publish("points3d", data=points3d, validity=100, timestamp=self.get_time_ms())
            
                
                user_pos = camera_positions[1, ...][:, 3]
                user_trajectory: np.array = normalize((camera_positions[1, ...] - camera_positions[0, ...])[:, 3])
                point_vectors = - np.subtract(points3d, user_pos)
                #  how far away are the points from the user?
                distances = np.linalg.norm(point_vectors, axis=1, keepdims=False)
                #  how close are the points to the trajectory of the user?
                alignment = np.dot(point_vectors, user_trajectory)
                # weigh the distance and alignment to obtain an estimate of how likely a collision is.
                critical_points = np.logical_and(distances < 3.0, alignment > 0.01)

                if np.count_nonzero(critical_points) > 0:
                    self.logger.critical("Collision Imminent")
                """

    @staticmethod
    def remove_outliers(points3d):
        zeroed = points3d * (np.absolute(points3d) < 2.0)
        return zeroed[~np.all(zeroed == 0, axis=1)]


    @staticmethod
    def extract_payload(payload: Dict) -> Tuple[np.array, np.array]:
        return payload["data"]["camera_positions"], payload["data"]["point_pairs"]

    def create_projection_matrices(self, camera_positions: np.array) -> Tuple[np.array, np.array]:
        pm1 = np.matmul(self.intrinsic_matrix, np.eye(3, 4))
        pm2 = np.matmul(self.intrinsic_matrix, camera_positions[1, ...])
        return pm1, pm2
