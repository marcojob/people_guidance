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
                                                 inputs=["visual_odometry_module:cloud"],
                                                 outputs=[],
                                                 log_dir=log_dir)

        self.point_buffer: List[np.array] = []
        self.request_counter = 0

        self.vo_pos_buffer: List[Tuple[np.array, np.array, Tuple[float, float]]] = []
        self.imu_pos_buffer: List[Tuple[np.array, Tuple[float, float]]] = []

    def start(self):
        criticality_smooth = 0.0
        while True:
            cloud = self.get("visual_odometry_module:cloud")
            if cloud:
                points3d = cloud["data"]["cloud"]
                homography = cloud["data"]["homography"]
                timestamps = cloud["data"]["timestamps"]

                rvec = cv2.Rodrigues(homography[:, :3])[0]
                tvec = homography[:, 3:]
                points2d = cv2.projectPoints(points3d, rvec, tvec, self.intrinsic_matrix, distCoeffs=None)[0]

                pink = (255, 153, 255)
                orange = (255, 128, 0)
                keypoints = [cv2.KeyPoint(points2d[i, 0, 0], points2d[i, 0, 1], 5) for i in range(points2d.shape[0])]
                # image = cv2.drawKeypoints(image, keypoints, None, color=pink, flags=0)
                # for i in range(point_pairs.shape[0]):
                #     image = cv2.line(image, tuple(point_pairs[1, :, i]), tuple(points2d[i, 0, :]), orange, 5)

                user_pos = np.array((0, 0, 0))
                user_trajectory: np.array = normalize(homography[:, 3])
                point_vectors = np.subtract(points3d, user_pos)
                point_vectors = point_vectors.reshape((point_vectors.shape[0], 3))
                
                #  how far away are the points from the user?
                distances = np.linalg.norm(point_vectors, axis=1, keepdims=False)
                distances.sort()

                #  how close are the points to the trajectory of the user?
                alignment = np.dot(normalize(point_vectors), normalize(user_trajectory))
                # weigh the distance and alignment to obtain an estimate of how likely a collision is.
                uncertainty = 1 / points3d.shape[0]
                criticality = (1 / distances) #* abs(alignment)
                criticality_smooth = 0.8 * criticality_smooth + 0.2 * criticality.mean()
                """
                plt.scatter(timestamps[0], criticality_smooth, c="r")
                plt.scatter(timestamps[0], uncertainty, c="g")
                plt.gca().set_ylim((0, 0.25))
                plt.pause(0.001)
                """

    def create_projection_matrices(self, homography) -> Tuple[np.array, np.array]:
        pm1 = np.matmul(self.intrinsic_matrix, np.eye(3, 4))
        pm2 = np.matmul(self.intrinsic_matrix, homography)
        return pm1, pm2
