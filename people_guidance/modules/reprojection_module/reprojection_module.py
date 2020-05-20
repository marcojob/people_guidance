import pathlib
from typing import Optional

import cv2
import copy
import matplotlib.pyplot as plt
import numpy as np

from ..module import Module
from ...utils import normalize, MovingAverageFilter


class ReprojectionModule(Module):
    def __init__(self, log_dir: pathlib.Path, args=None):
        super(ReprojectionModule, self).__init__(name="reprojection_module",
                                                 inputs=["position_module:homography"],
                                                 outputs=[("points3d", 10)],
                                                 log_dir=log_dir)

        self.average_filter = MovingAverageFilter()
        self.use_alignment = False

        self.last_update_ts = None
        self.origin_pm = np.matmul(self.intrinsic_matrix, np.eye(3, 4))
        self.origin = np.zeros(3)
        self.forward_direction = np.array((0., 0., 1.))

    def start(self):


        criticality_smooth = 0.0
        while True:
            homog_payload = self.get("position_module:homography")
            if homog_payload:
                homography = homog_payload["data"]["homography"]
                point_pairs = homog_payload["data"]["point_pairs"]
                timestamps = homog_payload["data"]["timestamps"]
                image = homog_payload["data"]["image"]

                offset_pm = np.matmul(self.intrinsic_matrix, homography)

                points_homo = cv2.triangulatePoints(self.origin_pm, offset_pm, point_pairs[0, ...], point_pairs[1, ...])
                points3d = cv2.convertPointsFromHomogeneous(points_homo.T)

                # Ensure that signs of points are correct
                for point in points3d:
                    # Matrix to vector
                    point_temp = copy.deepcopy(point[0])
                    point = point[0]

                    # Change coordinate system
                    point[0] =  point_temp[2]
                    point[1] = -point_temp[0]
                    point[2] = -point_temp[1]

                    # We only expect points in positive x direction
                    if point[0] < 0.0:
                        point[0] *= -1.0

                    # Same in z direction
                    if point[2] < 0.0:
                        point[2] *= -1.0

                self.publish("points3d", data=points3d, validity=100, timestamp=self.get_time_ms())

                points2d = self.project3dto2d(homography, points3d)

                """
                if cv2.waitKey(0) == ord('a'):
                    print("continue")
                """
                # cv2.waitKey(1)

                collision_probability = self.update_collision_probability(points3d, timestamps[1], image, homography)

                uncertainty = self.average_filter("uncertainty", self.update_uncertainty(points3d.shape[0], timestamps[1]))

                self.last_update_ts = timestamps[1]

    def project3dto2d(self, homography: np.array, points3d: np.array):
        rot_vec = cv2.Rodrigues(homography[:, :3])[0]
        trans_vec = homography[:, 3:]
        try:
            points2d = cv2.projectPoints(points3d, rot_vec, trans_vec, self.intrinsic_matrix, distCoeffs=None)[0]
        except Exception as e:
            points2d = np.array([])
        return points2d

    @staticmethod
    def visualize_reprojection(point_pairs: np.array, points2d: np.array, image: np.array):
        pink = (255, 153, 255)
        orange = (255, 128, 0)
        keypoints = [cv2.KeyPoint(points2d[i, 0, 0], points2d[i, 0, 1], 5) for i in range(points2d.shape[0])]
        image = cv2.drawKeypoints(image, keypoints, None, color=pink, flags=0)
        for i in range(point_pairs.shape[0]):
            image = cv2.line(image, tuple(point_pairs[1, :, i]), tuple(points2d[i, 0, :]), orange, 5)

        return image

    def update_collision_probability(self, points3d: np.array, timestamp: float, image: np.array, homography):

        point_vectors: np.array = points3d.reshape((points3d.shape[0], 3))

        distances = np.linalg.norm(point_vectors, axis=1, keepdims=False)

        alignment_vectors = np.cross(point_vectors, self.forward_direction)
        alignment = np.linalg.norm(alignment_vectors, axis=1, keepdims=False)

        if self.use_alignment:
            distances *= alignment

        # get the indices of the 10th percentile smallest distances
        n = int(0.1 * distances.shape[0])
        idxs = np.argpartition(distances, n)[:n]
        critical_points_3d = point_vectors[idxs, :]

        critical_points_2d = self.project3dto2d(homography, critical_points_3d)

        #  plot the critical points
        for i in range(critical_points_2d.shape[0]):
            image = cv2.circle(img=image, center=tuple(critical_points_2d[i, 0, :]), radius=10, color=(255, 153, 255), thickness=-1)

        #  show both the features and the reprojected critical points
        # cv2.imshow("visu", image)
        # cv2.imwrite(f"sandbox/plots/image_{forward_direction}_{timestamp}.png", image)

        # for i in range(3):
        #     for j in range(3):
        #         self.ax[i, j].clear()
        #         xlim = 8
        #         ylim = 8
        #         self.ax[i, j].set_xlim(-xlim, xlim)
        #         self.ax[i, j].set_ylim(-5, 5)
        #         self.ax[i, j].scatter(point_vectors[:, i], point_vectors[:, j], c="b")
        #         self.ax[i, j].scatter(np.zeros(100), np.linspace(-xlim, xlim, 100), c="g", s=1)
        #         self.ax[i, j].scatter(np.linspace(-ylim, ylim, 100), np.zeros(100), c="g", s=1)
        #         self.ax[i, j].scatter(point_vectors[idxs, i], point_vectors[idxs, j], c="r")

            #plt.pause(0.001)
            #self.fig.savefig(f"sandbox/plots/plot_{forward_direction}_{timestamp}.png")
        """
        image = self.project3dto2d()
        
        cv2.imshow("visu", image)
        cv2.imwrite(f"sandbox/plots/{timestamp}_image.png", image)

        
        point_vectors = np.subtract(points3d, user_pos)
        point_vectors = point_vectors.reshape((point_vectors.shape[0], 3))

        #  how far away are the points from the user?
        distances = np.linalg.norm(points3d, axis=1, keepdims=False)
        distances.sort()
        #  how close are the points to the trajectory of the user?
        alignment = np.dot(normalize(point_vectors), normalize(user_trajectory))
        # weigh the distance and alignment to obtain an estimate of how likely a collision is.
        uncertainty = 1 / points3d.shape[0]
        criticality = (1 / distances)  # * abs(alignment)
        criticality_smooth = 0.8 * criticality_smooth + 0.2 * criticality.mean()

        plt.scatter(timestamps[0], criticality_smooth, c="r")
        plt.scatter(timestamps[0], uncertainty, c="g")
        plt.pause(0.001)

        self.logger.info(f"Reconstructed points \n{criticality.shape}")
        """
        return 0.0

    def update_uncertainty(self, n_features: int, timestamp: float):
        # compares the expected number of features and the actual number of features.
        expected_n_features = self.average_filter("n_features", n_features, window_size=20)
        if n_features > expected_n_features:
            features_confidence = 1.0
        else:
            features_confidence = (expected_n_features - n_features) / expected_n_features

        assert 0 <= features_confidence <= 1.0, f"features_confidence should be in interval 0,1 but was {features_confidence}"

        if self.last_update_ts is not None:

            time_delta = timestamp - self.last_update_ts
            expected_time_delta = self.average_filter("time_delta", time_delta, window_size=20)

            if time_delta > expected_time_delta:
                time_confidence = 1.0
            else:
                time_confidence = (expected_time_delta - time_delta) / expected_time_delta
        else:
            time_confidence = 1.0

        assert 0 <= time_confidence <= 1.0, "time_confidence should be in interval 0,1"

        # take the average between the feature confidence and the time confidence
        confidence: float = (features_confidence + time_confidence) / 2

        uncertainty: float = 1. - confidence

        assert 0 <= uncertainty <= 1.0, "uncertainty should be in interval 0,1"
        return uncertainty
