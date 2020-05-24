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
                                                 outputs=[("points3d", 10), ("criticality", 100)],
                                                 log_dir=log_dir)

        self.average_filter = MovingAverageFilter()
        self.use_alignment = False

        self.last_update_ts = None
        self.P0 = np.dot(self.intrinsic_matrix, np.eye(3, 4))

        self.forward_direction = np.array((1., 0., 0.))

    def start(self):
        criticality_smooth = 0.0
        while True:
            homog_payload = self.get("position_module:homography")
            if homog_payload:
                homography = homog_payload["data"]["homography"]
                point_pairs = homog_payload["data"]["point_pairs"]
                timestamps = homog_payload["data"]["timestamps"]
                image = homog_payload["data"]["image"]

                P1 = np.dot(self.intrinsic_matrix, homography)

                points_homo = cv2.triangulatePoints(self.P0, P1, np.transpose(point_pairs[0]), np.transpose(point_pairs[1]))
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

                collision_probability = self.update_collision_probability(points3d, timestamps[1], image, homography)

                self.publish("points3d", data={"cloud": points3d, "crit": collision_probability}, validity=-1, timestamp=self.get_time_ms())

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

        smooth_mean_distance = self.average_filter("mean_distance", distances.mean(), 20)

        # get the indices of the 10th percentile smallest distances
        n = int(0.1 * distances.shape[0])
        idxs = np.argpartition(distances, n)[:n]
        critical_points_3d = point_vectors[idxs, :]

        alignment_vectors = np.cross(critical_points_3d, self.forward_direction)
        alignment = np.linalg.norm(alignment_vectors, axis=1, keepdims=False)

        smooth_critical_alignment = self.average_filter("critical_alignments", np.arctan(alignment.mean()) * 2 / np.pi, 20)

        probability = (np.arctan(1 / distances) * 2 / np.pi).mean()
        smooth_probability = self.average_filter("probability", probability, 50)

        # plt.scatter(timestamp, smooth_mean_distance, c="b")
        # plt.scatter(timestamp, smooth_critical_alignment , c="g")
        #plt.scatter(timestamp, smooth_probability, c="r")
        #plt.pause(0.001)

        return smooth_probability

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
