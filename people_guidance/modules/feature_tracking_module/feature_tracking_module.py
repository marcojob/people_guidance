import pathlib
import cv2
import numpy as np

from time import sleep
from scipy.spatial.transform import Rotation
from typing import Tuple

from people_guidance.modules.module import Module
from people_guidance.utils import project_path







class FeatureTrackingModule(Module):

    def __init__(self, log_dir: pathlib.Path, args=None):
        super(FeatureTrackingModule, self).__init__(name="feature_tracking_module", outputs=[("feature_point_pairs", 10), ("feature_point_pairs_vis", 10)],
                                                    inputs=["drivers_module:images"],
                                                    log_dir=log_dir)

    def start(self):
        self.old_timestamp = None
        self.old_keypoints = None
        self.old_descriptors = None
        self.intrinsic_matrix: Optional[np.array] = np.array([[2581.33211, 0, 320], [0, 2576, 240], [0, 0, 1]])

        # maximum numbers of keypoints to keep and calculate descriptors of,
        # reducing this number can improve computation time:
        self.max_num_keypoints = 1000

        # create cv2 ORB feature descriptor and brute force matcher object
        self.orb = cv2.ORB_create(nfeatures=self.max_num_keypoints)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

        while True:
            img_dict = self.get("drivers_module:images")

            if not img_dict:
                self.logger.warn("queue was empty")
            else:
                # extract the image data and time stamp
                img_encoded = img_dict["data"]["data"]
                timestamp = img_dict["data"]["timestamp"]

                self.logger.debug(f"Processing image with timestamp {timestamp} ...")

                img = cv2.imdecode(np.frombuffer(img_encoded, dtype=np.int8), flags=cv2.IMREAD_GRAYSCALE)
                keypoints, descriptors = self.extract_feature_descriptors(img)

                # only do feature matching if there were keypoints found in the new image, discard it otherwise
                if len(keypoints) == 0:
                    self.logger.warn(f"Didn't find any features in image with timestamp {timestamp}, skipping...")
                else:
                    if self.old_descriptors is not None:  # skip the matching step for the first image
                        # match the feature descriptors of the old and new image

                        inliers, delta_positions, total_nr_matches = self.match_features(keypoints, descriptors)

                        if total_nr_matches == 0:
                            # there were 0 inliers found, print a warning
                            self.logger.warn("Couldn't find enough matching features in the images with timestamps: " +
                                            f"{self.old_timestamp} and {timestamp}")
                        else:
                            visualization_img = self.visualize_matches(img, keypoints, inliers, total_nr_matches)
                            #visualization_img = cv2.resize(visualization_img, None, fx=0.85, fy=0.85)

                            self.publish("feature_point_pairs",
                                         {"camera_positions" : delta_positions,
                                          "image": visualization_img,
                                          "point_pairs": inliers,
                                          "timestamp_pair": (self.old_timestamp, timestamp)},
                                         1000)
                            self.publish("feature_point_pairs_vis",
                                         {"point_pairs": inliers,
                                          "img": img_encoded,
                                          "timestamp": timestamp},
                                         1000)

                    # store the date of the new image as old_img... for the next iteration
                    # If there are no features found in the new image this step is skipped
                    # This means that the next image will be compared witht he same old image again
                    self.old_timestamp = timestamp
                    self.old_keypoints = keypoints
                    self.old_descriptors = descriptors

    def extract_feature_descriptors(self, img: np.ndarray) -> (list, np.ndarray):
        # first detect the ORB keypoints and then compute the feature descriptors of those points
        keypoints = self.orb.detect(img, None)
        keypoints, descriptors = self.orb.compute(img, keypoints)
        self.logger.debug(f"Found {len(keypoints)} feautures")

        return (keypoints, descriptors)

    def match_features(self, keypoints: list, descriptors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        matches = self.matcher.match(self.old_descriptors, descriptors)

        if len(matches) > 10:
            # assemble the coordinates of the matched features into a numpy matrix for each image
            old_match_points = np.float32([self.old_keypoints[match.queryIdx].pt for match in matches])
            match_points = np.float32([keypoints[match.trainIdx].pt for match in matches])

            # if we found enough matches do a RANSAC search to find inliers corresponding to one homography
            # TODO: add camera pose info to improve matching
            H, mask = cv2.findHomography(old_match_points, match_points, cv2.RANSAC, 1.0)

            nb_solutions, H_rots, H_trans, H_norms = cv2.decomposeHomographyMat(H, self.intrinsic_matrix)

            if nb_solutions > 0:
                delta_positions = np.zeros((nb_solutions, 3, 4))

                for i in range(nb_solutions):
                    delta_positions[i, :, 0:3] = H_rots[i]
                    delta_positions[i, :, 3] = H_trans[i].ravel()

                old_match_points = old_match_points[mask.ravel().astype(bool)]
                match_points = match_points[mask.ravel().astype(bool)]
                # add the two matrixes together, first dimension are all the matches,
                # second dimension is image 1 and 2, thrid dimension is x and y
                # e.g. 4th match, 1st image, y-coordinate: matches_paired[3][0][1]
                #      8th match, 2nd image, x-coordinate: matches_paired[7][1][0]
                matches_paired = np.concatenate(
                (old_match_points.transpose().reshape(1, 2, -1),
                match_points.transpose().reshape(1, 2, -1)),
                axis=0)

                return (matches_paired, delta_positions, len(mask))
            else:
                self.logger.warn("Couldn't decompose homography")
        else:
            self.logger.warn(f"Only {len(matches)} matches found, not enough to calculate a homography")

        return (np.array([]), np.array([]), 0)


    def visualize_matches(self, img: np.ndarray, keypoints: list, inliers: np.ndarray, nb_matches: int) -> np.ndarray:
        visualization_img = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)
        for i in range(inliers.shape[2]):
            visualization_img = cv2.line(visualization_img,
                                         tuple(inliers[0,...,i]), tuple(inliers[1,...,i]),
                                         (255,0,0), 5)

        cv2.putText(visualization_img, f"Features: {len(keypoints)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(visualization_img, f"Total matches: {nb_matches}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(visualization_img, f"Inliers: {inliers.shape[2]}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return visualization_img
