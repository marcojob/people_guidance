import pathlib

from time import sleep

from people_guidance.modules.module import Module
from people_guidance.utils import project_path

import cv2
import numpy as np


class FeatureTrackingModule(Module):

    def __init__(self, log_dir: pathlib.Path, args=None):
        super(FeatureTrackingModule, self).__init__(name="feature_tracking_module", outputs=[("feature_point_pairs", 10)],
                                                    inputs=["drivers_module:images"], #requests=[("position_estimation_module:pose")]
                                                    log_dir=log_dir)

    def start(self):
        self.old_timestamp = None
        self.old_keypoints = None
        self.old_descriptors = None
        self.old_pose = None

        self.request_counter = 0

        # maximum numbers of keypoints to keep and calculate descriptors of,
        # reducing this number can improve computation time:
        self.max_num_keypoints = 100

        # create cv2 ORB feature descriptor and brute force matcher object
        self.orb = cv2.ORB_create(nfeatures=self.max_num_keypoints)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

        while True:
            img_dict = self.get("drivers_module:images")

            if not img_dict:
                sleep(1)
            else:
                # extract the image data and time stamp
                img_encoded = img_dict["data"]
                timestamp = img_dict["timestamp"]
                """
                # request the pose of the camera at this time stamp from the position_estimation_module
                self.make_request("position_estimation_module:pose", {"id" : self.request_counter, "payload": timestamp})
                self.request_counter += 1
                """
                self.logger.debug(f"Processing image with timestamp {timestamp} ...")

                keypoints, descriptors = self.extract_feature_descriptors(img_encoded)

                """
                # get the new pose and compute the difference to the old one
                pose_response = self.await_response("position_estimation_module:pose")
                pose = pose_response["payload"]
                """
                pose = 0
                
                # only do feature matching if there were keypoints found in the new image, discard it otherwise
                if len(keypoints) == 0:
                    self.logger.warn(f"Didn't find any features in image with timestamp {timestamp}, skipping...")
                else:
                    if self.old_descriptors is not None:  # skip the matching step for the first image
                        # match the feature descriptors of the old and new image

                        matches = self.match_features(keypoints, descriptors)
                        if matches.shape[0] == 0:
                            # there were 0 matches found, print a warning
                            self.logger.warn("Couldn't find any matching features in the images with timestamps: " +
                                             f"{old_timestamp} and {timestamp}")
                        else:
                            delta_pose = self.compute_delta_pose(pose)
                            self.publish("feature_point_pairs",
                                         {"timestamps": (self.old_timestamp, timestamp), 
                                          "matches": matches, "delta_pose": delta_pose},
                                         1000)

                    # store the date of the new image as old_img... for the next iteration
                    # If there are no features found in the new image this step is skipped
                    # This means that the next image will be compared witht he same old image again
                    self.old_timestamp = timestamp
                    self.old_keypoints = keypoints
                    self.old_descriptors = descriptors
                    self.old_pose = pose

    def extract_feature_descriptors(self, img_data: bytes) -> (list, np.ndarray):
        img = cv2.imdecode(np.frombuffer(img_data, dtype=np.int8), flags=cv2.IMREAD_COLOR)

        # first detect the ORB keypoints and then compute the feature descriptors of those points
        keypoints = self.orb.detect(img, None)
        keypoints, descriptors = self.orb.compute(img, keypoints)
        self.logger.debug(f"Found {len(keypoints)} feautures")

        return (keypoints, descriptors)
    
    def match_features(self, keypoints: list, descriptors: np.ndarray) -> np.ndarray:
        matches = self.matcher.match(self.old_descriptors, descriptors)

        # sort the matches by shortest distance first
        matches_sorted = sorted(matches, key=lambda x: x.distance)

        # assemble the coordinates of the matched features into a numpy matrix for each image
        old_match_points = np.float32([self.old_keypoints[match.queryIdx].pt for match in matches_sorted])
        match_points = np.float32([keypoints[match.trainIdx].pt for match in matches_sorted])

        # add the two matrixes together, first dimension are all the matches,
        # second dimension is image 1 and 2, thrid dimension is x and y
        # e.g. 4th match, 1st image, y-coordinate: matches_paired[3][0][1]
        #      8th match, 2nd image, x-coordinate: matches_paired[7][1][0]
        matches_paired = np.concatenate(
            (old_match_points.reshape(-1, 1, 2),
             match_points.reshape(-1, 1, 2)),
             axis=1)

        return matches_paired

    def compute_delta_pose(self, pose):
        # Do some fancy calculations here but in the end it's just
        return pose - self.old_pose # anyway
