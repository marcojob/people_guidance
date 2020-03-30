import pathlib

from time import sleep

from people_guidance.modules.module import Module
from people_guidance.utils import project_path

import cv2
import numpy as np


class FeatureTrackingModule(Module):

    def __init__(self, log_dir: pathlib.Path, args=None):
        super(FeatureTrackingModule, self).__init__(name="feature_tracking_module", outputs=[("feature_point_pairs", 10), ("feature_point_pairs_vis", 10)],
                                                    # requests=[("position_estimation_module:pose")]
                                                    inputs=[
                                                        "drivers_module:images"],
                                                    log_dir=log_dir)

    def start(self):
        self.old_img_timestamp = None
        self.old_img_keypoints = None
        self.old_img_descriptors = None
        self.old_img_pose = None

        self.request_counter = 0

        # maximum numbers of keypoints to keep and calculate descriptors of,
        # reducing this number can improve computation time:
        self.max_num_keypoints = 100

        # create cv2 ORB feature descriptor and brute force matcher object
        self.orb = cv2.ORB_create(nfeatures=self.max_num_keypoints)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

        while True:
            new_img_dict = self.get("drivers_module:images")

            if not new_img_dict:
                sleep(1)
            else:
                # extract the image data and time stamp
                new_img_encoded = new_img_dict["data"]
                new_img_timestamp = new_img_dict["timestamp"]
                """
                # request the pose of the camera at this time stamp from the position_estimation_module
                self.make_request("position_estimation_module:pose", {"id" : self.request_counter, "payload": new_img_timestamp})
                self.request_counter += 1
                """
                # self.logger.debug(
                #    f"Processing image with timestamp {new_img_timestamp} ...")
                # read the image jpg into an opencv matrix
                new_img = cv2.imdecode(np.frombuffer(
                    new_img_encoded, dtype=np.int8), flags=cv2.IMREAD_COLOR)

                # first detect the ORB keypoints and then compute the feature descriptors of those points
                new_img_keypoints = self.orb.detect(new_img, None)
                new_img_keypoints, new_img_descriptors = self.orb.compute(
                    new_img, new_img_keypoints)
                # self.logger.debug(
                #    f"Found {len(new_img_keypoints)} feautures in this image")
                """
                # get the new pose and compute the difference to the old one
                pose_response = self.await_response("position_estimation_module:pose")
                new_img_pose = pose_response["payload"]
                """
                new_img_pose = 0

                # only do feature matching if there were keypoints found in the new image, discard it otherwise
                if len(new_img_keypoints) == 0:
                    pass
                    #self.logger.warn(
                    #    f"Didn't find any features in image with timestamp {new_img_timestamp}, skipping...")
                else:
                    if self.old_img_descriptors is not None:  # skip the matching step for the first image
                        # match the feature descriptors of the old and new image
                        matches = self.matcher.match(new_img_descriptors,
                                                     self.old_img_descriptors)

                        if len(matches) == 0:
                            # there were 0 matches found, print a warning
                            self.logger.warn("Couldn't find any matching features in the images with timestamps: " +
                                             f"{old_img_timestamp} and {new_img_timestamp}")

                        else:
                            # sort the matches by shortest distance first
                            matches_sorted = sorted(
                                matches, key=lambda x: x.distance)

                            # assemble the coordinates of the matched features into a numpy matrix for each image
                            new_img_match_points = np.float32(
                                [new_img_keypoints[match.queryIdx].pt for match in matches_sorted])
                            old_img_match_points = np.float32(
                                [self.old_img_keypoints[match.trainIdx].pt for match in matches_sorted])

                            # add the two matrixes together and publish the data, first dimension are all the matches,
                            # second dimension is image 1 and 2, thrid dimension is x and y
                            # e.g. 4th match, 1st image, y-coordinate: matches_paired[3][0][1]
                            #      8th match, 2nd image, x-coordinate: matches_paired[7][1][0]
                            matches_paired = np.concatenate(
                                (old_img_match_points.reshape(-1, 1, 2),
                                 new_img_match_points.reshape(-1, 1, 2)),
                                axis=1)

                            delta_pose = self.compute_delta_pose(new_img_pose)
                            feature_point_pairs_dict = {"timestamps": (self.old_img_timestamp, new_img_timestamp),
                                                        "matches": matches_paired, "delta_pose": delta_pose}
                            self.publish("feature_point_pairs", feature_point_pairs_dict,
                                         1000, new_img_timestamp)

                            self.publish("feature_point_pairs_vis", feature_point_pairs_dict,
                                         1000, new_img_timestamp)

                    # store the date of the new image as old_img... for the next iteration
                    # If there are no features found in the new image this step is skipped
                    # This means that the next image will be compared witht he same old image again
                    self.old_img_timestamp = new_img_timestamp
                    self.old_img_keypoints = new_img_keypoints
                    self.old_img_descriptors = new_img_descriptors
                    self.old_img_pose = new_img_pose

    def compute_delta_pose(self, new_pose):
        # Do some fancy calculations here but in the end it's just
        return new_pose - self.old_img_pose  # anyway
