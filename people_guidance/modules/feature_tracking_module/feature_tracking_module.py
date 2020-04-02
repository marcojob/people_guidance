import pathlib

from time import sleep

from people_guidance.modules.module import Module
from people_guidance.utils import project_path

import cv2
import numpy as np
import matplotlib.pyplot as plt


class FeatureTrackingModule(Module):

    def __init__(self, log_dir: pathlib.Path, args=None):
        super(FeatureTrackingModule, self).__init__(name="feature_tracking_module", outputs=[("feature_point_pairs_orb", 10), ("feature_point_pairs_surf", 10)],
                                                    inputs=["drivers_module:images"], #requests=[("position_estimation_module:pose")]
                                                    log_dir=log_dir)

    def cleanup(self):
        plt.close('all')

    def start(self):
        self.old_timestamp = None
        self.old_keypoints = [None, None]
        self.old_descriptors = [None, None]
        self.old_pose = None

        self.inliers = [None, None]
        self.visualization = [None, None]

        self.request_counter = 0

        # maximum numbers of keypoints to keep and calculate descriptors of,
        # reducing this number can improve computation time:
        self.max_num_keypoints = 1000

        # create each an ORB and a SURF feature descriptor and brute force matcher object
        self.ORB = 0
        self.SURF = 1
        self.orb = cv2.ORB_create(nfeatures=self.max_num_keypoints)
        self.orb_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
        self.surf = cv2.xfeatures2d.SURF_create()
        self.surf_matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

        plt.ion()
        figure, (orb_plot, surf_plot) = plt.subplots(1,2)
        orb_plot.set_title("ORB")
        surf_plot.set_title("SURF")

        while True:
            img_dict = self.get("drivers_module:images")

            if not img_dict:
                sleep(0.1)
                self.logger.warn("queue was empty")
            else:
                # extract the image data and time stamp
                img_encoded = img_dict["data"]
                timestamp = img_dict["timestamp"]
                """
                # request the pose of the camera at this time stamp from the position_estimation_module
                self.make_request("position_estimation_module:pose", {"id" : self.request_counter, "payload": timestamp})
                self.request_counter += 1
                """
                # self.logger.debug(f"Processing image with timestamp {timestamp} ...")

                img = cv2.imdecode(np.frombuffer(img_encoded, dtype=np.int8), flags=cv2.IMREAD_GRAYSCALE)

                for i in range(2):
                    keypoints, descriptors = self.extract_feature_descriptors(img, i)

                    """
                    # get the new pose and compute the difference to the old one
                    pose_response = self.await_response("position_estimation_module:pose")
                    pose = pose_response["payload"]
                    """
                    pose = np.zeros((3,4))
                    
                    # only do feature matching if there were keypoints found in the new image, discard it otherwise
                    if len(keypoints) == 0:
                        pass
                        # self.logger.warn(f"Didn't find any features in image with timestamp {timestamp}, skipping...")
                    else:
                        if self.old_descriptors is not None:  # skip the matching step for the first image
                            # match the feature descriptors of the old and new image

                            self.inliers[i], total_nr_matches = self.match_features(keypoints, descriptors, i)

                            if  self.inliers[i].shape[2] == 0:
                                self.visualization[i] = img
                                pass
                                # there were 0 inliers found, print a warning
                                # self.logger.warn("Couldn't find any matching features in the images with timestamps: " +
                                #                 f"{old_timestamp} and {timestamp}")
                            else:
                                pose_pair = np.concatenate((self.old_pose[np.newaxis, :, :], pose[np.newaxis, :, :]), axis=0)
                                self.visualization[i] = self.visualize_matches(img, keypoints, self.inliers[i], total_nr_matches)

                        # store the date of the new image as old_img... for the next iteration
                        # If there are no features found in the new image this step is skipped
                        # This means that the next image will be compared witht he same old image again
                        self.old_timestamp = timestamp
                        self.old_keypoints[i] = keypoints
                        self.old_descriptors[i] = descriptors
                        self.old_pose = pose

                self.publish("feature_point_pairs_orb",
                            {"camera_positions" : (pose, pose),
                                "point_pairs": self.inliers[i]},
                                1000, timestamp)
                self.publish("feature_point_pairs_surf",
                            {"camera_positions" : (pose, pose),
                                "point_pairs": self.inliers[i]},
                                1000, timestamp)

                orb_plot.imshow(self.visualization[0])
                surf_plot.imshow(self.visualization[1])
                figure.show()
                plt.waitforbuttonpress(0.001)

    def extract_feature_descriptors(self, img: np.ndarray, method: int) -> (list, np.ndarray):
        if method == 0:
            # first detect the ORB keypoints and then compute the feature descriptors of those points
            keypoints = self.orb.detect(img, None)
            keypoints, descriptors = self.orb.compute(img, keypoints)
        elif method == 1:
            # surf detects and describes the feature in one function
            keypoints, descriptors = self.surf.detectAndCompute(img, None)
        else:
            self.logger.warn("Unknown feature extraction method, defaulting to orbs...")
            keypoints = self.orb.detect(img, None)
            keypoints, descriptors = self.orb.compute(img, keypoints)
        
        self.logger.debug(f"Found {len(keypoints)} feautures")

        return (keypoints, descriptors)
    
    def match_features(self, keypoints: list, descriptors: np.ndarray, method: int) -> np.ndarray:
        if method == self.ORB:
            matches = self.orb_matcher.match(self.old_descriptors[method], descriptors)
        elif method == self.SURF:
            matches = self.surf_matcher.match(self.old_descriptors[method], descriptors)
        else:
            self.logger.warn("Unknown matching method, defaulting to orbs...")
            matches = self.orb_matcher.match(self.old_descriptors[method], descriptors)
        # sort the matches by shortest distance first
        # matches_sorted = sorted(matches, key=lambda x: x.distance)

        # assemble the coordinates of the matched features into a numpy matrix for each image
        old_match_points = np.float32([self.old_keypoints[method][match.queryIdx].pt for match in matches])
        match_points = np.float32([keypoints[match.trainIdx].pt for match in matches])

        if len(matches) > 10:
            # if we found enough matches do a RANSAC search to find inliers corresponding to one homography
            # TODO: add camera pose info to improve matching
            _, mask = cv2.findHomography(old_match_points, match_points, cv2.RANSAC, 1.0)
            old_match_points = old_match_points[mask.ravel().astype(bool)]
            match_points = match_points[mask.ravel().astype(bool)]
        else:
            mask = list()

        # add the two matrixes together, first dimension are all the matches,
        # second dimension is image 1 and 2, thrid dimension is x and y
        # e.g. 4th match, 1st image, y-coordinate: matches_paired[3][0][1]
        #      8th match, 2nd image, x-coordinate: matches_paired[7][1][0]
        matches_paired = np.concatenate(
            (old_match_points.transpose().reshape(1, 2, -1),
             match_points.transpose().reshape(1, 2, -1)),
             axis=0)

        total_nr_matches = len(matches) if len(matches) <= 10 else len(mask)

        return (matches_paired, total_nr_matches)

    def visualize_matches(self, img, keypoints, inliers, nb_matches):
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