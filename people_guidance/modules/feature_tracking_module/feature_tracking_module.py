import pathlib
import cv2
import numpy as np
import platform

from time import sleep
from scipy.spatial.transform import Rotation
from typing import Tuple

from people_guidance.modules.module import Module
from people_guidance.utils import project_path

from .utils import *

# Need this to get cv imshow working on Ubuntu 20.04
if "Linux" in platform.system():
    import gi
    gi.require_version('Gtk', '2.0')


class FeatureTrackingModule(Module):

    def __init__(self, log_dir: pathlib.Path, args=None):
        super(FeatureTrackingModule, self).__init__(name="feature_tracking_module", outputs=[("feature_point_pairs", 10), ("feature_point_pairs_vis", 10)],
                                                    inputs=["drivers_module:images"],
                                                    log_dir=log_dir)

    def start(self):
        self.fm = featureMatcher(OF_MAX_NUM_FEATURES, self.logger, self.intrinsic_matrix, method=DETECTOR, use_OF=USE_OPTICAL_FLOW, use_E=USE_E)
        self.old_timestamp = 0

        while True:
            img_dict = self.get("drivers_module:images")

            if not img_dict:
                self.logger.info("queue was empty")
            else:
                # extract the image data and time stamp
                img_encoded = img_dict["data"]["data"]
                timestamp = img_dict["data"]["timestamp"]

                self.logger.debug(f"Processing image with timestamp {timestamp} ...")

                img = cv2.imdecode(np.frombuffer(img_encoded, dtype=np.int8), flags=cv2.IMREAD_GRAYSCALE)

                if self.fm.should_initialize:
                    self.fm.initialize(img)
                else:
                    mp1, mp2 = self.fm.match(img)
                    if mp1.shape[0] > 0:
                        inliers = np.concatenate((mp1.transpose().reshape(1, 2, -1),
                                                mp2.transpose().reshape(1, 2, -1)),
                                                axis=0)

                        
                        transformations = self.fm.getTransformations()

                        visualization_img = self.visualize_matches(img, keypoints, inliers, total_nr_matches)
                        if mp1.shape[0] > 0:
                            self.publish("feature_point_pairs",
                                        {"camera_positions" : transformations,
                                        "image": visualization_img,
                                        "point_pairs": inliers,
                                        "timestamp_pair": (self.old_timestamp, timestamp)},
                                        1000)
                            self.publish("feature_point_pairs_vis",
                                            {"point_pairs": inliers,
                                            "img": img_encoded,
                                            "timestamp": timestamp},
                                            1000)
                    self.old_timestamp = timestamp

    def visualize_matches(self, img: np.ndarray, inliers: np.ndarray) -> np.ndarray:
        for i in range(inliers.shape[2]):
            visualization_img = cv2.line(img,
                                         tuple(inliers[0,...,i]), tuple(inliers[1,...,i]),
                                         (255,0,0), 5)
            visualization_img = cv2.circle(visualization_img, tuple(inliers[1,...,i] ,1,(0,255,0),-1))

        return visualization_img

class featureMatcher:
    def __init__(self, max_num_features, logger, K, method='FAST', use_OF=False, use_E=True,):
        self.intrinsic_matrix = K
        self.logger = logger
        self.use_OF = use_OF
        self.use_E = use_E

        self.nb_transform_solutions = 0
        self.rotations = None
        self.translations = None

        self.should_initialize = True

        if not self.use_OF:
            matching_norm = None
            if method=='FAST':
                method='ORB'
                matching_norm = cv2.NORM_L2
            elif method=='ORB' or method=='SHI-TOMASI':
                matching_norm = cv2.NORM_L2
            elif method=='SIFT' or method=='SURF':
                matching_norm = cv2.NORM_HAMMING

            self.matcher = cv2.BFMatcher_create(matching_norm, crossCheck=True)

        self.detector = featureDetector(max_num_features, logger, method=method)
        self.prev_img = None
        self.prev_kp = None
        self.prev_desc = None

    def initialize(self, img):
        self.prev_img = img
        self.prev_kp, self.prev_desc = self.detector.detect(img)
        self.should_initialize = False

    def match(self, img):
        if self.use_OF:
            mp1, mp2, diff = self.KLT_featureTracking(img)
        else:
            mp1, mp2 = self.bruteForceMatching(img)

        mask = self.calcTransformation(kp1, kp2)

        mp1 = mp1[mask.ravel().astype(bool)]
        mp2 = mp2[mask.ravel().astype(bool)]

        return mp1, mp2

    def bruteForceMatching(self, new_img):
        new_kp, new_desc = self.detector.detect(new_img)
        matches = self.matcher.match(self.prev_desc, new_desc)

        old_match_points = np.float32([self.prev_kp[match.queryIdx].pt for match in matches])
        match_points = np.float32([new_kp[match.trainIdx].pt for match in matches])

        return old_match_points, match_points


    def KLT_featureTracking(self, new_img):
        """Feature tracking using the Kanade-Lucas-Tomasi tracker.
        """

        # Feature Correspondence with Backtracking Check
        kp2, status, error = cv2.calcOpticalFlowPyrLK(self.prev_img, new_img, self.prev_kp, None, **lk_params)
        kp1, status, error = cv2.calcOpticalFlowPyrLK(new_img, self.prev_img, kp2, None, **lk_params)

        d = abs(self.prev_kp - kp1).reshape(-1, 2).max(-1)  # Verify the absolute difference between feature points
        good = d < MIN_MATCHING_DIFF

        # Error Management
        if len(d) == 0:
            self.logger.warning('No point correspondance.')
        elif list(good).count(True) <= 5:  # If less than 5 good points, it uses the features obtain without the backtracking check
            self.logger.warning('Few point correspondances')
            return kp1, kp2, None

        # Create new lists with the good features
        n_kp1, n_kp2 = [], []
        for i, good_flag in enumerate(good):
            if good_flag:
                n_kp1.append(kp1[i])
                n_kp2.append(kp2[i])

        # Format the features into float32 numpy arrays
        n_kp1, n_kp2 = np.array(n_kp1, dtype=np.float32), np.array(n_kp2, dtype=np.float32)

        # Verify if the point correspondence points are in the same pixel coordinates
        d = abs(n_kp1 - n_kp2).reshape(-1, 2).max(-1)

        # The mean of the differences is used to determine the amount of distance between the pixels
        diff_mean = np.mean(d)

        if diff_mean > DIFF_THRESHOLD:
            self.prev_img = new_img
            self.prev_kp = n_kp2
            return n_kp1, n_kp2, diff_mean
        else:
            self.should_initialize = True
            return np.array([]), np.array([]), None

    def calcTransformation(self, mp1, mp2):
        if not self.use_E:
            # if we found enough matches do a RANSAC search to find inliers corresponding to one homography
            H, mask = cv2.findHomography(mp1, mp2, cv2.RANSAC, 1.0)
            self.nb_transform_solutions, self.rotations, self.translations, _ = cv2.decomposeHomographyMat(H, self.intrinsic_matrix)
            return mask
        else:
            E, mask = cv2.findEssentialMat(mp1, mp2, self.intrinsic_matrix, cv2.RANSAC, 0.999, 1.0, None)
            _, self.rotations, self.translations, _ = cv2.recoverPose(E, mp1, mp2, self.intrinsic_matrix, mask)
            self.nb_transform_solutions = 1
            return mask

    def getTransformations(self):
        if self.nb_transform_solutions > 0:
            transformations = np.zeros((nb_solutions, 3, 4))
            for i in range(self.nb_transform_solutions):
                transformations[i, :, 0:3] = self.rotations[i]
                transformations[i, :, 3] = self.translations[i].ravel()
            return transformations
        else:
            return np.zeros((1,3,4))


class featureDetector:
    def __init__(self, max_num_features, logger, method='FAST'):
        self.logger = logger
        self.method = method
        self.max_num_features = max_num_features
        self.detector = None
        methods = {'FAST': cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True),
                   'SIFT': cv2.xfeatures2d.SIFT_create(max_num_features),
                   'SURF': cv2.xfeatures2d.SURF_create(max_num_features),
                   'SHI-TOMASI': cv2.ORB_create(nfeatures=max_num_features),
                   'ORB': cv2.ORB_create(nfeatures=max_num_features)}
        try:
            self.detector = methods[method]
        except keyError:
            self.logger.warn(method + "detector is not available")

    def detect(self, img: np.array):
        keypoints = None
        descriptors = None

        if self.method == 'SHI-TOMASI':
            keypoints = cv2.goodFeaturesToTrack(img, **shi_tomasi_params)
            keypoints, descriptors = self.detector.compute(img, keypoints)
        elif self.method == 'ORB':
            keypoints = self.detector.detect(img, None)
            keypoints, descriptors = self.detector.compute(img, keypoints)
        elif self.method == 'FAST':
            keypoints = self.detector.detect(img, None)
        else:
            keypoints, descriptors = self.detector.detectAndCompute(img, None)
        
        self.logger.debug(f"Found {len(keypoints)} feautures")

        return (keypoints, descriptors)
