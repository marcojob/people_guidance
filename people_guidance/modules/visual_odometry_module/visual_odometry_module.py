import cv2
import numpy as np
import pathlib

from time import sleep
from scipy.spatial.transform import Rotation

from people_guidance.modules.module import Module
from people_guidance.utils import project_path

from .utils import *


class VisualOdometryModule(Module):

    def __init__(self, log_dir: pathlib.Path, args=None):
        super(VisualOdometryModule, self).__init__(name="visual_odometry_module",
                                                   outputs=[("position_vis", 100),
                                                            ("features_vis", 10)],
                                                   inputs=["drivers_module:images",
                                                           "drivers_module:accelerations"],
                                                   log_dir=log_dir)

    def start(self):
        img_id = 0

        vo = VisualOdometry(self.logger, self.intrinsic_matrix)
        #vo = VisualOdometry(self.intrinsic_matrix, 'SIFT', None)
        ts_prev = 0
        ts_cur = 0

        traj = np.zeros((600, 600, 3), dtype=np.uint8)

        while True:
            img_dict = self.get("drivers_module:images")
            if not img_dict:
                self.logger.info("queue was empty")
                sleep(0.001)
            else:
                # Convert img to grayscale
                img_encoded = img_dict["data"]["data"]
                img = cv2.imdecode(np.frombuffer(img_encoded, dtype=np.int8), flags=cv2.IMREAD_GRAYSCALE)
                clahe = cv2.createCLAHE(clipLimit=5.0)
                img = clahe.apply(img)

                # Update VO based on image
                vo.update(img, img_id)
                img_id += 1
                ts = 0

                # Publish to visualization
                if vo.stage == STAGE_DEFAULT:
                    euler = Rotation.from_matrix(vo.cur_r).as_euler('zyx', degrees=True)
                    data_dict = {
                        "x": vo.cur_t[2][0],
                        "y": vo.cur_t[0][0],
                        "z": vo.cur_t[1][0],
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 0.0,
                        "timestamp": ts
                    }
                    self.publish("position_vis", data_dict, 1000)

                    point_pairs = list()
                    for new, old in zip(vo.OFF_cur, vo.OFF_prev):
                        point_pairs.append((new, old))

                    self.publish("features_vis",
                                {"point_pairs": point_pairs,
                                 "img": img_encoded,
                                 "timestamp": ts}, 1000)

                    x, y, z = vo.cur_t[0], vo.cur_t[1], vo.cur_t[2]
                    traj = RT_trajectory_window(traj, x, y, z, img_id)  # Draw the trajectory window

class VisualOdometry:
    def __init__(self, logger, intrinsic_matrix, detector=DETECTOR):
        # Carry over elements from main class
        self.logger = logger
        self.intrinsic_matrix = intrinsic_matrix


        avail_detectors = {
                       'FAST': cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True),
                       'SIFT': cv2.xfeatures2d.SIFT_create(MAX_NUM_FEATURES),
                       'SURF': cv2.xfeatures2d.SURF_create(MAX_NUM_FEATURES),
                       'SHI-TOMASI': 'SHI-TOMASI'}
        self.detector = avail_detectors[detector]

        self.new_frame = None
        self.stage = STAGE_FIRST # Start with first stage

        self.prev_fts = None # Previous features
        self.cur_fts = None # Current features

        self.cur_r = None # Current rotation
        self.cur_t = None # Current translation

        # List of t vectors and R matrices
        self.t_vects = list()
        self.r_mats = list()

        # Point clouds
        self.new_cloud = np.array([])
        self.last_cloud = np.array([])

        # Scale
        self.scale = 0.0

        # Optical Flow Field
        self.OFF_cur = None
        self.OFF_prev = None

        # Frame skip flag
        self.do_frame_skip = False

    def update(self, img, frame_id):
        self.new_frame = img

        # Algorithm stage handling
        if self.stage == STAGE_DEFAULT:
            self.process_frame(frame_id)
        elif self.stage == STAGE_SECOND:
            self.process_second_frame()
        elif self.stage == STAGE_FIRST:
            self.process_first_frame()

        # Update last frame
        self.last_id = frame_id
        self.last_frame = self.new_frame

    def process_first_frame(self):
        """ Find feature points in first frame for Kanade-Lucas-Tomasi Tracker
        """
        self.prev_fts = self.detect_new_features(self.new_frame)

        # First t & R are zeros
        self.t_vects.append(np.zeros((3,1)))
        self.r_mats.append(np.zeros((3,3)))

        # Advance stage
        self.stage = STAGE_SECOND

    def process_second_frame(self):
        """ Processes first and second frame
        """

        # Update local img variables
        prev_img = self.last_frame
        cur_img = self.new_frame

        self.prev_fts, self.cur_fts, diff = self.KLT_featureTracking(prev_img, cur_img, self.prev_fts)

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(self.cur_fts, self.prev_fts, self.intrinsic_matrix, method=cv2.RANSAC, prob= 0.999, threshold=1.0)

        # Recover pose, meaning rotation and translation
        _, self.cur_r, self.cur_t, mask = cv2.recoverPose(E, self.cur_fts, self.prev_fts, self.intrinsic_matrix)

        # Keep track of rotations and translation
        self.t_vects.append(self.cur_r.dot(self.cur_t))
        self.r_mats.append(self.cur_r)

        # Triangulate points
        self.new_cloud = self.triangulatePoints(self.cur_r, self.cur_t)

        # Optical flow field vars
        self.OFF_prev = self.prev_fts
        self.OFF_cur = self.cur_fts

        # Advance to next stage
        self.stage = STAGE_DEFAULT

        # Update points
        self.prev_fts = self.cur_fts
        self.last_cloud = self.new_cloud


    def process_frame(self, frame_id):
        """ Processes general frames
        """

        # Update local img variables
        prev_img = self.last_frame
        cur_img = self.new_frame

        # If we don't have enough features, detect new ones
        if self.prev_fts.shape[0] < MIN_NUM_FEATURES:
            self.cur_fts = self.detect_new_features(cur_img)
            self.prev_fts = self.detect_new_features(prev_img)

        self.prev_fts, self.cur_fts, diff = self.KLT_featureTracking(prev_img, cur_img, self.prev_fts)

        # If the difference between images is small, then we can skip it
        self.do_frame_skip = self.skip_frame(diff)
        if self.do_frame_skip:
            self.logger.debug(f"Skipping frame with diff: {diff}")
            # If we decide to skip the feature, we try to detect more features to make it more robust
            if self.prev_fts.shape[0] < MIN_NUM_FEATURES:
                self.cur_fts = self.detect_new_features(prev_img)

                # Update fts and cloud
                self.prev_fts = self.cur_fts
                self.last_cloud = self.new_cloud
            return

        # If we don't skip, continue normally
        E, mask = cv2.findEssentialMat(self.cur_fts, self.prev_fts, self.intrinsic_matrix, method=cv2.RANSAC, prob= 0.999, threshold=1.0)

        # Recover pose, meaning rotation and translation
        _, r, t, mask = cv2.recoverPose(E, self.cur_fts, self.prev_fts, self.intrinsic_matrix)

        # Triangulate points
        self.new_cloud = self.triangulatePoints(self.cur_r, self.cur_t)

        # Get scale
        self.scale = self.get_relative_scale()

        # Continue tracking of movement
        self.cur_t = self.cur_t + self.scale * self.cur_r.dot(t)  # Concatenate the translation vectors
        self.cur_r = r.dot(self.cur_r)  # Concatenate the rotation matrix

        # Append vectors
        self.t_vects.append(self.cur_t)
        self.r_mats.append(self.cur_r)

        # Optical flow field vars
        self.OFF_prev = self.prev_fts
        self.OFF_cur = self.cur_fts

        # Update points
        self.prev_fts = self.cur_fts
        self.last_cloud = self.new_cloud


    def detect_new_features(self, img):
        """ Detect features using selected detector
        """
        if self.detector == 'SHI-TOMASI':
            feature_pts = cv2.goodFeaturesToTrack(img, **feature_params)
            feature_pts = np.array([x for x in feature_pts], dtype=np.float32).reshape((-1, 2))
        else:
            feature_pts = self.detector.detect(img, None)
            feature_pts = np.array([x.pt for x in feature_pts], dtype=np.float32)

        return feature_pts


    def KLT_featureTracking(self, prev_img, cur_img, prev_fts):
        """Feature tracking using the Kanade-Lucas-Tomasi tracker.
        """

        # Feature Correspondence with Backtracking Check
        kp2, status, error = cv2.calcOpticalFlowPyrLK(prev_img, cur_img, prev_fts, None, **lk_params)
        kp1, status, error = cv2.calcOpticalFlowPyrLK(cur_img, prev_img, kp2, None, **lk_params)

        d = abs(prev_fts - kp1).reshape(-1, 2).max(-1)  # Verify the absolute difference between feature points
        good = d < MIN_MATCHING_DIFF

        # Error Management
        if len(d) == 0:
            self.logger.warning('No point correspondance.')
        elif list(good).count(True) <= 5:  # If less than 5 good points, it uses the features obtain without the backtracking check
            self.logger.warning('Few point correspondances')
            return kp1, kp2, MIN_MATCHING_DIFF

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

        return n_kp1, n_kp2, diff_mean

    def triangulatePoints(self, R, t):
        """Triangulates the feature correspondence points with
        the camera intrinsic matrix, rotation matrix, and translation vector.
        It creates projection matrices for the triangulation process."""

        # The canonical matrix (set as the origin)
        P0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
        P0 = self.intrinsic_matrix.dot(P0)

        # Rotated and translated using P0 as the reference point
        P1 = np.hstack((R, t))
        P1 = self.intrinsic_matrix.dot(P1)

        # Reshaped the point correspondence arrays to cv2.triangulatePoints's format
        point1 = self.prev_fts.reshape(2, -1)
        point2 = self.cur_fts.reshape(2, -1)

        return cv2.triangulatePoints(P0, P1, point1, point2).reshape(-1, 4)[:, :3]

    def skip_frame(self, diff):
        """ Skip a frame if the difference is smaller than a certain value.
            Small difference means the frame almost did not change.
        """
        if diff == 0.0:
            return False
        else:
            return diff < DIFF_THRESHOLD

    def get_relative_scale(self):
        """ Returns the relative scale based on point cloud.
            This is a debug method, if we cannot get the real scale.
        """
        # Find the minimum number of elements of both clouds
        min_idx = min([self.new_cloud.shape[0], self.last_cloud.shape[0]])

        ratios = []  # List to obtain all the ratios of the distances

        # Iterate over the points
        for i in range(1, min_idx):
            Xk = self.new_cloud[i]
            p_Xk = self.new_cloud[i - 1]

            Xk_1 = self.last_cloud[i]
            p_Xk_1 = self.last_cloud[i - 1]

            # Calculate scaling
            if np.linalg.norm(p_Xk - Xk) != 0:
                ratios.append(np.linalg.norm(p_Xk_1 - Xk_1) / np.linalg.norm(p_Xk - Xk))

        # Take the median of ratios list as the final ratio
        d_ratio = np.median(ratios)
        return d_ratio
