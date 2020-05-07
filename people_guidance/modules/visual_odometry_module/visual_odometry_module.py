import cv2
import math
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
                                                            ("features_vis", 10),
                                                            ("cloud", 10)],
                                                   inputs=["drivers_module:images",
                                                           "drivers_module:accelerations"],
                                                   log_dir=log_dir)

    def start(self):
        cf = ComplementaryFilter()

        vo = VisualOdometry(cf, self.logger, self.intrinsic_matrix)

        img_id = 0
        traj = np.zeros((600, 600, 3), dtype=np.uint8)

        last_timestamp = 0
        timestamp = 0

        while True:
            # Process all the imu data, this loop usally takes max 2ms
            while True:
                imu_dict = self.get("drivers_module:accelerations")
                if imu_dict:
                    imu_data = imu_dict["data"]
                    frame = {"ax": imu_data["accel_x"],
                             "ay": imu_data["accel_y"],
                             "az": imu_data["accel_z"],
                             "gx": imu_data["accel_x"],
                             "gy": imu_data["gyro_x"],
                             "gz": imu_data["gyro_x"],
                             "ts": imu_data["timestamp"]}

                    cf.update(frame)
                else:
                    break

            img_dict = self.get("drivers_module:images")
            if img_dict:
                # Convert img to grayscale
                img_encoded = img_dict["data"]["data"]
                img = cv2.imdecode(np.frombuffer(img_encoded, dtype=np.int8), flags=cv2.IMREAD_GRAYSCALE)

                clahe = cv2.createCLAHE(clipLimit=5.0)
                img = clahe.apply(img)

                last_timestamp = timestamp
                timestamp = img_dict["data"]["timestamp"]

                # Update VO based on image
                vo.update(img, img_id, timestamp)
                img_id += 1

                # Publish to visualization
                if vo.stage == STAGE_DEFAULT:
                    euler = Rotation.from_matrix(vo.cur_r).as_euler('zyx', degrees=True)
                    data_dict = {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 0.0,
                        "timestamp": timestamp
                    }
                    self.publish("position_vis", data_dict, 1000)

                    point_pairs = list()
                    for new, old in zip(vo.OFF_cur, vo.OFF_prev):
                        point_pairs.append((new, old))

                    self.publish("features_vis",
                                {"point_pairs": point_pairs,
                                 "img": img_encoded,
                                 "cloud": vo.new_cloud,
                                 "timestamp": timestamp}, 1000)

                    x, y, z = vo.cur_t[0], vo.cur_t[1], vo.cur_t[2]
                    # traj = RT_trajectory_window(traj, x, y, z, img_id)  # Draw the trajectory window

                    # Publish to reprojection_module
                    self.publish("cloud",
                                {"cloud": vo.new_cloud,
                                 "homography": vo.homography,
                                 "timestamps": (last_timestamp, timestamp)}, 1000)


class VisualOdometry:
    def __init__(self, cf, logger, intrinsic_matrix, detector=DETECTOR):
        # Give access to complementary filter class instance
        self.cf = cf

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

        self.new_timestamp = 0
        self.last_timestamp = 0

        # Integrated accelerations
        self.ax_cur = 0.0
        self.ay_cur = 0.0
        self.az_cur = 0.0

        self.ax_prev = 0.0
        self.ay_prev = 0.0
        self.az_prev = 0.0

        self.homography = np.zeros((3,4))

    def update(self, img, frame_id, timestamp):
        self.new_frame = img
        self.new_timestamp = timestamp

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
        self.last_timestamp = timestamp

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

        # Create homography
        self.homography = np.concatenate((self.cur_r, self.cur_t), axis=1)

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

        # Get scale
        if USE_RELATIVE_SCALE:
            self.scale = self.get_relative_scale()
        else:
            self.scale = self.get_absolute_scale()

        # Continue tracking of movement
        self.cur_t = self.cur_t + self.scale * self.cur_r.dot(t)  # Concatenate the translation vectors
        self.cur_r = r.dot(self.cur_r)  # Concatenate the rotation matrix

        # Triangulate points
        self.new_cloud = self.triangulatePoints(self.cur_r, self.cur_t)

        # Create homography
        self.homography = np.concatenate((self.cur_r, self.cur_t), axis=1)

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

        cloud_homo = cv2.triangulatePoints(P0, P1, point1, point2)
        cloud = cv2.convertPointsFromHomogeneous(cloud_homo.T).reshape(-1, 3)
        return cloud

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

    def get_absolute_scale(self):
        ts_prev = self.last_timestamp
        ts_cur = self.new_timestamp

        # Robocentric, gravity corrected accelerations
        self.ax_prev = self.ax_cur
        self.ay_prev = self.ay_cur
        self.az_prev = self.az_cur

        self.ax_cur, self.ay_cur, self.az_cur = self.cf.integrate(ts_prev, ts_cur)

        # Scale
        scale = np.sqrt((self.ax_cur - self.ax_prev)**2 + (self.ay_cur - self.ay_prev)**2 + (self.az_cur - self.az_prev)**2)

        return scale

class ComplementaryFilter:
    # https://www.mdpi.com/1424-8220/15/8/19302/htm
    def __init__(self, ALPHA = 0.4):
        self.last_frame = None
        self.current_frame = {"ax": 0.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 0.0, "gz": 0.0, "ts": 0}
        self.alpha = ALPHA

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Orientation of global frame with respect to local frame
        self.q_g_l = np.array([1.0, 0.0, 0.0, 0.0])

        # Gravity corrected frames in IMU coordinate system
        self.frames_corrected = list()

    def integrate(self, ts_prev, ts_cur):
        ax_int = 0.0
        ay_int = 0.0
        az_int = 0.0
        for idx, frame in enumerate(self.frames_corrected):
            # Find all frames between the two timestamps
            if frame["ts"] >= ts_prev and frame["ts"] < ts_cur:
                # Add up accelerationss
                ax_int += frame["ax"]
                ay_int += frame["ay"]
                az_int += frame["az"]

                # Remove the element
                del self.frames_corrected[idx]

            elif frame["ts"] < ts_prev:
                # Discard too old frames immediately
                del self.frames_corrected[idx]

        return ax_int, ay_int, az_int

    def update(self, frame: dict):
        if self.last_frame is None:
            self.last_frame = frame
            self.current_frame = frame
        else:
            self.last_frame = self.current_frame
            self.current_frame = frame
            dt_s = (self.last_frame["ts"] - self.current_frame["ts"])/1000.0

            # Normalized accelerations, - frame.az is needed so that roll is not at 180 degrees
            a_l = np.array([frame["ax"], frame["ay"], -frame["az"]])
            a_l_norm = np.linalg.norm(a_l)
            a_l /= a_l_norm

            # PREDICTION
            w_q_l = np.array([0.0, frame["gx"], frame["gy"], frame["gz"]])*DEG_TO_RAD
            w_q_l /= np.linalg.norm(w_q_l)

            # Gyro based attitude velocity of global frame with respect to local frame
            q_w_dot_g_l = quaternion_multiply(-0.5*w_q_l, self.q_g_l)

            # Gyro based attitude
            q_w_g_l = self.q_g_l + q_w_dot_g_l * dt_s
            q_w_g_l /= np.linalg.norm(q_w_g_l)

            # Inverse gyro based attitude
            q_w_l_g = quaternion_conjugate(q_w_g_l)

            # CORRECTION
            R_q_w_l_g = quaternion_R(q_w_l_g)

            # Compute a prediction for the gravity vector
            g_predicted_g = np.dot(R_q_w_l_g, a_l)

            # Compute delta q acc
            gx, gy, gz = g_predicted_g
            gz_1 = gz + 1.0
            delta_q_acc = np.array([math.sqrt(gz_1/2.0), -gy/math.sqrt(2.0*gz_1), gx/math.sqrt(2.0*gz_1), 0.0])
            delta_q_acc_norm = np.linalg.norm(delta_q_acc)

            delta_q_acc /= delta_q_acc_norm

            q_identity = np.array([1.0, 0.0, 0.0, 0.0])

            # Omega is given by angle subtended by the two quaternions, in our case just:
            omega = delta_q_acc[0]

            if omega > LERP_THRESHOLD:
                delta_q_acc_hat = (1.0 - self.alpha)*q_identity + self.alpha*delta_q_acc
            else:
                delta_q_acc_hat = math.sin((1.0 - self.alpha)*omega)/math.sin(omega)*q_identity + math.sin(self.alpha*omega)/math.sin(omega)*delta_q_acc

            delta_q_acc_hat_norm = np.linalg.norm(delta_q_acc_hat)
            delta_q_acc_hat /= delta_q_acc_hat_norm

            # UPDATE
            self.q_g_l = quaternion_multiply(q_w_g_l, delta_q_acc_hat)

            error = abs(np.linalg.norm(np.array([frame["ax"], frame["ay"], frame["az"]])) - G_ACCEL) / G_ACCEL
            if error < ERROR_T_LOW:
                self.alpha = ALPHA_BAR*1.0
            elif error < ERROR_T_HIGH:
                self.alpha = ALPHA_BAR*error/(ERROR_T_HIGH - ERROR_T_LOW)
            else:
                self.alpha = 0.0

            local_gravity = quaternion_apply(self.q_g_l, [0, 0, 1])[1:] * G_ACCEL

            new_frame_corrected = {"ax": frame["ax"] - local_gravity[0], "ay": frame["ay"] - local_gravity[1], "az": frame["az"] - local_gravity[2], "ts": frame["ts"]}
            self.frames_corrected.append(new_frame_corrected)

# Quaternion multiply according to Valenti, 2015
def quaternion_multiply(p, q):
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    output = np.array([p0*q0 - p1*q1 - p2*q2 - p3*q3,
                       p0*q1 + p1*q0 + p2*q3 - p3*q2,
                       p0*q2 - p1*q3 + p2*q0 + p3*q1,
                       p0*q3 + p1*q2 - p2*q1 + p3*q0], dtype=np.float64)
    output /= np.linalg.norm(output)
    return output


def quaternion_R(q):
    q0, q1, q2, q3 = q
    R = np.array([[q0**2 + q1**2 - q2**2 - q3**2, 2.0*(q1*q2 - q0*q3), 2.0*(q1*q3 + q0*q2)],
                  [2.0*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2.0*(q2*q3 - q0*q1)],
                  [2.0*(q1*q3 - q0*q2), 2.0*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]])
    return R


def quaternion_apply(quaternion, vector):
    q2 = np.concatenate((np.array([0.0]), np.array(vector)))
    return quaternion_multiply(quaternion_multiply(quaternion, q2),
                               quaternion_conjugate(quaternion))


def quaternion_conjugate(quaternion):
    w, x, y, z = quaternion
    return np.array([w, -x, -y, -z])

def quaternion_to_euler(quaternion, degrees=True):
    return Rotation.from_quat(quaternion).as_euler('zyx', degrees=degrees)

def euler_to_quaternion(euler, degrees=True):
    return Rotation.from_euler('zyx', euler, degrees=degrees).as_quat()