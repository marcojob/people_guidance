from collections import namedtuple
from pathlib import Path
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from scipy.spatial.transform import Rotation


# DEFAULT_DATASET = Path("../../data/indoor_dataset_6")
# DEFAULT_DATASET = Path("../../data/outdoor_dataset_04")
DEFAULT_DATASET = Path("../../data/outdoor_dataset_04")
IMU_DATA_FILE = DEFAULT_DATASET / "imu_data.txt"
IMG_DATA_FILE = DEFAULT_DATASET / "img_data.txt"

IMU_RE_MASK = r'([0-9]*): accel_x: ([0-9.-]*), ' + \
    'accel_y: ([0-9.-]*), ' + \
    'accel_z: ([0-9.-]*), ' + \
    'gyro_x: ([0-9.-]*), ' + \
    'gyro_y: ([0-9.-]*), ' + \
    'gyro_z: ([0-9.-]*)'

ALPHA_CF = 0.9
RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0
FIGSIZE = (15, 12)
DPI = 100
G = -9.80600

RESIZED_IMAGE = (820, 616)
FAST_THRESHOLD = 30
INTRINSIC_MATRIX = np.array([[644.90127548, 0.0, 406.99519054], [
                            0.0, 644.99811417, 307.06244081], [0.0, 0.0, 1.0]])
DISTORTION_COEFFS = np.array(
    [[0.19956839, -0.49217089, -0.00235192, -0.00051292, 0.28251577]])
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
shi_tomasi_params = dict(
    maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)

# Config
method = 'REGULAR_GRID'
USE_ONLY_E = True
USE_RECOVER_POSE = True
MIN_NUM_FEATURES = 100
MAX_NUM_FEATURES = 1000
MAX_FRAME_DELTA = 5


fig = None
p1, p2, p3, p4 = None, None, None, None
cbar1, cbar2, cbar3 = None, None, None

img_window = list()

N_STATES = 10
N_MEAS = 6

class KF():
    def __init__(self):
        # World rotation
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.R_iw = None
        self.t_imu = np.array([0.0, 0.0, 0.0])

        # Timestamps
        self.timestamp_last = None
        self.timestamp = None
        self.dt = None
        self.dt_kf = None

        # State vector x: [pos x, vel x, acc x, pitch]
        self.z_prio = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(N_STATES, 1)
        self.z_post = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(N_STATES, 1)

        # Covariance matrices
        self.P_prio = np.zeros((N_STATES, N_STATES))
        self.P_post = np.zeros((N_STATES, N_STATES))

        # Process variance matrix
        self.Q = np.diag([0.5, 0.5, 0.5, 0.05, 0.05, 0.05, 0.005, 0.005, 0.005, 0.05])

        # State space model
        self.F = np.zeros((N_STATES, N_STATES))

        # Measurement matrix
        self.H = np.block([[np.eye(3), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3, 1))],
                           [np.zeros((3,3)), np.zeros((3,3)), np.eye(3), np.zeros((3,1))]])

        # Measurement variance matrix
        self.R = np.diag([0.25, 0.25, 0.25, 1.0, 1.0, 1.0])

        self.scale = 1.0

    def predict_state(self):
        f1 = self.dt_kf/self.scale
        f2 = self.dt_kf**2/(2.0*self.scale)
        f3 = -self.dt_kf/self.scale**2
        f4 = -self.dt_kf**2/(2.0*self.scale**2)

        # x(k) = F*x(k)
        self.F = np.block([[np.eye(3), f1*np.eye(3), f2*np.eye(3), f3*np.ones((3,1)) - f4*np.ones((3,1))],
                     [np.zeros((3,3)), np.eye(3), self.dt_kf*np.eye(3), np.zeros((3,1))],
                     [np.zeros((3,3)), np.zeros((3,3)), np.eye(3), np.zeros((3,1))],
                     [np.zeros((1,9)), 1]])

        # Predict state
        self.z_prio = np.dot(self.F, self.z_post)

        # Predict covariance
        self.P_prio = np.dot(np.dot(self.F, self.P_post), self.F.T) + self.Q

    def measurement_update(self, t_vo):
        # Inversion helper matrix
        inv = np.linalg.pinv(
            np.dot(np.dot(self.H, self.P_prio), self.H.T) + self.R)

        # Kalman gain
        self.K = np.dot(np.dot(self.P_prio, self.H.T), inv)

        # Innovation
        self.y_meas = np.block([self.z_post[0:3].reshape(3,) + t_vo, self.t_imu]).reshape((N_MEAS, 1))
        self.innovation = self.y_meas - \
            np.dot(self.H, self.z_prio)  # Innovation matrix

        # Posterior update
        self.z_post = self.z_prio + np.dot(self.K, self.innovation)

        # Update scale
        self.scale = self.z_post[-1]

        temp = np.eye(N_STATES) - np.dot(self.K, self.H)
        self.P_post = np.dot(np.dot(temp, self.P_prio), temp.T) + \
            np.dot(np.dot(self.K, self.R), self.K.T)


        print(self.P_post[0][0], self.P_post[1][1], self.P_post[2][2])

    def update(self, t_vo, dt_kf):
        # Update dt
        self.dt_kf = dt_kf

        # Prediction
        self.predict_state()

        # Correct
        self.measurement_update(t_vo)

    def complementary_filter(self, data):
        # Update dt
        if self.timestamp is None:
            self.timestamp = data["timestamp"]
            return
        self.timestamp_last = self.timestamp
        self.timestamp = data["timestamp"]
        self.dt = self.timestamp - self.timestamp_last

        # Extract data
        acc_x = data["accel_x"]
        acc_y = data["accel_y"]
        acc_z = data["accel_z"]
        gyro_x = data["gyro_x"]
        gyro_y = data["gyro_y"]
        gyro_z = data["gyro_z"]

        # Estimates
        pitch_accel = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2))
        roll_accel = np.arctan2(acc_x, np.sqrt(acc_y**2 + acc_z**2))

        roll_gyro = gyro_x + \
            gyro_y*np.sin(self.pitch)*np.tan(self.roll) + \
            gyro_z*np.cos(self.pitch)*np.tan(self.roll)

        pitch_gyro = gyro_y * \
            np.cos(self.pitch) - gyro_z*np.sin(self.pitch)

        # Yaw only from gyro
        yaw_gyro = gyro_y*np.sin(self.pitch)*1.0/np.cos(
            self.roll) + gyro_z*np.cos(self.pitch)*1.0/np.cos(self.roll)

        # Apply complementary filter
        self.roll = (1.0 - ALPHA_CF)*(self.roll + roll_gyro *
                                      self.dt) + ALPHA_CF * roll_accel
        self.pitch = (1.0 - ALPHA_CF)*(self.pitch +
                                       pitch_gyro*self.dt) + ALPHA_CF * pitch_accel
        self.yaw += yaw_gyro*self.dt

        # Construct world rotation at every update
        self.R_iw = Rotation.from_euler('xyz', [-self.roll, -self.pitch, -self.yaw])

        # World translation
        self.t_imu += np.dot(self.R_iw.as_matrix(), [acc_x, acc_y, acc_z]) + np.array([0.0, 0.0, G])

    def reset_t(self):
        self.t_imu = np.array([0.0, 0.0, 0.0])


def main(img1, img2):
    # Copy images
    img1_rgb = img1.copy()
    img2_rgb = img2.copy()

    # Grayscale imgs
    img1, img2 = grayscale_imgs(img1, img2)

    # Resize images
    img1, img2 = resize_imgs(img1, img2)
    img1_rgb, img2_rgb = resize_imgs(img1_rgb, img2_rgb)

    clahe = cv2.createCLAHE(clipLimit=5.0)
    img1 = clahe.apply(img1)
    img2 = clahe.apply(img2)

    # Detect keypoints
    detector = get_detector()
    prev_kps = detect_keypoints(method, detector, img1)

    # Undistort points
    prev_kps = cv2.undistortPoints(
        prev_kps, INTRINSIC_MATRIX, DISTORTION_COEFFS, R=None, P=INTRINSIC_MATRIX).reshape(-1, 2)

    # Lukas Kanade optical flow
    prev_kps, cur_kps, diff = KLT_featureTracking(img1, img2, prev_kps)

    # Fit essential matrix
    if USE_ONLY_E:
        E, mask = cv2.findEssentialMat(
            prev_kps, cur_kps, INTRINSIC_MATRIX, cv2.RANSAC, 0.99, 1, None)
    else:
        # Assume planar motion model
        H, mask = cv2.findHomography(prev_kps, cur_kps, cv2.RANSAC, 1.0)

        # Find points that belong to that model
        prev_kps_hom = prev_kps[mask.ravel().astype(bool)]
        cur_kps_hom = cur_kps[mask.ravel().astype(bool)]

        # Find essential matrix of this model
        E, mask = cv2.findEssentialMat(
            prev_kps_hom, cur_kps_hom, INTRINSIC_MATRIX, cv2.LMEDS, 0.99, 1, mask)

    if USE_RECOVER_POSE:
        # Recover pose
        _, rot, tran, mask_cheirality = cv2.recoverPose(
            E, prev_kps, cur_kps, INTRINSIC_MATRIX, None)

        # Only take points from right direction
        prev_kps = prev_kps[mask_cheirality.ravel().astype(bool)]
        cur_kps = cur_kps[mask_cheirality.ravel().astype(bool)]

        # Stack to homography
        homography = np.hstack((rot, tran))

        # Projection matrices
        P0 = np.dot(INTRINSIC_MATRIX, np.eye(3, 4))
        P1 = np.dot(INTRINSIC_MATRIX, homography)

    else:
        homography = get_homography(E, prev_kps, cur_kps)

        # Projection matrices
        P0 = np.dot(INTRINSIC_MATRIX, np.eye(3, 4))
        P1 = np.dot(INTRINSIC_MATRIX, homography[0])

    # Triangulate with all the keypoints
    if prev_kps.shape[0] > 0 and cur_kps.shape[0] > 0:
        points3d = cv2.triangulatePoints(
            P0, P1, np.transpose(prev_kps), np.transpose(cur_kps))
        points3d = cv2.convertPointsFromHomogeneous(points3d.T)
    else:
        points3d = None
    # Show images
    img1_rgb = visualize_matches(img1_rgb, prev_kps, cur_kps)
    img2_rgb = visualize_matches(img2_rgb, cur_kps, prev_kps)

    # Stack images
    img_vis = np.hstack((img1_rgb, img2_rgb))
    #cv2.imshow("img", img_vis)
    # cv2.waitKey(0)

    global fig, p1, p2, p3, p4, cbar1, cbar2, cbar3
    fig, p1, p2, p3, p4, cbar1, cbar2, cbar3 = plot_3d(
        points3d, img_vis, fig, p1, p2, p3, p4, cbar1, cbar2, cbar3)

    return len(prev_kps), homography


def detect_keypoints(method, detector, img):
    if method == 'SHI-TOMASI':
        keypoints = cv2.goodFeaturesToTrack(img, **shi_tomasi_params)
    elif method == 'ORB':
        keypoints = detector.detect(img, None)
        keypoints, descriptors = detector.compute(img, keypoints)
    elif method == 'FAST':
        keypoints = detector.detect(img, None)
    elif method == 'REGULAR_GRID':
        keypoints = regular_grid_detector(img)
    else:
        keypoints, descriptors = detector.detectAndCompute(img, None)

    if not method == 'SHI-TOMASI':
        keypoints = np.array([x.pt for x in keypoints],
                             dtype=np.float32).reshape((-1, 2))

    return keypoints


def resize_imgs(img1, img2):
    img1 = cv2.resize(img1, RESIZED_IMAGE)
    img2 = cv2.resize(img2, RESIZED_IMAGE)
    return img1, img2


def grayscale_imgs(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1, img2


def visualize_matches(img, prev_kps, cur_kps):
    THICKNESS = 1
    GREEN = (0, 255, 0)
    for i in range(min(cur_kps.shape[0], prev_kps.shape[0])):
        img = cv2.arrowedLine(img, tuple(cur_kps[i]), tuple(
            prev_kps[i]), GREEN, THICKNESS)
        #img = cv2.circle(img, tuple(cur_kps[i]), 1, (0,0,255), 1)
        #img = cv2.circle(img, tuple(prev_kps[i]), 1, (255, 0, 0), 1)

    return img


def get_detector():
    detector = None
    if method == 'FAST':
        detector = cv2.FastFeatureDetector_create(
            threshold=FAST_THRESHOLD, nonmaxSuppression=True)
    elif method == 'ORB':
        detector = cv2.ORB_create(nfeatures=MAX_NUM_FEATURES)
    elif method == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create(MAX_NUM_FEATURES)
    elif method == 'SURF':
        detector = cv2.xfeatures2d.SURF_create(MAX_NUM_FEATURES)
    elif method == 'SHI-TOMASI':
        detector = None
    elif method == 'REGULAR_GRID':
        regular_grid_max_pts = MAX_NUM_FEATURES
    else:
        print("Unknown method")
    return detector


def regular_grid_detector(img):
    """
    Very basic method of just sampling point from a regular grid
    """
    features = list()
    height = float(img.shape[0])
    width = float(img.shape[1])
    k = height/width

    n_col = int(np.sqrt(MAX_NUM_FEATURES/k))
    n_rows = int(n_col*k)

    h_cols = int(width/n_col)
    h_rows = int(height/n_rows)

    Kp = namedtuple("Kp", "pt")

    for c in range(n_col):
        for r in range(n_rows):
            features.append(Kp(pt=(c*h_cols, r*h_rows)))

    return features


def plot_3d(points3d, img_vis, fig, p1, p2, p3, p4, cbar1, cbar2, cbar3):
    FIGSIZE = (15, 12)
    DPI = 100
    PLOT_LIM = 30

    if fig is None:
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

        p4 = fig.add_subplot(2, 1, 1)
        p1 = fig.add_subplot(2, 3, 4)
        p2 = fig.add_subplot(2, 3, 5)
        p3 = fig.add_subplot(2, 3, 6, projection='3d')

    if points3d is not None:
        x = list()
        y = list()
        z = list()
        d = list()
        for i in range(points3d.shape[0]):
            x.append(points3d[i][0][2])
            y.append(points3d[i][0][0])
            z.append(-points3d[i][0][1])
            d.append(np.sqrt(points3d[i][0][0]**2 +
                             points3d[i][0][1]**2 + points3d[i][0][2]**2))

        p1.cla()
        im = p1.scatter(y, z, c=d, vmin=np.min(d), vmax=np.max(d))
        if cbar1 is not None:
            cbar1.remove()
        cbar1 = fig.colorbar(im, ax=p1)
        p1.set_title("front")
        p1.set_xlim((-PLOT_LIM, PLOT_LIM))
        p1.set_ylim((-PLOT_LIM, PLOT_LIM))

        p2.cla()
        im = p2.scatter(x, y, c=d, vmin=np.min(d), vmax=np.max(d))
        if cbar2 is not None:
            cbar2.remove()
        cbar2 = fig.colorbar(im, ax=p2)
        p2.set_title("top")
        p2.set_xlim((-PLOT_LIM, PLOT_LIM))
        p2.set_ylim((-PLOT_LIM, PLOT_LIM))

        p3.cla()
        im = p3.scatter(x, y, z, c=d, vmin=np.min(d), vmax=np.max(d))
        if cbar3 is not None:
            cbar3.remove()
        cbar3 = fig.colorbar(im, ax=p3)

    p4.imshow(img_vis[..., ::-1])

    plt.pause(0.1)

    return fig, p1, p2, p3, p4, cbar1, cbar2, cbar3


def KLT_featureTracking(prev_img, cur_img, prev_fts):
    """Feature tracking using the Kanade-Lucas-Tomasi tracker.
    """

    # Feature Correspondence with Backtracking Check
    MIN_MATCHING_DIFF = 1
    kp2, status, error = cv2.calcOpticalFlowPyrLK(
        prev_img, cur_img, prev_fts, None, **lk_params)
    kp1, status, error = cv2.calcOpticalFlowPyrLK(
        cur_img, prev_img, kp2, None, **lk_params)

    # Verify the absolute difference between feature points
    d = abs(prev_fts - kp1).reshape(-1, 2).max(-1)
    good = d < MIN_MATCHING_DIFF

    # Error Management
    if len(d) == 0:
        print('No point correspondance.')
    # If less than 5 good points, it uses the features obtain without the backtracking check
    elif list(good).count(True) <= 5:
        print('Few point correspondances')
        return kp1, kp2, MIN_MATCHING_DIFF

    # Create new lists with the good features
    n_kp1, n_kp2 = [], []
    for i, good_flag in enumerate(good):
        if good_flag:
            n_kp1.append(kp1[i])
            n_kp2.append(kp2[i])

    # Format the features into float32 numpy arrays
    n_kp1, n_kp2 = np.array(n_kp1, dtype=np.float32), np.array(
        n_kp2, dtype=np.float32)

    # Verify if the point correspondence points are in the same pixel coordinates
    d = abs(n_kp1 - n_kp2).reshape(-1, 2).max(-1)

    # The mean of the differences is used to determine the amount of distance between the pixels
    diff_mean = np.mean(d)

    return n_kp1, n_kp2, diff_mean


def get_homography(E, prev_kps, cur_kps):
    R1, R2, t = cv2.decomposeEssentialMat(E)

    # this gives us four canidate homgographies [R1,t], [R1,−t], [R2,t], [R2,−t].
    # we create only two canidates. if the best absolute score is negative, we now we must use
    # the negative translation vector

    candidates = (np.hstack((R1, t)), np.hstack((R1, -t)),
                  np.hstack((R2, t)), np.hstack((R2, -t)))
    scores = []
    masks = []
    forward_axis = 2
    for candidate_homo in candidates:
        points3d = cv2.triangulatePoints(
            np.eye(3, 4), candidate_homo, prev_kps.T, cur_kps.T)
        points3d = cv2.convertPointsFromHomogeneous(points3d.T)

        scores.append((points3d[:, 0, forward_axis] > 0).sum())
        masks.append(points3d[:, 0, forward_axis] > 0)

    idx = int(np.argmax(scores))

    return candidates[idx], masks[idx]


def adaptive_step(len_prev_kps, curr_img, prev_img, prev_id, new_img):
    """
    Control which images we consider currently
    """
    global img_window

    # Keep track of the img window
    img_window.append(curr_img)

    # Current length of the window
    len_window = len(img_window)

    if len_prev_kps < MIN_NUM_FEATURES and len_window < MAX_FRAME_DELTA:
            # We did not observe enough features,
            # keep the prev img the same img
        prev_img = prev_img

    elif len_prev_kps > MAX_NUM_FEATURES and len_window > 1:
        # We observed too many features, make
        # window smaller again
        img_window.pop(0)
        prev_img = img_window.pop(0)
        prev_id += 2

    else:
        # We observed enough features, advance normally
        prev_img = img_window.pop(0)
        prev_id += 1

    return curr_img, prev_img, prev_id


def load_img(id):
    img_file_path = DEFAULT_DATASET / "imgs" / f"img_{id:04d}.jpg"
    with open(img_file_path, 'rb') as fp:
        img_data = fp.read()
    img = cv2.imdecode(np.frombuffer(
        img_data, dtype=np.int8), flags=cv2.IMREAD_COLOR)
    return img


def extract_timestamp(prev_id, curr_id):
    MASK = r'([0-9]*): ([0-9]*)'
    prev_ts = 0
    curr_ts = 0

    with open(IMG_DATA_FILE) as f:
        for l in f.readlines():
            out = re.search(MASK, l)
            if out is None:
                continue

            img_id = int(out.group(1))
            ts = int(out.group(2))

            if img_id == prev_id:
                prev_ts = ts/1000.0
            elif img_id == curr_id:
                curr_ts = ts/1000.0
    return prev_ts, curr_ts


if __name__ == '__main__':
    TIME_BASED = True
    SAVE_FOLDER = "/home/marco/Downloads/vis"
    SAVE = False

    incr = 1
    img_id_1 = 75
    img_id_2 = img_id_1 + incr
    img_pair = [f"img_{img_id_1:04d}.jpg", f"img_{img_id_2:04d}.jpg"]
    counter = 0
    len_kps = 0

    prev_img = load_img(img_id_1)
    curr_img = load_img(img_id_2)
    img_cnt = img_id_1

    prev_id = img_id_1
    curr_id = img_id_2

    # Create Kalman Filter object
    kf = KF()

    # Empty data list
    data_list = list()

    # Process IMU data
    with IMU_DATA_FILE.open('r') as f:
        lines = f.readlines()

        # Iterate over all lines
        for line in lines:
            out = re.search(IMU_RE_MASK, line)

            # Assemble dict with current data
            data_now = dict()
            data_now["timestamp"] = int(out.group(1))/1000.0
            data_now["accel_z"] = -float(out.group(2))
            data_now["accel_y"] = float(out.group(3))
            data_now["accel_x"] = float(out.group(4))
            data_now["gyro_z"] = -float(out.group(5))*DEG_TO_RAD
            data_now["gyro_y"] = float(out.group(6))*DEG_TO_RAD
            data_now["gyro_x"] = float(out.group(7))*DEG_TO_RAD

            # Append to data list
            data_list.append(data_now)

    while True:
        if TIME_BASED:
            img_cnt += 1
            curr_id = img_cnt
            new_img = load_img(img_cnt)
            curr_img, prev_img, prev_id = adaptive_step(
                len_kps, curr_img, prev_img, prev_id, new_img)
            curr_img = new_img

            prev_ts, curr_ts = extract_timestamp(prev_id, curr_id)

        else:
            s = input("")
            if s == ".":
                img_id += incr
            elif s == ",":
                img_id -= incr

        # Get homography from imgs
        len_kps, homography = main(prev_img, curr_img)

        R_prev_world = None

        # Process IMU data
        for d in data_list:
            ts = d["timestamp"]
            if ts >= prev_ts and ts < curr_ts:
                kf.complementary_filter(d)

                # Get world rotation at prev img
                if R_prev_world is None and kf.R_iw is not None:
                    R_prev_world = kf.R_iw.as_matrix()

        # Get world to camera rotation
        R_vo = homography[:, 0:3]
        t_vo = homography[:, 3]

        R_ci = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

        t_vo_w = np.dot(R_prev_world, np.dot(R_ci, t_vo))

        # Update KF
        kf.update(t_vo_w, curr_ts - prev_ts)

        # Reset t after image pair
        kf.reset_t()

        if SAVE:
            counter += 1
            plt.savefig(Path(SAVE_FOLDER) / f"img_{counter:04d}.jpg")
