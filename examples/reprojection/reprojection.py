from pathlib import Path
from time import sleep

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DEFAULT_DATASET = Path("../../data/outdoor_dataset_19")
#DEFAULT_DATASET = Path("../../data/indoor_dataset_4")
RESIZED_IMAGE = (820, 616)
FAST_THRESHOLD = 30
INTRINSIC_MATRIX = np.array([[644.90127548, 0.0, 406.99519054], [0.0, 644.99811417, 307.06244081], [0.0, 0.0, 1.0]])
DISTORTION_COEFFS = np.array([[ 0.19956839 , -0.49217089, -0.00235192, -0.00051292, 0.28251577]])
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
shi_tomasi_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)

# Config
method = 'FAST'
USE_ONLY_E = True
USE_RECOVER_POSE = True
MIN_NUM_FEATURES = 1000
MAX_NUM_FEATURES = 3000
MAX_FRAME_DELTA = 10

fig = None
p1, p2, p3, p4 = None, None, None, None
cbar1, cbar2, cbar3 = None, None, None

img_window = list()

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

    #Detect keypoints
    detector = get_detector()
    prev_kps = detect_keypoints(method, detector, img1)

    # Undistort points
    prev_kps = cv2.undistortPoints(prev_kps, INTRINSIC_MATRIX, DISTORTION_COEFFS, R=None, P=INTRINSIC_MATRIX).reshape(-1, 2)

    # Lukas Kanade optical flow
    prev_kps, cur_kps, diff = KLT_featureTracking(img1, img2, prev_kps)

    # Fit essential matrix
    if USE_ONLY_E:
        E, mask = cv2.findEssentialMat(prev_kps, cur_kps, INTRINSIC_MATRIX, cv2.RANSAC, 0.99, 1, None)
    else:
        # Assume planar motion model
        H, mask = cv2.findHomography(prev_kps, cur_kps, cv2.RANSAC, 1.0)

        # Find points that belong to that model
        prev_kps_hom = prev_kps[mask.ravel().astype(bool)]
        cur_kps_hom = cur_kps[mask.ravel().astype(bool)]

        # Find essential matrix of this model
        E, mask = cv2.findEssentialMat(prev_kps_hom, cur_kps_hom, INTRINSIC_MATRIX, cv2.LMEDS, 0.99, 1, mask)

    if USE_RECOVER_POSE:
        # Recover pose
        _, rot, tran, mask_cheirality = cv2.recoverPose(E, prev_kps, cur_kps, INTRINSIC_MATRIX, None)

        # Only take points from right direction
        prev_kps = prev_kps[mask_cheirality.ravel().astype(bool)]
        cur_kps = cur_kps[mask_cheirality.ravel().astype(bool)]

        # Stack to homography
        homography = np.hstack((rot, tran))

        # Projection matrices
        P0 = np.dot(INTRINSIC_MATRIX, np.eye(3, 4))
        P1 = np.dot(INTRINSIC_MATRIX, homography)

        print(homography)

    else:
        homography = get_homography(E, prev_kps, cur_kps)

        # Projection matrices
        P0 = np.dot(INTRINSIC_MATRIX, np.eye(3, 4))
        P1 = np.dot(INTRINSIC_MATRIX, homography[0])

    # Triangulate with all the keypoints
    if prev_kps.shape[0] > 0 and cur_kps.shape[0] > 0:
        points3d = cv2.triangulatePoints(P0, P1, np.transpose(prev_kps), np.transpose(cur_kps))
        points3d = cv2.convertPointsFromHomogeneous(points3d.T)
    else:
        points3d = None
    # Show images
    img1_rgb = visualize_matches(img1_rgb, prev_kps, cur_kps)
    img2_rgb = visualize_matches(img2_rgb, cur_kps, prev_kps)

    # Stack images
    img_vis = np.hstack((img1_rgb, img2_rgb))
    #cv2.imshow("img", img_vis)
    #cv2.waitKey(0)

    global fig, p1, p2, p3, p4, cbar1, cbar2, cbar3
    fig, p1, p2, p3, p4, cbar1, cbar2, cbar3 = plot_3d(points3d, img_vis, fig, p1, p2, p3, p4, cbar1, cbar2, cbar3)

    return len(prev_kps)

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
        keypoints = np.array([x.pt for x in keypoints], dtype=np.float32).reshape((-1, 2))

    return keypoints

def resize_imgs(img1, img2):
    img1 = cv2.resize(img1, RESIZED_IMAGE)
    img2 = cv2.resize(img2, RESIZED_IMAGE)
    return img1, img2

def grayscale_imgs(img1, img2):
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    return img1, img2

def visualize_matches(img, prev_kps, cur_kps):
    THICKNESS = 1
    GREEN = (0, 255, 0)
    for i in range(min(cur_kps.shape[0], prev_kps.shape[0])):
        img = cv2.arrowedLine(img, tuple(cur_kps[i]), tuple(prev_kps[i]), GREEN, THICKNESS)
        #img = cv2.circle(img, tuple(cur_kps[i]), 1, (0,0,255), 1)
        #img = cv2.circle(img, tuple(prev_kps[i]), 1, (255, 0, 0), 1)

    return img

def get_detector():
    if method == 'FAST':
        detector = cv2.FastFeatureDetector_create(threshold=FAST_THRESHOLD, nonmaxSuppression=True)
    elif method == 'ORB':
        detector = cv2.ORB_create(nfeatures=max_num_features)
    elif method == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create(max_num_features)
    elif method == 'SURF':
        detector = cv2.xfeatures2d.SURF_create(max_num_features)
    elif method == 'SHI-TOMASI':
        detector = None
    elif method == 'REGULAR_GRID':
        regular_grid_max_pts = max_num_features
    else:
        print("Unknown method")
    return detector

def plot_3d(points3d, img_vis, fig, p1, p2, p3, p4, cbar1, cbar2, cbar3):
    FIGSIZE = (15,12)
    DPI = 100
    PLOT_LIM = 0.5

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
            y.append(-points3d[i][0][0])
            z.append(-points3d[i][0][1])
            d.append(np.sqrt(points3d[i][0][0]**2 + points3d[i][0][1]**2 + points3d[i][0][2]**2))

        p1.cla()
        im = p1.scatter(y, z, c=d, vmin=np.min(d), vmax=np.max(d))
        if cbar1 is not None:
            cbar1.remove()
        cbar1 = fig.colorbar(im, ax=p1)
        p1.set_title("front")

        p2.cla()
        im = p2.scatter(x, y, c=d, vmin=np.min(d), vmax=np.max(d))
        if cbar2 is not None:
            cbar2.remove()
        cbar2 = fig.colorbar(im, ax=p2)
        p2.set_title("top")

        p3.cla()
        im = p3.scatter(x, y, z, c=d, vmin=np.min(d), vmax=np.max(d))
        if cbar3 is not None:
            cbar3.remove()
        cbar3 = fig.colorbar(im, ax=p3)

    p4.imshow(img_vis[...,::-1])

    plt.pause(0.1)

    return fig, p1, p2, p3, p4, cbar1, cbar2, cbar3


def KLT_featureTracking(prev_img, cur_img, prev_fts):
    """Feature tracking using the Kanade-Lucas-Tomasi tracker.
    """

    # Feature Correspondence with Backtracking Check
    MIN_MATCHING_DIFF = 1
    kp2, status, error = cv2.calcOpticalFlowPyrLK(prev_img, cur_img, prev_fts, None, **lk_params)
    kp1, status, error = cv2.calcOpticalFlowPyrLK(cur_img, prev_img, kp2, None, **lk_params)

    d = abs(prev_fts - kp1).reshape(-1, 2).max(-1)  # Verify the absolute difference between feature points
    good = d < MIN_MATCHING_DIFF

    # Error Management
    if len(d) == 0:
        print('No point correspondance.')
    elif list(good).count(True) <= 5:  # If less than 5 good points, it uses the features obtain without the backtracking check
        print('Few point correspondances')
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

def get_homography(E, prev_kps, cur_kps):
    R1, R2, t = cv2.decomposeEssentialMat(E)

    # this gives us four canidate homgographies [R1,t], [R1,−t], [R2,t], [R2,−t].
    # we create only two canidates. if the best absolute score is negative, we now we must use
    # the negative translation vector

    candidates = (np.hstack((R1, t)), np.hstack((R1, -t)), np.hstack((R2, t)), np.hstack((R2, -t)))
    scores = []
    masks = []
    forward_axis = 2
    for candidate_homo in candidates:
        points3d = cv2.triangulatePoints(np.eye(3, 4), candidate_homo, prev_kps.T, cur_kps.T)
        points3d = cv2.convertPointsFromHomogeneous(points3d.T)

        scores.append((points3d[:, 0, forward_axis] > 0).sum())
        masks.append(points3d[:, 0, forward_axis] > 0)

    idx = int(np.argmax(scores))

    return candidates[idx], masks[idx]

def frame_selector(len_kps, img_id_1, img_id_2):
    if len_kps < MIN_FEATURES:
        img_id_2 += 1

    elif len_kps > MAX_FEATURES:
        img_id_1 += 1

    else:
        img_id_1 += 1
        img_id_2 += 1

    if img_id_1 == img_id_2:
        img_id_2 += 1

    print(img_id_1, img_id_2, img_id_2-img_id_1, len_kps)

    return img_id_1, img_id_2

def adaptive_step(len_prev_kps, curr_img, prev_img, new_img):
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

        elif len_prev_kps > MAX_NUM_FEATURES and len(img_window) > 1:
            # We observed too many features, make
            # window smaller again
            img_window.pop(0)
            prev_img = img_window.pop(0)

        else:
            # We observed enough features, advance normally
            prev_img = img_window.pop(0)

        print(len(img_window), len_prev_kps)

        return curr_img, prev_img

def load_img(id):
    img_file_path = DEFAULT_DATASET / "imgs" / f"img_{id:04d}.jpg"
    with open(img_file_path, 'rb') as fp:
            img_data = fp.read()
    img = cv2.imdecode(np.frombuffer(img_data, dtype=np.int8), flags=cv2.IMREAD_COLOR)
    return img


if __name__ == '__main__':
    TIME_BASED = True
    SAVE_FOLDER = "/home/marco/Downloads/vis"
    SAVE = False

    incr = 1
    img_id_1 = 200
    img_id_2 = img_id_1 + incr
    img_pair = [f"img_{img_id_1:04d}.jpg", f"img_{img_id_2:04d}.jpg"]
    counter = 0
    len_kps = 0

    prev_img = load_img(img_id_1)
    curr_img = load_img(img_id_2)
    img_cnt = img_id_1

    while True:
        if TIME_BASED:
            img_cnt += 1
            new_img = load_img(img_cnt)
            curr_img, prev_img = adaptive_step(len_kps, curr_img, prev_img, new_img)
            curr_img = new_img
        else:
            s = input("")
            if s == ".":
                img_id += incr
            elif s == ",":
                img_id -= incr

        img_pair = [f"img_{img_id_1:04d}.jpg", f"img_{img_id_2:04d}.jpg"]
        len_kps = main(prev_img, curr_img)



        if SAVE:
            counter += 1
            plt.savefig(Path(SAVE_FOLDER) / f"img_{counter:04d}.jpg")

    # ffmpeg -r 5 -i "img_%04d.jpg" -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" out.mp4
